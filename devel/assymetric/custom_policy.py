"""
This module defines a custom recurrent policy for the Asymmetric Actor-Critic (AAC) architecture.
"""
import torch
from torch import nn
from gymnasium import spaces
from typing import Type, List, Dict, Tuple

from sb3_contrib.common.recurrent.policies import RecurrentActorCriticPolicy
from sb3_contrib.common.recurrent.type_aliases import RNNStates
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor, MlpExtractor
from stable_baselines3.common.distributions import Distribution, DiagGaussianDistribution, CategoricalDistribution


class AsymmetricFeaturesExtractor(BaseFeaturesExtractor):
    """
    Feature extractor for the Asymmetric Actor-Critic architecture.
    It extracts features from a dictionary observation space with 'actor' and 'critic' keys.
    The actor's observation is passed through as-is to the LSTM, while the critic's
    observation is processed by an MLP.
    """
    def __init__(self, observation_space: spaces.Dict, critic_features_dim: int = 256):
        super().__init__(observation_space, features_dim=1) # Dummy features_dim

        actor_space = observation_space["actor"]
        critic_space = observation_space["critic"]

        # The actor's features are the raw observations that will be fed into the LSTM.
        self.actor_features_dim = actor_space.shape[0]

        # Critic network
        self.critic_net = nn.Sequential(
            nn.Linear(critic_space.shape[0], 256),
            nn.ReLU(),
            nn.Linear(256, critic_features_dim),
            nn.ReLU()
        )
        self.critic_features_dim = critic_features_dim

    def forward(self, observations: dict) -> Dict[str, torch.Tensor]:
        """
        Forward pass for the feature extractor.
        """
        return {
            "actor": observations["actor"],
            "critic": self.critic_net(observations["critic"])
        }


class AsymmetricRecurrentCriticPolicy(RecurrentActorCriticPolicy):
    """
    A custom recurrent policy for the Asymmetric Actor-Critic (AAC) architecture.
    The actor uses an LSTM to process sequential partial observations, while the
    critic uses an MLP with access to the full state.
    """
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule,
        net_arch: Dict[str, List[int]] = None,
        activation_fn: Type[nn.Module] = nn.Tanh,
        critic_features_dim: int = 256,
        n_lstm_layers: int = 1,
        lstm_hidden_size: int = 64,
        **kwargs,
    ):
        self.critic_features_dim = critic_features_dim
        self.n_lstm_layers = n_lstm_layers
        self.lstm_hidden_size = lstm_hidden_size
        # Pass LSTM related parameters to the parent class
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch,
            activation_fn,
            features_extractor_class=AsymmetricFeaturesExtractor,
            features_extractor_kwargs=dict(critic_features_dim=critic_features_dim),
            n_lstm_layers=n_lstm_layers,
            lstm_hidden_size=lstm_hidden_size,
            **kwargs,
        )
        self._build(lr_schedule)

    def _build(self, lr_schedule) -> None:
        """
        Build the networks.
        """
        # This is called after the feature extractor is created.
        # The actor's input to the LSTM is the raw observation.
        actor_input_dim = self.features_extractor.actor_features_dim
        
        # The critic's input to its MLP is the output of its feature extractor.
        critic_input_dim = self.features_extractor.critic_features_dim

        # Default LSTM architecture
        self.lstm_actor = nn.LSTM(
            input_size=actor_input_dim,
            hidden_size=self.lstm_hidden_size,
            num_layers=self.n_lstm_layers,
            batch_first=True
        )

        # Default MLP architecture for critic
        self.critic_net = nn.Sequential(
            nn.Linear(critic_input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU()
        )

        # Action and value heads
        self.action_net, self.log_std = self.action_dist.proba_distribution_net(latent_dim=self.lstm_hidden_size)
        self.value_net = nn.Linear(64, 1)

        # Setup optimizer with model parameters
        self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)

    def _get_action_dist_from_latent(self, latent_pi: torch.Tensor) -> Distribution:
        mean_actions = self.action_net(latent_pi)
        return self.action_dist.proba_distribution(mean_actions, self.log_std)

    def forward(self, obs: Dict[str, torch.Tensor], lstm_states: Tuple[torch.Tensor, ...], episode_starts: torch.Tensor, deterministic: bool = False):
        """
        Forward pass for the policy.
        """
        features = self.features_extractor(obs)
        actor_features = features["actor"]
        critic_features = features["critic"]

        # Actor (LSTM)
        # Reshape actor_features for LSTM: (batch_size, sequence_length, input_size)
        # Here, sequence_length is 1 as we process one step at a time
        actor_features = actor_features.unsqueeze(1) # Add sequence length dimension
        
        # lstm_states here is (actor_hidden_cell_tuple, critic_hidden_cell_tuple)
        # Since critic is not recurrent, critic_hidden_cell_tuple will be None
        actor_lstm_states, _ = lstm_states # Unpack the outer tuple

        if actor_lstm_states is None:
            # Initialize LSTM states to zeros if None (start of episode)
            hidden_state = torch.zeros(self.n_lstm_layers, actor_features.size(0), self.lstm_hidden_size).to(self.device)
            cell_state = torch.zeros(self.n_lstm_layers, actor_features.size(0), self.lstm_hidden_size).to(self.device)
        else:
            # actor_lstm_states is (hidden_state_tensor, cell_state_tensor)
            hidden_state, cell_state = actor_lstm_states[0], actor_lstm_states[1]
            # Check if batch size matches current input batch size
            if hidden_state.size(1) != actor_features.size(0):
                # Reinitialize if batch size changed (e.g., during training with variable batch sizes)
                hidden_state = torch.zeros(self.n_lstm_layers, actor_features.size(0), self.lstm_hidden_size).to(self.device)
                cell_state = torch.zeros(self.n_lstm_layers, actor_features.size(0), self.lstm_hidden_size).to(self.device)

        latent_pi, (hidden_state, cell_state) = self.lstm_actor(actor_features, (hidden_state, cell_state))
        latent_pi = latent_pi.squeeze(1) # Remove sequence length dimension

        # Repack lstm_states into RNNStates object for return
        # The hidden_state and cell_state from LSTM output are already (n_layers, batch, hidden_size)
        vf_hidden = torch.zeros_like(hidden_state)
        vf_cell = torch.zeros_like(cell_state)
        lstm_states = RNNStates(pi=(hidden_state, cell_state), vf=(vf_hidden, vf_cell))
        
        # Critic (MLP)
        latent_vf = self.critic_net(critic_features)
        
        values = self.value_net(latent_vf)
        distribution = self._get_action_dist_from_latent(latent_pi)
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)
        
        return actions, values, log_prob, lstm_states

    def evaluate_actions(self, obs: Dict[str, torch.Tensor], actions: torch.Tensor, lstm_states: Tuple[torch.Tensor, ...], episode_starts: torch.Tensor):
        """
        Evaluate actions according to the current policy.
        """
        features = self.features_extractor(obs)
        actor_features = features["actor"]
        critic_features = features["critic"]

        # Actor (LSTM)
        actor_features = actor_features.unsqueeze(1) # Add sequence length dimension
        
        if lstm_states is None:
            # Initialize LSTM states to zeros if None (start of episode)
            hidden_state = torch.zeros(self.n_lstm_layers, actor_features.size(0), self.lstm_hidden_size).to(self.device)
            cell_state = torch.zeros(self.n_lstm_layers, actor_features.size(0), self.lstm_hidden_size).to(self.device)
        else:
            # Correctly unpack nested lstm_states
            hidden_state, cell_state = lstm_states.pi[0], lstm_states.pi[1]
            # Check if batch size matches current input batch size
            if hidden_state.size(1) != actor_features.size(0):
                # Reinitialize if batch size changed
                hidden_state = torch.zeros(self.n_lstm_layers, actor_features.size(0), self.lstm_hidden_size).to(self.device)
                cell_state = torch.zeros(self.n_lstm_layers, actor_features.size(0), self.lstm_hidden_size).to(self.device)

        latent_pi, _ = self.lstm_actor(actor_features, (hidden_state, cell_state))
        latent_pi = latent_pi.squeeze(1) # Remove sequence length dimension
        
        # Critic (MLP)
        latent_vf = self.critic_net(critic_features)

        values = self.value_net(latent_vf)
        distribution = self._get_action_dist_from_latent(latent_pi)
        log_prob = distribution.log_prob(actions)
        
        return values, log_prob, distribution.entropy()

    def predict_values(self, obs: Dict[str, torch.Tensor], lstm_states: Tuple[torch.Tensor, ...], episode_starts: torch.Tensor) -> torch.Tensor:
        """
        Get the estimated values according to the current policy given the observations.
        """
        features = self.features_extractor(obs)
        critic_features = features["critic"]
        latent_vf = self.critic_net(critic_features)
        return self.value_net(latent_vf)