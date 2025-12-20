import torch as th
from torch import nn
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from sb3_contrib.common.recurrent.policies import RecurrentActorCriticPolicy

class AsymmetricFeatureExtractor(BaseFeaturesExtractor):
    """
    Dynamic Feature Extractor that builds networks based on provided architecture.
    """
    def __init__(
        self, 
        observation_space: spaces.Box, 
        blind_dim: int, 
        features_dim: int = 256,
        actor_arch: list = None,
        critic_arch: list = None
    ):
        # Standard initialization with the target output size (must match LSTM size)
        super().__init__(observation_space, features_dim=features_dim)
        
        self.blind_dim = blind_dim
        full_dim = observation_space.shape[0]

        # Default architectures if none provided
        if actor_arch is None: actor_arch = [64]
        if critic_arch is None: critic_arch = [64]

        # --- Build Actor Network (Blind) ---
        self.actor_net = self._build_mlp(self.blind_dim, actor_arch, features_dim)
        
        # --- Build Critic Network (Privileged) ---
        self.critic_net = self._build_mlp(full_dim, critic_arch, features_dim)

    def _build_mlp(self, input_dim, hidden_layers, output_dim):
        """Helper to create a dynamic MLP."""
        layers = []
        last_dim = input_dim
        
        for h_dim in hidden_layers:
            layers.append(nn.Linear(last_dim, h_dim))
            layers.append(nn.Tanh())
            last_dim = h_dim
        
        # Final layer maps to the LSTM size (features_dim)
        layers.append(nn.Linear(last_dim, output_dim))
        layers.append(nn.Tanh())
        
        return nn.Sequential(*layers)

    def forward(self, observations: th.Tensor) -> tuple[th.Tensor, th.Tensor]:
        # 1. Actor Slicing: Blind
        actor_input = observations[..., :self.blind_dim]
        actor_features = self.actor_net(actor_input)
        
        # 2. Critic Slicing: Privileged
        critic_input = observations
        critic_features = self.critic_net(critic_input)
        
        return actor_features, critic_features

class ExtractActor(nn.Module):
    """Wrapper to extract only the Actor tensor from the tuple."""
    def __init__(self, extractor):
        super().__init__()
        self.extractor = extractor
    def forward(self, obs):
        return self.extractor(obs)[0]

class ExtractCritic(nn.Module):
    """Wrapper to extract only the Critic tensor from the tuple."""
    def __init__(self, extractor):
        super().__init__()
        self.extractor = extractor
    def forward(self, obs):
        return self.extractor(obs)[1]

class AsymmetricLstmPolicy(RecurrentActorCriticPolicy):
    """
    Asymmetric Policy that patches the extractors to be compatible with SB3 logic.
    """
    def __init__(
        self, 
        observation_space, 
        action_space, 
        lr_schedule, 
        share_features_extractor=True, # Catch this arg
        **kwargs
    ):
        # 1. Extract custom args
        blind_dim = kwargs.pop("blind_dim", None)
        # We look for a dictionary 'extractor_arch' containing 'pi' and 'vf' lists
        extractor_arch = kwargs.pop("extractor_arch", {})
        
        if blind_dim is None:
            raise ValueError("AsymmetricLstmPolicy requires 'blind_dim'")

        # 2. Determine LSTM Size (needed to tell extractor how big the final output is)
        # We check if user passed lstm_hidden_size, otherwise default to 256
        lstm_hidden_size = kwargs.get("lstm_hidden_size", 256)

        # 3. Configure Extractor
        kwargs["features_extractor_class"] = AsymmetricFeatureExtractor
        kwargs["features_extractor_kwargs"] = dict(
            blind_dim=blind_dim,
            features_dim=lstm_hidden_size, # Output of extractor matches LSTM input
            actor_arch=extractor_arch.get("pi", [64]), # Default [64]
            critic_arch=extractor_arch.get("vf", [64]) # Default [64]
        )
        
        # 4. Force share_features_extractor=False
        super().__init__(
            observation_space, 
            action_space, 
            lr_schedule, 
            share_features_extractor=False, 
            **kwargs
        )

        # 5. Apply Wrappers
        self.pi_features_extractor = ExtractActor(self.pi_features_extractor)
        self.vf_features_extractor = ExtractCritic(self.vf_features_extractor)