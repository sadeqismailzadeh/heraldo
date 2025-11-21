import torch as th
from torch import nn
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from sb3_contrib.common.recurrent.policies import RecurrentActorCriticPolicy

class AsymmetricFeatureExtractor(BaseFeaturesExtractor):
    """
    The internal network that performs the slicing.
    It contains BOTH Actor and Critic networks.
    """
    def __init__(self, observation_space: spaces.Box, blind_dim: int):
        super().__init__(observation_space, features_dim=256)
        
        self.blind_dim = blind_dim
        full_dim = observation_space.shape[0]
        
        # --- Actor Network (Blind) ---
        self.actor_net = nn.Sequential(
            nn.Linear(self.blind_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 256),
            nn.Tanh()
        )
        
        # --- Critic Network (Privileged) ---
        self.critic_net = nn.Sequential(
            nn.Linear(full_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 256),
            nn.Tanh()
        )

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
        if blind_dim is None:
            raise ValueError("AsymmetricLstmPolicy requires 'blind_dim'")

        # 2. Setup Feature Extractor Class
        kwargs["features_extractor_class"] = AsymmetricFeatureExtractor
        kwargs["features_extractor_kwargs"] = dict(blind_dim=blind_dim)
        
        # 3. Force share_features_extractor=False
        # This tells SB3 to create TWO instances of our extractor:
        # self.pi_features_extractor AND self.vf_features_extractor
        super().__init__(
            observation_space, 
            action_space, 
            lr_schedule, 
            share_features_extractor=False, 
            **kwargs
        )

        # 4. THE FIX: Wrap the extractors!
        # Currently, both extractors return (Actor, Critic) tuples.
        # This causes the crash. We wrap them so:
        # - pi_features_extractor returns ONLY Actor Tensor
        # - vf_features_extractor returns ONLY Critic Tensor
        
        self.pi_features_extractor = ExtractActor(self.pi_features_extractor)
        self.vf_features_extractor = ExtractCritic(self.vf_features_extractor)