"""
This module defines a custom policy for the Asymmetric Actor-Critic (AAC) architecture.
"""

import torch
from torch import nn
from gymnasium import spaces
from typing import Type, Union, List, Dict, Any, Tuple

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor, MlpExtractor
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.distributions import Distribution, DiagGaussianDistribution, CategoricalDistribution


class AsymmetricFeaturesExtractor(BaseFeaturesExtractor):
    """
    Feature extractor for the Asymmetric Actor-Critic architecture.
    It extracts features from a dictionary observation space with 'actor' and 'critic' keys.
    """
    def __init__(self, observation_space: spaces.Dict, actor_features_dim: int = 64, critic_features_dim: int = 256):
        # We have to set features_dim to something, but it will be ignored.
        # The real feature dimensions will be handled by the custom MlpExtractor.
        super().__init__(observation_space, features_dim=1)

        actor_space = observation_space["actor"]
        critic_space = observation_space["critic"]

        # Actor network
        self.actor_net = nn.Sequential(
            nn.Linear(actor_space.shape[0], 128),
            nn.ReLU(),
            nn.Linear(128, actor_features_dim),
            nn.ReLU()
        )

        # Critic network
        self.critic_net = nn.Sequential(
            nn.Linear(critic_space.shape[0], 256),
            nn.ReLU(),
            nn.Linear(256, critic_features_dim),
            nn.ReLU()
        )
        
        # This is not used by the policy but can be useful for debugging
        self.actor_features_dim = actor_features_dim
        self.critic_features_dim = critic_features_dim

    def forward(self, observations: dict) -> Dict[str, torch.Tensor]:
        """
        Forward pass for the feature extractor.
        Returns a dictionary with separate features for the actor and critic.
        """
        return {
            "actor": self.actor_net(observations["actor"]),
            "critic": self.critic_net(observations["critic"])
        }


class AsymmetricMlpExtractor(nn.Module):
    """
    Custom MLP extractor for the Asymmetric Actor-Critic architecture.
    It takes a dictionary of features ("actor" and "critic") and processes them
    with separate networks.
    """
    def __init__(
        self,
        feature_dim: int, # This is not used, but required by the SB3 API
        net_arch: Dict[str, List[int]],
        activation_fn: Type[nn.Module],
        device: torch.device,
        actor_features_dim: int,
        critic_features_dim: int,
    ):
        super().__init__()
        self.actor_net = self._build_net(actor_features_dim, net_arch["pi"], activation_fn)
        self.critic_net = self._build_net(critic_features_dim, net_arch["vf"], activation_fn)
        self.device = device

    def _build_net(self, in_features: int, layer_sizes: List[int], activation_fn: Type[nn.Module]) -> nn.Sequential:
        """Helper to build a simple MLP."""
        layers = []
        last_size = in_features
        for size in layer_sizes:
            layers.append(nn.Linear(last_size, size))
            layers.append(activation_fn())
            last_size = size
        return nn.Sequential(*layers)

    def forward_actor(self, features: Dict[str, torch.Tensor]) -> torch.Tensor:
        return self.actor_net(features["actor"])

    def forward_critic(self, features: Dict[str, torch.Tensor]) -> torch.Tensor:
        return self.critic_net(features["critic"])


class AsymmetricCriticPolicy(ActorCriticPolicy):
    """
    A custom policy for the Asymmetric Actor-Critic (AAC) architecture.
    """
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule,
        net_arch: Dict[str, List[int]] = None,
        activation_fn: Type[nn.Module] = nn.Tanh,
        actor_features_dim: int = 64,
        critic_features_dim: int = 256,
        **kwargs,
    ):
        self.actor_features_dim = actor_features_dim
        self.critic_features_dim = critic_features_dim
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch,
            activation_fn,
            # Pass the custom feature extractor and its kwargs
            features_extractor_class=AsymmetricFeaturesExtractor,
            features_extractor_kwargs=dict(
                actor_features_dim=actor_features_dim,
                critic_features_dim=critic_features_dim,
            ),
            **kwargs,
        )

    def _build_mlp_extractor(self) -> None:
        """
        Create the custom MLP extractor.
        """
        self.mlp_extractor = AsymmetricMlpExtractor(
            0, # feature_dim is not used
            net_arch=self.net_arch,
            activation_fn=self.activation_fn,
            device=self.device,
            actor_features_dim=self.actor_features_dim,
            critic_features_dim=self.critic_features_dim,
        )

    def _build(self, lr_schedule) -> None:
        """
        Build the networks.
        """
        # The features_extractor is already built by the parent's __init__
        # We need to re-assign it to ensure it's our custom one
        self.features_extractor = AsymmetricFeaturesExtractor(
            self.observation_space,
            actor_features_dim=self.actor_features_dim,
            critic_features_dim=self.critic_features_dim,
        )
        self._build_mlp_extractor()

        # Determine the output size of the actor and critic networks
        actor_last_layer_dim = self.net_arch["pi"][-1] if self.net_arch["pi"] else self.actor_features_dim
        critic_last_layer_dim = self.net_arch["vf"][-1] if self.net_arch["vf"] else self.critic_features_dim

        # Create the action distribution networks
        self.action_net, self.log_std = self.action_dist.proba_distribution_net(latent_dim=actor_last_layer_dim)
        self.value_net = nn.Linear(critic_last_layer_dim, 1)

        # Setup optimizer with model parameters
        self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)

    def _get_action_dist_from_latent(self, latent_pi: torch.Tensor) -> Distribution:
        """
        Retrieve action distribution given the latent codes.
        """
        mean_actions = self.action_net(latent_pi)

        if isinstance(self.action_dist, DiagGaussianDistribution):
            return self.action_dist.proba_distribution(mean_actions, self.log_std)
        elif isinstance(self.action_dist, CategoricalDistribution):
            # For discrete actions, action_net directly outputs logits
            return self.action_dist.proba_distribution(mean_actions)
        else:
            raise NotImplementedError("Unsupported action distribution")

    def forward(self, obs: Dict[str, torch.Tensor], deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass in all the networks (actor and critic)
        """
        features = self.features_extractor(obs)
        latent_pi = self.mlp_extractor.forward_actor(features)
        latent_vf = self.mlp_extractor.forward_critic(features)

        # Evaluate the values for the critic
        values = self.value_net(latent_vf)
        
        # Generate actions for the actor
        distribution = self._get_action_dist_from_latent(latent_pi)
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)
        
        return actions, values, log_prob

    def evaluate_actions(self, obs: Dict[str, torch.Tensor], actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate actions according to the current policy,
        given the observations.
        """
        features = self.features_extractor(obs)
        latent_pi = self.mlp_extractor.forward_actor(features)
        latent_vf = self.mlp_extractor.forward_critic(features)
        
        distribution = self._get_action_dist_from_latent(latent_pi)
        log_prob = distribution.log_prob(actions)
        values = self.value_net(latent_vf)
        
        return values, log_prob, distribution.entropy()

    def predict_values(self, obs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Get the estimated values according to the current policy given the observations.
        """
        features = self.features_extractor(obs)
        latent_vf = self.mlp_extractor.forward_critic(features)
        return self.value_net(latent_vf)
