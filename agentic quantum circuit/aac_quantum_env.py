"""
This module defines the AACQuantumCircuitEnv, a wrapper around the QuantumCircuitEnv
that adapts it for an Asymmetric Actor-Critic (AAC) architecture.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from quantum_circuit_env import QuantumCircuitEnv

class AACQuantumCircuitEnv(gym.Env):
    """
    A wrapper for the QuantumCircuitEnv that provides separate observations
    for an actor and a critic, suitable for an Asymmetric Actor-Critic (AAC)
    architecture.

    - The actor receives a partial, realistic observation (the number of detected photons).
    - The critic receives a privileged, full observation (the quantum state vector).
    """
    def __init__(self, **kwargs):
        """
        Initializes the AACQuantumCircuitEnv.

        Args:
            **kwargs: Keyword arguments to be passed to the underlying QuantumCircuitEnv.
        """
        self.env = QuantumCircuitEnv(**kwargs)
        self.cutoff_dim = self.env.cutoff_dim

        # Define the observation space as a dictionary
        self.observation_space = spaces.Dict({
            "actor": spaces.Box(low=0, high=self.cutoff_dim, shape=(1,), dtype=np.float32),
            "critic": self.env.observation_space
        })

        # The action space is the same as the underlying environment
        self.action_space = self.env.action_space

    def reset(self, seed=None, options=None):
        """
        Resets the environment and returns the initial observation dictionary.

        Returns:
            tuple: A tuple containing the initial observation dictionary and info dictionary.
        """
        obs, info = self.env.reset(seed=seed, options=options)
        
        # Initial actor observation: 0 detected photons
        actor_obs = np.array([0.0], dtype=np.float32)
        
        # Critic observation is the full state vector
        critic_obs = obs
        
        return {"actor": actor_obs, "critic": critic_obs}, info

    def step(self, action):
        """
        Steps the environment with the given action.

        Args:
            action: The action to take.

        Returns:
            tuple: A tuple containing the observation dictionary, reward, terminated flag,
                   truncated flag, and info dictionary.
        """
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # Actor observation is the number of detected photons
        actor_obs = np.array([info.get('detected_photons', 0.0)], dtype=np.float32)
        
        # Critic observation is the full state vector
        critic_obs = obs
        
        return {"actor": actor_obs, "critic": critic_obs}, reward, terminated, truncated, info

    def render(self):
        """Renders the environment."""
        return self.env.render()

    def close(self):
        """Closes the environment."""
        return self.env.close()
