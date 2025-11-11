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
        self.max_steps = self.env.max_steps

        # The action space is the same as the underlying environment
        self.action_space = self.env.action_space
        self.action_dim = self.action_space.shape[0]

        # Photons normalized to [0, 1] by dividing by cutoff_dim
        low_photons = np.array([0.0], dtype=np.float32)
        high_photons = np.array([1.0], dtype=np.float32)

        # Normalized step bounds [0, 1]
        low_step = np.array([0.0], dtype=np.float32)
        high_step = np.array([1.0], dtype=np.float32)

        # Get bounds for the 'previous_action' part from the environment's action space
        low_action = self.action_space.low
        high_action = self.action_space.high

        # Concatenate bounds to form the complete actor observation space
        actor_obs_low = np.concatenate((low_photons, low_step, low_action))
        actor_obs_high = np.concatenate((high_photons, high_step, high_action))

        actor_obs_shape = (2 + self.action_dim,)
        
        # Define the observation space as a dictionary
        self.observation_space = spaces.Dict({
            "actor": spaces.Box(
                low=actor_obs_low,
                high=actor_obs_high,
                shape=actor_obs_shape,
                dtype=np.float32
            ),
            "critic": self.env.observation_space
        })

        # To store the last action taken
        self.previous_action = np.zeros(self.action_dim, dtype=np.float32)

    def reset(self, seed=None, options=None):
        """
        Resets the environment and returns the initial observation dictionary.

        Returns:
            tuple: A tuple containing the initial observation dictionary and info dictionary.
        """
        obs, info = self.env.reset(seed=seed, options=options)

        # Reset the previous action to zeros
        self.previous_action = np.zeros(self.action_dim, dtype=np.float32)

        # Initial actor observation: 0 detected photons (normalized), normalized step 0, and a zero vector for the previous action
        initial_photons = np.array([0.0], dtype=np.float32)  # 0 photons normalized is 0
        initial_step = np.array([0.0], dtype=np.float32)  # Step 0 normalized to 0
        actor_obs = np.concatenate([initial_photons, initial_step, self.previous_action])

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


        # Update the previous_action for the NEXT step's observation
        self.previous_action = action.astype(np.float32)

        # Actor observation is the number of detected photons from the CURRENT step (normalized),
        # combined with the normalized current step, and the action from the PREVIOUS step.
        detected_photons_raw = info.get('detected_photons', 0.0)
        detected_photons_normalized = np.array([detected_photons_raw / self.cutoff_dim], dtype=np.float32)
        # Normalize current step to [0, 1]
        current_step_normalized = np.array([float(self.env.current_step) / self.max_steps], dtype=np.float32)
        actor_obs = np.concatenate([detected_photons_normalized, current_step_normalized, self.previous_action])

        # Critic observation is the full state vector
        critic_obs = obs

        return {"actor": actor_obs, "critic": critic_obs}, reward, terminated, truncated, info

    def render(self):
        """Renders the environment."""
        return self.env.render()

    def close(self):
        """Closes the environment."""
        return self.env.close()
