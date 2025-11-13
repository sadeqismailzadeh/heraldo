import gymnasium as gym
from gymnasium import spaces
import numpy as np

# Import your original, fully observable environment
from quantum_circuit_env import QuantumCircuitEnv

class PartiallyObservableQuantumEnv(QuantumCircuitEnv):
    """
    A partially observable wrapper for the QuantumCircuitEnv.

    This environment inherits the underlying physics simulation from QuantumCircuitEnv
    but provides the agent with a limited, more realistic observation. The agent
    does NOT see the full quantum state.

    The observation consists of three parts, ALL NORMALIZED to the [-1, 1] range:
    1. The detected photon number from the PNR measurement.
    2. The action taken by the agent in the PREVIOUS step.
    3. The current step number within the episode.
    
    The reward mechanism remains the same (dense, fidelity-based) as the parent.
    """
    def __init__(self, **kwargs):
        """
        Initializes the partially observable environment.

        Args:
            **kwargs: Keyword arguments to be passed to the parent QuantumCircuitEnv,
                      e.g., cutoff_dim, max_steps, etc.
        """
        # --- 1. Initialize the parent environment to handle all the physics ---
        super().__init__(**kwargs)
        print("--- Initializing Partially Observable Quantum Environment Wrapper ---")

        # --- 2. Define the NEW, limited observation space with CORRECT bounds ---
        action_dim = self.action_space.shape[0]
        
        # The total size of our observation vector:
        # 1 (for PNR) + action_dim (for last_action) + 1 (for timestep)
        obs_dim = 1 + action_dim + 1
        
        # CORRECTED OBSERVATION SPACE:
        # Since all components of the observation vector are normalized to the
        # range [-1, 1], the observation space must reflect this.
        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(obs_dim,),
            dtype=np.float32
        )
        
        # --- 3. Internal state for tracking the last action ---
        self._last_action = np.zeros(action_dim, dtype=np.float32)

    def _normalize_value(self, value, min_val, max_val):
        """Helper to normalize a value to the [-1, 1] range."""
        return 2.0 * ((value - min_val) / (max_val - min_val)) - 1.0

    def _construct_observation(self, detected_photons, last_action, current_step):
        """
        Constructs the observation vector, normalizing all components.
        """
        # Normalize photon count to [-1, 1]
        norm_photons = self._normalize_value(detected_photons, 0, self.cutoff_dim - 1)
        
        # Normalize the last action to [-1, 1] for each component
        norm_action = self._normalize_value(last_action, self.action_space.low, self.action_space.high)
        
        # Normalize timestep to [-1, 1]
        norm_step = self._normalize_value(current_step, 0, self.max_steps)
        
        return np.concatenate([
            np.array([norm_photons]),
            norm_action,
            np.array([norm_step])
        ]).astype(np.float32)

    def reset(self, seed=None, options=None):
        """
        Resets the environment and returns the initial partial observation.
        """
        # Call the parent's reset to reset the physics simulation.
        super().reset(seed=seed, options=options)
        
        # Reset the last action to zeros
        self._last_action = np.zeros(self.action_space.shape[0], dtype=np.float32)
        
        # The initial observation corresponds to 0 detected photons, a zero action,
        # and step 0.
        initial_observation = self._construct_observation(
            detected_photons=0,
            last_action=self._last_action,
            current_step=0
        )
        
        return initial_observation, {}

    def step(self, action):
        """
        Runs one timestep of the environment's dynamics.
        """
        # Call the parent's step method to run the quantum simulation
        _, reward, terminated, truncated, info = super().step(action)
        
        # Extract the information we need from the info dict
        detected_photons = info.get('detected_photons', 0)
        
        # Construct our new, partial observation
        observation = self._construct_observation(
            detected_photons=detected_photons,
            last_action=self._last_action, # Use the action from the previous step
            current_step=self.current_step
        )
        
        # Update the last action for the *next* step's observation
        self._last_action = action
        
        # Return the results
        return observation, reward, terminated, truncated, info