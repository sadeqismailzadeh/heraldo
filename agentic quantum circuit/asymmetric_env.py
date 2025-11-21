import numpy as np
from gymnasium import spaces
from partially_observable_env import PartiallyObservableQuantumEnv

class AsymmetricTrainingEnv(PartiallyObservableQuantumEnv):
    """
    An environment wrapper for Asymmetric Actor-Critic training.
    
    Observation Space:
    [ --- BLIND PART (For Actor) --- | --- PRIVILEGED PART (For Critic) --- ]
    [ Photons, Actions, Step         | Real/Imag parts of Quantum Ket       ]
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # 1. Determine dimensions
        # The blind part comes from the parent class
        self.blind_dim = self.observation_space.shape[0]
        
        # The privileged part is the full ket (Real + Imag)
        # 2 * cutoff_dim
        self.priv_dim = 2 * self.cutoff_dim
        
        # 2. Expand the observation space to hold BOTH
        self.observation_space = spaces.Box(
            low=-np.inf, # Use loose bounds for the combined vector
            high=np.inf,
            shape=(self.blind_dim + self.priv_dim,),
            dtype=np.float32
        )
        
        print(f"--- Asymmetric Env Initialized ---")
        print(f"Blind Dim (Actor): {self.blind_dim}")
        print(f"Privileged Dim (Critic): {self.priv_dim}")

    def _get_combined_obs(self, blind_obs):
        """Helper to concatenate blind obs with current quantum state."""
        # Get the privileged info (The Ket)
        # We use the helper from QuantumCircuitEnv to get the flattened ket
        privileged_obs = self._ket_to_observation(self.current_ket)
        
        # Glue them together
        return np.concatenate([blind_obs, privileged_obs]).astype(np.float32)

    def reset(self, seed=None, options=None):
        # Get the standard blind observation
        blind_obs, info = super().reset(seed=seed, options=options)
        
        # Append the privileged info
        combined_obs = self._get_combined_obs(blind_obs)
        return combined_obs, info

    def step(self, action):
        # Run physics
        blind_obs, reward, terminated, truncated, info = super().step(action)
        
        # Fix for 'Lagging Action' bug (ensuring we use the fix locally)
        # If your base file already has this, this line is redundant but harmless.
        self._last_action = action 
        
        # Re-construct blind obs with the correct action if needed, 
        # or just use what super() returned if you fixed the base file.
        # Assuming super() returns valid blind_obs.
        
        # Append privileged info
        combined_obs = self._get_combined_obs(blind_obs)
        
        return combined_obs, reward, terminated, truncated, info