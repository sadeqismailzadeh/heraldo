import gymnasium as gym
from gymnasium import spaces
import numpy as np
import matplotlib.pyplot as plt

class HypersphereNavigationEnv(gym.Env):
    """
    A robust, deterministic surrogate environment for quantum control PPO tuning.
    
    Mimicry of Quantum Mechanics:
    -----------------------------
    1. State: Normalized vector on N-dimensional hypersphere (Pure State |psi>).
    2. Dynamics: Unitary Evolution via Rotations.
       Instead of adding vectors, we apply a sequence of Givens Rotations (planes).
       This preserves the norm exactly (Unitary) and introduces non-commutativity.
       
       s_{t+1} = R_drift * [ Product_k R_k(theta_k) ] * s_t
       
       where theta_k = action_scale * action[k].
       
    This creates a control landscape that is curved, periodic, and non-linear,
    much closer to optimizing gate parameters in a quantum circuit than simple vector addition.
    """
    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(
        self,
        n_dims=50,
        m_actions=3,  # Ignored, fixed to 3 for [mu, theta, sigma]
        max_steps=50,
        target_fidelity=0.99,
        terminate_on_success=True,
        drift_scale=0.02, # Small constant rotation per step
        action_scale=0.3, # Max rotation angle per step (radians)
        seed=42
    ):
        super().__init__()
        self.n_dims = n_dims
        self.m_actions = 3
        self.max_steps = max_steps
        self.target_fidelity = target_fidelity
        self.terminate_on_success = terminate_on_success
        self.drift_scale = drift_scale
        self.action_scale = action_scale 
        
        # Use a fixed generator for the structural elements (The "Physics")
        rng_struct = np.random.default_rng(seed)
        
        # 1. Target State (e.g. Fock |0>)
        self.target_state = np.zeros(n_dims, dtype=np.float32)
        self.target_state[0] = 1.0
        
        # 2. Define "Gates" (Control Hamiltonian Axes)
        # STAR TOPOLOGY: Connect index 0 (Target) to all other indices 1..N-1
        self.gate_planes = [(0, i) for i in range(1, n_dims)]
            
        # 3. Define Drift (Hamiltonian Drift)
        # Drift acts on the target index to force the agent to actively correct
        self.drift_plane = (0, 1) 

        # Spaces
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(3,), dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(n_dims,), dtype=np.float32
        )
        
        self.current_step = 0
        self.state = None

    def set_difficulty(self, fidelity):
        """Curriculum Learning Interface."""
        self.target_fidelity = float(fidelity)
        return self.target_fidelity

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Start with a random state
        state = self.np_random.standard_normal(self.n_dims).astype(np.float32)
        state /= np.linalg.norm(state)
        
        # Force start "far away" (dot product < 0) to ensure non-trivial episode
        if np.dot(state, self.target_state) > 0.1:
             state *= -1.0
             
        self.state = state
        self.current_step = 0
        
        return self.state.copy(), {}
    
    def _apply_rotation(self, vec, u, v, theta):
        """
        Applies a Givens rotation in the (u, v) plane by angle theta.
        This is an O(1) update (only 2 indices change), representing a sparse unitary gate.
        """
        c = np.cos(theta)
        s = np.sin(theta)
        
        val_u = vec[u]
        val_v = vec[v]
        
        vec[u] = c * val_u - s * val_v
        vec[v] = s * val_u + c * val_v
        return vec

    def _calculate_shaped_val(self, fidelity):
        """Mimics the specific shaping in LogFidelityReward."""
        safe_fid = min(fidelity, 1.0 - 1e-4)
        infidelity = 1.0 - safe_fid
        log_term = -np.log10(infidelity)
        return ((safe_fid))**2

    def step(self, action):
        self.current_step += 1
        
        # Clip action
        action = np.clip(action, -1.0, 1.0).astype(np.float32)
        
        # Parse Action: [mu, theta, sigma]
        # 1. Map mu from [-1, 1] to [0, n_dims-1]
        c_mu = (action[0] + 1.0) / 2.0 * (self.n_dims - 1)
        
        # 2. Map theta from [-1, 1] (scaled by action_scale)
        c_theta = action[1] * self.action_scale
        
        # 3. Map sigma from [-1, 1] to [0.5, 5.0]
        c_sigma = 0.5 + (action[2] + 1.0) / 2.0 * (4.5)

        # 1. Dynamics: Apply Variable Focus Scanning
        # Apply rotations to all star-topology planes based on Gaussian weights
        for (u, v) in self.gate_planes:
            # u is always 0, v is the other dimension index i
            # Gaussian weight based on distance of v from mu
            weight = np.exp(-((v - c_mu) ** 2) / (2 * c_sigma ** 2))
            angle = c_theta * weight
            self.state = self._apply_rotation(self.state, u, v, angle)
            
        # 2. Dynamics: Apply Drift (The "Hamiltonian Evolution")
        self.state = self._apply_rotation(self.state, *self.drift_plane, self.drift_scale)
        
        # Note: No manual normalization needed! Rotations preserve norm.
        # But purely for numerical stability over long episodes:
        if self.current_step % 10 == 0:
             self.state /= np.linalg.norm(self.state)
        
        # 3. Fidelity Calculation
        # Fidelity = |<psi|target>|^2
        overlap = np.dot(self.state, self.target_state)
        fidelity = float(overlap ** 2)
        
        # 4. Reward Calculation
        max_val = self._calculate_shaped_val(1.0) # Reference max value
        shaped_val = self._calculate_shaped_val(fidelity)
        
        # Base reward: negative penalty for being away from max
        # reward = shaped_val - max_val
        reward = shaped_val
        terminated = False
        is_success = fidelity >= self.target_fidelity
        
        if is_success:
            if self.terminate_on_success:
                terminated = True
            # Success Bonus
            reward = 1.0 * max_val + 9.0 * shaped_val
        
        # Normalize reward
        reward /= (11.0 * max_val)
        
        truncated = (self.current_step >= self.max_steps)
        
        info = {
            "fidelity": fidelity,
            "is_success": is_success,
            "target_fidelity": self.target_fidelity
        }
        
        return self.state.copy(), reward, terminated, truncated, info

def plot_hypersphere_trajectory(fidelity_history, title="Hypersphere Trajectory"):
    """
    Helper to visualize a single episode's fidelity.
    
    Args:
        fidelity_history (list or np.array): Sequence of fidelity values.
        title (str): Plot title.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(fidelity_history, label="Fidelity", linewidth=2, color='teal')
    
    # Plot guide lines
    plt.axhline(1.0, color='grey', linestyle='--', alpha=0.5, label='Max Fidelity')
    if len(fidelity_history) > 0:
         plt.axhline(max(fidelity_history), color='orange', linestyle=':', alpha=0.8, label=f'Max Reached: {max(fidelity_history):.4f}')

    plt.xlabel("Step")
    plt.ylabel("Fidelity |<s|target>|^2")
    plt.title(title)
    plt.ylim(-0.05, 1.05)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.show()
