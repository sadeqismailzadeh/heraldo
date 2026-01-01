import gymnasium as gym
from gymnasium import spaces
import numpy as np
import matplotlib.pyplot as plt

class HypersphereNavigationEnv(gym.Env):
    """
    A robust, deterministic surrogate environment for quantum control PPO tuning.
    
    State: Normalized vector on N-dimensional hypersphere (simulating a pure quantum state).
    Action: M-dimensional continuous vector (simulating control voltages).
    Dynamics: 
        s' = normalize(s + B@action + drift)
        where B is a fixed random projection matrix and drift is a fixed vector.
        This simulates the interplay between control fields and environmental noise/drift.
    """
    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(
        self,
        n_dims=50,
        m_actions=4,
        max_steps=50,
        target_fidelity=0.99,
        terminate_on_success=True,
        drift_scale=0.05,
        action_scale=0.1,
        seed=42
    ):
        super().__init__()
        self.n_dims = n_dims
        self.m_actions = m_actions
        self.max_steps = max_steps
        self.target_fidelity = target_fidelity
        self.terminate_on_success = terminate_on_success
        self.drift_scale = drift_scale
        self.action_scale = action_scale
        
        # Use a fixed generator for the structural elements of the environment
        # to ensure the "physics" (B matrix, drift vector) are consistent across resets.
        rng_struct = np.random.default_rng(seed)
        
        # 1. Target State (The "North Pole")
        # In a real quantum env, this might be a specific Fock state or GKP state.
        self.target_state = np.zeros(n_dims, dtype=np.float32)
        self.target_state[0] = 1.0
        
        # 2. Action Projection Matrix (Fixed)
        # Simulates how M control parameters couple to the N-dimensional Hilbert space.
        self.B = rng_struct.standard_normal((n_dims, m_actions)).astype(np.float32)
        
        # Normalize columns of B for consistent scaling
        self.B /= np.linalg.norm(self.B, axis=0)
        
        # 3. Drift Vector (Fixed)
        # Represents deterministic decoherence or unitary drift.
        # We make it orthogonal to the target to ensure it pushes the agent "sideways"
        # rather than helping it.
        drift = rng_struct.standard_normal(n_dims).astype(np.float32)
        drift[0] = 0 # Ensure orthogonality to target
        self.drift_vector = drift / np.linalg.norm(drift)

        # Spaces
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(m_actions,), dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(n_dims,), dtype=np.float32
        )
        
        self.current_step = 0
        self.state = None

    def set_difficulty(self, fidelity):
        """
        Curriculum Learning Interface.
        Allows the CurriculumCallback to adjust the difficulty dynamically.
        """
        self.target_fidelity = float(fidelity)
        return self.target_fidelity

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # We allow a random start state to facilitate exploration,
        # but the environment dynamics (B, drift) remain constant.
        
        # Generate random state
        state = self.np_random.standard_normal(self.n_dims).astype(np.float32)
        state /= np.linalg.norm(state)
        
        # Optional: Start in the "Southern Hemisphere" (opposite to target)
        # to guarantee a minimum difficulty.
        if np.dot(state, self.target_state) > 0:
            state *= -1.0

        self.state = state
        self.current_step = 0
        
        return self.state.copy(), {}

    def _calculate_shaped_val(self, fidelity):
        """Mimics the specific shaping in LogFidelityReward."""
        # Shaping formula: ((F^2) * -log10(1-F))^2
        # Clamp fidelity to avoid log(0)
        safe_fid = min(fidelity, 1.0 - 1e-4)
        infidelity = 1.0 - safe_fid
        log_term = -np.log10(infidelity)
        return ((safe_fid**2) * log_term)**2

    def step(self, action):
        self.current_step += 1
        
        # Clip action
        action = np.clip(action, -1.0, 1.0).astype(np.float32)
        
        # 1. Dynamics Update
        # S_new = S_old + (Scale * B @ Action) + (Drift_Scale * Drift)
        control_force = self.action_scale * (self.B @ action)
        drift_force = self.drift_scale * self.drift_vector
        
        new_unscaled = self.state + control_force + drift_force
        
        # Normalize to stay on Hypersphere
        self.state = new_unscaled / np.linalg.norm(new_unscaled)
        
        # 2. Fidelity Calculation
        # Fidelity = |<psi|target>|^2. For real vectors: (dot)^2
        overlap = np.dot(self.state, self.target_state)
        fidelity = float(overlap ** 2)
        
        # 3. Reward Calculation (Identical to LogFidelityReward)
        max_val = self._calculate_shaped_val(1.0) # Reference max value
        shaped_val = self._calculate_shaped_val(fidelity)
        
        # Base reward: negative penalty for being away from max
        reward = shaped_val - max_val
        
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