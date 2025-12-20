"""Base classes for quantum circuit Gymnasium environments."""

# 1. Import the module we need to patch
import scipy.integrate

# 2. Check if the patch is needed to avoid errors
if not hasattr(scipy.integrate, 'simps'):
    print("Monkey patching scipy.integrate: 'simps' not found. Pointing to 'simpson'.")
    # 3. Create the 'simps' attribute and point it to the existing 'simpson' function.
    scipy.integrate.simps = scipy.integrate.simpson
else:
    print("'simps' already exists in scipy.integrate. No patch needed.")


import abc
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import scipy.sparse as sp

# Import Strawberry Fields
import strawberryfields as sf
from strawberryfields import ops
from strawberryfields.ops import *


# disable caching to save memory for large cutoff dims
from quantum_agent.patches.sf_operations_no_cache import disable_fock_caching
disable_fock_caching()

# optimized loss channel
from quantum_agent.patches.monitored_loss_measure_fock_patch import MonitoredLossMeasureFock, patch_fock_backend, decode_measurement_result
patch_fock_backend()

from quantum_agent.patches.beamsplitter_patch import patch_beamsplitter
patch_beamsplitter()

# --- Utility Functions ---


# --- Pure State Fidelity Function ---
def fidelity_pure_state(target_ket, state_ket):
    """Return fidelity between two pure states.

    For two pure states |φ⟩ and |ψ⟩, the fidelity is:
    F(|φ⟩, |ψ⟩) = |⟨φ|ψ⟩|²

    Args:
        target_ket (np.ndarray): Target state vector.
        state_ket (np.ndarray): Current state vector.

    Returns:
        float: Clipped fidelity value in [0, 1].
    """
    target_ket = np.asarray(target_ket, dtype=np.complex128).flatten()
    state_ket = np.asarray(state_ket, dtype=np.complex128).flatten()

    # Fidelity: F = |⟨φ|ψ⟩|²
    overlap = np.vdot(target_ket, state_ket)  # ⟨φ|ψ⟩
    fidelity = np.abs(overlap) ** 2
    return np.clip(fidelity, 0.0, 1.0)


def fidelity_max_rotation(target_ket, state_ket, n_fft=2048):
    """
    Calculates the maximum fidelity between state_ket and target_ket
    optimizing over any global phase space rotation z-rotation R(phi).

    Args:
        target_ket (np.ndarray): Target state vector (Fock basis).
        state_ket (np.ndarray): Current state vector (Fock basis).
        n_fft (int): Resolution of the angle search. Higher = more accurate.
                     2048 is usually plenty for cutoff_dim ~ 25.

    Returns:
        float: The maximum achievable fidelity.
    """
    # Ensure inputs are 1D arrays
    t = np.asarray(target_ket, dtype=np.complex128).flatten()
    s = np.asarray(state_ket, dtype=np.complex128).flatten()

    # Pad to matching lengths if necessary
    max_len = max(len(t), len(s))
    if len(t) < max_len:
        t = np.pad(t, (0, max_len - len(t)))
    if len(s) < max_len:
        s = np.pad(s, (0, max_len - len(s)))

    # 1. Calculate the element-wise product: h[n] = s[n]* . t[n]
    # We conjugate s and not t (or vice versa), the magnitude result is the same.
    h = np.conj(s) * t

    # 2. Use FFT to compute sum(h[n] * e^{-i*n*phi}) for discrete phi
    # Zero-padding (n_fft > len(h)) interpolates the spectrum, effectively
    # searching more angles for a finer resolution.
    fft_values = np.fft.fft(h, n=n_fft)

    # 3. The Fidelity is the square of the maximum magnitude of the overlap
    max_overlap = np.max(np.abs(fft_values))
    return np.clip(max_overlap**2, 0.0, 1.0)


def db_to_r(db_value):
    """Converts squeezing level from Decibels (dB) to the squeezing parameter r."""
    return db_value / (20 * np.log10(np.e))


from quantum_agent.core.interfaces import TargetGenerator, CircuitContext, RewardMechanism


class ModularQuantumEnv(gym.Env):
    """
    A modular quantum optical circuit environment that delegates core responsibilities
    to pluggable modules: TargetGenerator, CircuitContext, and RewardMechanism.
    """
    metadata = {"render_modes": [], "render_fps": 0}

    def __init__(
        self, 
        target_gens,  # Can be a single TargetGenerator or list of TargetGenerators
        circuit_context: CircuitContext,
        reward_mech: RewardMechanism,
        cutoff_dim=25,
        max_steps=10,
        loss_channel=1.0,
        initial_target_fidelity=0.8,
        **kwargs
    ):
        super().__init__()
        self.cutoff_dim = cutoff_dim
        self.max_steps = max_steps
        self.loss_channel = loss_channel
        
        # Composition: normalize target_gens to a list
        if isinstance(target_gens, (list, tuple)):
            self.target_gens = list(target_gens)
        else:
            self.target_gens = [target_gens]
        
        self.circuit_context = circuit_context
        self.reward_mech = reward_mech
        
        # --- Internal State ---
        self.current_step = 0
        self.current_ket = None
        self.past_ket = None
        self.min_inner_product = 1.0
        self.current_state = None
        self.past_fidelity = 0.0
        
        # Initialize
        self.action_keys = self.circuit_context.action_keys
        self.action_ranges = self.circuit_context.action_ranges
        self.action_space = self.circuit_context.get_action_space()
        
        # Observation space (Standardized)
        obs_size = 2 * self.cutoff_dim
        self._obs_buffer = np.zeros(obs_size, dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=-1, high=1, shape=(obs_size,), dtype=np.float32)

        # Lazy load target states (in case it's expensive)
        self.target_kets = [target_gen.get_target_ket(self.cutoff_dim) for target_gen in self.target_gens]
        self.target_fidelity = initial_target_fidelity

        # --- Precompute operators for optional metrics ---
        self._precompute_quadrature_operators()

        # Compute target non-Gaussianity (use first target)
        self.target_ng_score = self.compute_non_gaussianity(self.target_kets[0])
        print(f"target ng = {self.target_ng_score:.2f}")



    def set_difficulty(self, fidelity):
        """
        Explicit setter for curriculum learning.
        This ensures the attribute is updated within the subprocess.
        """
        self.target_fidelity = float(fidelity)
        return self.target_fidelity

    def _ket_to_observation(self, state_ket):
        """Converts a pure state ket into a normalized observation vector."""
        if state_ket is None:
            self._obs_buffer.fill(0)
            return self._obs_buffer.copy()

        state_ket = state_ket.flatten()

        if len(state_ket) < self.cutoff_dim:
            padded = np.zeros(self.cutoff_dim, dtype=np.complex128)
            padded[:len(state_ket)] = state_ket
            state_ket = padded
        elif len(state_ket) > self.cutoff_dim:
            state_ket = state_ket[:self.cutoff_dim]

        self._obs_buffer[:self.cutoff_dim] = state_ket.real
        self._obs_buffer[self.cutoff_dim:] = state_ket.imag
        return self._obs_buffer.copy()

    def _denormalize_to_dict(self, action_array):
        """Maps [-1, 1] array to a dictionary of named physical values."""
        denorm_dict = {}
        for i, key in enumerate(self.action_keys):
            low, high = self.action_ranges[key]
            # Scale from [-1, 1] to [low, high]
            val = ((high - low) / 2.0) * action_array[i] + ((high + low) / 2.0)
            denorm_dict[key] = np.clip(val, low, high)
        return denorm_dict

    def reset(self, seed=None, options=None):
        """Resets the environment to an initial state."""
        super().reset(seed=seed)
        
        self.eng = sf.Engine("fock", backend_options={"cutoff_dim": self.cutoff_dim})
        self.current_step = 0
        self.min_inner_product = 1.0
        
        # Delegate to Circuit Strategy
        prog = self.circuit_context.build_reset_program()
        result = self.eng.run(prog)
        self.current_state = result.state
        self.current_ket = self.circuit_context._get_current_ket(self.current_state)
        self.past_ket = self.current_ket
        
        inner_product = np.abs(np.vdot(self.current_ket, self.current_ket))
        self.min_inner_product = min(self.min_inner_product, inner_product)
        
        return self._ket_to_observation(self.current_ket), {}

    def step(self, action):
        """Executes one time step within the environment."""
        self.current_step += 1

        denorm_action_dict = self._denormalize_to_dict(action)
        
        # 1. Delegate Circuit Execution
        programs  = self.circuit_context.build_step_program(denorm_action_dict)

        # Safety check: Ensure programs is iterable (list), even if single program returned 
        if isinstance(programs, sf.Program):
            programs = [programs]
    
        result = None

        
        # 2. Execute sequentially and monitor state between stages
        for prog in programs:
            result = self.eng.run(prog)
            
            # Update state tracking immediately after every sub-program
            self.current_state = result.state
            self.current_ket = self.circuit_context._get_current_ket(self.current_state)
            
            # Check inner product (Critical for the "split" logic)
            full_ket = result.state.ket()
            inner_product = np.abs(np.vdot(full_ket.flatten(), full_ket.flatten()))
            self.min_inner_product = min(self.min_inner_product, inner_product)

        # Get the current ket from the result
        self.current_state = result.state
        self.current_ket = self.circuit_context._get_current_ket(self.current_state)
        
        inner_product = np.abs(np.vdot(self.current_ket, self.current_ket))
        self.min_inner_product = min(self.min_inner_product, inner_product)
        
        # 2. Delegate Reward Calculation
        # Pass context like step number, max steps, etc.
        step_info = {
            "step": self.current_step, 
            "max_steps": self.max_steps,
            "samples": result.samples if hasattr(result, 'samples') else None,
            "result": result,
            "past_ket": self.past_ket,
            "ng_score": self.compute_non_gaussianity(self.current_ket)
        }
        
        reward, terminated, info = self.reward_mech.compute(
            self.current_ket, 
            self.target_kets,  # Pass all targets for reward calculation
            step_info,
            self.target_fidelity 
        )
        
        truncated = self.current_step >= self.max_steps
        obs = self._ket_to_observation(self.current_ket)
        
        self.past_ket = self.current_ket
        
        if terminated or truncated:
            info['final_ket'] = self.current_ket
            info['min_inner_product'] = self.min_inner_product
            info['episode_len'] = self.current_step
        
        return obs, reward, terminated, truncated, info

    def render(self):
        """Not implemented."""
        pass

    def close(self):
        """Clean up resources."""
        pass

    # --- Gadget-Specific Methods (available to all subclasses) ---
    def _precompute_quadrature_operators(self):
        """Pre-computes sparse quadrature operators for non-Gaussianity calculations."""
        dim = self.cutoff_dim
        sqrt_n = np.sqrt(np.arange(1, dim))

        self.a_op_sparse = sp.diags([sqrt_n], [1], shape=(dim, dim), format='csr')
        self.a_dag_op_sparse = self.a_op_sparse.T

        self.x_op_sparse = (self.a_op_sparse + self.a_dag_op_sparse) / np.sqrt(2)
        self.x2_sparse = self.x_op_sparse.dot(self.x_op_sparse)
        self.x3_sparse = self.x2_sparse.dot(self.x_op_sparse)
        self.x4_sparse = self.x2_sparse.dot(self.x2_sparse)

        self.p_op_sparse = 1j * (self.a_dag_op_sparse - self.a_op_sparse) / np.sqrt(2)
        self.p2_sparse = self.p_op_sparse.dot(self.p_op_sparse)
        self.p3_sparse = self.p2_sparse.dot(self.p_op_sparse)
        self.p4_sparse = self.p2_sparse.dot(self.p2_sparse)

        xp_sparse = self.x_op_sparse.dot(self.p_op_sparse)
        px_sparse = self.p_op_sparse.dot(self.x_op_sparse)
        self.xp_plus_px_sparse = xp_sparse + px_sparse

    def compute_non_gaussianity(self, state_ket):
        """Computes a non-Gaussianity score based on moments of quadratures."""
        ket = state_ket.astype(np.complex128)

        def expect_sparse(sparse_op):
            op_psi = sparse_op.dot(ket)
            return np.real(np.vdot(ket, op_psi))

        def get_quadrature_score(m1_op, m2_op, m3_op, m4_op):
            m1, m2, m3, m4 = expect_sparse(m1_op), expect_sparse(m2_op), expect_sparse(m3_op), expect_sparse(m4_op)

            var = m2 - m1**2
            if var < 1e-6:
                return 0.0

            sigma = np.sqrt(var)
            moment3_central = m3 - 3*m1*m2 + 2*(m1**3)
            moment4_central = m4 - 4*m1*m3 + 6*(m1**2)*m2 - 3*(m1**4)
            skewness = moment3_central / (sigma**3)
            excess_kurtosis = (moment4_central / (var**2)) - 3.0

            return np.abs(skewness) +  np.abs(excess_kurtosis)

        score_x = get_quadrature_score(self.x_op_sparse, self.x2_sparse, self.x3_sparse, self.x4_sparse)
        score_p = get_quadrature_score(self.p_op_sparse, self.p2_sparse, self.p3_sparse, self.p4_sparse)

        return score_x + score_p
