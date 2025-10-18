# Documentation for `quantum_circuit_env.py`

## 📜 Script Description

This script defines the `QuantumCircuitEnv`, a custom reinforcement learning environment built using the `gymnasium` library. It simulates a quantum optical circuit based on the Strawberry Fields library. The primary goal for the RL agent is to learn how to control the parameters of this circuit to generate one of four specific target quantum states, known as "squeezed cat states." The agent's performance is measured by the fidelity between the state it produces and the closest target state.

This environment is the core component of the project, providing the simulation space where the agent interacts, learns, and is evaluated.

---

## 💻 Code with Explanations

```python
# ==============================================================================
# === 1. MONKEY PATCH FOR SCIPY COMPATIBILITY ==================================
# ==============================================================================
# Some newer versions of SciPy renamed the `simps` (Simpson's rule) function
# to `simpson`. The Strawberry Fields library, however, may still rely on the
# old name. This block of code acts as a compatibility layer. It checks if

# `scipy.integrate.simps` exists. If it doesn't, it creates it by pointing it
# to the new `scipy.integrate.simpson` function. This prevents potential
# crashes and ensures the code runs smoothly across different library versions.

import scipy.integrate

if not hasattr(scipy.integrate, 'simps'):
    print("Monkey patching scipy.integrate: 'simps' not found. Pointing to 'simpson'.")
    scipy.integrate.simps = scipy.integrate.simpson
else:
    print("'simps' already exists in scipy.integrate. No patch needed.")

# ==============================================================================
# === 2. LIBRARY IMPORTS =======================================================
# ==============================================================================
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from scipy.linalg import sqrtm 

# Import Strawberry Fields, the core library for quantum optical simulation
import strawberryfields as sf
from strawberryfields import ops
from strawberryfields.ops import Sgate, BSgate, MeasureFock, Catstate, Rgate

# ==============================================================================
# === 3. HELPER FUNCTIONS FOR FIDELITY CALCULATION =============================
# ==============================================================================
# Fidelity is a measure of "closeness" between two quantum states. A fidelity
# of 1.0 means the states are identical, while 0.0 means they are completely
# different (orthogonal). Calculating it efficiently is crucial for the reward
# signal in the RL environment.

def compute_matrix_sqrt(rho):
    """
    Computes the matrix square root of a density matrix robustly.
    
    The fidelity calculation requires finding the square root of the density
    matrix of the target state. Since the target states are fixed, we can
    pre-compute these square roots once during initialization to save significant
    computation time during the training loop.

    Args:
        rho (np.ndarray): The density matrix (a complex, Hermitian matrix).

    Returns:
        np.ndarray: The matrix square root of rho.
    """
    rho = np.asarray(rho, dtype=np.complex128)
    
    # Due to floating-point inaccuracies, a matrix that should be perfectly
    # Hermitian (equal to its own conjugate transpose) might not be.
    # Enforcing Hermiticity improves numerical stability.
    rho = 0.5 * (rho + rho.T.conj())
    
    # For Hermitian matrices, `np.linalg.eigh` is more efficient and stable
    # than the general `np.linalg.eig`. It returns eigenvalues and eigenvectors.
    e_vals_rho, e_vecs_rho = np.linalg.eigh(rho)
    
    # Eigenvalues of a density matrix should be non-negative. Numerical errors
    # can sometimes produce tiny negative values. We clip these to zero.
    e_vals_rho_clipped = np.maximum(e_vals_rho.real, 0)
    
    # The square root of the eigenvalues can now be safely calculated.
    sqrt_e_vals_rho = np.sqrt(e_vals_rho_clipped)
    
    # Reconstruct the matrix square root using the eigenvectors and the
    # square root of the eigenvalues.
    rho_sqrt = e_vecs_rho @ np.diag(sqrt_e_vals_rho) @ e_vecs_rho.T.conj()
    
    return rho_sqrt

def fidelity_with_sqrt(rho_sqrt, sigma):
    """
    Calculates the Uhlmann-Jozsa fidelity using a pre-computed sqrt(rho).
    
    This is the most common fidelity measure for mixed quantum states. The formula is:
    F(rho, sigma) = (Tr[sqrt(sqrt(rho) * sigma * sqrt(rho))])^2
    
    By pre-calculating sqrt(rho), we make this function much faster inside the
    environment's `step` method, which is called thousands of times.

    Args:
        rho_sqrt (np.ndarray): The pre-computed square root of the target density matrix.
        sigma (np.ndarray): The density matrix of the current state from the simulation.

    Returns:
        float: The fidelity, a value between 0.0 and 1.0.
    """
    sigma = np.asarray(sigma, dtype=np.complex128)
    
    # Enforce Hermiticity on the current state's DM for stability.
    sigma = 0.5 * (sigma + sigma.T.conj())
    
    # Calculate the intermediate matrix K = sqrt(rho) * sigma * sqrt(rho).
    K = rho_sqrt @ sigma @ rho_sqrt
    K = 0.5 * (K + K.T.conj()) # Ensure K is also Hermitian.
    
    # The trace of sqrt(K) is the sum of the square roots of K's eigenvalues.
    # `eigvalsh` is used for Hermitian matrices.
    e_vals_K = np.linalg.eigvalsh(K)
    e_vals_K_clipped = np.maximum(e_vals_K.real, 0) # Clip for safety.
    
    trace_val = np.sum(np.sqrt(e_vals_K_clipped))
    
    # The final fidelity is the square of this trace.
    fidelity = trace_val**2
    
    # Clip the final result to the valid [0, 1] range to handle any floating-point errors.
    return np.clip(fidelity, 0.0, 1.0)

# ==============================================================================
# === 4. THE QUANTUM CIRCUIT ENVIRONMENT CLASS =================================
# ==============================================================================
class QuantumCircuitEnv(gym.Env):
    """
    A gymnasium environment for the quantum optical circuit described in the paper.
    The agent's goal is to control circuit parameters to generate a target squeezed cat state.
    """
    metadata = {"render_modes": [], "render_fps": 0}

    def __init__(self, cutoff_dim=25, max_steps=10, reward_power=2, tunable_r=False, is_agent_able_to_terminate=False):
        super(QuantumCircuitEnv, self).__init__()

        # --- Environment Parameters ---
        # The cutoff dimension for the Fock space simulation. A higher value means
        # a more accurate simulation but is exponentially more demanding on memory
        # and CPU. `25` is a reasonable balance for this problem.
        self.cutoff_dim = cutoff_dim
        # The maximum number of steps the agent can take in one episode.
        self.max_steps = max_steps
        # A factor to shape the reward function. reward = fidelity^reward_power.
        # A value > 1 makes the reward signal stronger for high fidelities,
        # encouraging the agent to achieve near-perfect states.
        self.reward_power = reward_power
        # A boolean to change the agent's action space. If True, the agent
        # controls the squeezing parameter `r`. If False, `r` is fixed.
        self.tunable_r = tunable_r
        self.initial_squeezing = 1.38 # Fixed `r` value from the paper.
        # A threshold for the beam splitter's transmissivity. If it drops below
        # this, the agent is considered to be "terminating" the episode.
        self.termination_threshold = 0.001
        self.is_agent_able_to_terminate = is_agent_able_to_terminate

        # The Strawberry Fields engine is initialized in `reset()` for each episode.
        self.eng = None

        # --- Pre-calculate Target States ---
        # This is a key optimization. We generate the four target density matrices
        # and their square roots only once when the environment is created.
        print("Pre-calculating target density matrices and their square roots...")
        self.target_dms, self.target_sqrts = self._initialize_target_states()
        print("Target states and square roots initialized.")
        
        # --- Define Observation and Action Spaces ---
        # The observation is a flattened representation of the quantum state's
        # density matrix. Since the matrix is Hermitian, we only need the
        # upper triangular part to reconstruct it fully, which saves space.
        obs_size = self.cutoff_dim**2
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(obs_size,), dtype=np.float32
        )

        # The action space defines what the agent can control.
        if self.tunable_r:
            # 3D action: [squeezing_r, BS_angle, squeezing_phase]
            self.action_space = spaces.Box(
                low=np.array([0.0, 0.0, -np.pi]),
                high=np.array([2.0, np.pi/2, np.pi]),
                shape=(3,),
                dtype=np.float32
            )
        else:
            # 2D action: [BS_angle, squeezing_phase] (squeezing is fixed)
            self.action_space = spaces.Box(
                low=np.array([0.0, -np.pi]),
                high=np.array([np.pi/2, np.pi]),
                shape=(2,),
                dtype=np.float32
            )
        
        # --- Internal State ---
        self.current_step = 0
        self.current_dm = None # Holds the density matrix of the evolving state.

    def _initialize_target_states(self):
        """Generates the four target squeezed cat state density matrices and their square roots."""
        alpha = 3.0
        r = 1.38
        targets = []
        target_sqrts = []

        # Use a temporary engine for this one-off calculation.
        temp_eng = sf.Engine("fock", backend_options={"cutoff_dim": self.cutoff_dim})
        
        # The four target states are variations of a "squeezed cat state".
        # They are generated using different initial cat states (p=0 or p=1)
        # and an optional final rotation gate.
        
        # Target 1: rho_plus
        prog = sf.Program(1)
        with prog.context as q:
            Catstate(alpha, p=0) | q[0]
            Sgate(r) | q[0]
        dm = temp_eng.run(prog).state.dm()
        targets.append(dm)
        target_sqrts.append(compute_matrix_sqrt(dm))
        
        # Target 2: rho_minus
        prog = sf.Program(1)
        with prog.context as q:
            Catstate(alpha, p=1) | q[0]
            Sgate(r) | q[0]
        dm = temp_eng.run(prog).state.dm()
        targets.append(dm)
        target_sqrts.append(compute_matrix_sqrt(dm))

        # Target 3: rho_plus_rot
        prog = sf.Program(1)
        with prog.context as q:
            Catstate(alpha, p=0) | q[0]
            Sgate(r) | q[0]
            Rgate(np.pi/2) | q[0]
        dm = temp_eng.run(prog).state.dm()
        targets.append(dm)
        target_sqrts.append(compute_matrix_sqrt(dm))
        
        # Target 4: rho_minus_rot
        prog = sf.Program(1)
        with prog.context as q:
            Catstate(alpha, p=1) | q[0]
            Sgate(r) | q[0]
            Rgate(np.pi/2) | q[0]
        dm = temp_eng.run(prog).state.dm()
        targets.append(dm)
        target_sqrts.append(compute_matrix_sqrt(dm))
        
        return targets, target_sqrts

    def _dm_to_observation(self, dm):
        """
        Converts a density matrix to a flattened observation vector using the upper triangular part.
        """
        # This pre-allocates a buffer to avoid creating new numpy arrays on every step.
        self._obs_buffer = np.zeros(self.cutoff_dim**2, dtype=np.float32)

        # Extract the diagonal (real numbers) and the upper triangle's real and imaginary parts.
        diag_elements = np.real(np.diag(dm))
        iu1 = np.triu_indices(self.cutoff_dim, k=1)
        off_diag_elements = dm[iu1]
        real_parts = off_diag_elements.real
        imag_parts = off_diag_elements.imag
        
        # Concatenate them into a single flat vector.
        return np.concatenate([diag_elements, real_parts, imag_parts]).astype(np.float32)

    def reset(self, seed=None, options=None):
        """
        Resets the environment to an initial state for the start of a new episode.
        """
        super().reset(seed=seed)

        # Each episode gets a fresh simulation engine.
        self.eng = sf.Engine("fock", backend_options={"cutoff_dim": self.cutoff_dim})
        self.current_step = 0
        
        # The circuit starts with two modes, both in the vacuum state.
        # We immediately measure mode 0 to ensure it's a classical vacuum.
        prog = sf.Program(2)
        with prog.context as q:
            MeasureFock() | q[0]

        self.current_state = self.eng.run(prog).state
        # The agent only observes mode 0. We get its state by taking the "partial trace".
        self.current_dm = self.current_state.reduced_dm(modes=[0])
        
        observation = self._dm_to_observation(self.current_dm)
        
        return observation, {}

    def step(self, action):
        """
        Executes one time step within the environment.
        """
        self.current_step += 1

        # 1. Unpack the agent's action.
        if self.tunable_r:
            squeezing_r, theta_1, squeezing_phase = action
        else:
            squeezing_r = self.initial_squeezing
            theta_1, squeezing_phase = action

        # 2. Build the quantum circuit for this step.
        prog = sf.Program(2)
        with prog.context as q:
            # A squeezed vacuum state is injected into mode 1.
            Sgate(squeezing_r, squeezing_phase) | q[1]
            # The two modes interact at a beam splitter. The agent controls the angle `theta_1`.
            BSgate(theta_1, 0) | (q[0], q[1])
            # A photon measurement is made on mode 0. This is a key part of the protocol.
            MeasureFock() | q[0]
            # The modes are swapped using a 50/50 beam splitter (a mirror).
            BSgate(np.pi/2, 0) | (q[0], q[1])        
        
        # 3. Run the simulation.
        result = self.eng.run(prog)
        self.current_state = result.state
        self.current_dm = self.current_state.reduced_dm(modes=[0])

        # 4. Convert the new state to an observation.
        observation = self._dm_to_observation(self.current_dm)

        # 5. Calculate the reward.
        # We check the fidelity against all four target states and take the maximum.
        fidelities = np.array([fidelity_with_sqrt(sqrt, self.current_dm) for sqrt in self.target_sqrts])
        max_fidelity = np.max(fidelities)
        # The reward is shaped by `reward_power` to encourage high fidelities.
        reward = max_fidelity ** self.reward_power

        # 6. Check for termination conditions.
        terminated = False
        # The agent can choose to terminate by making the beam splitter fully reflective.
        transmissivity = np.cos(theta_1)**2
        if self.is_agent_able_to_terminate and transmissivity < self.termination_threshold:
            terminated = True
        
        # Truncation occurs if the episode runs for too long.
        truncated = self.current_step >= self.max_steps

        # A large bonus reward is given at the end of an episode to incentivize
        # achieving a high final fidelity.
        if truncated or terminated:
            terminal_bonus = (max_fidelity ** self.reward_power) * 10
            # An additional bonus is given for exceeding a high fidelity threshold (e.g., 0.9).
            fidelity_threshold = 0.9
            excess_fidelity = max(max_fidelity - fidelity_threshold, 0)
            rescaled_excess = excess_fidelity / (1 - fidelity_threshold)
            terminal_bonus += (rescaled_excess ** self.reward_power) * 10
            reward += terminal_bonus

        # The `info` dictionary is used to pass diagnostic data to callbacks or wrappers.
        info = {
            'measured_photons': result.samples[0][0],
            'max_fidelity': max_fidelity
        }
        if truncated or terminated:
            info['final_dm'] = self.current_dm # Pass the final state for evaluation.

        return observation, reward, terminated, truncated, info

    def render(self):
        # Graphical rendering is not implemented for this environment.
        pass

    def close(self):
        # No special cleanup is needed.
        pass
```