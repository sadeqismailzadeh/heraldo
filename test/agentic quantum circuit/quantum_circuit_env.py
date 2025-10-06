# 1. Import the module we need to patch
import scipy.integrate

# 2. Check if the patch is needed to avoid errors
if not hasattr(scipy.integrate, 'simps'):
    print("Monkey patching scipy.integrate: 'simps' not found. Pointing to 'simpson'.")
    # 3. Create the 'simps' attribute and point it to the existing 'simpson' function.
    scipy.integrate.simps = scipy.integrate.simpson
else:
    print("'simps' already exists in scipy.integrate. No patch needed.")

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from scipy.linalg import sqrtm 

# Import Strawberry Fields
import strawberryfields as sf
from strawberryfields.ops import Sgate, BSgate, MeasureFock, Catstate, Rgate


# --- Helper Function for Fidelity (from your code) ---
def fidelity(rho, sigma):
    """
    Calculates the Uhlmann-Jozsa fidelity with enhanced numerical robustness.

    This implementation explicitly uses eigendecomposition and includes steps to
    handle common floating-point precision issues.
    """
    # --- 1. Input Validation and Conditioning ---
    rho = np.asarray(rho, dtype=np.complex128)
    sigma = np.asarray(sigma, dtype=np.complex128)
    
    if rho.shape != sigma.shape or rho.ndim != 2 or rho.shape[0] != rho.shape[1]:
        raise ValueError("Input density matrices must be square and have the same shape.")

    # Enforce Hermiticity on inputs to remove numerical noise
    rho = 0.5 * (rho + rho.T.conj())
    sigma = 0.5 * (sigma + sigma.T.conj())

    # --- 2. Calculate sqrt(rho) Robustly ---
    # eigh is best for Hermitian matrices
    e_vals_rho, e_vecs_rho = np.linalg.eigh(rho)
    
    # Clip small negative eigenvalues to 0 due to numerical instability
    e_vals_rho_clipped = np.maximum(e_vals_rho.real, 0)
    
    # Calculate square root of eigenvalues
    sqrt_e_vals_rho = np.sqrt(e_vals_rho_clipped)
    
    # Reconstruct sqrt(rho) = U * sqrt(D) * U_dagger
    rho_sqrt = e_vecs_rho @ np.diag(sqrt_e_vals_rho) @ e_vecs_rho.T.conj()
    
    # --- 3. Calculate the product matrix K and ensure it's Hermitian ---
    K = rho_sqrt @ sigma @ rho_sqrt
    K = 0.5 * (K + K.T.conj()) # Enforce Hermiticity on the result

    # --- 4. Calculate Tr(sqrt(K)) Robustly ---
    # We only need the eigenvalues of K. Use eigvalsh for efficiency.
    e_vals_K = np.linalg.eigvalsh(K)
    
    # Clip again before the final square root
    e_vals_K_clipped = np.maximum(e_vals_K.real, 0)
    
    # The trace of sqrt(K) is the sum of the square roots of K's eigenvalues
    trace_val = np.sum(np.sqrt(e_vals_K_clipped))
    
    # --- 5. Calculate and Clip Final Fidelity ---
    fidelity = trace_val**2
    
    # Clip the final result to the valid [0, 1] range
    return np.clip(fidelity, 0.0, 1.0)

class QuantumCircuitEnv(gym.Env):
    """
    A gymnasium environment for the quantum optical circuit described in the paper.
    The agent's goal is to control circuit parameters to generate a target squeezed cat state.
    """
    metadata = {"render_modes": [], "render_fps": 0}

    def __init__(self, cutoff_dim=20, max_steps=10, reward_power=50):
        super(QuantumCircuitEnv, self).__init__()

        # --- Environment Parameters ---
        self.cutoff_dim = cutoff_dim
        self.max_steps = max_steps
        self.reward_power = reward_power  # Controls reward curve steepness (higher = harder)
        self.initial_squeezing = 1.38 # r0 from the paper

        # --- Strawberry Fields Engine ---
        self.eng = None # Will be initialized in reset()

        # --- Pre-calculate Target States (Reward States) ---
        print("Pre-calculating target density matrices...")
        self.target_dms = self._initialize_target_states()
        print("Target states initialized.")

        # --- Define Observation and Action Spaces ---
        # OBSERVATION SPACE: The flattened density matrix (real and imaginary parts).
        # Shape is 2 * (cutoff_dim * cutoff_dim).
        obs_size = 2 * self.cutoff_dim**2
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(obs_size,), dtype=np.float32
        )

        # ACTION SPACE: A vector [squeezing_r, BS angle, squeezing_phase].
        # Squeezing 'r' is between 0 and 2.
        # BS angle is between 0 (perfectly transparent) and pi/2 (perfect mirror).
        # Squeezing phase is between -pi and pi.
        self.action_space = spaces.Box(
            low=np.array([0.0, 0.0, -np.pi]),
            high=np.array([2.0, np.pi/2,  np.pi]),
            shape=(3,),
            dtype=np.float32
        )
        
        # Internal state of the environment
        self.current_step = 0
        self.current_dm = None # This will hold the density matrix of mode 1

    def _initialize_target_states(self):
        """Generates the four target squeezed cat state density matrices."""
        alpha = 3.0
        r = 1.38
        targets = []

        temp_eng = sf.Engine("fock", backend_options={"cutoff_dim": self.cutoff_dim})
        
        # Target 1: rho_plus
        prog = sf.Program(1)
        with prog.context as q:
            Catstate(alpha, p=0) | q[0]
            Sgate(r) | q[0]
        targets.append(temp_eng.run(prog).state.dm())
        
        # Target 2: rho_minus
        prog = sf.Program(1)
        with prog.context as q:
            Catstate(alpha, p=1) | q[0]
            Sgate(r) | q[0]
        targets.append(temp_eng.run(prog).state.dm())

        # Target 3: rho_plus_rot
        prog = sf.Program(1)
        with prog.context as q:
            Catstate(alpha, p=0) | q[0]
            Sgate(r) | q[0]
            Rgate(np.pi/2) | q[0]
        targets.append(temp_eng.run(prog).state.dm())
        
        # Target 4: rho_minus_rot
        prog = sf.Program(1)
        with prog.context as q:
            Catstate(alpha, p=1) | q[0]
            Sgate(r) | q[0]
            Rgate(np.pi/2) | q[0]
        targets.append(temp_eng.run(prog).state.dm())
        
        return targets

    def _dm_to_observation(self, dm):
        """Converts a density matrix to a flattened observation vector."""
        if dm is None or dm.shape != (self.cutoff_dim, self.cutoff_dim):
             # Return a zero vector if DM is invalid
            return np.zeros(2 * self.cutoff_dim**2, dtype=np.float32)
        real_part = dm.real.flatten()
        imag_part = dm.imag.flatten()
        return np.concatenate([real_part, imag_part]).astype(np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Create a new engine for the new episode
        self.eng = sf.Engine("fock", backend_options={"cutoff_dim": self.cutoff_dim})
        
        # Reset the step counter
        self.current_step = 0
        
        # Prepare the initial circuit
        prog = sf.Program(2)
        with prog.context as q:
            # Initialize mode 1 with a squeezed vacuum state
            Sgate(self.initial_squeezing) | q[1]

            # Apply variable beam splitter (VBS1). 
            # initially perfect transmitive. no entanglement
            BSgate(0, 0) | (q[0], q[1])

            # Photon-number-resolving measurement (PNR)
            # does basically nothing
            MeasureFock() | q[0]

            # Fully reflective mirror  
            # the mode q[1] is now q[0]
            BSgate(np.pi/2, 0) | (q[0], q[1]) 

            # the final result is q[1] becomes q[0]
            # the squeezed mode only went into the loop      

        self.current_state = self.eng.run(prog).state
        self.current_dm = self.current_state.reduced_dm(modes=[0])
        
        # Convert the initial DM to an observation
        observation = self._dm_to_observation(self.current_dm)
        
        return observation, {}

    def step(self, action):
        self.current_step += 1

        # 1. Unpack and clip the agent's action
        squeezing_r = np.clip(action[0], 0, 2)
        theta_1 = action[1]
        squeezing_phase = action[2]

        # 2. Build the Strawberry Fields program for one step
        prog = sf.Program(2)
        with prog.context as q:
            # Initialize mode 1 with a squeezed vacuum state
            Sgate(squeezing_r, squeezing_phase) | q[1]

            # Apply variable beam splitter (VBS1).
            BSgate(theta_1, 0) | (q[0], q[1])

            # Photon-number-resolving measurement (PNR)
            MeasureFock() | q[0]

            # Fully reflective mirror  
            # the mode q[1] is now q[0]
            BSgate(np.pi/2, 0) | (q[0], q[1])        
        
        # 3. Run the simulation
        result = self.eng.run(prog)

        # The new state is the state of mode 0 after the interaction
        self.current_state = result.state
        self.current_dm = self.current_state.reduced_dm(modes=[0]) # Get partial trace for mode 0

        # 4. Convert the new state to an observation for the agent
        observation = self._dm_to_observation(self.current_dm)

        # 5. Calculate the reward
        # Enable debug mode to see potential numerical issues
        fidelities = [fidelity(sigma=self.current_dm, rho=target) for target in self.target_dms]
        max_fidelity = np.max(fidelities) if len(fidelities) > 0 else 0.0
        reward = max_fidelity ** self.reward_power

        # 6. Check for termination/truncation
        # The episode ends when the maximum number of steps is reached

        # The episode is NEVER terminated early by the environment.
        # The agent must learn to stabilize a high-fidelity state.
        terminated = False 

        truncated = self.current_step >= self.max_steps

        
        # The 'info' dictionary is the standard place for diagnostic information.
        # result.samples[0][0] holds the measured photon number from q[0].
        info = {
                'measured_photons': result.samples[0][0],
                'max_fidelity': max_fidelity  # It's good practice to log this
                }

        return observation, reward, terminated, truncated, info

    def render(self):
        # We won't implement graphical rendering for this complex environment
        pass

    def close(self):
        # No special cleanup needed
        pass