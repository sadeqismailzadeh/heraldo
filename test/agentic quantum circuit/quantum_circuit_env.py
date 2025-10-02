import gymnasium as gym
from gymnasium import spaces
import numpy as np
from scipy.linalg import sqrtm

# Import Strawberry Fields
import strawberryfields as sf
from strawberryfields.ops import Sgate, BSgate, MeasureFock, Load, Catstate, Rgate

# --- Helper Function for Fidelity (from your code) ---
def uhlmann_jozsa_fidelity(rho, sigma):
    """Calculates the Uhlmann-Jozsa fidelity between two density matrices."""
    # Ensure inputs are valid density matrices
    if rho is None or sigma is None:
        return 0.0
    rho = np.array(rho)
    sigma = np.array(sigma)
    
    # Handle potential numerical instability by ensuring matrices are positive semi-definite
    rho = (rho + rho.conj().T) / 2
    sigma = (sigma + sigma.conj().T) / 2
    
    sqrt_rho = sqrtm(rho)
    sqrt_product = sqrtm(sqrt_rho @ sigma @ sqrt_rho)
    
    # Fidelity is the squared trace of the result
    return (np.trace(sqrt_product).real)**2

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
        self.reward_power = reward_power
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

        initial_state = self.eng.run(prog).state
        self.current_dm = initial_state.reduced_dm(modes=[0])
        
        # Convert the initial DM to an observation
        observation = self._dm_to_observation(self.current_dm)
        
        return observation, {}

    def step(self, action):
        self.current_step += 1

        # 1. Unpack and clip the agent's action
        squeezing_r = action[0]
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
        self.current_dm = result.state.reduced_dm(modes=[0]) # Get partial trace for mode 0

        # 4. Convert the new state to an observation for the agent
        observation = self._dm_to_observation(self.current_dm)

        # 5. Calculate the reward
        fidelities = [uhlmann_jozsa_fidelity(self.current_dm, target) for target in self.target_dms]
        max_fidelity = np.max(fidelities) if len(fidelities) > 0 else 0.0
        reward = max_fidelity ** self.reward_power

        # 6. Check for termination/truncation
        # The episode ends when the maximum number of steps is reached
        terminated = False
        truncated = self.current_step >= self.max_steps
        
        # Optional: Add bonus reward if a high fidelity is achieved
        if max_fidelity > 0.95:
             reward += 1.0 # Give a small bonus for getting close

        
        # The 'info' dictionary is the standard place for diagnostic information.
        # result.samples[0][0] holds the measured photon number from q[0].
        info = {'measured_photons': result.samples[0][0]}

        return observation, reward, terminated, truncated, info

    def render(self):
        # We won't implement graphical rendering for this complex environment
        pass

    def close(self):
        # No special cleanup needed
        pass