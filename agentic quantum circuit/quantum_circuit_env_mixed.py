"""Custom Gymnasium environment that simulates a squeezed-cat generation circuit."""

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
from strawberryfields import ops
from strawberryfields.ops import Sgate, BSgate, MeasureFock, Catstate, Rgate


# disable caching to save memory for large cutoff dims
from sf_operations_no_cache import disable_fock_caching
disable_fock_caching() 

# optimized loss channel
from loss_measure_fock_patch import LossMeasureFock, patch_fock_backend
patch_fock_backend()

# --- Helper Function for Computing Matrix Square Root ---
def compute_matrix_sqrt(rho):
    """Return a numerically stable matrix square root of a density matrix.

    Args:
        rho (np.ndarray): Hermitian density matrix for which to compute
            :math:`\sqrt{\rho}`.

    Returns:
        np.ndarray: Hermitian square root of ``rho`` with negative eigenvalues
        clipped to zero.
    """
    rho = np.asarray(rho, dtype=np.complex128)
    
    # Enforce Hermiticity on input to remove numerical noise
    rho = 0.5 * (rho + rho.T.conj())
    
    # eigh is best for Hermitian matrices
    e_vals_rho, e_vecs_rho = np.linalg.eigh(rho)
    
    # Clip small negative eigenvalues to 0 due to numerical instability
    e_vals_rho_clipped = np.maximum(e_vals_rho.real, 0)
    
    # Calculate square root of eigenvalues
    sqrt_e_vals_rho = np.sqrt(e_vals_rho_clipped)
    
    # Reconstruct sqrt(rho) = U * sqrt(D) * U_dagger
    rho_sqrt = e_vecs_rho @ np.diag(sqrt_e_vals_rho) @ e_vecs_rho.T.conj()
    
    return rho_sqrt

# --- Optimized Fidelity Function (with pre-computed sqrt) ---
def fidelity_with_sqrt(rho_sqrt, sigma):
    """Return Uhlmann fidelity using a pre-computed target square root.

    Args:
        rho_sqrt (np.ndarray): Square root of a target density matrix.
        sigma (np.ndarray): Candidate density matrix produced by the agent.

    Returns:
        float: Clipped fidelity value in :math:`[0, 1]`.
    """
    sigma = np.asarray(sigma, dtype=np.complex128)
    
    # Enforce Hermiticity on sigma
    sigma = 0.5 * (sigma + sigma.T.conj())
    
    # Calculate the product matrix K and ensure it's Hermitian
    K = rho_sqrt @ sigma @ rho_sqrt
    K = 0.5 * (K + K.T.conj())
    
    # Calculate Tr(sqrt(K)) robustly
    e_vals_K = np.linalg.eigvalsh(K)
    
    # Clip before the final square root
    e_vals_K_clipped = np.maximum(e_vals_K.real, 0)
    
    # The trace of sqrt(K) is the sum of the square roots of K's eigenvalues
    trace_val = np.sum(np.sqrt(e_vals_K_clipped))
    
    # Calculate and clip final fidelity
    fidelity = trace_val**2
    
    return np.clip(fidelity, 0.0, 1.0)


class QuantumCircuitEnv(gym.Env):
    """A gymnasium environment for a quantum optical circuit.

    This environment simulates the quantum optical circuit described in the paper.
    The agent's goal is to control the circuit parameters to generate a target
    squeezed cat state.

    Attributes:
        cutoff_dim (int): The Fock-space cutoff dimension for the simulation.
        max_steps (int): The maximum number of steps per episode.
        reward_power (int): The exponent applied to the fidelity for reward shaping.
        tunable_r (bool): Whether the agent can tune the squeezing parameter 'r'.
        initial_squeezing (float): The initial squeezing parameter 'r0'.
        termination_threshold (float): The transmissivity threshold for early termination.
        is_agent_able_to_terminate (bool): Whether the agent can terminate an episode early.
        eng (sf.Engine): The Strawberry Fields engine for the simulation.
        target_dms (list): A list of target density matrices.
        target_sqrts (list): A list of the square roots of the target density matrices.
        observation_space (gym.spaces.Box): The observation space for the environment.
        action_space (gym.spaces.Box): The action space for the environment.
        current_step (int): The current step in the episode.
        current_dm (np.ndarray): The current density matrix of the system.
    """
    metadata = {"render_modes": [], "render_fps": 0}
    # agent can terminate
    def __init__(self, cutoff_dim=25, max_steps=10, reward_power=2, tunable_r=False,
                 is_loss_channel=False, loss_channel=0.99):
        """Initializes the quantum circuit environment.

        This method sets up the simulation parameters, pre-calculates the target
        states, and defines the observation and action spaces for the reinforcement
        learning agent.

        Args:
            cutoff_dim (int): The Fock-space cutoff dimension used for the
                Strawberry Fields simulations.
            max_steps (int): The maximum number of control steps per episode.
            reward_power (int): The exponent applied to the fidelity to shape the
                PPO rewards.
            tunable_r (bool): A flag to determine if the agent controls the
                squeezing strength 'r'.
            is_agent_able_to_terminate (bool): A flag that allows the agent to
                drive an early stopping of the simulation when the
                transmissivity falls below the `termination_threshold`.
        
        Example:
            >>> env = QuantumCircuitEnv(cutoff_dim=20, max_steps=15)
            >>> print(env.observation_space)
            Box(-1.0, 1.0, (400,), float32)
        """
        super(QuantumCircuitEnv, self).__init__()

        # --- Environment Parameters ---
        self.cutoff_dim = cutoff_dim
        self.max_steps = max_steps
        self.reward_power = reward_power  # Controls reward curve steepness (higher = harder)
        self.tunable_r = tunable_r  # Toggle: True = action controls r, False = fixed r
        self.initial_squeezing = 1.38 # r0 from the paper (used when tunable_r=False)
        self.termination_threshold = 0.001 # e.g., less than 0.1% transmissivity
        self.is_loss_channel=is_loss_channel
        self.loss_channel=loss_channel


        # --- Strawberry Fields Engine ---
        self.eng = None # Will be initialized in reset()

        # --- Pre-calculate Target States (Reward States) ---
        print("Pre-calculating target density matrices and their square roots...")
        self.target_dms, self.target_sqrts = self._initialize_target_states()
        print("Target states and square roots initialized.")
        
        # --- Pre-allocate observation buffer for efficiency ---
        obs_size = self.cutoff_dim**2
        self._obs_buffer = np.zeros(obs_size, dtype=np.float32)

        # --- Define Observation and Action Spaces ---
        # OBSERVATION SPACE: The upper triangular part of the density matrix.
        # Since the DM is Hermitian, this is sufficient to describe the state.
        # It consists of the real diagonal elements, and the real and imaginary
        # parts of the off-diagonal elements in the upper triangle.
        # Shape is cutoff_dim**2.
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(obs_size,), dtype=np.float32
        )

        # ACTION SPACE: Depends on tunable_r setting
        # If tunable_r=True: [squeezing_r, BS_angle, squeezing_phase] (3D)
        # If tunable_r=False: [BS_angle, squeezing_phase] (2D)
        # Squeezing 'r' is between 0 and 2.
        # BS angle is between 0 (perfectly transparent) and pi/2 (perfect mirror).
        # Squeezing phase is between -pi and pi.
        if self.tunable_r:
            # 3D action space: agent controls squeezing_r, BS angle, and phase
            self.action_space = spaces.Box(
                low=np.array([0.0, 0.0, -np.pi]),
                high=np.array([1.38, np.pi/2, np.pi]),
                shape=(3,),
                dtype=np.float32
            )
        else:
            # 2D action space: squeezing_r is fixed, agent controls BS angle and phase only
            self.action_space = spaces.Box(
                low=np.array([0.0, -np.pi]),
                high=np.array([np.pi/2, np.pi]),
                shape=(2,),
                dtype=np.float32
            )
        
        # Internal state of the environment
        self.current_step = 0
        self.current_dm = None # This will hold the density matrix of mode 1

    def _initialize_target_states(self):
        """Generates and caches the target squeezed-cat states.

        This method creates the canonical squeezed-cat states that the agent
        will be trained to generate. It also pre-computes and caches the square
        roots of their density matrices to speed up the fidelity calculations.

        Returns:
            tuple[list[np.ndarray], list[np.ndarray]]: A tuple containing two lists.
                The first list holds the density matrices of the target states,
                and the second list holds the corresponding square roots of these
                matrices.
        """
        alpha = 3.0
        r = 1.38
        targets = []
        target_sqrts = []

        temp_eng = sf.Engine("fock", backend_options={"cutoff_dim": self.cutoff_dim})
        
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
        """Flattens the density matrix into an observation vector.

        This method converts the density matrix of the quantum state into a
        one-dimensional vector that can be used as an observation by the PPO
        agent. The vector contains the real and imaginary parts of the upper
        triangular elements of the density matrix.

        Args:
            dm (np.ndarray | None): The density matrix for mode `0`. If the state
                is invalid, this can be `None`.

        Returns:
            np.ndarray: A real-valued observation vector that contains the
                diagonal, upper-triangular real, and imaginary parts of the
                density matrix.
        """
        if dm is None or dm.shape != (self.cutoff_dim, self.cutoff_dim):
            # Return a zero vector if DM is invalid
            self._obs_buffer.fill(0)
            return self._obs_buffer.copy()

        # Extract the diagonal elements (which are real)
        diag_elements = np.real(np.diag(dm))

        # Extract the real and imaginary parts of the upper triangular elements (excluding the diagonal)
        iu1 = np.triu_indices(self.cutoff_dim, k=1)
        off_diag_elements = dm[iu1]
        real_parts = off_diag_elements.real
        imag_parts = off_diag_elements.imag
        
        # Concatenate into the observation buffer
        len_diag = len(diag_elements)
        len_real = len(real_parts)
        
        self._obs_buffer[:len_diag] = diag_elements
        self._obs_buffer[len_diag:len_diag + len_real] = real_parts
        self._obs_buffer[len_diag + len_real:] = imag_parts
        
        return self._obs_buffer.copy()

    def reset(self, seed=None, options=None):
        """Resets the environment to its initial state.

        This method is called at the beginning of each episode. It resets the
        Strawberry Fields engine and returns the initial observation, which
        corresponds to the vacuum state.

        Args:
            seed (int, optional): The seed for the random number generator.
            options (dict, optional): Additional options for resetting the environment.

        Returns:
            tuple[np.ndarray, dict]: A tuple containing the initial observation
                and an empty dictionary for additional information.
        
        Example:
            >>> env = QuantumCircuitEnv()
            >>> observation, info = env.reset()
            >>> print(observation.shape)
            (625,)
        """
        super().reset(seed=seed)

        # Create a new engine for the new episode
        self.eng = sf.Engine("fock", backend_options={"cutoff_dim": self.cutoff_dim})
        
        # Reset the step counter
        self.current_step = 0
        
        # Prepare the initial circuit
        prog = sf.Program(2)

        with prog.context as q:
            # MeasureFock() | q[0]  # Start with vacuum in mode 0
            Sgate(self.initial_squeezing) | q[0]
            # MeasureFock() | q[1]  # Start with vacuum in mode 1

        self.current_state = self.eng.run(prog).state
        self.current_dm = self.current_state.reduced_dm(modes=[0])
        
        # Convert the initial DM to an observation
        observation = self._dm_to_observation(self.current_dm)
        
        return observation, {}

    def step(self, action):
        """Executes one time step in the environment.

        This method applies the agent's action to the quantum circuit, evolves
        the state, and calculates the reward.

        Args:
            action (np.ndarray): The control parameters for the squeezing and
                beam-splitter operations, as determined by the `tunable_r` setting.

        Returns:
            tuple: A tuple containing the observation, the shaped reward, a
                termination flag, a truncation flag, and a dictionary with
                diagnostic information, compatible with the Gymnasium API.
        
        Example:
            >>> env = QuantumCircuitEnv()
            >>> obs, info = env.reset()
            >>> action = env.action_space.sample()
            >>> obs, reward, terminated, truncated, info = env.step(action)
            >>> print(reward)
            0.0
        """
        self.current_step += 1

        # 1. Unpack and clip the agent's action based on tunable_r setting
        if self.tunable_r:
            # 3D action: [squeezing_r, BS_angle, squeezing_phase]
            squeezing_r = np.clip(action[0], 0, 1.38)
            theta_1 = np.clip(action[1], 0, np.pi/2)
            squeezing_phase = np.clip(action[2], -np.pi, np.pi)
        else:
            # 2D action: [BS_angle, squeezing_phase], squeezing_r is fixed
            squeezing_r = self.initial_squeezing
            theta_1 = np.clip(action[0], 0, np.pi/2)
            squeezing_phase = np.clip(action[1], -np.pi, np.pi)

        # 2. Build the Strawberry Fields program for one step
        prog = sf.Program(2)
        with prog.context as q:
            # Initialize mode 1 with a squeezed vacuum state
            Sgate(squeezing_r, squeezing_phase) | q[1]

            # Apply variable beam splitter (VBS1).
            BSgate(theta_1, 0) | (q[0], q[1])

            if self.is_loss_channel:
                LossMeasureFock(self.loss_channel) | q[0]
            else:
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

        # 5. Calculate the reward by computing fidelities for all target states
        fidelities = np.array([fidelity_with_sqrt(sqrt, self.current_dm) for sqrt in self.target_sqrts])
        max_fidelity = np.max(fidelities)
        reward = max_fidelity ** self.reward_power

        # 6. Check for termination/truncation
        # The episode ends when the maximum number of steps is reached
       
        terminated = False
        truncated = self.current_step >= self.max_steps

        # # Add a large, shaped bonus on the final step of the episode.
        terminal_bonus = 0

        if truncated:
            terminal_bonus += (max_fidelity ** self.reward_power) * 10

            fidelity_threshold = 0.9
            # Your proposed bonus function:
            excess_fidelity = max(max_fidelity - fidelity_threshold, 0)
            # Rescale the excess from [0, 0.1] to [0, 1]
            rescaled_excess = excess_fidelity / (1 - fidelity_threshold)
            # Apply non-linear shaping and final scaling
            terminal_bonus += (rescaled_excess ** self.reward_power) * 10
            
            reward += terminal_bonus

        
        # The 'info' dictionary is the standard place for diagnostic information.
        # result.samples[0][0] holds the measured photon number from q[0].
        info = {
                'measured_photons': result.samples[0][0],
                'max_fidelity': max_fidelity  # It's good practice to log this
                }
        
        if truncated:
            info['terminal_bonus'] = terminal_bonus
            info['final_dm'] = self.current_dm

        return observation, reward, terminated, truncated, info

    def render(self):
        """Renders the environment.

        This method is not implemented for this environment.
        """

    def close(self):
        """Closes the environment.

        This method does not require any special cleanup.
        """