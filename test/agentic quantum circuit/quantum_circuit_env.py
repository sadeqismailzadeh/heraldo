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

# Import Strawberry Fields
import strawberryfields as sf
from strawberryfields import ops
from strawberryfields.ops import Sgate, BSgate, MeasureFock, Catstate, Rgate


# disable caching to save memory for large cutoff dims
from sf_operations_no_cache import disable_fock_caching
disable_fock_caching() 

# optimized loss channel
from monitored_loss_measure_fock_patch import MonitoredLossMeasureFock, patch_fock_backend, decode_measurement_result
patch_fock_backend()

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
        print("Pre-calculating target state kets...")
        self.target_kets = self._initialize_target_states()
        print("Target state kets initialized.")
        
        # --- Pre-allocate observation buffer for efficiency ---
        # Pure states: 2*cutoff_dim (real + imaginary parts of ket)
        obs_size = 2 * self.cutoff_dim
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
        self.current_ket = None  # This will hold the state vector of mode 0

    def _initialize_target_states(self):
        """Generates and caches the target squeezed-cat states.

        This method creates the canonical squeezed-cat states that the agent
        will be trained to generate. Target states are stored as state vectors (kets)
        for efficient pure state fidelity calculation.

        Returns:
            list[np.ndarray]: List of state vectors (kets) of the target states.
        """
        alpha = 3.0
        r = 1.38
        targets = []

        temp_eng = sf.Engine("fock", backend_options={"cutoff_dim": self.cutoff_dim})
        
        # Target 1: ket_plus
        prog = sf.Program(1)
        with prog.context as q:
            Catstate(alpha, p=0) | q[0]
            Sgate(r) | q[0]
        ket = temp_eng.run(prog).state.ket()
        targets.append(ket)
        
        # Target 2: ket_minus
        prog = sf.Program(1)
        with prog.context as q:
            Catstate(alpha, p=1) | q[0]
            Sgate(r) | q[0]
        ket = temp_eng.run(prog).state.ket()
        targets.append(ket)

        # Target 3: ket_plus_rot
        prog = sf.Program(1)
        with prog.context as q:
            Catstate(alpha, p=0) | q[0]
            Sgate(r) | q[0]
            Rgate(np.pi/2) | q[0]
        ket = temp_eng.run(prog).state.ket()
        targets.append(ket)
        
        # Target 4: ket_minus_rot
        prog = sf.Program(1)
        with prog.context as q:
            Catstate(alpha, p=1) | q[0]
            Sgate(r) | q[0]
            Rgate(np.pi/2) | q[0]
        ket = temp_eng.run(prog).state.ket()
        targets.append(ket)
        
        return targets

    def _ket_to_observation(self, state_ket):
        """Converts a pure state ket into an observation vector.
        
        This method extracts the state vector for a single mode and flattens it
        into a real-valued observation by separating real and imaginary parts.
        
        Args:
            state_ket (np.ndarray): The state vector (ket) for the mode.
            
        Returns:
            np.ndarray: A real-valued observation vector containing the real
                and imaginary parts of the state coefficients.
        """
        if state_ket is None:
            self._obs_buffer.fill(0)
            return self._obs_buffer.copy()
        
        state_ket = state_ket.flatten()
        
        # Extract coefficients up to cutoff_dim
        if len(state_ket) < self.cutoff_dim:
            # Pad with zeros if state is smaller than cutoff
            padded = np.zeros(self.cutoff_dim, dtype=np.complex128)
            padded[:len(state_ket)] = state_ket
            state_ket = padded
        elif len(state_ket) > self.cutoff_dim:
            # Truncate if larger
            state_ket = state_ket[:self.cutoff_dim]
        
        # Split into real and imaginary parts
        real_parts = state_ket.real
        imag_parts = state_ket.imag
        
        # Concatenate into observation buffer
        self._obs_buffer[:self.cutoff_dim] = real_parts
        self._obs_buffer[self.cutoff_dim:] = imag_parts
        
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
        
        # Pure state - extract mode 0 ket
        full_ket = self.current_state.ket()
        if len(full_ket.shape) == 1:
            # Single mode
            self.current_ket = full_ket
        else:
            # Multi-mode: tensor[i,j] = coefficient for |i⟩_mode0 ⊗ |j⟩_mode1
            # To get state of mode 0, we take the slice when mode 1 is in vacuum: full_ket[:, 0]
            self.current_ket = full_ket[:, 0]
        observation = self._ket_to_observation(self.current_ket)
        
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

            # Monitored lossy photon-number-resolving measurement (PNR)
            MonitoredLossMeasureFock(self.loss_channel) | q[0]

            # Fully reflective mirror  
            # the mode q[1] is now q[0]
            BSgate(np.pi/2, 0) | (q[0], q[1])        
        
        # 3. Run the simulation
        result = self.eng.run(prog)

        # The new state is the state of mode 0 after the interaction
        self.current_state = result.state
        
        # 4. Extract pure state and convert to observation
        full_ket = self.current_state.ket()
        if len(full_ket.shape) == 1:
            # Single mode
            self.current_ket = full_ket
        else:
            # Multi-mode: tensor[i,j] = coefficient for |i⟩_mode0 ⊗ |j⟩_mode1
            # After MonitoredLossMeasureFock on q[0] and BSgate(π/2) swap:
            # - Mode 0 is measured and placed in vacuum, then swapped to mode 1
            # - Mode 1 (unmeasured) is swapped to mode 0
            # The state of the current mode 0 is: full_ket[:, 0] (mode 1 in vacuum)
            self.current_ket = full_ket[:, 0]
        observation = self._ket_to_observation(self.current_ket)

        # 5. Calculate the reward by computing fidelities for all target states
        fidelities = np.array([fidelity_pure_state(target_ket, self.current_ket) 
                               for target_ket in self.target_kets])
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
        # Decode the monitored loss measurement result
        encoded_result = result.samples[0][0]
        lost_photons, detected_photons = decode_measurement_result(encoded_result)
        info = {
            'lost_photons': lost_photons,
            'detected_photons': detected_photons,
            'total_photons': lost_photons + detected_photons,
            'max_fidelity': max_fidelity
        }
        
        if truncated:
            info['terminal_bonus'] = terminal_bonus
            info['final_ket'] = self.current_ket

        return observation, reward, terminated, truncated, info

    def render(self):
        """Renders the environment.

        This method is not implemented for this environment.
        """

    def close(self):
        """Closes the environment.

        This method does not require any special cleanup.
        """