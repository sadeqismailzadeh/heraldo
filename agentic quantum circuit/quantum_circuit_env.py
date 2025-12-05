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
    if len(t) < max_len: t = np.pad(t, (0, max_len - len(t)))
    if len(s) < max_len: s = np.pad(s, (0, max_len - len(s)))

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
        action_space (gym.spaces.Box): The normalized action space in [-1, 1] for the environment.
        current_step (int): The current step in the episode.
        current_dm (np.ndarray): The current density matrix of the system.
    """
    metadata = {"render_modes": [], "render_fps": 0}
    # agent can terminate
    def __init__(self, cutoff_dim=25, max_steps=10, reward_power=2, tunable_r=True,
                 is_loss_channel=False, loss_channel=1, initial_fidelity_threshold=0.70):
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
        self.max_squeezing = 1.38 # r0 from the paper (used when tunable_r=False)
        self.termination_threshold = 0.001 # e.g., less than 0.1% transmissivity
        self.is_loss_channel=is_loss_channel
        self.loss_channel=loss_channel

         # --- CURRICULUM PARAMETERS ---
        # D(v): The current difficulty level defined by target fidelity
        self.target_fidelity = initial_fidelity_threshold 

        # --- Strawberry Fields Engine ---
        self.eng = None # Will be initialized in reset()

        # --- Pre-calculate Target States (Reward States) ---
        print("Pre-calculating target state kets...")
        self.target_kets = self._initialize_target_states_no_rotate()
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

        # ACTION SPACE: Normalized to [-1, 1] for all dimensions
        # Internally denormalized to original ranges
        # If tunable_r=True: [squeezing_r, BS_angle]
        # If tunable_r=False: [squeezing_phase, BS_angle]
        # Squeezing 'r' is between -max_squeezing and max_squeezing (when tunable).
        # BS angle is between 0 (perfectly transparent) and pi/2 (perfect mirror).
        # Squeezing phase is between -pi and pi (when tunable).
        self.action_ranges = {
            'squeezing_r': (-self.max_squeezing, self.max_squeezing),
            'theta_1': (0, np.pi/2),
            'squeezing_phase': (-np.pi, np.pi)
        }
        if self.tunable_r:
            self.action_keys = ['squeezing_r', 'theta_1']
        else:
            self.action_keys = ['squeezing_phase', 'theta_1']

        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(len(self.action_keys),), dtype=np.float32
        )
        
        # Internal state of the environment
        self.current_step = 0
        self.current_ket = None  # This will hold the state vector of mode 0
        self.min_inner_product = 1.0  # Track minimum inner product during episode


    def _initialize_target_states_no_rotate(self):
        alpha = 3.0
        r = 1.38
        targets = []

        temp_eng = sf.Engine("fock", backend_options={"cutoff_dim": self.cutoff_dim})
        
        # Target 1: Even Parity (Cat+)
        prog = sf.Program(1)
        with prog.context as q:
            Catstate(alpha, p=0) | q[0]
            Sgate(r) | q[0]
        targets.append(temp_eng.run(prog).state.ket())
        
        # Target 2: Odd Parity (Cat-)
        prog = sf.Program(1)
        with prog.context as q:
            Catstate(alpha, p=1) | q[0]
            Sgate(r) | q[0]
        targets.append(temp_eng.run(prog).state.ket())

        return targets

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
        final_state = temp_eng.run(prog).state
        ket = final_state.ket()
        assert final_state.is_pure
        targets.append(ket)
        
        assert not np.any(targets == None)  , "Value should not be None" 
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

    def _denormalize_action(self, action):
        """Denormalize action from [-1, 1] to original ranges.

        This method performs a linear transformation to scale normalized actions
        back to their physical ranges for use in the quantum circuit operations.
        The transformation is: f(x) = a*x + b, where f(-1) = low and f(1) = high.
        Solving: a = (high - low)/2, b = (high + low)/2
        Thus: f(x) = (high - low)/2 * x + (high + low)/2
        Which simplifies to: low + (x + 1) * (high - low) / 2

        Designed to be reusable and overridable in derived environment classes.

        Args:
            action (np.ndarray): Normalized action array in [-1, 1].

        Returns:
            np.ndarray: Denormalized action array in original ranges.
        """
        denorm_action = np.zeros_like(action, dtype=np.float32)
        for i, key in enumerate(self.action_keys):
            low, high = self.action_ranges[key]
            # Linear transformation: f(x) = a*x + b with f(-1)=low, f(1)=high
            a = (high - low) / 2
            b = (high + low) / 2
            denorm_action[i] = a * action[i] + b
        return denorm_action

    def _calculate_log_reward(self, fidelity):
        """
        Helper to calculate the specific Log Reward value for a given fidelity.
        Used for both the current step and the bonus calculation. 
        """
        # 1. Get parameters based on the paper's piecewise function
        # 2. Calculate Log Error
        # prevent log(0) with a tiny epsilon
        infidelity = max(1.0 - fidelity, 1e-5)
        log_val = - np.log10(infidelity)
        power_val = fidelity
        # 3. Compute Reward
        reward =  power_val * log_val
        return reward

   

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
        
        # Reset the minimum inner product tracker
        self.min_inner_product = 1.0
        
        # Prepare the initial circuit
        prog = sf.Program(2)

        with prog.context as q:
            # MeasureFock() | q[0]  # Start with vacuum in mode 0
            Sgate(self.max_squeezing) | q[0]
            # MeasureFock() | q[1]  # Start with vacuum in mode 1

        self.current_state = self.eng.run(prog).state


        
        # Pure state - extract mode 0 ket
        full_ket = self.current_state.ket()

        # Multi-mode: tensor[i,j] = coefficient for |i⟩_mode0 ⊗ |j⟩_mode1
        # To get state of mode 0, we take the slice when mode 1 is in vacuum: full_ket[:, 0]
        self.current_ket = full_ket[:, 0]
        observation = self._ket_to_observation(self.current_ket)

        # Calculate inner product of state with itself (should be ~1 for normalized states)
        inner_product = np.abs(np.vdot(self.current_ket, self.current_ket))
        self.min_inner_product = min(self.min_inner_product, inner_product)
        self.past_ket = self.current_ket
        
        return observation, {}

    def step(self, action):
        """Executes one time step in the environment.

        This method applies the agent's action to the quantum circuit, evolves
        the state, and calculates the reward.

        Args:
            action (np.ndarray): The normalized control parameters in [-1, 1] for the squeezing and
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

        # 1. Denormalize action from [-1, 1] to original ranges
        action = self._denormalize_action(action)

        # 2. Unpack and clip the agent's action based on tunable_r setting
        if self.tunable_r:
            # 3D action: [squeezing_r, BS_angle, squeezing_phase]
            squeezing_r = np.clip(action[0], -self.max_squeezing, self.max_squeezing)
            theta_1 = np.clip(action[1], 0, np.pi/2)
            squeezing_phase = 0
        else:
            # 2D action: [BS_angle, squeezing_phase], squeezing_r is fixed
            squeezing_r = self.max_squeezing
            theta_1 = np.clip(action[1], 0, np.pi/2)
            squeezing_phase = np.clip(action[0], -np.pi, np.pi)

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
        # Multi-mode: tensor[i,j] = coefficient for |i⟩_mode0 ⊗ |j⟩_mode1
        # After MonitoredLossMeasureFock on q[0] and BSgate(π/2) swap:
        # - Mode 0 is measured and placed in vacuum, then swapped to mode 1
        # - Mode 1 (unmeasured) is swapped to mode 0
        # The state of the current mode 0 is: full_ket[:, 0] (mode 1 in vacuum)
        self.current_ket = full_ket[:, 0]
        observation = self._ket_to_observation(self.current_ket)

        # Calculate inner product of state with itself (should be ~1 for normalized states)
        inner_product = np.abs(np.vdot(self.current_ket, self.current_ket))
        self.min_inner_product = min(self.min_inner_product, inner_product)

        # 5. Calculate the reward by computing fidelities for all target states
        fidelities = np.array([fidelity_max_rotation(target_ket, self.current_ket) 
                               for target_ket in self.target_kets])
    
        fidelity = np.max(fidelities)
        
        
        # 5. SMART EPISODE LOGIC (Definition 3 in paper)
        # Terminate if max steps reached OR current performance surpasses target fidelity
        # 2. Calculate Immediate Step Reward

        # 3. Check Termination
        terminated = False
        target_fidelity = 0.9
        hit_target = (fidelity >  target_fidelity)
        
        max_reward = self._calculate_log_reward(target_fidelity)
        reward = self._calculate_log_reward(fidelity)
        reward -= max_reward

        self_fidelity = fidelity_max_rotation(self.past_ket, self.current_ket)
        if self_fidelity > 0.95:
            reward -= max_reward

        self.past_ket = self.current_ket

        if hit_target:
            reward += 10 * max_reward
            terminated = True
        
        truncated = self.current_step >= self.max_steps
        # if truncated and not hit_target:
        #     reward -= 2

        # # 4. TARGET-BASED COMPLETION BONUS
        # if hit_target:
        #     # A. Determine the "Value" of the Target
        #     # We calculate what the reward IS at exactly the target threshold.
        #     # This standardizes the bonus relative to the difficulty.
        #     target_value = self._calculate_log_reward(self.target_fidelity)
            
        #     # B. The "Big Win" Bonus (10x the target value)
        #     big_win_bonus = 10.0
            
        #     # C. The "Time Savings" Bonus (simulate getting the target value for the rest of time)
        #     steps_remaining = self.max_steps - self.current_step
        #     time_saving_bonus = steps_remaining 
            
        #     # Total Bonus added to the current step
        #     reward += (big_win_bonus + time_saving_bonus)


        
        # The 'info' dictionary is the standard place for diagnostic information.
        # Decode the monitored loss measurement result
        encoded_result = result.samples[0][0]
        lost_photons, detected_photons = decode_measurement_result(encoded_result)
        info = {
            'photon_loss': lost_photons,
            'detected_photons': detected_photons,
            'is_success': hit_target, # Flag for Curriculum Manager
            'total_photons': lost_photons + detected_photons,
            'fidelity': fidelity
        }
        
        if truncated:
            info['terminal_bonus'] = 0
            info['final_ket'] = self.current_ket
            info['min_inner_product'] = self.min_inner_product
        
        if terminated or truncated:
            info['is_success'] = hit_target
            info['curriculum_difficulty'] = self.target_fidelity
            # Helpful for debugging:
            info['episode_len'] = self.current_step 

        return observation, reward, terminated, truncated, info

    def render(self):
        """Renders the environment.

        This method is not implemented for this environment.
        """

    def close(self):
        """Closes the environment.

        This method does not require any special cleanup.
        """