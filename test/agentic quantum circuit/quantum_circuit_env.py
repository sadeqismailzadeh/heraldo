import gymnasium as gym
from gymnasium import spaces
import numpy as np
from scipy.linalg import sqrtm

# Import Strawberry Fields
import strawberryfields as sf
from strawberryfields.ops import Sgate, BSgate, MeasureFock, Catstate, Rgate

# --- Helper Function for Fidelity (from your code) ---
def uhlmann_jozsa_fidelity(rho, sigma, debug=False):
    """Calculates the Uhlmann-Jozsa fidelity between two density matrices.
    
    Args:
        rho: First density matrix
        sigma: Second density matrix
        debug: If True, prints debug information about potential errors
    
    Returns:
        Fidelity value between 0 and 1
    """
    
    def check_density_matrix_validity(dm, name="density matrix"):
        """Check if a matrix is a valid density matrix."""
        errors = []
        warnings = []
        
        # Check if matrix is square
        if dm.shape[0] != dm.shape[1]:
            errors.append(f"{name} is not square: shape {dm.shape}")
            return errors, warnings
        
        # Check hermiticity
        hermiticity_error = np.max(np.abs(dm - dm.conj().T))
        if hermiticity_error > 1e-10:
            warnings.append(f"{name} hermiticity error: {hermiticity_error:.2e}")
        
        # Check trace
        trace = np.trace(dm)
        trace_error = abs(trace - 1.0)
        if trace_error > 1e-6:
            warnings.append(f"{name} trace = {trace:.6f} (should be 1.0, error: {trace_error:.2e})")
        
        # Check positive semi-definiteness
        try:
            eigenvalues = np.linalg.eigvalsh(dm)
            min_eigenvalue = np.min(eigenvalues)
            if min_eigenvalue < -1e-10:
                errors.append(f"{name} has negative eigenvalue: {min_eigenvalue:.2e}")
            elif min_eigenvalue < -1e-14:
                warnings.append(f"{name} has slightly negative eigenvalue: {min_eigenvalue:.2e}")
        except np.linalg.LinAlgError as e:
            errors.append(f"{name} eigenvalue computation failed: {e}")
        
        return errors, warnings
    
    # Ensure inputs are valid
    if rho is None or sigma is None:
        if debug:
            print("ERROR: One or both density matrices are None")
        return 0.0
    

    if debug:
        # Validate density matrices
        rho = np.array(rho, dtype=np.complex128)
        sigma = np.array(sigma, dtype=np.complex128)
        rho_errors, rho_warnings = check_density_matrix_validity(rho, "rho (first argument)")
        sigma_errors, sigma_warnings = check_density_matrix_validity(sigma, "sigma (second argument)")
        print("\n=== Density Matrix Validation ===")
        if rho_errors:
            print("ERRORS in rho (first argument):")
            for error in rho_errors:
                print(f"  - {error}")
        if rho_warnings:
            print("WARNINGS in rho (first argument):")
            for warning in rho_warnings:
                print(f"  - {warning}")
        
        if sigma_errors:
            print("ERRORS in sigma (second argument):")
            for error in sigma_errors:
                print(f"  - {error}")
        if sigma_warnings:
            print("WARNINGS in sigma (second argument):")
            for warning in sigma_warnings:
                print(f"  - {warning}")
        
        if not rho_errors and not sigma_errors and not rho_warnings and not sigma_warnings:
            print("Both density matrices appear valid.")
    
    # Handle potential numerical instability by ensuring matrices are hermitian
    rho_hermitian = (rho + rho.conj().T) / 2
    sigma_hermitian = (sigma + sigma.conj().T) / 2
    
    try:
        # Calculate sqrt(rho)
        sqrt_rho = sqrtm(rho_hermitian)
        # Check for numerical issues in sqrt_rho
        if debug:
            rho_hermitian_nan = np.isnan(rho_hermitian).any()
            rho_hermitian_inf = np.isinf(rho_hermitian).any()
            if rho_hermitian_nan or rho_hermitian_inf:
                print(f"\nWARNING: sqrt(rho) contains NaN={rho_hermitian_nan}, Inf={rho_hermitian_inf}")
            sqrt_rho_nan = np.isnan(sqrt_rho).any()
            sqrt_rho_inf = np.isinf(sqrt_rho).any()
            if sqrt_rho_nan or sqrt_rho_inf:
                print(f"\nWARNING: sqrt(rho) contains NaN={sqrt_rho_nan}, Inf={sqrt_rho_inf}")
        
        # Calculate the product
        product = sqrt_rho @ sigma_hermitian @ sqrt_rho
        
        # Enforce hermiticity of product
        product_hermitian = (product + product.conj().T) / 2
        
        # Check eigenvalues of product before taking sqrt
        if debug:
            try:
                product_eigenvalues = np.linalg.eigvalsh(product_hermitian)
                min_prod_eigenvalue = np.min(product_eigenvalues)
                if min_prod_eigenvalue < -1e-10:
                    print(f"\nWARNING: Product matrix has negative eigenvalue: {min_prod_eigenvalue:.2e}")
                    print("This will cause issues in sqrt computation.")
            except np.linalg.LinAlgError:
                print("\nWARNING: Could not compute eigenvalues of product matrix")
        
        # Calculate sqrt of product
        sqrt_product = sqrtm(product_hermitian)
        
        # Check for numerical issues in sqrt_product
        if debug:
            sqrt_prod_nan = np.isnan(sqrt_product).any()
            sqrt_prod_inf = np.isinf(sqrt_product).any()
            if sqrt_prod_nan or sqrt_prod_inf:
                print(f"\nWARNING: sqrt(product) contains NaN={sqrt_prod_nan}, Inf={sqrt_prod_inf}")
        
        # Calculate trace and fidelity
        trace_value = np.trace(sqrt_product)
        
        if debug:
            print(f"\nTrace of sqrt(product): {trace_value}")
            if np.abs(trace_value.imag) > 1e-10:
                print(f"WARNING: Trace has significant imaginary part: {trace_value.imag:.2e}")
        
        # Using np.abs() handles potential small imaginary parts from numerical noise
        fidelity = (np.abs(trace_value))**2
        
        # Clamp fidelity to valid range [0, 1]
        if fidelity > 1.0:
            if debug and fidelity > 1.001:
                print(f"\nWARNING: Fidelity {fidelity:.6f} exceeds 1.0, clamping to 1.0")
            fidelity = 1.0
        elif fidelity < 0.0:
            if debug:
                print(f"\nERROR: Fidelity {fidelity:.6f} is negative, setting to 0.0")
            fidelity = 0.0
        
        if debug:
            print(f"\nFinal fidelity: {fidelity:.6f}")
            print("=" * 40)
        
        return fidelity
        
    except Exception as e:
        print(f"\nERROR in fidelity calculation: {type(e).__name__}: {e}")
        print("Returning fidelity = 0.0")
        if debug:
            import traceback
            traceback.print_exc()
        return 0.0

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

        self.current_state = self.eng.run(prog).state
        self.current_dm = self.current_state.reduced_dm(modes=[0])
        
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
        self.current_state = result.state
        self.current_dm = self.current_state.reduced_dm(modes=[0]) # Get partial trace for mode 0

        # 4. Convert the new state to an observation for the agent
        observation = self._dm_to_observation(self.current_dm)

        # 5. Calculate the reward
        # Enable debug mode to see potential numerical issues
        fidelities = [uhlmann_jozsa_fidelity(sigma=self.current_dm, rho=target, debug=False) for target in self.target_dms]
        max_fidelity = np.max(fidelities) if len(fidelities) > 0 else 0.0
        reward = max_fidelity ** self.reward_power

        # 6. Check for termination/truncation
        # The episode ends when the maximum number of steps is reached

        # Give a large bonus reward if a high fidelity is achieved
        terminated = False
        if max_fidelity > 0.95:
             terminated = True
             reward += 10.0 

        truncated = self.current_step >= self.max_steps

        
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