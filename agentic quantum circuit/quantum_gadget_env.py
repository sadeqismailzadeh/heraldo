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
from strawberryfields.ops import *

import scipy.sparse as sp

# disable caching to save memory for large cutoff dims
from sf_operations_no_cache import disable_fock_caching
disable_fock_caching() 

# optimized loss channel
from monitored_loss_measure_fock_patch import MonitoredLossMeasureFock, patch_fock_backend, decode_measurement_result
patch_fock_backend()

# Import your original, fully observable environment
from quantum_circuit_env import QuantumCircuitEnv, fidelity_pure_state


def db_to_r(db_value):
        """
        Converts squeezing level from Decibels (dB) to the squeezing parameter r.
        
        Formula: dB = 10 * log10(exp(2r)) => r = dB / (20 * log10(e))
        Approx: r = dB / 8.686
        
        Args:
            db_value (float): Squeezing in dB (e.g., 10.0 for 10dB squeezing).
            
        Returns:
            float: The dimensionless parameter r for the Sgate.
        """
        # 20 * np.log10(np.e) is approximately 8.685889
        r_value = db_value / (20 * np.log10(np.e))
        return r_value


class QuantumGadgetEnv(QuantumCircuitEnv):
    """
    Implements the full 'Modified Article 2' architecture with Time-Multiplexing
    and Displacement control.
    
    Architecture:
    1. RESET: Simulates the Article 2 Gadget (Two displaced squeezed beams collide).
       One output is measured, the other enters the loop.
    2. STEP:  A FRESH ancilla is generated with Squeezing AND Displacement.
       It interacts with the loop, is measured, and the loop continues.
    """

    def __init__(self, **kwargs):
        """
        Args:
            max_displacement (float): Maximum magnitude for the displacement alpha.
            gadget_initial_params (dict): Parameters for the t=0 gadget setup.
                                          {'r': 1.0, 'alpha': 1.0, 'theta': pi/4}
            **kwargs: Arguments for the parent QuantumCircuitEnv.
        """
        # Initialize parent
        super().__init__(**kwargs)
        
        self.max_disp = 2.0
        self.max_squeezing = db_to_r(10)

        self.non_gaussian_weight = 0.6

        self._precompute_quadrature_operators()
        
        # Default gadget parameters (Article 2 style) if none provided

        # --- REDEFINE ACTION SPACE ---
        # The agent now controls 5 parameters per step:
        # 1. Squeezing Magnitude (r)
        # 2. Beam Splitter Angle (theta)
        # 3. Squeezing Phase (phi_sq)
        # 4. Displacement Magnitude (alpha_mag) 
        # 5. Displacement Phase (alpha_phi)    
        self.action_space = spaces.Box(
            low=np.array([0.0, 0.0, -np.pi, 0.0, -np.pi]),
            high=np.array([self.max_squeezing, np.pi/2, np.pi, self.max_disp, np.pi]),
            shape=(5,),
            dtype=np.float32
        )


    def _precompute_quadrature_operators(self):
        """
        Pre-computes the X quadrature operator as a sparse matrix.
        X = (a + a_dag) / sqrt(2)
        """
        dim = self.cutoff_dim
        
        # 1. Create the data for the 'a' operator (creation)
        # It only has values on the first super-diagonal (k=1)
        sqrt_n = np.sqrt(np.arange(1, dim))
        
        # 2. Construct sparse 'a' and 'a_dag'
        # diags(data, offsets, shape)
        # offset=1 is super-diagonal, offset=-1 is sub-diagonal
        self.a_op_sparse = sp.diags([sqrt_n], [1], shape=(dim, dim), format='csr')
        self.a_dag_op_sparse = self.a_op_sparse.T  # Transpose of real matrix
        
        # 3. Construct Sparse X operator
        # Note: Adding sparse matrices preserves sparsity
        self.x_op_sparse = (self.a_op_sparse + self.a_dag_op_sparse) / np.sqrt(2)
        
        # 4. Pre-compute powers of X (X^2, X^3, X^4) 
        # Why? Because computing them once here is faster than doing dot products every step.
        # Since X is tridiagonal, X^2 is pentadiagonal, etc. Still very sparse.
        self.x2_sparse = self.x_op_sparse.dot(self.x_op_sparse)
        self.x3_sparse = self.x2_sparse.dot(self.x_op_sparse)
        self.x4_sparse = self.x2_sparse.dot(self.x2_sparse)

    def compute_non_gaussianity(self, state_ket):
        """
        Computes Non-Gaussianity using sparse matrix multiplication.
        """
        # Ensure complex128 for precision
        ket = state_ket.astype(np.complex128)
        ket_conj = np.conj(ket)

        # Helper for expectation value <O> = <psi|O|psi>
        # sparse_matrix.dot(vector) returns a dense vector
        def expect_sparse(sparse_op):
            # 1. Sparse Matrix x Vector -> Vector (Fast)
            op_psi = sparse_op.dot(ket) 
            # 2. Vector dot Vector -> Scalar
            return np.real(np.vdot(ket, op_psi))

        # 1. Calculate Raw Moments <X^n> directly using precomputed sparse matrices
        m1_raw = expect_sparse(self.x_op_sparse)
        m2_raw = expect_sparse(self.x2_sparse)
        m3_raw = expect_sparse(self.x3_sparse)
        m4_raw = expect_sparse(self.x4_sparse)

        # 2. Convert to Central Moments (Standard Statistics Formulas)
        # This is faster than centering the matrix itself.
        
        # Variance: E[X^2] - E[X]^2
        var = m2_raw - m1_raw**2
        
        if var < 1e-6: return 0.0

        sigma = np.sqrt(var)
        
        # Skewness: (E[X^3] - 3*mu*sigma^2 - mu^3) / sigma^3
        # Simplified central moment formula: E[(X-mu)^3]
        moment3_central = m3_raw - 3*m1_raw*m2_raw + 2*(m1_raw**3)
        skewness = moment3_central / (sigma**3)

        # Kurtosis: E[(X-mu)^4]
        moment4_central = m4_raw - 4*m1_raw*m3_raw + 6*(m1_raw**2)*m2_raw - 3*(m1_raw**4)
        excess_kurtosis = (moment4_central / (var**2)) - 3.0

        # 3. Return Score
        return np.abs(skewness) + 0.1 * np.abs(excess_kurtosis)


    def _precompute_quadrature_operators2(self):
        """
        Pre-computes the X quadrature operator in the Fock basis for fast moment calculation.
        X = (a + a_dag) / sqrt(2)
        """
        # Creation and Annihilation operators
        # a |n> = sqrt(n) |n-1>
        dim = self.cutoff_dim
        
        # Diagonals for creation/annihilation
        sqrt_n = np.sqrt(np.arange(1, dim))
        
        # Construct 'a' matrix (upper diagonal)
        self.a_op = np.zeros((dim, dim), dtype=np.float64)
        np.fill_diagonal(self.a_op[:, 1:], sqrt_n)
        
        # Construct 'a_dag' matrix (lower diagonal)
        self.a_dag_op = self.a_op.T
        
        # Construct X operator
        self.x_op = (self.a_op + self.a_dag_op) / np.sqrt(2)

    def compute_non_gaussianity2(self, state_ket):
        """
        Computes a 'Non-Gaussianity Score' based on Skewness and Kurtosis of the X quadrature.
        Gaussian states will score ~0. Non-Gaussian states will score > 0.
        """
        # Ensure pure state density matrix for expectation values: rho = |psi><psi|
        # But we can do vector multiplication <psi|Op|psi> which is faster.
        ket = state_ket.astype(np.complex128)
        
        # Helper for expectation value <O> = <psi|O|psi>
        def expect(op_matrix):
            # op_matrix is real symmetric, ket is complex
            # result = dot(conj(ket), dot(op, ket))
            return np.real(np.vdot(ket, op_matrix.dot(ket)))

        # 1. Calculate Moments of X
        # We need <X>, <X^2>, <X^3>, <X^4>
        
        # <X>
        mu = expect(self.x_op)
        
        # Center the operator: X_centered = X - mu*I
        # It's faster to compute moments and correct them, but for stability:
        x_centered = self.x_op - np.eye(self.cutoff_dim) * mu
        
        # Calculate centered moments
        x2_op = x_centered.dot(x_centered)
        var = expect(x2_op) # Variance (<X^2>)
        
        x3_op = x2_op.dot(x_centered)
        m3 = expect(x3_op)  # 3rd Central Moment
        
        x4_op = x2_op.dot(x2_op)
        m4 = expect(x4_op)  # 4th Central Moment
        
        # 2. Calculate Skewness and Kurtosis
        # Skewness = m3 / sigma^3
        # Kurtosis = m4 / sigma^4 - 3
        
        sigma = np.sqrt(var)
        if sigma < 1e-6: return 0.0 # Avoid division by zero for unphysical states
        
        skewness = m3 / (sigma ** 3)
        excess_kurtosis = (m4 / (var ** 2)) - 3.0
        
        # 3. Combine into a Score
        # For Cubic Phase states (Article 2), Skewness is the dominant feature.
        # For Cat states, Kurtosis is dominant.
        # We sum absolute values to reward ANY non-Gaussianity.
        score = np.abs(skewness) + 0.1 * np.abs(excess_kurtosis)
        
        return score

    def _initialize_target_states(self):
        """
        Generates the Cubic Phase Resource State from Eq. (1) of Article 2.
        Target = N * (|0> + i*a*sqrt(1.5)|1> + i*a|3>)
        """

        print("Target state is cubic Phase Resource State")
        # Parameter 'a' from the paper (e.g., 0.3, 0.61, etc.)
        # You should probably pass this in __init__, but hardcoding 0.61 is fine for testing.
        a = 0.61

        # 1. Define coefficients
        c0 = 1.0 + 0j
        c1 = 0.0 + 1j * a * np.sqrt(1.5) # i * a * sqrt(3/2)
        c2 = 0.0 + 0j                    # No |2> component
        c3 = 0.0 + 1j * a                # i * a

        # 2. Construct vector
        target_ket = np.zeros(self.cutoff_dim, dtype=np.complex128)
        target_ket[0] = c0
        target_ket[1] = c1
        target_ket[2] = c2
        target_ket[3] = c3

        # 3. Normalize
        norm = np.linalg.norm(target_ket)
        target_ket = target_ket / norm

        # Return as a list (standard format for your Env)
        return [target_ket]

    def reset(self, seed=None, options=None):
        """
        Initializes the loop using the Displaced Gadget logic (Article 2, Fig 2).
        """
        # Call parent reset logic (seeding, etc) but skip the circuit part
        super(QuantumCircuitEnv, self).reset(seed=seed)
        
        self.eng = sf.Engine("fock", backend_options={"cutoff_dim": self.cutoff_dim})
        self.current_step = 0
        self.min_inner_product = 1.0

        # --- STEP 0: THE GADGET INITIALIZATION ---
        prog = sf.Program(2)
        with prog.context as q: 
            MonitoredLossMeasureFock(self.loss_channel) | q[0]

        result = self.eng.run(prog)
        self.current_state = result.state
        
        # Extract Observation
        self.current_ket = self.current_state.ket()[:, 0]
        observation = self._ket_to_observation(self.current_ket)
        
        # Inner product check
        inner = np.abs(np.vdot(self.current_ket, self.current_ket))
        self.min_inner_product = min(self.min_inner_product, inner)

        return observation, {}

    def step(self, action):
        """
        Executes one RL correction step using a Displaced Ancilla.
        """
        self.current_step += 1

        # 1. Unpack 5D Action
        # [r, theta, phi_sq, alpha_mag, alpha_phi]
        r_val = np.clip(action[0], 0, self.max_squeezing)
        theta_val = np.clip(action[1], 0, np.pi/2)
        phi_sq_val = np.clip(action[2], -np.pi, np.pi)
        alpha_mag = np.clip(action[3], 0, self.max_disp)
        alpha_phi = np.clip(action[4], -np.pi, np.pi)

        # 2. Build Circuit
        prog = sf.Program(2)
        with prog.context as q:
            # --- ANCILLA PREPARATION (q[1]) ---
            # Squeezing
            Sgate(r_val, phi_sq_val) | q[1]
            # Displacement (NEW: The user diagram explicitly asks for |z, alpha>)
            Dgate(alpha_mag, alpha_phi) | q[1]

            # --- INTERACTION ---
            # q[0] is the Loop State, q[1] is the Fresh Ancilla
            BSgate(theta_val, 0) | (q[0], q[1])

            # --- MEASUREMENT ---
            MonitoredLossMeasureFock(self.loss_channel) | q[0]

            # --- LOOP RECYCLE (SWAP) ---
            BSgate(np.pi/2, 0) | (q[0], q[1])

        # 3. Run
        result = self.eng.run(prog)
        self.current_state = result.state
        self.current_ket = self.current_state.ket()[:, 0]
        observation = self._ket_to_observation(self.current_ket)

        # 4. Reward Calculation (Identical to parent)
        inner = np.abs(np.vdot(self.current_ket, self.current_ket))
        self.min_inner_product = min(self.min_inner_product, inner)

        fidelities = np.array([fidelity_pure_state(target, self.current_ket) 
                               for target in self.target_kets])
        max_fidelity = np.max(fidelities)
    

        # 2. Calculate NG Score
        ng_score = self.compute_non_gaussianity(self.current_ket)
        
        # 3. Define Parameters for the Multiplier
        # Lambda: Strength of the effect. 1.0 means NG can double or nullify the reward.
        ng_weight = 1.0 
        
        # Threshold: Below this = Penalty, Above this = Bonus
        # 0.2 is a good baseline for "Definitely not Gaussian"
        ng_threshold = 0.2 
        
        # 4. Calculate the Proportional Multiplier
        # Logic: Multiplier = 1 + weight * (Score - Threshold)
        # Example: Score=0.0 (Vacuum) -> Mult = 1 + 1*(-0.2) = 0.8 (20% Penalty)
        # Example: Score=0.5 (Target) -> Mult = 1 + 1*(0.3)  = 1.3 (30% Bonus)
        ng_multiplier = 1.0 + ng_weight * (ng_score - ng_threshold)
        
        # Safety Clip: Don't let penalty go below 0 (negative rewards can be unstable)
        # We ensure the multiplier is at least 0.1 (10% of original fidelity)
        ng_multiplier = max(0.1, ng_multiplier)

        # 5. Final Reward Calculation
        fid_term = max_fidelity ** self.reward_power
        reward = fid_term * ng_multiplier




        # 5. Termination Logic
        terminated = False
        truncated = self.current_step >= self.max_steps
        
        if truncated:
             # Add terminal bonus
             terminal_bonus = 0
             if max_fidelity > 0.0:
                 terminal_bonus += (max_fidelity ** self.reward_power) * 10 * ng_multiplier
             
             fidelity_threshold = 0.9
             if max_fidelity > fidelity_threshold:
                 excess = (max_fidelity - fidelity_threshold) / (1 - fidelity_threshold)
                 terminal_bonus += (excess ** self.reward_power) * 100 * ng_multiplier
             
             reward += terminal_bonus

        # Info
        encoded_result = result.samples[0][0]
        lost, detected = decode_measurement_result(encoded_result)
        info = {
            'photon_loss': lost,
            'detected_photons': detected,
            'total_photons': lost + detected,
            'fidelity': max_fidelity,
            'ng_score': ng_score
        }
        
        if truncated:
            info['terminal_bonus'] = terminal_bonus
            info['final_ket'] = self.current_ket
            info['min_inner_product'] = self.min_inner_product

        return observation, reward, terminated, truncated, info

# --- USAGE EXAMPLE ---
# env = DisplacedLoopedGadgetEnv(
#     cutoff_dim=25, 
#     max_steps=10, 
#     tunable_r=True, # Ignored, this class enforces its own action space
#     gadget_initial_params={'r': 0.5, 'alpha': 1.2, 'theta': 0.8, 'phi': 0.0}
# )