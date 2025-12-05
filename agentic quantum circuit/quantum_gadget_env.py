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
from quantum_circuit_env import QuantumCircuitEnv, fidelity_pure_state, fidelity_max_rotation

# TODO clip on step use boundaries defiend on init

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

    (Memory / Loop Mode)
      q[0] (t-1) ═════════════════════════════════════════[ PNR ]═══(Classical Data n_t)
                                                │        (Measure)
                                                │         
                                            [ VBS ] (θ_t)
                                                │
                                                │
      q[1] (New) ════[ S(r,φ) ]═══[ D(α,φ) ]═══════════════> Becomes q[0] (t)
       (Vacuum)      (Squeezing)  (Displacement)             (Feedback to Loop)
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
        self.max_squeezing = db_to_r(8)

        
        self.target_kets = self._initialize_target_states()
        self._precompute_quadrature_operators()
        
        
        self.target_ng_scores = []
        print("\n--- Target State Analysis ---")
        for i, ket in enumerate(self.target_kets):
            score = self.compute_non_gaussianity(ket)
            self.target_ng_scores.append(score)
            
            # Identify the label for the print
            labels = ["0 deg", "90 deg", "180 deg", "270 deg"]
            print(f"Target {i} ({labels[i]}): NG Score = {score:.5f}")
            
        # We just take the score of the first one as the reference for rewards
        self.target_ng_score = self.target_ng_scores[0]
        print("-----------------------------\n")

        # Default gadget parameters (Article 2 style) if none provided

        # --- REDEFINE ACTION SPACE ---
        # Normalized to [-1, 1] for all dimensions, denormalized internally
        # The agent now controls 5 parameters per step:
        # 1. Squeezing Magnitude (r)
        # 2. Beam Splitter Angle (theta)
        # 3. Squeezing Phase (phi_sq)
        # 4. Displacement Magnitude (alpha_mag)
        # 5. Displacement Phase (alpha_phi)
        self.action_ranges = {
            'squeezing_r': (0, self.max_squeezing),
            'squeezing_phase': (-np.pi, np.pi),
            'theta_1': (0, np.pi/2),
            'phi_1': (-np.pi, np.pi),
            'displacement_magnitude': (0, self.max_disp),
            'displacement_phase': (-np.pi, np.pi)
        }
        self.action_keys = ['squeezing_r', 
                            'squeezing_phase', 
                            'theta_1', 
                            'phi_1', 
                            'displacement_magnitude',
                            'displacement_phase']

        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(6,), dtype=np.float32
        )

    
    def compute_non_gaussianity2(self, state_ket):
        """
        Computes non-Gaussianity based on the entropy of the reference Gaussian state.
        For pure states: NG > 0 implies the state is non-Gaussian.
        """
        # 1. Get Covariance Matrix (2x2 for single mode) from SF backend
        # Note: check your SF version, usually state.cov() works. 
        # If using Fock backend, this computes 2nd moments from the Fock distribution.
        cov = self.compute_covariance(state_ket) 
        
        # 2. Calculate Determinant (Symplectic eigenvalue squared)
        det_cov = np.linalg.det(cov)
        
        # 3. For vacuum/squeezed/coherent, det_cov approx 1.0.
        # For cubic phase/cat states, det_cov > 1.0.
        # We use log to scale it nicely.
        ng_score = np.log(det_cov) 
        
        # Clip negative noise (numerical precision errors can give 0.999)
        return max(0.0, ng_score)
    
    def compute_covariance(self, ket):
        """
        Computes the 2x2 Covariance Matrix for a given state vector (ket)
        using pre-computed sparse matrices.
        
        Returns:
            cov_matrix (2x2 np.array): [[Var(X), Cov(X,P)], [Cov(P,X), Var(P)]]
        """
        # Ensure ket is complex for correct math
        ket = ket.astype(np.complex128)
        
        # Helper for expectation value <O> = <psi|O|psi>
        def expect_sparse(sparse_op):
            op_psi = sparse_op.dot(ket) 
            return np.real(np.vdot(ket, op_psi))

        # 1. First Moments (Means)
        mu_x = expect_sparse(self.x_op_sparse)
        mu_p = expect_sparse(self.p_op_sparse)

        # 2. Second Moments
        x2 = expect_sparse(self.x2_sparse)
        p2 = expect_sparse(self.p2_sparse)
        
        # Expectation of Anti-commutator <XP + PX>
        xp_plus_px = expect_sparse(self.xp_plus_px_sparse)

        # 3. Calculate Variances and Covariance
        # Var(X) = <X^2> - <X>^2
        var_x = x2 - mu_x**2
        var_p = p2 - mu_p**2
        
        # Cov(X, P) = 0.5 * <XP + PX> - <X><P>
        cov_xp = 0.5 * xp_plus_px - (mu_x * mu_p)

        # 4. Construct Matrix
        cov_matrix = np.array([
            [var_x, cov_xp],
            [cov_xp, var_p]
        ])
        
        return cov_matrix


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

        # P Operator (Note the 1j)
        # We use P because the target state has 'i' coefficients
        self.p_op_sparse = 1j * (self.a_dag_op_sparse - self.a_op_sparse) / np.sqrt(2)
        
        # Pre-compute Powers of P
        self.p2_sparse = self.p_op_sparse.dot(self.p_op_sparse)
        self.p3_sparse = self.p2_sparse.dot(self.p_op_sparse)
        self.p4_sparse = self.p2_sparse.dot(self.p2_sparse)


        # Pre-compute operators needed for the off-diagonal terms of Covariance
        # XP and PX
        self.xp_sparse = self.x_op_sparse.dot(self.p_op_sparse)
        self.px_sparse = self.p_op_sparse.dot(self.x_op_sparse)
        
        # Anti-commutator {X, P} = XP + PX
        # We use this for the covariance element C_xp
        self.xp_plus_px_sparse = self.xp_sparse + self.px_sparse

    def compute_non_gaussianity(self, state_ket):
        """
        Computes Non-Gaussianity by summing the Negentropy proxies (Skew/Kurtosis)
        of both the Position (X) and Momentum (P) quadratures.
        """
        # Ensure complex128 for precision
        ket = state_ket.astype(np.complex128)

        # Helper for expectation value <O> = <psi|O|psi>
        def expect_sparse(sparse_op):
            op_psi = sparse_op.dot(ket) 
            return np.real(np.vdot(ket, op_psi))

        def get_quadrature_score(m1_op, m2_op, m3_op, m4_op):
            # 1. Raw Moments
            m1 = expect_sparse(m1_op)
            m2 = expect_sparse(m2_op)
            m3 = expect_sparse(m3_op)
            m4 = expect_sparse(m4_op)

            # 2. Central Moments
            # Variance: E[X^2] - E[X]^2
            var = m2 - m1**2
            
            # Safety check for highly squeezed states (avoid division by zero)
            if var < 1e-6: 
                return 0.0

            sigma = np.sqrt(var)
            
            # 3rd Central Moment (Skewness numerator): E[(X-mu)^3]
            # = m3 - 3*mu*m2 + 2*mu^3
            moment3_central = m3 - 3*m1*m2 + 2*(m1**3)
            
            # 4th Central Moment (Kurtosis numerator): E[(X-mu)^4]
            # = m4 - 4*mu*m3 + 6*mu^2*m2 - 3*mu^4
            moment4_central = m4 - 4*m1*m3 + 6*(m1**2)*m2 - 3*(m1**4)

            # 3. Normalized Cumulants
            skewness = moment3_central / (sigma**3)
            excess_kurtosis = (moment4_central / (var**2)) - 3.0

            # Return absolute deviation from Gaussian (0.0)
            return np.abs(skewness) + 0.1 * np.abs(excess_kurtosis)

        # --- Calculate Score for Position (X) ---
        score_x = get_quadrature_score(
            self.x_op_sparse, self.x2_sparse, self.x3_sparse, self.x4_sparse
        )

        # --- Calculate Score for Momentum (P) ---
        score_p = get_quadrature_score(
            self.p_op_sparse, self.p2_sparse, self.p3_sparse, self.p4_sparse
        )

        # Summing them ensures we catch the non-Gaussianity regardless of rotation
        return score_x + score_p
    
    def _initialize_target_states(self):
        """
        Generates the Cubic Phase Resource State and 3 rotated variants.
        Variants:
        1. Standard (0 rad)
        2. Momentum Gate (pi/2 rad)
        3. Negative Strength Gate (pi rad)
        4. Inverse Momentum Gate (3pi/2 rad)
        """
        print("Initializing 4 Cardinal Cubic Phase Target States...")
        
        # Parameter 'a' from Article 2 (e.g., 0.61)
        a = 0.61
        cutoff = self.cutoff_dim

        # --- 1. Construct the Base State (0 degrees) ---
        # Target = N * (|0> + i*a*sqrt(1.5)|1> + i*a|3>)
        base_ket = np.zeros(cutoff, dtype=np.complex128)
        base_ket[0] = 1.0 + 0j
        base_ket[1] = 0.0 + 1j * a * np.sqrt(1.5)
        base_ket[3] = 0.0 + 1j * a
        
        # Normalize
        norm = np.linalg.norm(base_ket)
        base_ket = base_ket / norm

        targets = []
        
        # --- 2. Generate Rotated Variants ---
        # Angles: 0, 90, 180, 270 degrees
        # rotation_angles = [0.0, np.pi/2, np.pi, 3*np.pi/2]
        rotation_angles = [0.0]

        # Create vector of photon numbers [0, 1, 2, ..., cutoff-1]
        n_vec = np.arange(cutoff)

        for theta in rotation_angles:
            # The rotation operator R(theta) adds phase e^(-i * n * theta) to Fock state |n>
            # We construct a phase vector to multiply element-wise with the base ket
            phase_factors = np.exp(1j * n_vec * theta)
            
            # Apply rotation
            rotated_ket = base_ket * phase_factors
            
            targets.append(rotated_ket)

        return targets

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
        self.past_ket = self.current_ket

        fidelities = np.array([fidelity_max_rotation(target, self.current_ket) 
                               for target in self.target_kets])
        fidelity = np.max(fidelities)

        self.past_fidelity = fidelity


        return observation, {}

    def step(self, action):
        """
        Executes one RL correction step using a Displaced Ancilla.
        """
        self.current_step += 1

        # 1. Denormalize action from [-1, 1] to original ranges
        action = self._denormalize_action(action)

        # 2. Unpack 5D Action
        # [r, theta, phi_sq, alpha_mag, alpha_phi]
        r_val = np.clip(action[0], 0, self.max_squeezing)
        phi_sq_val = np.clip(action[1], -np.pi, np.pi)
        theta_val = np.clip(action[2], 0, np.pi/2)
        phi_val = np.clip(action[3], -np.pi, np.pi)
        alpha_mag = np.clip(action[4], 0, self.max_disp)
        alpha_phi = np.clip(action[5], -np.pi, np.pi)

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
            BSgate(theta_val, phi_val) | (q[0], q[1])

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

        fidelities = np.array([fidelity_max_rotation(target, self.current_ket) 
                               for target in self.target_kets])
        fidelity = np.max(fidelities)
    

        # 2. Compute Current Non-Gaussianity
        current_ng_score = self.compute_non_gaussianity(self.current_ket)
        
        # 3. NEW MULTIPLIER LOGIC (Gaussian RBF)
        # We want a function that is 1.0 when current == target
        # And approaches 0.01 when distance is large.


        # 5. Final Reward Calculation 
        terminated = False
        target_fidelity = 0.95
        hit_target = (fidelity >  target_fidelity)
        
        max_reward = self._calculate_log_reward(target_fidelity)
        reward = self._calculate_log_reward(fidelity)
        reward -= max_reward

        if current_ng_score < 1:
            reward -= max_reward

        self_fidelity = fidelity_max_rotation(self.past_ket, self.current_ket)
        if self_fidelity > 0.95:
            reward -= max_reward
        self.past_ket = self.current_ket


        if abs(fidelity - self.past_fidelity) < 0.05:
            reward -= max_reward
        self.past_fidelity = fidelity

        if hit_target:
            reward += 10 * max_reward
            terminated = True
        
        truncated = self.current_step >= self.max_steps

        # Info
        encoded_result = result.samples[0][0]
        lost, detected = decode_measurement_result(encoded_result)
        info = {
            'photon_loss': lost,
            'detected_photons': detected,
            'total_photons': lost + detected,
            'fidelity': fidelity,
            'ng_score': current_ng_score,
            'is_success': hit_target, # Flag for Curriculum Manager
            'self_fidelity': self_fidelity
        }
        
        if truncated:
            # info['terminal_bonus'] = terminal_bonus
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