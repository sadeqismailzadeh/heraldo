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


import gymnasium as gym
from gymnasium import spaces
import numpy as np
import strawberryfields as sf
from strawberryfields.ops import *
import strawberryfields.ops as ops
import qutip as qt
from scipy.special import factorial
import matplotlib.pyplot as plt # Needed for the demo

# Import your existing base infrastructure
from base_quantum_env import BaseQuantumEnv, fidelity_max_rotation, decode_measurement_result
from monitored_loss_measure_fock_patch import MonitoredLossMeasureFock

# --- HELPER: QuTiP Target Generators (from previous discussion) ---

def sqrGKP_qutip(mu, d, delta, cutoff, nmax=25):  
    """Generates Square GKP target using QuTiP."""
    n1 = np.arange(-nmax, nmax+1)[:, None]
    n2 = np.arange(-nmax, nmax+1)[None, :]

    # Lattice spacing L = sqrt(4*pi) for square
    # arg1 handles the phase checkerboard pattern for logical states
    arg1 = 1j * np.pi * n2 * (d * n1 + mu) / d
    amplitude = (np.exp(arg1)).flatten()[:, None]

    alpha = np.sqrt(np.pi / d) * ((d * n1 + mu - 1j * n2))
    alpha = alpha.flatten()[:, None]
    n = np.arange(cutoff)[None, :]
    
    coherent = np.exp(-0.5 * np.abs(alpha)**2) * alpha**n / np.sqrt(factorial(n))
    state_vector = np.sum(amplitude * coherent * np.exp(-n * delta**2), axis=0).reshape(-1, 1)
    
    return qt.Qobj(state_vector).unit()



def hexGKP(mu, d, delta, cutoff, nmax=20):
    r"""Hexagonal GKP code state (QuTiP 5 compatible)."""
    n1 = np.arange(-nmax, nmax+1)[:, None]
    n2 = np.arange(-nmax, nmax+1)[None, :]

    n1sq = n1**2
    n2sq = n2**2

    sqrt3 = np.sqrt(3)

    # Complex phase and envelope arguments
    arg1 = -1j * np.pi * n2 * (d * n1 + mu) / d
    arg2 = -np.pi * (d**2 * n1sq + n2sq - d * n1 * (n2 - 2 * mu) - n2 * mu + mu**2) / (sqrt3 * d)
    arg2 *= 1 - np.exp(-2 * delta**2)

    amplitude = (np.exp(arg1)).flatten()[:, None]

    # Hexagonal lattice displacement amplitudes
    alpha = np.sqrt(np.pi / (2 * sqrt3 * d)) * (sqrt3 * (d * n1 + mu) - 1j * (d * n1 - 2 * n2 + mu))
    alpha = alpha.flatten()[:, None]

    n = np.arange(cutoff)[None, :]
    coherent = np.exp(-0.5 * np.abs(alpha)**2) * alpha**n / np.sqrt(factorial(n))
    
    # Sum and Reshape for QuTiP 5
    state_vector = np.sum(amplitude * coherent * np.exp(-n * delta**2), axis=0).reshape(-1, 1)
    
    return qt.Qobj(state_vector).unit()

# --- MAIN ENVIRONMENT ---

class GKPStateEnv(BaseQuantumEnv):
    """
    RL Environment for GKP State Preparation (Square & Hex).
    
    Action Space (5 Continuous params):
    1. Squeezing Magnitude (r)
    2. Squeezing Angle (phi)
    3. Displacement Magnitude (|alpha|)
    4. Displacement Angle (arg(alpha))
    5. Beamsplitter Transmissivity (theta)
    """

    def __init__(self, 
                 gkp_type='square',   # 'square' or 'hex'
                 mu=0,             # Logical 0 or 1
                 delta=0.58,        # Finite energy envelope
                 cutoff_dim=80,    # GKP requires high cutoff!
                 max_steps=50, 
                 loss_channel=1.0, 
                 **kwargs):
        
        self.gkp_type = gkp_type.lower()
        self.mu = mu
        self.delta = delta
        

        self.fixed_measure_beta = 2 # Applied as i*2.5
        
        self.max_sq_r = 1.15
        self.max_disp_alpha = 1.5
        
        super().__init__(cutoff_dim=cutoff_dim, max_steps=max_steps, loss_channel=loss_channel, 
                         **kwargs)

    def _define_action_space(self):
        """
        Defines the 5-element continuous action space.
        Normalized to [-1, 1] for the agent, scaled in _build_step_program.
        """        
        self.action_ranges = {
            'r_mag':    (0.0, self.max_sq_r),
            'r_phi':    (0.0, 2 * np.pi),
            'd_mag':    (0.0, self.max_disp_alpha),
            'd_phi':    (0.0, 2 * np.pi),
            'd_mag2':    (0.0, self.fixed_measure_beta),
            'd_phi2':    (0.0, 2 * np.pi),
            'bs_theta': (0.0, np.pi/2)
        }

        self.action_keys = list(self.action_ranges.keys())

        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(len(self.action_keys),), dtype=np.float32
        )

    def set_curriculum_params(self, fidelity_threshold, envelope_delta):
        """
        Updates the curriculum parameters.
        If delta changes, we must re-generate the target ket.
        """
        # 1. Update Fidelity Threshold
        self.target_fidelity = float(fidelity_threshold)
        
        # 2. Update Envelope (Delta) if changed
        # We use a small epsilon for float comparison
        if abs(self.delta - envelope_delta) > 1e-6:
            print(f"Worker updating Delta: {self.delta:.4f} -> {envelope_delta:.4f}")
            self.delta = float(envelope_delta)
            
            # 3. RE-GENERATE TARGET STATE
            # This is crucial because the physical target definition changed
            self.target_kets = self._initialize_target_states()

            self.target_ng_score =self.compute_non_gaussianity(self.target_kets[0])
            print(f"target ng = {self.target_ng_score:.2f}")
            
        return self.target_fidelity, self.delta


    
    # def _initialize_target_states(self):
    #     self.target_delta = 0.05
    #     self.target_s_r = 1
    #     """
    #     Generates the Quartic Phase State Target manually.
    #     |δ, s⟩ = exp(i * δ * x^4) S(s) |0⟩
    #     """
    #     print(f"Generating Target Quartic Phase State (delta={self.target_delta}, cutoff={self.cutoff_dim})...")
        
    #     # 1. Generate Squeezed State S(s)|0> using SF
    #     prog = sf.Program(1)
    #     with prog.context as q:
    #         Sgate(-self.target_s_r) | q[0]
    #         # Rgate(-1) | q[0]

    #     eng = sf.Engine("fock", backend_options={"cutoff_dim": self.cutoff_dim})
    #     state = eng.run(prog).state
    #     ket_init = state.ket() # Shape (cutoff_dim,)

    #     # 2. REUSE PRECOMPUTED X4
    #     # We convert sparse -> dense because scipy.linalg.expm is generally 
    #     # more robust/faster for full matrix exponentiation at dim ~60 than sparse routines.
    #     x4_dense = self.x4_sparse.toarray()
        
    #     # 3. Matrix Exponentiation U = exp(i * delta * x^4)
    #     U_quartic = scipy.linalg.expm(1j * self.target_delta * x4_dense)
        
        
    #     # 3. Apply U to ket
    #     target_ket = U_quartic @ ket_init
        
    #     # 4. Normalize (numerical stability)
    #     norm = np.linalg.norm(target_ket)

    #     print(f'norm target before renormalize={norm}')
    #     target_ket = target_ket / norm
        
    #     # Return as a list (standard format for our env)
    #     return [target_ket]
    
    # def _initialize_target_states(self):
    #     # Paper constants
    #     self.target_gamma = -0.2
    #     self.target_r = -0.7
    #     self.target_alpha = 1.25  # Magnitude (applied as i*1.25)
    #     """
    #     Generates the Cubic Phase State Target.
    #     Equation 3: |γ, r, α⟩ = D(α) exp(iγQ^3) S(r) |0⟩
    #     """
    #     print("Generating Target Cubic Phase State...")
        
    #     prog = sf.Program(1)
    #     with prog.context as q:
    #         # Note on ordering: The equation applies operators right-to-left on vacuum.
    #         # 1. Squeezing S(r)
    #         Sgate(self.target_r) | q[0]
            
    #         # 2. Cubic Phase Gate V(gamma) = exp(i * gamma * x^3)
    #         # Note: SF uses hbar=2 convention by default. 
    #         # Vgate matches the definition of Q^3 interaction.
    #         Vgate(self.target_gamma*2) | q[0]
            
    #         # 3. Displacement D(alpha)
    #         # Paper Fig 2 caption says alpha = i1.25
    #         # numpy complex to polar

    #         z = 1j * self.target_alpha

    #         # Convert to polar
    #         r = np.abs(z)
    #         theta = np.angle(z)

    #         Dgate(r, theta) | q[0]

    #     eng = sf.Engine("fock", backend_options={"cutoff_dim": self.cutoff_dim})
    #     state = eng.run(prog).state
    #     assert state.is_pure
        
    #     # We return a list containing the single target ket
    #     return [state.ket()]
    
    def _initialize_target_states(self):
        """
        Generates the target GKP state using QuTiP, then converts to numpy
        compatible with Strawberry Fields.
        """
        print(f"Generating {self.gkp_type.upper()} GKP Target (mu={self.mu}, delta={self.delta}, N={self.cutoff_dim})...")
        
        if 'hex' in self.gkp_type:
            qobj_tgt = hexGKP(self.mu, 2, self.delta, self.cutoff_dim)
        else:
            qobj_tgt = sqrGKP_qutip(self.mu, 2, self.delta, self.cutoff_dim) # d=2 for qubit

        # Convert QuTiP Qobj to Numpy Array (flattened for SF)
        # QuTiP shape is (N, 1), we need (N,)
        target_np = qobj_tgt.full().flatten()
        
        # Ensure it's normalized
        target_np /= np.linalg.norm(target_np)
        
        return [target_np]

    def _build_reset_program(self):
        """
        Initializes the loop with a squeezed state.
        GKP preparation often starts with a squeezed state.
        """
        prog = sf.Program(2)
        with prog.context as q:
            # Initialize with standard squeezing along X
            Sgate(self.max_sq_r) | q[0]
        return prog

    def _get_current_ket(self, state):
        """Extracts ket of the loop mode."""
        return state.ket()[:, 0] 

    def _build_step_program(self, action):
        """
        Builds the circuit for one time step.
        Action: [r_mag, r_phi, d_mag, d_phi, bs_theta]
        """
        # Unpack actions
        r_mag = action[0]
        r_phi = action[1]
        
        d_mag = action[2]
        d_phi = action[3]
        
        d_mag2 = action[4]
        d_phi2 = action[5]

        bs_theta = action[6]

        prog = sf.Program(2)
        with prog.context as q:
            # q[0] = Loop Memory
            # q[1] = Fresh Input (Ancilla)
            
            # 1. Prepare Ancilla
            # Full control over Squeezing (magnitude and ANGLE)
            Sgate(r_mag, r_phi) | q[1]
            
            # Full control over Displacement (magnitude and ANGLE)
            Dgate(d_mag, d_phi) | q[0]
            
            # 2. Interact
            BSgate(bs_theta, 0) | (q[0], q[1])
            
            # 3. Measurement Setup
            # For GKP, sometimes specific measurement bases are needed, 
            # but usually the adaptive nature of RL handles the displacements.
            # We keep the measurement displacement learned by the agent (via the loop logic)
            # or we could add a fixed measurement parameter. 
            # For now, we assume the agent controls the state *before* this point.
            # Full control over Displacement (magnitude and ANGLE)
            Dgate(d_mag2, d_phi2) | q[0]

        self.current_state = self.eng.run(prog).state
        self.current_ket = self.current_state.ket()
        inner_product = np.real(np.vdot(self.current_ket, self.current_ket))
        self.min_inner_product = min(self.min_inner_product, inner_product)
        
        # PNR Measurement Step
        prog_meas = sf.Program(2)
        with prog_meas.context as q:
            MonitoredLossMeasureFock(self.loss_channel) | q[0]
            # Swap logic to keep the state in q[0]
            BSgate(np.pi/2, 0) | (q[0], q[1])
            
        return prog_meas

    def _calculate_fidelity(self, state_ket):
        """
        Calculates fidelity. 
        Note: GKP states are rotationally symmetric (modularly), but absolute rotation matters.
        fidelity_max_rotation helps the agent learn the shape first, then lock phase.
        """
        return fidelity_max_rotation(self.target_kets[0], state_ket)

    def _calculate_reward_and_termination(self, fidelity, result):
        terminated = False
        
        
        # max_reward = self._calculate_reward(self.target_fidelity)
        max_reward = self._calculate_reward(0.99)
        reward = self._calculate_reward(fidelity)
        reward -= max_reward
        current_ng_score = self.compute_non_gaussianity(self.current_ket)
        # self.target_ng_score = self.compute_non_gaussianity(self.current_ket)

        # reward -= 1 # for time
        self_fidelity = fidelity_max_rotation(self.past_ket, self.current_ket)
        if self_fidelity > 0.95:
            reward -= max_reward

        hit_target = (fidelity > self.target_fidelity) and (current_ng_score > self.target_ng_score /2 )
        if hit_target:
            reward += 10 * max_reward
            terminated = True

        encoded_result = result.samples[0][0]
        lost_photons, detected_photons = decode_measurement_result(encoded_result)

        info = {
            'photon_loss': lost_photons,
            'detected_photons': detected_photons,
            'is_success': hit_target,
            'fidelity': fidelity,
            'ng_score': current_ng_score,
        }

        return reward, terminated, info
    


if __name__ == "__main__":
    # sf.hbar = 1
    # 1. Setup
    print("Initializing Environment...")
    # Use Hex GKP, Logical 0, cutoff 60
    env = GKPStateEnv(gkp_type='square', cutoff_dim=60)
    
    # 2. Get Target State
    target_ket = env.target_kets[0]
    
    # Calculate Fock Probabilities
    # |<n|psi>|^2
    probs = np.abs(target_ket)**2
    
    # 3. Calculate Wigner Function (using temporary SF engine)
    prog = sf.Program(1)
    with prog.context as q:
        ops.Ket(target_ket) | q[0]
    eng = sf.Engine("fock", backend_options={"cutoff_dim": 60})
    state = eng.run(prog).state
    
    print("Calculating Wigner Function (High Res)...")
    grid_size = 200
    limit = 7 * np.sqrt(2) 
    xvec = np.linspace(-limit, limit, grid_size)
    W = state.wigner(mode=0, xvec=xvec, pvec=xvec)
    
    # 4. Improved Plotting
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # --- Plot 1: Wigner Function ---
    # Use pcolormesh for better rendering of the grid, or contourf with many levels
    # Centering the colormap at 0 is crucial
    max_w = np.max(np.abs(W))
    
    c = ax1.contourf(xvec, xvec, W, 120, cmap='RdBu', 
                     vmin=-max_w, vmax=max_w)
    
    ax1.set_title("Target State Wigner Function (Hex GKP)")
    ax1.set_xlabel("x (Position)")
    ax1.set_ylabel("p (Momentum)")
    ax1.set_aspect('equal') # Fixes the "squishing" issue
    
    # Add subtle grid lines to guide the eye
    ax1.axhline(0, color='black', alpha=0.2, linestyle='--')
    ax1.axvline(0, color='black', alpha=0.2, linestyle='--')
    fig.colorbar(c, ax=ax1, label='W(x,p)')
    
    # --- Plot 2: Fock Probabilities ---
    # GKP states have specific "teeth" in Fock space
    ax2.bar(range(len(probs)), probs, color='purple', alpha=0.8, edgecolor='black', width=0.8)
    ax2.set_title("Fock State Distribution")
    ax2.set_xlabel("Photon Number |n>")
    ax2.set_ylabel("Probability")
    ax2.set_xlim(-0.5, 60.5)
    
    plt.tight_layout()
    plt.show()

    # 5. Quick Step Check
    print("Testing Step...")
    obs, _ = env.reset()
    # No-op action (0s in normalized space -> middle of ranges)
    # [r=0.7, phi=pi, d=1.5, phi=pi, bs=pi/4]
    action = np.zeros(5, dtype=np.float32) 
    obs, rew, term, trunc, info = env.step(action)
    print(f"Initial Step Fidelity: {info['fidelity']:.5f}")