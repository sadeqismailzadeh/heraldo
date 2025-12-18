import scipy.linalg
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import strawberryfields as sf
from strawberryfields.ops import *

# Import the base class and utilities from your existing setup
from base_quantum_env import BaseQuantumEnv, fidelity_pure_state, fidelity_max_rotation, decode_measurement_result
from monitored_loss_measure_fock_patch import MonitoredLossMeasureFock

class QuarticPhaseEnv(BaseQuantumEnv):
    """
    RL Environment for generating Quartic Phase States.
    
    Based on the paper: "Deep reinforcement learning for near-deterministic preparation of 
    cubic- and quartic-phase gates" (Anteneh et al.)
    
    Target State (Eq 4): |δ, s⟩ = exp(i * δ * Q^4) S(s) |0⟩
    
    Differences from Cubic:
    1. Requires higher cutoff_dim (>= 60).
    2. Target generated via matrix exponentiation (no native SF gate).
    3. Quartic unitary is exp(i * delta * x^4).
    """

    def __init__(self, 
                 cutoff_dim=60,  # Paper explicitly says cutoff must be raised to >= 60 for Quartic
                 max_steps=50, 
                 loss_channel=1.0,
                 **kwargs):
        
        self.target_delta = 0.05
        self.target_s_r = 1
        
        # Fixed measurement displacement (similar to cubic strategy, can be tuned)
        self.fixed_measure_beta = 1.5 
        
        # Action constraints
        self.max_sq_r = 1 # Slightly higher squeezing allowance for Quartic
        self.max_disp_alpha = 2
        
        super().__init__(cutoff_dim=cutoff_dim, max_steps=max_steps, loss_channel=loss_channel, 
                         **kwargs)

    def _define_action_space(self):
        """
        Defines the 3-continuous action space:
        1. Input Squeezing (r)
        2. Beam Splitter Transmissivity (via theta)
        3. Input Displacement (alpha)
        """
        self.action_keys = ['r', 'theta', 'alpha']
        
        self.action_ranges = {
            'r': (-self.max_sq_r, self.max_sq_r),
            'theta': (0, np.pi/2),
            'alpha': (-self.max_disp_alpha, self.max_disp_alpha) 
        }

        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(3,), dtype=np.float32
        )

    def _initialize_target_states(self):
        """
        Generates the Quartic Phase State Target manually.
        |δ, s⟩ = exp(i * δ * x^4) S(s) |0⟩
        """
        print(f"Generating Target Quartic Phase State (delta={self.target_delta}, cutoff={self.cutoff_dim})...")
        
        # 1. Generate Squeezed State S(s)|0> using SF
        prog = sf.Program(1)
        with prog.context as q:
            Sgate(-self.target_s_r) | q[0]
            # Rgate(-1) | q[0]

        eng = sf.Engine("fock", backend_options={"cutoff_dim": self.cutoff_dim})
        state = eng.run(prog).state
        ket_init = state.ket() # Shape (cutoff_dim,)

        # 2. REUSE PRECOMPUTED X4
        # We convert sparse -> dense because scipy.linalg.expm is generally 
        # more robust/faster for full matrix exponentiation at dim ~60 than sparse routines.
        x4_dense = self.x4_sparse.toarray()
        
        # 3. Matrix Exponentiation U = exp(i * delta * x^4)
        U_quartic = scipy.linalg.expm(1j * self.target_delta * x4_dense)
        
        
        # 3. Apply U to ket
        target_ket = U_quartic @ ket_init
        
        # 4. Normalize (numerical stability)
        norm = np.linalg.norm(target_ket)

        print(f'norm target before renormalize={norm}')
        target_ket = target_ket / norm
        
        # Return as a list (standard format for our env)
        return [target_ket]

    def _build_reset_program(self):
        """
        Initializes the loop.
        Using a standard high-squeezed state as the resource to start the loop.
        """
        prog = sf.Program(2)
        with prog.context as q:
            Sgate(self.max_sq_r) | q[0]
        return prog

    def _get_current_ket(self, state):
        """Extracts ket of mode 0 (the loop memory)."""
        full_ket = state.ket()
        return full_ket[:, 0] 

    def _build_step_program(self, action):
        """
        Builds the circuit for one time step.
        """
        r_val = np.clip(action[0], -self.max_sq_r, self.max_sq_r)
        theta_val = np.clip(action[1], 0, np.pi/2)
        alpha_val = np.clip(action[2], -self.max_disp_alpha, self.max_disp_alpha)

        prog = sf.Program(2)
        with prog.context as q:
            # Prepare Input (Fresh resource)
            Sgate(r_val) | q[1]
            
            # Displacement (Imaginary axis strategy)
            z = 1j * alpha_val
            r = np.abs(z)
            theta_disp = np.angle(z)
            Dgate(r, theta_disp) | q[1]
            
            # Interact (VBS)
            BSgate(theta_val, 0) | (q[0], q[1])
            
            # # Measurement preparation
            # z_meas = 1j * self.fixed_measure_beta
            # r_meas = np.abs(z_meas)
            # theta_meas = np.angle(z_meas)
            # Dgate(r_meas, theta_meas) | q[0]
            

        self.current_state = self.eng.run(prog).state
        self.current_ket = self.current_state.ket()
        
        # Update internal min_inner_product tracking
        inner_product = np.real(np.vdot(self.current_ket, self.current_ket))
        self.min_inner_product = min(self.min_inner_product, inner_product)
        
        prog_meas = sf.Program(2)
        with prog_meas.context as q:
            # PNR Detection
            MonitoredLossMeasureFock(self.loss_channel) | q[0]
            
            # Swap logic (keep surviving state in q[0])
            BSgate(np.pi/2, 0) | (q[0], q[1])
            
        return prog_meas

    def _calculate_fidelity(self, state_ket):
        """
        Calculates fidelity with the quartic phase target.
        """
        # We use max_rotation because the quartic state might be rotated 
        # in phase space relative to the target depending on exact accumulation of phases.
        return fidelity_max_rotation(self.target_kets[0], state_ket)

    def _calculate_reward_and_termination(self, fidelity, result):
        """Calculates the reward and determines if the episode should terminate."""
        
        terminated = False
        hit_target = (fidelity > self.target_fidelity)
        max_reward = self._calculate_reward(self.target_fidelity)
        reward = self._calculate_reward(fidelity)
        reward -= max_reward

        # self_fidelity = fidelity_max_rotation(self.past_ket, self.current_ket)
        # if self_fidelity > 0.95:
        #     reward -= max_reward

        if hit_target:
            reward += 10 * max_reward
            terminated = True

        encoded_result = result.samples[0][0]
        lost_photons, detected_photons = decode_measurement_result(encoded_result)

        info = {
            'photon_loss': lost_photons,
            'detected_photons': detected_photons,
            'is_success': hit_target,
            'total_photons': lost_photons + detected_photons,
            'fidelity': fidelity,
            'target_fidelity': self.target_fidelity
        }

        return reward, terminated, info