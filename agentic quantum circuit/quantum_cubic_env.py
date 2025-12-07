"""
Cubic Phase State Generation Environment.
Reference: "Deep reinforcement learning for near-deterministic preparation of 
cubic- and quartic-phase gates" (Anteneh et al., 2024/2025)
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import strawberryfields as sf
from strawberryfields.ops import *

# Reuse the patched MonitoredLoss from previous code if available, 
# otherwise standard MeasureFock would work for lossless (loss_channel=1.0)
from monitored_loss_measure_fock_patch import MonitoredLossMeasureFock

# Import the base class
from base_quantum_env import BaseQuantumEnv, fidelity_pure_state, fidelity_max_rotation, decode_measurement_result

class CubicPhaseEnv(BaseQuantumEnv):
    """
    RL Environment for generating Cubic Phase States.
    
    Paper Highlights:
    - Target: Cubic Phase State with gamma=-0.2
    - Circuit: Input Squeezing + Input Displacement + Fixed Measurement Displacement
    - Reward: Fidelity^55
    """

    def __init__(self, 
                 cutoff_dim=31, # Paper suggests 31, using 35 for safety 
                 max_steps=50, 
                 loss_channel=1.0, # Default to lossless as per primary result
                 **kwargs):
        
        # Paper constants
        self.target_gamma = -0.2
        self.target_r = -0.7
        self.target_alpha = 1.25  # Magnitude (applied as i*1.25)
        
        self.fixed_measure_beta = 2.5 # Applied as i*2.5
        
        self.max_sq_r = 1.15
        self.max_disp_alpha = 2.5

        initial_target_fidelity = 0.8
        
        super().__init__(cutoff_dim=cutoff_dim, max_steps=max_steps, loss_channel=loss_channel, 
                         initial_target_fidelity= initial_target_fidelity, **kwargs)

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
            'theta': (0, np.pi/2),         # 0=Transmissive, pi/2=Reflective
            'alpha': (-self.max_disp_alpha, self.max_disp_alpha) 
        }

        # Box space [-1, 1] for the agent, scaled later
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(3,), dtype=np.float32
        )

    def _initialize_target_states(self):
        """
        Generates the Cubic Phase State Target.
        Equation 3: |γ, r, α⟩ = D(α) exp(iγQ^3) S(r) |0⟩
        """
        print("Generating Target Cubic Phase State...")
        
        prog = sf.Program(1)
        with prog.context as q:
            # Note on ordering: The equation applies operators right-to-left on vacuum.
            # 1. Squeezing S(r)
            Sgate(self.target_r) | q[0]
            
            # 2. Cubic Phase Gate V(gamma) = exp(i * gamma * x^3)
            # Note: SF uses hbar=2 convention by default. 
            # Vgate matches the definition of Q^3 interaction.
            Vgate(self.target_gamma*2) | q[0]
            
            # 3. Displacement D(alpha)
            # Paper Fig 2 caption says alpha = i1.25
            # numpy complex to polar

            z = 1j * self.target_alpha

            # Convert to polar
            r = np.abs(z)
            theta = np.angle(z)

            Dgate(r, theta) | q[0]

        eng = sf.Engine("fock", backend_options={"cutoff_dim": self.cutoff_dim})
        state = eng.run(prog).state
        assert state.is_pure
        
        # We return a list containing the single target ket
        return [state.ket()]

    def _build_reset_program(self):
        """
        Initializes the loop.
        Paper: "The initial input state rho_0 for each episode was a 10 dB squeezed state (r=1.15)."
        """
        prog = sf.Program(2)
        with prog.context as q:
            # q[0] is the loop memory, q[1] is the fresh input port (unused in reset)
            # Initialize loop with high squeezing
            Sgate(1.15) | q[0]
        return prog

    def _get_current_ket(self, state):
        """Extracts ket of mode 0 (the loop memory)."""
        full_ket = state.ket()
        # Assuming mode 1 was measured/traced out, we extract mode 0

        return full_ket[:, 0] 

    def _build_step_program(self, action):
        """
        Builds the circuit for one time step (Fig 1).
        
        Action: [r, theta, alpha]
        """
        # Unpack actions
        r_val = np.clip(action[0], -self.max_sq_r, self.max_sq_r)
        theta_val = np.clip(action[1], 0, np.pi/2)
        alpha_val = np.clip(action[2], -self.max_disp_alpha, self.max_disp_alpha)

        prog = sf.Program(2)
        with prog.context as q:
            # q[0] = Loop Memory (from previous step)
            # q[1] = Fresh Input
            
            # --- 1. Prepare Input State ---
            # Squeeze
            Sgate(r_val) | q[1]
            # Displace (Paper implies imaginary axis alignment based on target/measure)

            z = 1j * alpha_val
            # Convert to polar
            r = np.abs(z)
            theta = np.angle(z)
            Dgate(r, theta) | q[1]
            
            # --- 2. Interference (VBS) ---
            BSgate(theta_val, 0) | (q[0], q[1])
            
            # --- 3. Measurement Arm (q[0]) ---
            # Apply FIXED displacement D(i * 2.5) before measurement

            z = 1j * self.fixed_measure_beta
            # Convert to polar
            r = np.abs(z)
            theta = np.angle(z)
            Dgate(r, theta) | q[0]
            
            # PNR Detection
            MonitoredLossMeasureFock(self.loss_channel) | q[0]
            
            # --- 4. Logic Swap ---
            # The state to be preserved is now in q[1] (reflected/transmitted part).
            # We swap it back to q[0] to maintain the 'loop' abstraction.
            BSgate(np.pi/2, 0) | (q[0], q[1])
            
        return prog

    def _calculate_fidelity(self, state_ket):
        """
        Calculates fidelity with the cubic phase target.
        Paper Eq (2): F = Tr(rho * rho_target) -> |<psi|target>|^2 for pure states.
        """
        # In this paper, they don't seem to optimize over rotation in the reward
        # (unlike the Cat state paper). We use direct overlap.
        return fidelity_pure_state(self.target_kets[0], state_ket)

    def _calculate_reward_and_termination(self, fidelity, result):
        """Calculates the reward and determines if the episode should terminate."""
        
        terminated = False
        hit_target = (fidelity > self.target_fidelity)
        max_reward = self._calculate_reward(self.target_fidelity)
        reward = self._calculate_reward(fidelity)
        reward -= max_reward

        self_fidelity = fidelity_max_rotation(self.past_ket, self.current_ket)
        if self_fidelity > 0.95:
            reward -= max_reward

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