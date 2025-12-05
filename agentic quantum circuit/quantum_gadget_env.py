import numpy as np
import strawberryfields as sf
from gymnasium import spaces
from strawberryfields.ops import Sgate, Dgate, BSgate

from base_quantum_env import BaseQuantumEnv, fidelity_max_rotation, db_to_r, MonitoredLossMeasureFock, decode_measurement_result


class QuantumGadgetEnv(BaseQuantumEnv):
    """
    Implements the 2-Mode Time-Multiplexed architecture with Displacement control.
    """

    def __init__(self, cutoff_dim=25, max_steps=10, **kwargs):
        self.max_disp = 2.0
        self.max_squeezing = db_to_r(8)
        self.past_fidelity = 0.0
        super().__init__(cutoff_dim=cutoff_dim, max_steps=max_steps, **kwargs)

    def _define_action_space(self):
        """Defines the action space for the 2-mode gadget."""
        self.action_ranges = {
            'squeezing_r': (0, self.max_squeezing),
            'squeezing_phase': (-np.pi, np.pi),
            'theta_1': (0, np.pi/2),
            'phi_1': (-np.pi, np.pi),
            'displacement_magnitude': (0, self.max_disp),
            'displacement_phase': (-np.pi, np.pi)
        }
        self.action_keys = list(self.action_ranges.keys())
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(len(self.action_keys),), dtype=np.float32
        )

    def _initialize_target_states(self):
        """Generates the Cubic Phase Resource State."""
        print("Initializing Cubic Phase Target State for QuantumGadgetEnv...")
        a = 0.61
        cutoff = self.cutoff_dim
        base_ket = np.zeros(cutoff, dtype=np.complex128)
        base_ket[0] = 1.0
        base_ket[1] = 1j * a * np.sqrt(1.5)
        base_ket[3] = 1j * a
        base_ket /= np.linalg.norm(base_ket)
        return [base_ket]

    def _calculate_fidelity(self, state_ket):
        """Calculates the max fidelity over all target states, optimizing for phase."""
        fidelities = np.array([fidelity_max_rotation(target, state_ket) 
                               for target in self.target_kets])
        return np.max(fidelities)

    def _build_reset_program(self):
        """Builds the Strawberry Fields program for the initial state."""
        # Start with vacuum and let the first step create the state
        prog = sf.Program(2)
        with prog.context as q: 
            MonitoredLossMeasureFock(self.loss_channel) | q[0]
        return prog

    def _get_current_ket(self, state):
        """Extracts the ket of the primary mode from the state."""
        return state.ket()[:, 0]

    def _build_step_program(self, action):
        """Builds the Strawberry Fields program for one step."""
        r_val = np.clip(action[0], 0, self.max_squeezing)
        phi_sq_val = np.clip(action[1], -np.pi, np.pi)
        theta_val = np.clip(action[2], 0, np.pi/2)
        phi_val = np.clip(action[3], -np.pi, np.pi)
        alpha_mag = np.clip(action[4], 0, self.max_disp)
        alpha_phi = np.clip(action[5], -np.pi, np.pi)

        prog = sf.Program(2)
        with prog.context as q:
            Sgate(r_val, phi_sq_val) | q[1]
            Dgate(alpha_mag, alpha_phi) | q[1]
            BSgate(theta_val, phi_val) | (q[0], q[1])
            MonitoredLossMeasureFock(self.loss_channel) | q[0]
            BSgate(np.pi/2, 0) | (q[0], q[1])
        return prog

    def _calculate_reward_and_termination(self, fidelity, result):
        """Calculates the reward and determines if the episode should terminate."""
        terminated = False
        target_fidelity = 0.95
        hit_target = (fidelity > target_fidelity)

        max_reward = self._calculate_reward(target_fidelity)
        reward = self._calculate_reward(fidelity)
        reward -= max_reward
        
        current_ng_score = self.compute_non_gaussianity(self.current_ket)
        if current_ng_score < 1:
            reward -= max_reward

        self_fidelity = fidelity_max_rotation(self.past_ket, self.current_ket)
        if self_fidelity > 0.95:
            reward -= max_reward
        
        if abs(fidelity - self.past_fidelity) < 0.05:
            reward -= max_reward
        self.past_fidelity = fidelity

        if hit_target:
            reward += 10 * max_reward
            terminated = True
        
        encoded_result = result.samples[0][0]
        lost, detected = decode_measurement_result(encoded_result)
        info = {
            'photon_loss': lost,
            'detected_photons': detected,
            'total_photons': lost + detected,
            'fidelity': fidelity,
            'ng_score': current_ng_score,
            'is_success': hit_target,
            'self_fidelity': self_fidelity
        }
        
        return reward, terminated, info