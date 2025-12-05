import numpy as np
import strawberryfields as sf
from gymnasium import spaces
from strawberryfields.ops import Sgate, Dgate, BSgate

from base_quantum_env import BaseQuantumEnv, fidelity_max_rotation, db_to_r, MonitoredLossMeasureFock, decode_measurement_result


class ThreeModeGadgetEnv(BaseQuantumEnv):
    """
    Implements the 3-Mode Time-Multiplexed architecture from Article 2.
    """

    def __init__(self, tunable_bs_phase=True, cutoff_dim=25, max_steps=10, **kwargs):
        self.tunable_bs_phase = tunable_bs_phase
        self.r_max = db_to_r(8)
        self.d_max = 2.0
        self.pi_val = np.pi
        self.past_fidelity = 0.0
        super().__init__(cutoff_dim=cutoff_dim, max_steps=max_steps, **kwargs)
        print("3 mode circuit is being used")

    def _define_action_space(self):
        """Defines the action space for the 3-mode gadget."""
        self.action_ranges = {
            'r1': (0.0, self.r_max), 'pr1': (-self.pi_val, self.pi_val),
            'a1': (0.0, self.d_max), 'pa1': (-self.pi_val, self.pi_val),
            'r2': (0.0, self.r_max), 'pr2': (-self.pi_val, self.pi_val),
            'a2': (0.0, self.d_max), 'pa2': (-self.pi_val, self.pi_val),
        }
        if self.tunable_bs_phase:
            self.action_ranges.update({
                'th1': (0.0, self.pi_val/2), 'ph1': (-self.pi_val, self.pi_val),
                'th2': (0.0, self.pi_val/2), 'ph2': (-self.pi_val, self.pi_val),
                'th3': (0.0, self.pi_val/2), 'ph3': (-self.pi_val, self.pi_val),
            })
        else:
            self.action_ranges.update({
                'th1': (0.0, self.pi_val/2),
                'th2': (0.0, self.pi_val/2),
                'th3': (0.0, self.pi_val/2),
            })
        self.action_keys = list(self.action_ranges.keys())
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(len(self.action_keys),), dtype=np.float32
        )

    def _initialize_target_states(self):
        """Generates the Cubic Phase Resource State."""
        print("Initializing Cubic Phase Target State for ThreeModeGadgetEnv...")
        a = 0.61
        cutoff = self.cutoff_dim
        base_ket = np.zeros(cutoff, dtype=np.complex128)
        base_ket[0] = 1.0
        base_ket[1] = 1j * a * np.sqrt(1.5)
        base_ket[3] = 1j * a
        base_ket /= np.linalg.norm(base_ket)
        return [base_ket]

    def _get_current_ket(self, state):
        """Extracts the ket of the primary mode from the 3-mode state."""
        return state.ket()[:, 0, 0]

    def _calculate_fidelity(self, state_ket):
        """Calculates the max fidelity over all target states, optimizing for phase."""
        fidelities = np.array([fidelity_max_rotation(target, state_ket) 
                               for target in self.target_kets])
        return np.max(fidelities)

    def _build_reset_program(self):
        """Builds the Strawberry Fields program for the initial state."""
        # Initialize Loop q[0] as Vacuum
        prog = sf.Program(3)
        with prog.context as q:
            MonitoredLossMeasureFock(self.loss_channel) | q[0]
        return prog


    def _build_step_program(self, action):
        """Builds the Strawberry Fields program for one step."""
        for i, key in enumerate(self.action_keys):
            low, high = self.action_ranges[key]
            action[i] = np.clip(action[i], low, high)
        r1, pr1, a1, pa1, r2, pr2, a2, pa2 = action[:8]
        
        if self.tunable_bs_phase:
            th1, ph1, th2, ph2, th3, ph3 = action[8:]
        else:
            th1, th2, th3 = action[8:]
            ph1, ph2, ph3 = 0.0, 0.0, 0.0

        prog = sf.Program(3)
        with prog.context as q:
            # Prepare Ancillas
            Sgate(r1, pr1) | q[1]
            Dgate(a1, pa1) | q[1]
            Sgate(r2, pr2) | q[2]
            Dgate(a2, pa2) | q[2]
            
            # Optical Interactions
            BSgate(th1, ph1) | (q[0], q[1])
            BSgate(th2, ph2) | (q[1], q[2])
            BSgate(th3, ph3) | (q[0], q[1])
            
            # Measurements and Swap
            MonitoredLossMeasureFock(self.loss_channel) | q[0]
            MonitoredLossMeasureFock(self.loss_channel) | q[1]
            BSgate(np.pi/2, 0) | (q[0], q[2])
            
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

        raw_samples = result.samples[0]
        lost1, n1 = decode_measurement_result(raw_samples[0])
        lost2, n2 = decode_measurement_result(raw_samples[1])

        info = {
            'photon_loss': lost1 + lost2,
            'detected_photons': n1 + n2,
            'total_photons': lost1 + lost2 + n1 + n2,
            'fidelity': fidelity,
            'ng_score': current_ng_score,
            'is_success': hit_target,
        }
        
        return reward, terminated, info