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



from base_quantum_env import BaseQuantumEnv, fidelity_max_rotation, fidelity_pure_state


class QuantumCircuitEnv(BaseQuantumEnv):
    """A gymnasium environment for a quantum optical circuit.

    This environment simulates the quantum optical circuit described in the paper.
    The agent's goal is to control the circuit parameters to generate a target
    squeezed cat state.
    """

    def __init__(self, cutoff_dim=25, max_steps=10, reward_power=2, tunable_r=True,
                 is_loss_channel=False, loss_channel=1, initial_target_fidelity=0.9, **kwargs):
        """Initializes the quantum circuit environment."""
        self.tunable_r = tunable_r
        self.max_squeezing = 1.38
        self.reward_power = reward_power

        super().__init__(cutoff_dim=cutoff_dim, max_steps=max_steps, loss_channel=loss_channel, 
                         initial_target_fidelity=initial_target_fidelity)

    def _define_action_space(self):
        """Defines the action space for the environment."""
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

    def _initialize_target_states(self):
        """Generates and caches the target squeezed-cat states."""
        print("Pre-calculating target state kets for QuantumCircuitEnv...")

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

    def _calculate_fidelity(self, state_ket):
        """Calculates the max fidelity over all target states, optimizing for phase."""
        fidelities = np.array([fidelity_max_rotation(target, state_ket)
                               for target in self.target_kets])
        return np.max(fidelities)

    def _build_reset_program(self):
        """Builds the Strawberry Fields program for the initial state."""
        prog = sf.Program(2)
        with prog.context as q:
            Sgate(self.max_squeezing) | q[0]
        return prog

    def _get_current_ket(self, state):
        """Extracts the ket of the primary mode from the state."""
        full_ket = state.ket()
        # The state of mode 0 is the slice where mode 1 is in vacuum
        return full_ket[:, 0]

    def _build_step_program(self, action):
        """Builds the Strawberry Fields program for one step."""
        if self.tunable_r:
            squeezing_r = np.clip(action[0], -self.max_squeezing, self.max_squeezing)
            theta_1 = np.clip(action[1], 0, np.pi/2)
            squeezing_phase = 0
        else:
            squeezing_r = self.max_squeezing
            theta_1 = np.clip(action[1], 0, np.pi/2)
            squeezing_phase = np.clip(action[0], -np.pi, np.pi)

        prog = sf.Program(2)
        with prog.context as q:
            Sgate(squeezing_r, squeezing_phase) | q[1]
            BSgate(theta_1, 0) | (q[0], q[1])


        self.current_state = self.eng.run(prog).state
        self.current_ket =self.current_state.ket()
        inner_product = np.real(np.vdot(self.current_ket, self.current_ket))
        self.min_inner_product = min(self.min_inner_product, inner_product)
        
        prog = sf.Program(2)
        with prog.context as q:
            MonitoredLossMeasureFock(self.loss_channel) | q[0]
            BSgate(np.pi/2, 0) | (q[0], q[1])
        return prog

    def _calculate_reward_and_termination(self, fidelity, result):
        """Calculates the reward and determines if the episode should terminate."""
        reward = 0
        terminated = False
        hit_target = (fidelity > self.target_fidelity)
        max_reward = self._calculate_reward(self.target_fidelity)
        reward += self._calculate_reward(fidelity)
        reward -= max_reward
        reward -= 0.5 * max_reward # for time
        self_fidelity = fidelity_max_rotation(self.past_ket, self.current_ket)
        if self_fidelity > 0.95:
            reward -= max_reward

        bounus = 3 * max_reward
        if hit_target:
            reward += bounus
            terminated = True

            # reward = (fidelity)**50

        reward /= 4 * (max_reward)
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