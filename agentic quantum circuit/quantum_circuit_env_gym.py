"""Custom Gym environment compatible with OpenAI Gym (classic Gym API).

This is a copy of `quantum_circuit_env.py` adapted to the Gym API so it can
be used with RL-Games which expects `gym` (not `gymnasium`). Key API changes:
- uses `import gym` instead of `gymnasium`
- `reset()` returns the observation only (no `info`)
- `step()` returns `(obs, reward, done, info)` where `done` = terminated||truncated
"""

# 1. Import the module we need to patch
import scipy.integrate

# 2. Check if the patch is needed to avoid errors
if not hasattr(scipy.integrate, 'simps'):
    print("Monkey patching scipy.integrate: 'simps' not found. Pointing to 'simpson'.")
    # 3. Create the 'simps' attribute and point it to the existing 'simpson' function.
    scipy.integrate.simps = scipy.integrate.simpson
else:
    print("'simps' already exists in scipy.integrate. No patch needed.")

import gym
from gym import spaces
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
    """
    target_ket = np.asarray(target_ket, dtype=np.complex128).flatten()
    state_ket = np.asarray(state_ket, dtype=np.complex128).flatten()
    
    # Fidelity: F = |⟨φ|ψ⟩|²
    overlap = np.vdot(target_ket, state_ket)  # ⟨φ|ψ⟩
    fidelity = np.abs(overlap) ** 2
    
    return np.clip(fidelity, 0.0, 1.0)


class QuantumCircuitEnvGym(gym.Env):
    """A `gym.Env` compatible environment for a quantum optical circuit.

    This mirrors the behavior of the original `QuantumCircuitEnv` written for
    `gymnasium`, but follows the classic Gym API: `reset()` returns observation
    only, and `step()` returns `(obs, reward, done, info)`.
    """
    metadata = {"render_modes": [], "render_fps": 0}

    def __init__(self, cutoff_dim=25, max_steps=10, reward_power=2, tunable_r=False,
                 is_loss_channel=False, loss_channel=0.99):
        super(QuantumCircuitEnvGym, self).__init__()

        # --- Environment Parameters ---
        self.cutoff_dim = cutoff_dim
        self.max_steps = max_steps
        self.reward_power = reward_power
        self.tunable_r = tunable_r
        self.max_squeezing = 1.38
        self.termination_threshold = 0.001
        self.is_loss_channel = is_loss_channel
        self.loss_channel = loss_channel

        # --- Strawberry Fields Engine ---
        self.eng = None

        # --- Pre-calculate Target States (Reward States) ---
        print("Pre-calculating target state kets (gym version)...")
        self.target_kets = self._initialize_target_states()
        print("Target state kets initialized.")

        # --- Pre-allocate observation buffer for efficiency ---
        obs_size = 2 * self.cutoff_dim
        self._obs_buffer = np.zeros(obs_size, dtype=np.float32)

        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(obs_size,), dtype=np.float32
        )

        if self.tunable_r:
            self.action_space = spaces.Box(
                low=np.array([0.0, 0.0, -np.pi]),
                high=np.array([self.max_squeezing, np.pi/2, np.pi]),
                shape=(3,),
                dtype=np.float32
            )
        else:
            self.action_space = spaces.Box(
                low=np.array([0.0, -np.pi]),
                high=np.array([np.pi/2, np.pi]),
                shape=(2,),
                dtype=np.float32
            )

        self.current_step = 0
        self.current_ket = None
        self.min_inner_product = 1.0

    def _initialize_target_states(self):
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
        ket = temp_eng.run(prog).state.ket()
        targets.append(ket)

        return targets

    def _ket_to_observation(self, state_ket):
        if state_ket is None:
            self._obs_buffer.fill(0)
            return self._obs_buffer.copy()
        
        state_ket = state_ket.flatten()
        
        if len(state_ket) < self.cutoff_dim:
            padded = np.zeros(self.cutoff_dim, dtype=np.complex128)
            padded[:len(state_ket)] = state_ket
            state_ket = padded
        elif len(state_ket) > self.cutoff_dim:
            state_ket = state_ket[:self.cutoff_dim]
        
        real_parts = state_ket.real
        imag_parts = state_ket.imag
        
        self._obs_buffer[:self.cutoff_dim] = real_parts
        self._obs_buffer[self.cutoff_dim:] = imag_parts
        
        return self._obs_buffer.copy()

    def reset(self, seed=None, options=None):
        """Reset environment and return initial observation (Gym API).

        Note: This method intentionally returns the observation only to match
        classic Gym expectations (used by RL-Games).
        """
        # Do not call super().reset() because classic `gym` may not accept seed
        if seed is not None:
            np.random.seed(seed)

        # Create a new engine for the new episode
        self.eng = sf.Engine("fock", backend_options={"cutoff_dim": self.cutoff_dim})
        self.current_step = 0
        self.min_inner_product = 1.0

        prog = sf.Program(2)

        with prog.context as q:
            Sgate(self.max_squeezing) | q[0]

        self.current_state = self.eng.run(prog).state

        full_ket = self.current_state.ket()
        self.current_ket = full_ket[:, 0]
        observation = self._ket_to_observation(self.current_ket)

        inner_product = np.abs(np.vdot(self.current_ket, self.current_ket))
        self.min_inner_product = min(self.min_inner_product, inner_product)

        return observation

    def step(self, action):
        self.current_step += 1

        if self.tunable_r:
            squeezing_r = np.clip(action[0], 0, self.max_squeezing)
            theta_1 = np.clip(action[1], 0, np.pi/2)
            squeezing_phase = np.clip(action[2], -np.pi, np.pi)
        else:
            squeezing_r = self.max_squeezing
            theta_1 = np.clip(action[0], 0, np.pi/2)
            squeezing_phase = np.clip(action[1], -np.pi, np.pi)

        prog = sf.Program(2)
        with prog.context as q:
            Sgate(squeezing_r, squeezing_phase) | q[1]
            BSgate(theta_1, 0) | (q[0], q[1])
            MonitoredLossMeasureFock(self.loss_channel) | q[0]
            BSgate(np.pi/2, 0) | (q[0], q[1])

        result = self.eng.run(prog)
        self.current_state = result.state

        full_ket = self.current_state.ket()
        self.current_ket = full_ket[:, 0]
        observation = self._ket_to_observation(self.current_ket)

        inner_product = np.abs(np.vdot(self.current_ket, self.current_ket))
        self.min_inner_product = min(self.min_inner_product, inner_product)

        fidelities = np.array([fidelity_pure_state(target_ket, self.current_ket) 
                               for target_ket in self.target_kets])
        max_fidelity = np.max(fidelities)
        reward = max_fidelity ** self.reward_power

        terminated = False
        truncated = self.current_step >= self.max_steps

        terminal_bonus = 0
        if truncated:
            terminal_bonus += (max_fidelity ** self.reward_power) * 10
            fidelity_threshold = 0.9
            excess_fidelity = max(max_fidelity - fidelity_threshold, 0)
            rescaled_excess = excess_fidelity / (1 - fidelity_threshold)
            terminal_bonus += (rescaled_excess ** self.reward_power) * 100
            reward += terminal_bonus

        encoded_result = result.samples[0][0]
        lost_photons, detected_photons = decode_measurement_result(encoded_result)
        info = {
            'photon_loss': lost_photons,
            'detected_photons': detected_photons,
            'total_photons': lost_photons + detected_photons,
            'fidelity': max_fidelity
        }

        if truncated:
            info['terminal_bonus'] = terminal_bonus
            info['final_ket'] = self.current_ket
            info['min_inner_product'] = self.min_inner_product

        done = terminated or truncated
        return observation, reward, done, info

    def render(self):
        pass

    def close(self):
        pass
