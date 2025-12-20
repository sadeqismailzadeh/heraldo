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

from base_quantum_env import BaseQuantumEnv, fidelity_max_rotation, db_to_r, MonitoredLossMeasureFock, decode_measurement_result


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

class ThreeModeGadgetEnv(BaseQuantumEnv):
    """
    Implements the 3-Mode Time-Multiplexed architecture from Article 2.
    """

    def __init__(self, 
                gkp_type='square',   # 'square' or 'hex'
                mu=0,             # Logical 0 or 1
                delta=0.4,        # Finite energy envelope
                tunable_bs_phase=False,
                cutoff_dim=25,
                max_steps=10,
                **kwargs):
        
        self.gkp_type = gkp_type.lower()
        self.mu = mu
        self.delta = delta
        

    
        self.tunable_bs_phase = tunable_bs_phase
        self.r_max = db_to_r(8)
        self.d_max = 1
        self.pi_val = np.pi
        super().__init__(cutoff_dim=cutoff_dim, max_steps=max_steps, **kwargs)

        self.target_ng_score =self.compute_non_gaussianity(self.target_kets[0])
        print(f"target ng = {self.target_ng_score:.2f}")

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

    # def _initialize_target_states(self):
    #     """Generates the Cubic Phase Resource State."""
    #     print("Initializing Cubic Phase Target State for ThreeModeGadgetEnv...")
    #     a = 0.61
    #     cutoff = self.cutoff_dim
    #     base_ket = np.zeros(cutoff, dtype=np.complex128)
    #     base_ket[0] = 1.0
    #     base_ket[1] = 1j * a * np.sqrt(1.5)
    #     base_ket[3] = 1j * a
    #     base_ket /= np.linalg.norm(base_ket)
    #     return [base_ket]

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


        self.current_state = self.eng.run(prog).state
        self.current_ket =self.current_state.ket()
        inner_product = np.real(np.vdot(self.current_ket, self.current_ket))
        self.min_inner_product = min(self.min_inner_product, inner_product)
 

        prog = sf.Program(3)
        with prog.context as q:
            # Measurements and Swap
            MonitoredLossMeasureFock(self.loss_channel) | q[0]
            MonitoredLossMeasureFock(self.loss_channel) | q[1]
            BSgate(np.pi/2, 0) | (q[0], q[2])
            
        return prog

    def _calculate_reward_and_termination(self, fidelity, result):
        """Calculates the reward and determines if the episode should terminate."""
        terminated = False
        target_fidelity = self.target_fidelity
        
        max_reward = self._calculate_reward(target_fidelity)
        reward = self._calculate_reward(fidelity)
        reward -= max_reward
        
        # current_ng_score = self.compute_non_gaussianity(self.current_ket)
        # if current_ng_score < self.target_ng_score/4:
        #     reward -= max_reward

        # self_fidelity = fidelity_max_rotation(self.past_ket, self.current_ket)
        # if self_fidelity > 0.95:
        #     reward -= max_reward
        
        # if abs(fidelity - self.past_fidelity) < 0.05:
        #     reward -= max_reward
        # self.past_fidelity = fidelity

        hit_target = (fidelity > target_fidelity)
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
            'ng_score': 0,
            'is_success': hit_target,
            'self_fidelity': 0,
            'target_fidelity': self.target_fidelity
        }
        
        return reward, terminated, info