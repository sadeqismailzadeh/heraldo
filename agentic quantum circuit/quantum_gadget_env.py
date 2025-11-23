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


class DisplacedLoopedGadgetEnv(QuantumCircuitEnv):
    """
    Implements the full 'Modified Article 2' architecture with Time-Multiplexing
    and Displacement control.
    
    Architecture:
    1. RESET: Simulates the Article 2 Gadget (Two displaced squeezed beams collide).
       One output is measured, the other enters the loop.
    2. STEP:  A FRESH ancilla is generated with Squeezing AND Displacement.
       It interacts with the loop, is measured, and the loop continues.
    """

    def __init__(self, max_displacement=2.0, **kwargs):
        """
        Args:
            max_displacement (float): Maximum magnitude for the displacement alpha.
            gadget_initial_params (dict): Parameters for the t=0 gadget setup.
                                          {'r': 1.0, 'alpha': 1.0, 'theta': pi/4}
            **kwargs: Arguments for the parent QuantumCircuitEnv.
        """
        # Initialize parent
        super().__init__(**kwargs)
        
        self.max_disp = max_displacement
        self.max_squeezing = db_to_r(10)
        
        # Default gadget parameters (Article 2 style) if none provided

        # --- REDEFINE ACTION SPACE ---
        # The agent now controls 5 parameters per step:
        # 1. Squeezing Magnitude (r)
        # 2. Beam Splitter Angle (theta)
        # 3. Squeezing Phase (phi_sq)
        # 4. Displacement Magnitude (alpha_mag)  <-- NEW
        # 5. Displacement Phase (alpha_phi)      <-- NEW
        self.action_space = spaces.Box(
            low=np.array([0.0, 0.0, -np.pi, 0.0, -np.pi]),
            high=np.array([self.max_squeezing, np.pi/2, np.pi, self.max_disp, np.pi]),
            shape=(5,),
            dtype=np.float32
        )

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
        reward = max_fidelity ** self.reward_power

        # 5. Termination Logic
        terminated = False
        truncated = self.current_step >= self.max_steps
        
        if truncated:
             # Add terminal bonus
             terminal_bonus = 0
             if max_fidelity > 0.0:
                 terminal_bonus += (max_fidelity ** self.reward_power) * 10
             
             fidelity_threshold = 0.9
             if max_fidelity > fidelity_threshold:
                 excess = (max_fidelity - fidelity_threshold) / (1 - fidelity_threshold)
                 terminal_bonus += (excess ** self.reward_power) * 100
             
             reward += terminal_bonus

        # Info
        encoded_result = result.samples[0][0]
        lost, detected = decode_measurement_result(encoded_result)
        info = {
            'photon_loss': lost,
            'detected_photons': detected,
            'total_photons': lost + detected,
            'fidelity': max_fidelity
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