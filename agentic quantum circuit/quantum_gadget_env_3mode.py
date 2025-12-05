# 1. Import the module we need to patch
import scipy.integrate

# 2. Check if the patch is needed to avoid errors
if not hasattr(scipy.integrate, 'simps'):
    print("Monkey patching scipy.integrate: 'simps' not found. Pointing to 'simpson'.")
    # 3. Create the 'simps' attribute and point it to the existing 'simpson' function.
    scipy.integrate.simps = scipy.integrate.simpson
else:
    print("'simps' already exists in scipy.integrate. No patch needed.")

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import strawberryfields as sf
from strawberryfields.ops import *
from monitored_loss_measure_fock_patch import MonitoredLossMeasureFock, decode_measurement_result

# Import the parent class
from quantum_circuit_env import QuantumCircuitEnv, fidelity_pure_state, fidelity_max_rotation
from quantum_gadget_env import QuantumGadgetEnv, db_to_r

class ThreeModeGadgetEnv(QuantumGadgetEnv):
    """
    Implements the 3-Mode Time-Multiplexed architecture from Article 2.
    
    Circuit Topology (Per Step):
    - q[0]: Loop Mode (Input from previous step / Output to next step)
    - q[1]: Middle Ancilla (Fresh Displaced Squeezed State)
    - q[2]: Top Ancilla (Fresh Displaced Squeezed State)
    
    Gate Sequence:
    1. Prepare q[1], q[2]
    2. BS1(q[2], q[1])  <- Top interacts with Middle
    3. BS2(q[1], q[0])  <- Middle interacts with Loop (The Memory Step)
    4. BS3(q[2], q[1])  <- Top interacts with Middle again
    5. Measure q[2], q[1]
    6. q[0] survives to next step.

    (Input 1: Top Wire)
      q[2] (New) ─────[S,D]────▼───────────────▼─────[ PNR 1 ]── n1
                               │               │      (Exit)
                            [ BS1 ]         [ BS3 ]
                               │               │
                  (Input 2: Middle Wire)       │
      q[1] (New) ─────[S,D]────▲───────▼───────▲─────[ PNR 2 ]── n2
                                       │              (Exit)
                                    [ BS2 ]
                                       │
                  (Loop: Bottom Wire)  │
      q[0] (Old) ══════════════════════▲══════════════════════════> Loops to Next Step
    """

    def __init__(self, tunable_bs_phase = True,**kwargs):
        super().__init__(**kwargs)
        self.tunable_bs_phase = tunable_bs_phase

        print("3 mode circuit is being used")

        # Limits
        self.r_max = db_to_r(8)
        self.d_max = 2.0
        self.pi_val = np.pi

        # Action ranges for normalization
        self.action_ranges = {
            'r1': (0.0, self.r_max),
            'pr1': (-self.pi_val, self.pi_val),
            'a1': (0.0, self.d_max),
            'pa1': (-self.pi_val, self.pi_val),
            'r2': (0.0, self.r_max),
            'pr2': (-self.pi_val, self.pi_val),
            'a2': (0.0, self.d_max),
            'pa2': (-self.pi_val, self.pi_val),
        }

        if self.tunable_bs_phase:
            self.action_ranges.update({
                'th1': (0.0, self.pi_val/2),
                'ph1': (-self.pi_val, self.pi_val),
                'th2': (0.0, self.pi_val/2),
                'ph2': (-self.pi_val, self.pi_val),
                'th3': (0.0, self.pi_val/2),
                'ph3': (-self.pi_val, self.pi_val),
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

    def reset(self, seed=None, options=None):
        """
        Resets the environment. 
        Unlike the 2-mode env, we start with a clean vacuum in the loop (q0)
        and let the agent's first step define the first gadget in the stack.
        """
        # Call grandparent reset (skipping the parent's specific 2-mode init)
        super(QuantumCircuitEnv, self).reset(seed=seed)
        
        # Initialize Engine
        self.eng = sf.Engine("fock", backend_options={"cutoff_dim": self.cutoff_dim})
        self.current_step = 0
        self.min_inner_product = 1.0

        # Initialize Loop q[0] as Vacuum
        # We run a dummy program just to get the initial vacuum state object
        prog = sf.Program(3)
        with prog.context as q: 
            MonitoredLossMeasureFock(self.loss_channel) | q[0]
        result = self.eng.run(prog)
        self.current_state = result.state
        
        # Extract Observation (Mode 0)
        self.current_ket = self.current_state.ket()[:, 0, 0]  # Single mode ket
        observation = self._ket_to_observation(self.current_ket)

        self.past_ket = self.current_ket

        fidelities = np.array([fidelity_max_rotation(target, self.current_ket) 
                               for target in self.target_kets])
        fidelity = np.max(fidelities)

        self.past_fidelity = fidelity
        
        return observation, {}

    def step(self, action):
        self.current_step += 1

        # Denormalize action from [-1, 1] to original ranges
        action = self._denormalize_action(action)

         # --- 1. UNPACK ACTIONS WITH CLIPPING ---

        # Ancillas are always the first 8 indices
        r1 = np.clip(action[0], 0.0, self.r_max)
        pr1 = np.clip(action[1], -self.pi_val, self.pi_val)
        a1 = np.clip(action[2], 0.0, self.d_max)
        pa1 = np.clip(action[3], -self.pi_val, self.pi_val)
        r2 = np.clip(action[4], 0.0, self.r_max)
        pr2 = np.clip(action[5], -self.pi_val, self.pi_val)
        a2 = np.clip(action[6], 0.0, self.d_max)
        pa2 = np.clip(action[7], -self.pi_val, self.pi_val)

        # Logic branching for Beam Splitters
        if self.tunable_bs_phase:
            # Unpack both Theta and Phi from action vector
            th1 = np.clip(action[8], 0.0, self.pi_val/2)
            ph1 = np.clip(action[9], -self.pi_val, self.pi_val)
            th2 = np.clip(action[10], 0.0, self.pi_val/2)
            ph2 = np.clip(action[11], -self.pi_val, self.pi_val)
            th3 = np.clip(action[12], 0.0, self.pi_val/2)
            ph3 = np.clip(action[13], -self.pi_val, self.pi_val)
        else:
            # Unpack only Theta; fix Phi to 0.0
            th1 = np.clip(action[8], 0.0, self.pi_val/2)
            th2 = np.clip(action[9], 0.0, self.pi_val/2)
            th3 = np.clip(action[10], 0.0, self.pi_val/2)

            ph1 = 0.0
            ph2 = 0.0
            ph3 = 0.0


        # --- 2. BUILD 3-MODE CIRCUIT ---
        prog = sf.Program(3)
        with prog.context as q:
            # q[0] is the Loop (Memory)
            # q[1] is Middle Ancilla
            # q[2] is Top Ancilla
            
            # --- Prepare Ancillas ---
            # Middle (q[1])
            Sgate(r1, pr1) | q[1]
            Dgate(a1, pa1) | q[1]
            
            # Top (q[2])
            Sgate(r2, pr2) | q[2]
            Dgate(a2, pa2) | q[2]
            
            # --- Optical Interaction Sequence ---
            # 1. Top mixes with Middle
            BSgate(th1, ph1) | (q[0], q[1])
            
            # 2. Middle mixes with Loop (The Memory Interaction)
            # Note: q[0] enters from the "left" (history)
            BSgate(th2, ph2) | (q[1], q[2])
            
            # 3. Top mixes with Middle again
            BSgate(th3, ph3) | (q[0], q[1])
            
            # --- Measurements ---
            # Measure the two ancillas
            MonitoredLossMeasureFock(self.loss_channel) | q[0]
            MonitoredLossMeasureFock(self.loss_channel) | q[1]

            # --- LOOP RECYCLE (SWAP) ---
            BSgate(np.pi/2, 0) | (q[0], q[2])
            
            # q[0] is NOT measured; it loops to the next step.

        # --- 3. EXECUTE ---
        result = self.eng.run(prog)
        
        # --- 4. EXTRACT STATE ---
        self.current_state = result.state
        full_ket = self.current_state.ket()
        
        # SF Tensor shape: [dim, dim, dim] for modes [q0, q1, q2]
        # But q1 and q2 were measured (collapsed to specific Fock states).
        # In the Simulator, 'ket()' usually returns the full tensor.
        # We need to slice it. Since q1 and q2 are projected, 
        # we extract the q0 component.
        # Note: If measured, SF simulator puts them in the vacuum state.
        # We extract Mode 0 (Loop).
        self.current_ket = full_ket[:, 0, 0] 


        observation = self._ket_to_observation(self.current_ket)

        # --- 5. CALCULATE REWARD ---
        # (This logic is inherited from the Article 2 Non-Gaussianity logic)
        
        # A. Inner Product Check (Simulation Stability)
        inner = np.abs(np.vdot(self.current_ket, self.current_ket))
        self.min_inner_product = min(self.min_inner_product, inner)

        # B. Fidelity to Cubic Phase Target
        fidelities = np.array([fidelity_max_rotation(target, self.current_ket) 
                               for target in self.target_kets])
        fidelity = np.max(fidelities)




        current_ng_score = self.compute_non_gaussianity(self.current_ket)

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
        

        # --- 7. INFO ---
        # Extract measurement results for diagnostics
        # samples shape: [shots, modes] -> [1, 2] (q2, q1)
        raw_samples = result.samples[0]
        # Decode custom MonitoredLoss encoding if used
        lost1, n1 = decode_measurement_result(raw_samples[0]) # q2
        lost2, n2 = decode_measurement_result(raw_samples[1]) # q1

        info = {
            'photon_loss': lost1 + lost2,
            'detected_photons': n1+n2,
            'total_photons': lost1 + lost2 + n1+n2,
            'fidelity': fidelity,
            'ng_score': current_ng_score,
            'is_success': hit_target, # Flag for Curriculum Manager
        }
        
        if truncated:
            # info['terminal_bonus'] = terminal_bonus
            info['final_ket'] = self.current_ket
            info['min_inner_product'] = self.min_inner_product

        return observation, reward, terminated, truncated, info