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

import scipy.sparse as sp

# disable caching to save memory for large cutoff dims
from sf_operations_no_cache import disable_fock_caching
disable_fock_caching() 

# optimized loss channel
from monitored_loss_measure_fock_patch import MonitoredLossMeasureFock, patch_fock_backend, decode_measurement_result
patch_fock_backend()

# Import your original, fully observable environment
from quantum_circuit_env import QuantumCircuitEnv, fidelity_pure_state
from quantum_gadget_env import QuantumGadgetEnv, fidelity_pure_state

class SeededQuantumGadgetEnv(QuantumGadgetEnv):
    """
    A variant of the QuantumGadgetEnv that resets to the output state of 
    the optimal Two-Mode Gadget (Table 1) instead of vacuum/random.
    """
    def __init__(self, target_a=0.61, **kwargs):
        # Pass kwargs to parent
        super().__init__(**kwargs)
        
        # Store the target 'a' parameter to look up Table 1
        self.target_a = target_a
        self.seed_params = self._get_table1_params(target_a)
        
        print(f"--- Environment Seeded with Table 1 Gadget (a={target_a}) ---")
        print(f"Params: {self.seed_params}")

    def _get_table1_params(self, a):
        """Finds the closest matching parameters from Table 1."""
        # Extracted from Table 1 of Sabapathy et al. (Two-mode architecture, m=2)
        # q[1] = Top Mode (Measured, Index 1)
        # q[0] = Bottom Mode (Output, Index 2)
        TABLE_1_PARAMS = [
            {"a": 0.30, "r1": 0.37, "r2": -0.37, "pr1": -3.08, "pr2": 0.59, "d1": -0.25, "d2": 0.35, "pd1": 0.77, "pd2": -0.98, "theta": -0.64, "phi": 2.26},
            {"a": 0.38, "r1": -0.43, "r2": -0.43, "pr1": -1.72, "pr2": -1.64, "d1": -0.36, "d2": -0.30, "pd1": -2.05, "pd2": -3.25, "theta": 2.25, "phi": -1.93},
            {"a": 0.46, "r1": -0.48, "r2": -0.48, "pr1": -1.34, "pr2": 0.59, "d1": 0.34, "d2": -0.39, "pd1": -0.14, "pd2": -0.93, "theta": 2.49, "phi": 2.85},
            {"a": 0.53, "r1": -0.50, "r2": 0.46, "pr1": 1.73, "pr2": 0.37, "d1": 0.40, "d2": -0.29, "pd1": -0.64, "pd2": -1.20, "theta": 0.97, "phi": -0.77},
            {"a": 0.61, "r1": -0.54, "r2": -0.58, "pr1": 0.55, "pr2": 0.65, "d1": 0.39, "d2": -0.41, "pd1": 0.75, "pd2": 2.28, "theta": -0.68, "phi": -1.26},
            {"a": 0.67, "r1": -0.34, "r2": -0.48, "pr1": 3.10, "pr2": 0.00, "d1": -0.14, "d2": 0.39, "pd1": 1.55, "pd2": -1.57, "theta": 0.61, "phi": -3.12},
            {"a": 0.77, "r1": 0.35, "r2": -0.51, "pr1": 0.85, "pr2": 0.01, "d1": 0.14, "d2": -0.40, "pd1": -1.14, "pd2": 1.57, "theta": 0.63, "phi": 2.72},
            {"a": 0.84, "r1": 0.28, "r2": -0.46, "pr1": -0.57, "pr2": 0.00, "d1": 0.07, "d2": 0.38, "pd1": 4.42, "pd2": -1.57, "theta": 0.62, "phi": -2.86},
            {"a": 0.92, "r1": -0.53, "r2": 0.34, "pr1": -1.84, "pr2": 0.02, "d1": -0.40, "d2": 0.12, "pd1": -2.49, "pd2": 7.90, "theta": -0.93, "phi": 0.93},
            {"a": 1.00, "r1": -0.23, "r2": 0.43, "pr1": -0.97, "pr2": 3.14, "d1": 0.01, "d2": -0.38, "pd1": -0.47, "pd2": -1.57, "theta": 3.76, "phi": -1.09}
        ]
        # Simple lookup, finding the params with the closest 'a'
        closest = min(TABLE_1_PARAMS, key=lambda x: abs(x['a'] - a))
        return closest

    def reset(self, seed=None, options=None):
        """
        Initializes the loop by running the Two-Mode Gadget Simulation.
        """
        super(QuantumCircuitEnv, self).reset(seed=seed)
        
        self.eng = sf.Engine("fock", backend_options={"cutoff_dim": self.cutoff_dim})
        self.current_step = 0
        self.min_inner_product = 1.0

        # --- STEP 0: Run Table 1 Circuit ---
        # We need 2 modes:
        # q[0]: The Output Mode (Bottom wire in Fig 2 top)
        # q[1]: The Measured Mode (Top wire in Fig 2 top)
        prog = sf.Program(2)
        p = self.seed_params
        
        with prog.context as q:
            # 1. Prepare Top Mode (q[1]) - Index 1 in Table
            Sgate(p['r1'], p['pr1']) | q[1]
            Dgate(p['d1'], p['pd1']) | q[1]
            
            # 2. Prepare Bottom Mode (q[0]) - Index 2 in Table
            Sgate(p['r2'], p['pr2']) | q[0]
            Dgate(p['d2'], p['pd2']) | q[0]
            
            # 3. Interfere
            # Note: SF BSgate(theta, phi) convention usually mixes (q[0], q[1]).
            # We apply it to match the schematic.
            BSgate(p['theta'], p['phi']) | (q[1], q[0])
            
            # 4. Measure Top Mode (q[1])
            # The paper specifies PNR m=2 for the Two-Mode case.
            # We use select=2 to FORCE the success branch.
            MeasureFock() | q[1]
            
            # Note: We do NOT measure q[0] yet. q[0] is our resource state.

        # Run the initialization circuit
        result = self.eng.run(prog)
        
        # Extract the state of the system
        # Since q[1] was measured/collapsed, q[0] holds our pure resource state
        self.current_state = result.state
        
        # Extract ket for Mode 0 (The Output)
        # Note: We used select=2, so the simulator state is projected.
        # We take the ket slice for mode 0, given mode 1 is at Fock |2>
        full_ket = self.current_state.ket()
        # Shape is [cutoff, cutoff]. Index 1 is fixed to 2.
        self.current_ket = full_ket[:, 0]
        
        # Generate Observation
        observation = self._ket_to_observation(self.current_ket)
        
        # Inner product tracking
        inner = np.abs(np.vdot(self.current_ket, self.current_ket))
        self.min_inner_product = inner
        
        # Calculate initial Non-Gaussianity for debugging/info
        initial_ng = self.compute_non_gaussianity(self.current_ket)
        
        return observation, {"initial_ng": initial_ng}