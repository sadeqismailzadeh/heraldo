import numpy as np
import strawberryfields as sf
from strawberryfields.ops import Sgate, Dgate, BSgate

from quantum_agent.optimization.interfaces import OptimizableCircuit

class TwoModeGadget(OptimizableCircuit):
    """
    2-Mode Gadget from 'two_mode.py'.
    
    Architecture:
    1. Squeezing + Displacement on Mode 0
    2. Squeezing + Displacement on Mode 1
    3. Beamsplitter interaction
    4. Post-selection on one mode (handled in extract_output)
    
    Parameters (10):
    sq0_r, sq0_phi, disp0_r, disp0_phi, sq1_r, sq1_phi, disp1_r, disp1_phi, theta, phi
    """
    
    def __init__(self, clip_size=1.0):
        self.clip_size = clip_size
        self._param_names = [
            'sq0_r', 'sq0_phi', 'disp0_r', 'disp0_phi',
            'sq1_r', 'sq1_phi', 'disp1_r', 'disp1_phi',
            'bs_theta', 'bs_phi'
        ]
        # Standard bounds typically used in these optimizations
        # r in [0, clip], angles in [-pi, pi] or similar.
        self._bounds = [
            (0.0, self.clip_size), (-np.pi, np.pi), # sq0
            (0.0, self.clip_size), (-np.pi, np.pi), # disp0
            (0.0, self.clip_size), (-np.pi, np.pi), # sq1
            (0.0, self.clip_size), (-np.pi, np.pi), # disp1
            (0.0, 2*np.pi), (-np.pi, np.pi)         # BS
        ]

    @property
    def parameter_names(self) -> list[str]:
        return self._param_names

    @property
    def parameter_bounds(self) -> list[tuple[float, float]]:
        return self._bounds

    def run_circuit(self, params: np.ndarray, engine: sf.Engine) -> sf.backends.BaseState:
        # Unpack parameters
        sq0_r, sq0_phi, disp0_r, disp0_phi, \
        sq1_r, sq1_phi, disp1_r, disp1_phi, \
        theta, phi = params

        prog = sf.Program(2)
        with prog.context as q:
            Sgate(sq0_r, sq0_phi) | q[0]
            Dgate(disp0_r, disp0_phi) | q[0]
            
            Sgate(sq1_r, sq1_phi) | q[1]
            Dgate(disp1_r, disp1_phi) | q[1]
            
            BSgate(theta, phi) | (q[0], q[1])
            
        result = engine.run(prog)
        return result.state

    def extract_output(self, state: sf.backends.BaseState, post_select_dict: dict) -> tuple[np.ndarray, float]:
        """
        Projects the 2-mode pure state based on post-selection.
        
        Args:
            state: SF state object
            post_select_dict: e.g. {0: 1} means measure mode 0, find Fock |1>
        
        Returns:
            (ket, probability)
        """
        # Get the full tensor (ket)
        # Shape: (cutoff, cutoff) for 2 modes
        # Ordering is (mode0, mode1)
        full_ket = state.ket()
        
        # Assume only one mode is post-selected for the 2-mode gadget
        if len(post_select_dict) != 1:
            raise ValueError("TwoModeGadget expects exactly one post-selection constraint.")
            
        measure_mode, measure_val = list(post_select_dict.items())[0]
        
        if measure_mode == 0:
            # Project mode 0 onto |n>. We slice the first dimension.
            # Remaining vector corresponds to mode 1.
            projected_ket = full_ket[measure_val, :]
        else:
            # Project mode 1 onto |n>. We slice the second dimension.
            projected_ket = full_ket[:, measure_val]
            
        # Probability is the squared norm of the projected vector
        prob = np.linalg.norm(projected_ket)**2
        
        if prob < 1e-12:
            # Avoid division by zero, return zero vector and zero prob
            return np.zeros_like(projected_ket), 0.0
            
        # Normalize
        normalized_ket = projected_ket / np.sqrt(prob)
        
        return normalized_ket, prob