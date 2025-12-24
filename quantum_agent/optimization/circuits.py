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


class ThreeModeSqueezeOnly(OptimizableCircuit):
    """
    3-Mode Gadget with Squeezing only (No Displacement).

    Architecture:
    1. Squeezing on Modes 0, 1, 2
    2. Interferometer with 3 Beamsplitters
       - BS1 on (0, 1)
       - BS2 on (1, 2)
       - BS3 on (0, 1)
    3. Post-selection on two modes (handled in extract_output)

    Parameters (12):
    sq_r(0,1,2), sq_phi(0,1,2), bs_theta(1,2,3), bs_phi(1,2,3)
    """

    def __init__(self, clip_size=1.0):
        self.clip_size = clip_size
        self._param_names = [
            'sq0_r', 'sq1_r', 'sq2_r',
            'sq0_phi', 'sq1_phi', 'sq2_phi',
            'bs_theta1', 'bs_theta2', 'bs_theta3',
            'bs_phi1', 'bs_phi2', 'bs_phi3'
        ]

        self._bounds = []
        # sq_r
        self._bounds.extend([(0.0, self.clip_size)] * 3)
        # sq_phi
        self._bounds.extend([(-np.pi, np.pi)] * 3)
        # bs_theta
        self._bounds.extend([(0.0, 2 * np.pi)] * 3)
        # bs_phi
        self._bounds.extend([(-np.pi, np.pi)] * 3)

    @property
    def parameter_names(self) -> list[str]:
        return self._param_names

    @property
    def parameter_bounds(self) -> list[tuple[float, float]]:
        return self._bounds

    def run_circuit(self, params: np.ndarray, engine: sf.Engine) -> sf.backends.BaseState:
        # Unpack parameters
        sq_r = params[:3]
        sq_phi = params[3:6]
        bs_theta = params[6:9]
        bs_phi = params[9:]

        prog = sf.Program(3)
        with prog.context as q:
            # Squeezing
            for k in range(3):
                Sgate(sq_r[k], sq_phi[k]) | q[k]

            # Interferometer
            BSgate(bs_theta[0], bs_phi[0]) | (q[0], q[1])
            BSgate(bs_theta[1], bs_phi[1]) | (q[1], q[2])
            BSgate(bs_theta[2], bs_phi[2]) | (q[0], q[1])

        result = engine.run(prog)
        return result.state

    def extract_output(self, state: sf.backends.BaseState, post_select_dict: dict) -> tuple[np.ndarray, float]:
        """
        Projects the 3-mode pure state based on post-selection.

        Args:
            state: SF state object
            post_select_dict: {mode_idx: fock_val, ...} (must have 2 entries)
        """
        full_ket = state.ket()

        if len(post_select_dict) != 2:
            raise ValueError("ThreeModeSqueezeOnly expects exactly two post-selection constraints.")

        # Construct slices for projection
        indices = [slice(None)] * 3

        for mode, val in post_select_dict.items():
            indices[mode] = val

        # This reduces the array to 1D (the unmeasured mode)
        projected_ket = full_ket[tuple(indices)]

        prob = np.linalg.norm(projected_ket)**2

        if prob < 1e-12:
            return np.zeros_like(projected_ket), 0.0

        normalized_ket = projected_ket / np.sqrt(prob)

        return normalized_ket, prob


class TwoModeSqueezeOnly(OptimizableCircuit):
    """
    2-Mode Gadget without displacement (Squeezing only).
    
    Architecture:
    1. Squeezing on Mode 0
    2. Squeezing on Mode 1
    3. Beamsplitter interaction
    4. Post-selection on one mode (handled in extract_output)
    
    Parameters (6):
    sq0_r, sq0_phi, sq1_r, sq1_phi, bs_theta, bs_phi
    """
    
    def __init__(self, clip_size=1.0):
        self.clip_size = clip_size
        self._param_names = [
            'sq0_r', 'sq0_phi',
            'sq1_r', 'sq1_phi',
            'bs_theta', 'bs_phi'
        ]
        self._bounds = [
            (0.0, self.clip_size), (-np.pi, np.pi), # sq0
            (0.0, self.clip_size), (-np.pi, np.pi), # sq1
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
        sq0_r, sq0_phi, \
        sq1_r, sq1_phi, \
        theta, phi = params

        prog = sf.Program(2)
        with prog.context as q:
            Sgate(sq0_r, sq0_phi) | q[0]
            Sgate(sq1_r, sq1_phi) | q[1]
            
            BSgate(theta, phi) | (q[0], q[1])
            
        result = engine.run(prog)
        return result.state

    def extract_output(self, state: sf.backends.BaseState, post_select_dict: dict) -> tuple[np.ndarray, float]:
        """
        Projects the 2-mode pure state based on post-selection.
        """
        full_ket = state.ket()
        
        if len(post_select_dict) != 1:
            raise ValueError("TwoModeSqueezeOnly expects exactly one post-selection constraint.")
            
        measure_mode, measure_val = list(post_select_dict.items())[0]
        
        if measure_mode == 0:
            projected_ket = full_ket[measure_val, :]
        else:
            projected_ket = full_ket[:, measure_val]
            
        prob = np.linalg.norm(projected_ket)**2
        
        if prob < 1e-12:
            return np.zeros_like(projected_ket), 0.0
            
        normalized_ket = projected_ket / np.sqrt(prob)
        
        return normalized_ket, prob


class ThreeModeGadget(OptimizableCircuit):
    """
    3-Mode Gadget from 'three_mode.py'.
    
    Architecture:
    1. Squeezing + Displacement on Modes 0, 1, 2
    2. Interferometer with 3 Beamsplitters
       - BS1 on (0, 1)
       - BS2 on (1, 2)
       - BS3 on (0, 1)
    3. Post-selection on two modes (handled in extract_output)
    
    Parameters (15):
    sq_r(0,1,2), sq_phi(0,1,2), d_r(0,1,2), bs_theta(1,2,3), bs_phi(1,2,3)
    """
    
    def __init__(self, clip_size=1.0):
        self.clip_size = clip_size
        self._param_names = [
            'sq0_r', 'sq1_r', 'sq2_r',
            'sq0_phi', 'sq1_phi', 'sq2_phi',
            'd0_r', 'd1_r', 'd2_r',
            'bs_theta1', 'bs_theta2', 'bs_theta3',
            'bs_phi1', 'bs_phi2', 'bs_phi3'
        ]
        
        self._bounds = []
        # sq_r
        self._bounds.extend([(0.0, self.clip_size)] * 3)
        # sq_phi
        self._bounds.extend([(-np.pi, np.pi)] * 3)
        # d_r (displacement is real in original script)
        self._bounds.extend([(0.0, self.clip_size)] * 3)
        # bs_theta
        self._bounds.extend([(0.0, 2*np.pi)] * 3)
        # bs_phi
        self._bounds.extend([(-np.pi, np.pi)] * 3)

    @property
    def parameter_names(self) -> list[str]:
        return self._param_names

    @property
    def parameter_bounds(self) -> list[tuple[float, float]]:
        return self._bounds

    def run_circuit(self, params: np.ndarray, engine: sf.Engine) -> sf.backends.BaseState:
        # Unpack parameters
        sq_r = params[:3]
        sq_phi = params[3:6]
        d_r = params[6:9]
        bs_theta = params[9:12]
        bs_phi = params[12:]

        prog = sf.Program(3)
        with prog.context as q:
            # Squeezing + Displacement
            for k in range(3):
                Sgate(sq_r[k], sq_phi[k]) | q[k]
                Dgate(d_r[k]) | q[k] # Phase is 0

            # Interferometer
            BSgate(bs_theta[0], bs_phi[0]) | (q[0], q[1])
            BSgate(bs_theta[1], bs_phi[1]) | (q[1], q[2])
            BSgate(bs_theta[2], bs_phi[2]) | (q[0], q[1])
            
        result = engine.run(prog)
        return result.state

    def extract_output(self, state: sf.backends.BaseState, post_select_dict: dict) -> tuple[np.ndarray, float]:
        """
        Projects the 3-mode pure state based on post-selection.
        
        Args:
            state: SF state object
            post_select_dict: {mode_idx: fock_val, ...} (must have 2 entries)
        
        Returns:
            (ket, probability)
        """
        full_ket = state.ket()
        
        if len(post_select_dict) != 2:
            raise ValueError("ThreeModeGadget expects exactly two post-selection constraints.")
            
        # Construct slices for projection
        indices = [slice(None)] * 3
        
        for mode, val in post_select_dict.items():
            indices[mode] = val
            
        # This reduces the array to 1D (the unmeasured mode)
        projected_ket = full_ket[tuple(indices)]
            
        prob = np.linalg.norm(projected_ket)**2
        
        if prob < 1e-12:
            return np.zeros_like(projected_ket), 0.0
            
        normalized_ket = projected_ket / np.sqrt(prob)
        
        return normalized_ket, prob
