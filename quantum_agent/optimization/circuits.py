import numpy as np
import strawberryfields as sf
from strawberryfields.ops import Sgate, Dgate, BSgate, Fock

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
    
    def __init__(self, clip_size=1.0, num_single_photon=0):
        self.clip_size = clip_size
        self.num_single_photon = num_single_photon
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
            if self.num_single_photon >= 1:
                Fock(1) | q[1]

            Sgate(sq0_r, sq0_phi) | q[0]
            Dgate(disp0_r, disp0_phi) | q[0]
            
            Sgate(sq1_r, sq1_phi) | q[1]
            Dgate(disp1_r, disp1_phi) | q[1]
            
            BSgate(theta, phi) | (q[0], q[1])
            
        result = engine.run(prog)
        return result.state






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

    def __init__(self, clip_size=1.0, num_single_photon=0):
        self.clip_size = clip_size
        self.num_single_photon = num_single_photon
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
            if self.num_single_photon >= 1:
                Fock(1) | q[1]
            if self.num_single_photon >= 2:
                Fock(1) | q[2]

            # Squeezing
            for k in range(3):
                Sgate(sq_r[k], sq_phi[k]) | q[k]

            # Interferometer
            BSgate(bs_theta[0], bs_phi[0]) | (q[0], q[1])
            BSgate(bs_theta[1], bs_phi[1]) | (q[1], q[2])
            BSgate(bs_theta[2], bs_phi[2]) | (q[0], q[1])

        result = engine.run(prog)
        return result.state






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
    
    def __init__(self, clip_size=1.0, num_single_photon=0):
        self.clip_size = clip_size
        self.num_single_photon = num_single_photon
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
            if self.num_single_photon >= 1:
                Fock(1) | q[1]

            Sgate(sq0_r, sq0_phi) | q[0]
            Sgate(sq1_r, sq1_phi) | q[1]
            
            BSgate(theta, phi) | (q[0], q[1])
            
        result = engine.run(prog)
        return result.state






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
    
    def __init__(self, clip_size=1.0, num_single_photon=0):
        self.clip_size = clip_size
        self.num_single_photon = num_single_photon
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
            if self.num_single_photon >= 1:
                Fock(1) | q[1]
            if self.num_single_photon >= 2:
                Fock(1) | q[2]

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




