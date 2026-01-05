"""
Concrete implementation of TimeMultiplexedCircuit for time-domain optimization.
"""
import numpy as np
import strawberryfields as sf
from strawberryfields.ops import Sgate, Dgate, BSgate, Fock

from quantum_agent.optimization.time_interfaces import TimeMultiplexedCircuit

class TwoModeTimeDomainGadget(TimeMultiplexedCircuit):
    """
    Time-domain gadget with Loop (Mode 0) and Ancilla (Mode 1).
    
    Architecture per step:
    1. Prepare Ancilla (Mode 1) in Fock |1>.
    2. Squeeze Ancilla.
    3. Displace Ancilla.
    4. BS Interaction between Loop (0) and Ancilla (1).
    """
    
    def __init__(self, steps: int, time_invariant: bool = False, clip_size: float = 2.0, measure_fock_cutoff: int = 5):
        super().__init__(steps, time_invariant)
        self.clip_size = clip_size
        self.measure_fock_cutoff = measure_fock_cutoff
        
        self._param_names = [
            'sq_r', 'sq_phi', 
            'disp_r', 'disp_phi', 
            'bs_theta', 'bs_phi'
        ]
        
        self._bounds = [
            (0.0, self.clip_size),  # sq_r
            (-np.pi, np.pi),        # sq_phi
            (0.0, self.clip_size),  # disp_r
            (-np.pi, np.pi),        # disp_phi
            (0.0, 2 * np.pi),       # bs_theta
            (-np.pi, np.pi)         # bs_phi
        ]

    @property
    def per_step_parameter_names(self) -> list[str]:
        return self._param_names

    @property
    def per_step_parameter_bounds(self) -> list[tuple[float, float]]:
        return self._bounds

    def get_measurement_specs(self) -> list[tuple[int, int]]:
        """
        Returns list of (measurement_mode_index, max_fock_cutoff).
        """
        return [(1, self.measure_fock_cutoff)]

    def run_step(self, state, step_idx: int, step_params: np.ndarray, engine: sf.Engine):
        """
        Executes one step of the time-domain circuit.
        """
        # Unpack parameters (6 params)
        sq_r, sq_phi, disp_r, disp_phi, bs_theta, bs_phi = step_params
        
        prog = sf.Program(2)
        with prog.context as q:
            # 1. Prepare Ancilla |1> (Reset mode 1)
            Fock(1) | q[1]
            
            # 2. Squeeze Ancilla
            Sgate(sq_r, sq_phi) | q[1]
            
            # 3. Displace Ancilla
            Dgate(disp_r, disp_phi) | q[1]
            
            # 4. Interaction
            BSgate(bs_theta, bs_phi) | (q[0], q[1])
            
        return engine.run(prog)


class TwoModeTimeDomainSqueezeOnly(TimeMultiplexedCircuit):
    """
    Time-domain gadget with Squeezing on both Loop (0) and Ancilla (1), no displacement.
    
    Architecture per step:
    1. Prepare Ancilla (Mode 1).
    2. Squeeze Loop (Mode 0).
    3. Squeeze Ancilla (Mode 1).
    4. BS Interaction between Loop (0) and Ancilla (1).
    """
    
    def __init__(self, steps: int, time_invariant: bool = False, clip_size: float = 2.0, measure_fock_cutoff: int = 5, num_single_photon=0):
        super().__init__(steps, time_invariant)
        self.clip_size = clip_size
        self.measure_fock_cutoff = measure_fock_cutoff
        self.num_single_photon = num_single_photon
        
        self._param_names = [
            'sq0_r', 'sq0_phi', 
            'sq1_r', 'sq1_phi', 
            'bs_theta', 'bs_phi'
        ]
        
        self._bounds = [
            (0.0, self.clip_size),  # sq0_r
            (-np.pi, np.pi),        # sq0_phi
            (0.0, self.clip_size),  # sq1_r
            (-np.pi, np.pi),        # sq1_phi
            (0.0, 2 * np.pi),       # bs_theta
            (-np.pi, np.pi)         # bs_phi
        ]

    @property
    def per_step_parameter_names(self) -> list[str]:
        return self._param_names

    @property
    def per_step_parameter_bounds(self) -> list[tuple[float, float]]:
        return self._bounds

    def get_measurement_specs(self) -> list[tuple[int, int]]:
        return [(1, self.measure_fock_cutoff)]

    def run_step(self, state, step_idx: int, step_params: np.ndarray, engine: sf.Engine):
        sq0_r, sq0_phi, sq1_r, sq1_phi, bs_theta, bs_phi = step_params
        
        prog = sf.Program(2)
        with prog.context as q:
            # 1. Prepare Ancilla
            if self.num_single_photon >= 1:
                Fock(1) | q[1]
            
            # 2. Ops
            Sgate(sq0_r, sq0_phi) | q[0]
            Sgate(sq1_r, sq1_phi) | q[1]
            
            BSgate(bs_theta, bs_phi) | (q[0], q[1])
            
        return engine.run(prog)


class ThreeModeTimeDomainSqueezeOnly(TimeMultiplexedCircuit):
    """
    Time-domain gadget with 1 Loop (0) and 2 Ancillas (1, 2).
    Squeezing on all modes, 3 BS interactions.
    
    Architecture per step:
    1. Reset Ancillas 1, 2.
    2. Squeeze 0, 1, 2.
    3. BS(0,1), BS(1,2), BS(0,1).
    """
    
    def __init__(self, steps: int, time_invariant: bool = False, clip_size: float = 2.0, measure_fock_cutoff: int = 5, num_single_photon=0):
        super().__init__(steps, time_invariant)
        self.clip_size = clip_size
        self.measure_fock_cutoff = measure_fock_cutoff
        self.num_single_photon = num_single_photon
        
        self._param_names = [
            'sq0_r', 'sq1_r', 'sq2_r',
            'sq0_phi', 'sq1_phi', 'sq2_phi',
            'bs_theta1', 'bs_theta2', 'bs_theta3',
            'bs_phi1', 'bs_phi2', 'bs_phi3'
        ]
        
        self._bounds = []
        self._bounds.extend([(0.0, self.clip_size)] * 3) # sq_r
        self._bounds.extend([(-np.pi, np.pi)] * 3)       # sq_phi
        self._bounds.extend([(0.0, 2 * np.pi)] * 3)      # bs_theta
        self._bounds.extend([(-np.pi, np.pi)] * 3)       # bs_phi

    @property
    def per_step_parameter_names(self) -> list[str]:
        return self._param_names

    @property
    def per_step_parameter_bounds(self) -> list[tuple[float, float]]:
        return self._bounds

    def get_measurement_specs(self) -> list[tuple[int, int]]:
        # Returns specs for both ancillas (Mode 1 and Mode 2)
        return [(1, self.measure_fock_cutoff), (2, self.measure_fock_cutoff)]

    def run_step(self, state, step_idx: int, step_params: np.ndarray, engine: sf.Engine):
        sq_r = step_params[:3]
        sq_phi = step_params[3:6]
        bs_theta = step_params[6:9]
        bs_phi = step_params[9:]
        
        prog = sf.Program(3)
        with prog.context as q:
            # 1. Reset Ancillas
            if self.num_single_photon >= 1:
                Fock(1) | q[1]
            if self.num_single_photon >= 2:
                Fock(1) | q[2]
            
            # 2. Squeezing
            for k in range(3):
                Sgate(sq_r[k], sq_phi[k]) | q[k]
                
            # 3. Interferometer
            BSgate(bs_theta[0], bs_phi[0]) | (q[0], q[1])
            BSgate(bs_theta[1], bs_phi[1]) | (q[1], q[2])
            BSgate(bs_theta[2], bs_phi[2]) | (q[0], q[1])
            
        return engine.run(prog)
