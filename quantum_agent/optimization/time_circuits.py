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

    def get_measurement_spec(self) -> tuple[int, int]:
        """
        Returns (measurement_mode_index, max_fock_cutoff).
        """
        return (1, self.measure_fock_cutoff)

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