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
    1. Prepare Ancilla (Mode 1) (Optional Fock(1)).
    2. Squeeze Ancilla.
    3. Displace Ancilla.
    4. BS Interaction between Loop (0) and Ancilla (1).
    """
    
    def __init__(self, steps: int, time_invariant: bool = False, clip_size: float = 2.0, 
                 measure_fock_cutoff: int = 5, num_single_photon: int = 0,
                 train_initial_state: bool = False, initial_r: float = 0.0):
        super().__init__(steps, time_invariant)
        self.clip_size = clip_size
        self.measure_fock_cutoff = measure_fock_cutoff
        self.num_single_photon = num_single_photon
        self.train_initial_state = train_initial_state
        self.initial_r = initial_r
        
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
        return [(1, self.measure_fock_cutoff)]

    # --- Initial Parameter Logic ---

    @property
    def num_initial_parameters(self) -> int:
        # Returns 2 (r, phi) if we are training the initial state of Mode 0
        return 2 if self.train_initial_state else 0

    @property
    def initial_parameter_bounds(self) -> list[tuple[float, float]]:
        if self.train_initial_state:
            return [(0.0, self.clip_size), (-np.pi, np.pi)]
        return []

    def get_initial_state_ket(self, init_params: np.ndarray, cutoff_dim: int) -> np.ndarray:
        """Generates the starting state for the loop mode (Mode 0)."""
        if self.train_initial_state:
            r, phi = init_params[0], init_params[1]
        else:
            r, phi = self.initial_r, 0.0

        if abs(r) < 1e-6:
            ket = np.zeros(cutoff_dim, dtype=np.complex128)
            ket[0] = 1.0
            return ket

        # Generate Squeezed State for initialization
        prog = sf.Program(1)
        with prog.context as q:
            Sgate(r, phi) | q[0]
        
        eng = sf.Engine("fock", backend_options={"cutoff_dim": cutoff_dim})
        result = eng.run(prog)
        return result.state.ket().flatten()

    # --- Execution Logic ---

    def run_step(self, state, step_idx: int, step_params: np.ndarray, engine: sf.Engine):
        """Executes one step of the time-domain circuit."""
        # Unpack parameters (6 params)
        sq_r, sq_phi, disp_r, disp_phi, bs_theta, bs_phi = step_params
        
        prog = sf.Program(2)
        with prog.context as q:
            # 1. Conditionally Prepare Ancilla |1>
            if self.num_single_photon >= 1:
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
    Time-domain gadget with Squeezing on Ancilla (1) only, no displacement.
    
    Architecture per step:
    1. Prepare Ancilla (Mode 1).
    2. Squeeze Ancilla (Mode 1).
    3. BS Interaction between Loop (0) and Ancilla (1).
    """
    
    def __init__(self, steps: int, time_invariant: bool = False, clip_size: float = 2.0, measure_fock_cutoff: int = 5, num_single_photon=0,
                 train_initial_state: bool = False,
                 initial_r: float = 0.0):
        super().__init__(steps, time_invariant)
        self.clip_size = clip_size
        self.measure_fock_cutoff = measure_fock_cutoff
        self.num_single_photon = num_single_photon

        self.train_initial_state = train_initial_state
        self.initial_r = initial_r

        
        self._param_names = [
            'sq_r', 'sq_phi', 
            'bs_theta', 'bs_phi'
        ]
        
        self._bounds = [
            (0.0, self.clip_size),  # sq_r
            (-np.pi, np.pi),        # sq_phi
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

    @property
    def num_initial_parameters(self) -> int:
        # If training, we need 2 params: r and phi
        return 2 if self.train_initial_state else 0

    @property
    def initial_parameter_bounds(self) -> list[tuple[float, float]]:
        if self.train_initial_state:
            return [(0.0, self.clip_size), (-np.pi, np.pi)] # r, phi bounds
        return []

    def get_initial_state_ket(self, init_params: np.ndarray, cutoff_dim: int) -> np.ndarray:
        # Determine R and Phi
        if self.train_initial_state:
            r, phi = init_params[0], init_params[1]
        else:
            # Manual mode
            r, phi = self.initial_r, 0.0

        if abs(r) < 1e-6:
            # Return vacuum if squeezing is zero
            ket = np.zeros(cutoff_dim, dtype=np.complex128)
            ket[0] = 1.0
            return ket

        # Generate Squeezed State using a temporary engine
        prog = sf.Program(1)
        with prog.context as q:
            Sgate(r, phi) | q[0]
        
        eng = sf.Engine("fock", backend_options={"cutoff_dim": cutoff_dim})
        result = eng.run(prog)
        return result.state.ket().flatten()

    def run_step(self, state, step_idx: int, step_params: np.ndarray, engine: sf.Engine):
        sq_r, sq_phi, bs_theta, bs_phi = step_params
        
        prog = sf.Program(2)
        with prog.context as q:
            # 1. Prepare Ancilla
            if self.num_single_photon >= 1:
                Fock(1) | q[1]
            
            # 2. Ops
            Sgate(sq_r, sq_phi) | q[1]
            
            BSgate(bs_theta, bs_phi) | (q[0], q[1])
            
        return engine.run(prog)


class ThreeModeTimeDomainSqueezeOnly(TimeMultiplexedCircuit):
    """
    Time-domain gadget with 1 Loop (0) and 2 Ancillas (1, 2).
    Squeezing on ancillas 1 & 2 only, 3 BS interactions.
    
    Architecture per step:
    1. Reset Ancillas 1, 2.
    2. Squeeze 1, 2.
    3. BS(0,1), BS(1,2), BS(0,1).
    """
    
    def __init__(self, steps: int, time_invariant: bool = False, clip_size: float = 2.0, measure_fock_cutoff: int = 5, num_single_photon=0,
                train_initial_state: bool = False,
                 initial_r: float = 0.0):
        super().__init__(steps, time_invariant)
        self.clip_size = clip_size
        self.measure_fock_cutoff = measure_fock_cutoff
        self.num_single_photon = num_single_photon

        self.train_initial_state = train_initial_state
        self.initial_r = initial_r
        
        self._param_names = [
            'sq1_r', 'sq2_r',
            'sq1_phi', 'sq2_phi',
            'bs_theta1', 'bs_theta2', 'bs_theta3',
            'bs_phi1', 'bs_phi2', 'bs_phi3'
        ]
        
        self._bounds = []
        self._bounds.extend([(0.0, self.clip_size)] * 2) # sq_r (only 2 now)
        self._bounds.extend([(-np.pi, np.pi)] * 2)       # sq_phi (only 2 now)
        self._bounds.extend([(0.0, np.pi/2)] * 3)      # bs_theta
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
    
    @property
    def num_initial_parameters(self) -> int:
        # If training, we need 2 params: r and phi
        return 2 if self.train_initial_state else 0

    @property
    def initial_parameter_bounds(self) -> list[tuple[float, float]]:
        if self.train_initial_state:
            return [(0.0, self.clip_size), (-np.pi, np.pi)] # r, phi bounds
        return []

    def get_initial_state_ket(self, init_params: np.ndarray, cutoff_dim: int) -> np.ndarray:
        # Determine R and Phi
        if self.train_initial_state:
            r, phi = init_params[0], init_params[1]
        else:
            # Manual mode
            r, phi = self.initial_r, 0.0

        if abs(r) < 1e-6:
            # Return vacuum if squeezing is zero
            ket = np.zeros(cutoff_dim, dtype=np.complex128)
            ket[0] = 1.0
            return ket

        # Generate Squeezed State using a temporary engine
        prog = sf.Program(1)
        with prog.context as q:
            Sgate(r, phi) | q[0]
        
        eng = sf.Engine("fock", backend_options={"cutoff_dim": cutoff_dim})
        result = eng.run(prog)
        return result.state.ket().flatten()

    def run_step(self, state, step_idx: int, step_params: np.ndarray, engine: sf.Engine):
        sq_r = step_params[:2]
        sq_phi = step_params[2:4]
        bs_theta = step_params[4:7]
        bs_phi = step_params[7:]
        
        prog = sf.Program(3)
        with prog.context as q:
            # 1. Reset Ancillas
            if self.num_single_photon >= 1:
                Fock(1) | q[1]
            if self.num_single_photon >= 2:
                Fock(1) | q[2]
            
            # 2. Squeezing (ancillas only)
            Sgate(sq_r[0], sq_phi[0]) | q[1]
            Sgate(sq_r[1], sq_phi[1]) | q[2]
                
            # 3. Interferometer
            BSgate(bs_theta[0], bs_phi[0]) | (q[0], q[1])
            BSgate(bs_theta[1], bs_phi[1]) | (q[1], q[2])
            BSgate(bs_theta[2], bs_phi[2]) | (q[0], q[1])
            
        return engine.run(prog)
