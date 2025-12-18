"""Circuit implementations implementing the updated CircuitContext interface."""

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

from quantum_modules import CircuitContext
from monitored_loss_measure_fock_patch import MonitoredLossMeasureFock

class GKPCircuit(CircuitContext):
    """
    GKP circuit implementation with squeezing, displacement, and beamsplitter.
    
    Action Space (5 Continuous):
    1. Squeezing Magnitude (r)
    2. Squeezing Angle (phi)
    3. Displacement Magnitude (|alpha|)
    4. Displacement Angle (arg(alpha))
    5. Beamsplitter Transmissivity (theta)
    """
    def __init__(self, max_sq_r=1.38, max_disp_mag=2.5):
        self.max_sq_r = max_sq_r
        self.max_disp_mag = max_disp_mag
        
        self._action_keys = ['r_mag', 'r_phi', 'd_mag', 'd_phi', 'bs_theta']
        self._action_ranges = {
            'r_mag':    (0.0, self.max_sq_r),
            'r_phi':    (0.0, 2 * np.pi),
            'd_mag':    (0.0, self.max_disp_mag),
            'd_phi':    (0.0, 2 * np.pi),
            'bs_theta': (0.0, np.pi/2)
        }

    @property
    def action_keys(self): return self._action_keys

    @property
    def action_ranges(self): return self._action_ranges

    def get_action_space(self) -> gym.spaces.Box:
        return spaces.Box(low=-1.0, high=1.0, shape=(len(self._action_keys),), dtype=np.float32)
    
    def build_reset_program(self) -> sf.Program:
        prog = sf.Program(2)
        with prog.context as q:
            Sgate(self.max_sq_r) | q[0]
        return prog
    
    def build_step_program(self, action_dict: dict) -> sf.Program:
        """
        Builds the circuit for one time step.
        Action: [r_mag, r_phi, d_mag, d_phi, bs_theta] (normalized to [-1, 1])
        """
        prog = sf.Program(2)
        with prog.context as q:
            # Prepare Ancilla in q[1]
            Sgate(action_dict['r_mag'], action_dict['r_phi']) | q[1]
            Dgate(action_dict['d_mag'], action_dict['d_phi']) | q[1]
            # Interact
            BSgate(action_dict['bs_theta'], 0) | (q[0], q[1])
            
            # Measurement and reset are critical for loop stability
            MonitoredLossMeasureFock(1) | q[0]
            BSgate(np.pi/2, 0) | (q[0], q[1])
            
        return prog

    def _get_current_ket(self, state):
        return state.ket()[:, 0]


class GeneralLoopCircuit(CircuitContext):
    """
    General loop circuit for squeezed cat generation.
    Handles tunable squeezing parameter.
    
    Action Space:
    - If tunable_r: [squeezing_r, theta_1]
    - If not tunable_r: [squeezing_phase, theta_1]
    """
    
    def __init__(self, tunable_r=True, max_squeezing=1.38):
        self.tunable_r = tunable_r
        self.max_squeezing = max_squeezing
        
        if self.tunable_r:
            self._action_keys = ['squeezing_r', 'theta_1']
            self._action_ranges = {
                'squeezing_r': (-self.max_squeezing, self.max_squeezing),
                'theta_1': (0, np.pi/2)
            }
        else:
            self._action_keys = ['squeezing_phase', 'theta_1']
            self._action_ranges = {
                'squeezing_phase': (-np.pi, np.pi),
                'theta_1': (0, np.pi/2)
            }
    
    @property
    def action_keys(self): return self._action_keys

    @property
    def action_ranges(self): return self._action_ranges

    def get_action_space(self) -> gym.spaces.Box:
        return spaces.Box(low=-1.0, high=1.0, shape=(len(self._action_keys),), dtype=np.float32)
    
    def build_reset_program(self) -> sf.Program:
        """Initializes the loop with high squeezing."""
        prog = sf.Program(2)
        with prog.context as q:
            Sgate(self.max_squeezing) | q[0]
        return prog
    
    def build_step_program(self, action_dict: dict) -> sf.Program:
        prog = sf.Program(2)
        with prog.context as q:
            # 1. Prepare Ancilla (q[1])
            if self.tunable_r:
                Sgate(action_dict['squeezing_r'], 0) | q[1]
            else:
                Sgate(self.max_squeezing, action_dict['squeezing_phase']) | q[1]
            
            # 2. Interact Data(q[0]) and Ancilla(q[1])
            BSgate(action_dict['theta_1'], 0) | (q[0], q[1])

            # 3. Measure and Reset (CRITICAL FIX)
            # We measure q[0] (which contains the 'waste' after interaction in this setup)
            # and then swap q[1] (which momentarily holds the data) back to q[0].
            MonitoredLossMeasureFock(1) | q[0]
            BSgate(np.pi/2, 0) | (q[0], q[1])

        return prog

    def _get_current_ket(self, state):
        # Even after measurement, the state object handles the projection.
        # We extract the ket of the active mode (q[0] after swap).
        return state.ket()[:, 0]


class CubicSpecificCircuit(CircuitContext):
    """
    Cubic Phase specific circuit with fixed measurement displacement.
    
    Action Space (3 Continuous):
    1. Input Squeezing (r)
    2. Beam Splitter Transmissivity (theta)
    3. Input Displacement (alpha)
    """
    
    def __init__(self, max_sq_r=1.15, max_disp_alpha=1.0, fixed_measure_beta=2.0):
        self.max_sq_r = max_sq_r
        self.max_disp_alpha = max_disp_alpha
        self.fixed_measure_beta = fixed_measure_beta
        
        self._action_keys = ['r', 'theta', 'alpha']
        self._action_ranges = {
            'r': (-self.max_sq_r, self.max_sq_r),
            'theta': (0, np.pi/2),
            'alpha': (-self.max_disp_alpha, self.max_disp_alpha) 
        }

    @property
    def action_keys(self): return self._action_keys

    @property
    def action_ranges(self): return self._action_ranges

    def get_action_space(self) -> gym.spaces.Box:
        return spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
    
    def build_reset_program(self) -> sf.Program:
        """Initialize loop with squeezed state."""
        prog = sf.Program(2)
        with prog.context as q:
            Sgate(self.max_sq_r) | q[0]
        return prog
    
    def build_step_program(self, action_dict: dict) -> sf.Program:
        prog = sf.Program(2)
        with prog.context as q:
            # Prepare Input
            Sgate(action_dict['r']) | q[1]
            # Displacement on imaginary axis
            Dgate(np.abs(1j * action_dict['alpha']), np.angle(1j * action_dict['alpha'])) | q[1]
            
            # Interaction
            BSgate(action_dict['theta'], 0) | (q[0], q[1])
            
            # Fixed displacement in measurement arm (q[0])
            Dgate(np.abs(1j * self.fixed_measure_beta), np.angle(1j * self.fixed_measure_beta)) | q[0]
            
            # Measure and Swap
            MonitoredLossMeasureFock(1) | q[0]
            BSgate(np.pi/2, 0) | (q[0], q[1])

        return prog

    def _get_current_ket(self, state):
        return state.ket()[:, 0]


class GadgetCircuit(CircuitContext):
    """
    2-Mode Time-Multiplexed Gadget circuit with full displacement control.
    
    Action Space (8 Continuous):
    1. Squeezing magnitude (r)
    2. Squeezing phase (phi)
    3. Beamsplitter theta
    4. Beamsplitter phi
    5. Displacement magnitude (alpha)
    6. Displacement phase (phi_alpha)
    7. Second displacement magnitude (alpha2)
    8. Second displacement phase (phi_alpha2)
    """
    
    def __init__(self, max_squeezing=0.347, max_disp=1.0):
        self.max_squeezing = max_squeezing
        self.max_disp = max_disp
        
        self._action_keys = [
            'squeezing_r', 'squeezing_phase', 'theta_1', 'phi_1',
            'disp_mag1', 'disp_phi1', 'disp_mag2', 'disp_phi2'
        ]
        self._action_ranges = {
            'squeezing_r': (0, self.max_squeezing),
            'squeezing_phase': (-np.pi, np.pi),
            'theta_1': (0, np.pi/2),
            'phi_1': (-np.pi, np.pi),
            'disp_mag1': (0, self.max_disp),
            'disp_phi1': (-np.pi, np.pi),
            'disp_mag2': (0, self.max_disp),
            'disp_phi2': (-np.pi, np.pi)
        }
    
    @property
    def action_keys(self): return self._action_keys

    @property
    def action_ranges(self): return self._action_ranges

    def get_action_space(self) -> gym.spaces.Box:
        return spaces.Box(low=-1.0, high=1.0, shape=(8,), dtype=np.float32)
    
    def build_reset_program(self) -> sf.Program:
        return sf.Program(2) # Starts in vacuum
    
    def build_step_program(self, action_dict: dict) -> sf.Program:
        prog = sf.Program(2)
        with prog.context as q:
            # Prepare input on q[1]
            Sgate(action_dict['squeezing_r'], action_dict['squeezing_phase']) | q[1]
            Dgate(action_dict['disp_mag1'], action_dict['disp_phi1']) | q[1]
            
            # BS Interaction
            BSgate(action_dict['theta_1'], action_dict['phi_1']) | (q[0], q[1])
            
            # Output displacement on loop mode
            Dgate(action_dict['disp_mag2'], action_dict['disp_phi2']) | q[0]

            # Measure and Swap
            MonitoredLossMeasureFock(1) | q[0]
            BSgate(np.pi/2, 0) | (q[0], q[1])

        return prog

    def _get_current_ket(self, state):
        return state.ket()[:, 0]


class QuarticSpecificCircuit(CircuitContext):
    """Quartic Phase specific circuit."""
    def __init__(self, max_sq_r=1.0, max_disp_alpha=2.0):
        self.max_sq_r = max_sq_r
        self.max_disp_alpha = max_disp_alpha
        
        self._action_keys = ['r', 'theta', 'alpha']
        self._action_ranges = {
            'r': (-self.max_sq_r, self.max_sq_r),
            'theta': (0, np.pi/2),
            'alpha': (-self.max_disp_alpha, self.max_disp_alpha) 
        }

    @property
    def action_keys(self): return self._action_keys

    @property
    def action_ranges(self): return self._action_ranges

    def get_action_space(self) -> gym.spaces.Box:
        return spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
    
    def build_reset_program(self) -> sf.Program:
        prog = sf.Program(2)
        with prog.context as q:
            Sgate(self.max_sq_r) | q[0]
        return prog
    
    def build_step_program(self, action_dict: dict) -> sf.Program:
        prog = sf.Program(2)
        with prog.context as q:
            Sgate(action_dict['r']) | q[1]
            # Displacement (Imaginary axis)
            alpha_z = 1j * action_dict['alpha']
            Dgate(np.abs(alpha_z), np.angle(alpha_z)) | q[1]
            
            BSgate(action_dict['theta'], 0) | (q[0], q[1])
            
            # Measure and Swap
            MonitoredLossMeasureFock(1) | q[0]
            BSgate(np.pi/2, 0) | (q[0], q[1])

        return prog

    def _get_current_ket(self, state):
        return state.ket()[:, 0]