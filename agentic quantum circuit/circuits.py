"""Circuit implementations implementing the CircuitContext interface."""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import strawberryfields as sf
from strawberryfields.ops import Sgate, Dgate, BSgate, Vgate

from quantum_modules import CircuitContext


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
    
    def get_action_space(self) -> gym.spaces.Box:
        """Returns the action space: 5D continuous [-1, 1]."""
        return spaces.Box(
            low=-1.0, high=1.0, shape=(5,), dtype=np.float32
        )
    
    def build_reset_program(self) -> sf.Program:
        """Initializes the loop with a squeezed state."""
        prog = sf.Program(2)
        with prog.context as q:
            Sgate(self.max_sq_r) | q[0]
        return prog
    
    def build_step_program(self, action: np.ndarray) -> sf.Program:
        """
        Builds the circuit for one time step.
        Action: [r_mag, r_phi, d_mag, d_phi, bs_theta] (normalized to [-1, 1])
        """
        # Denormalize action from [-1, 1] to physical ranges
        r_mag = np.clip(action[0] * self.max_sq_r, -self.max_sq_r, self.max_sq_r)
        r_phi = np.clip(action[1] * np.pi, 0, 2 * np.pi)
        d_mag = np.clip(action[2] * self.max_disp_mag, 0, self.max_disp_mag)
        d_phi = np.clip(action[3] * np.pi, 0, 2 * np.pi)
        bs_theta = np.clip(action[4] * np.pi/2, 0, np.pi/2)

        prog = sf.Program(2)
        with prog.context as q:
            # q[0] = Loop Memory
            # q[1] = Fresh Input (Ancilla)
            
            # 1. Prepare Ancilla
            Sgate(r_mag, r_phi) | q[1]
            Dgate(d_mag, d_phi) | q[1]
            
            # 2. Interact (beamsplitter)
            BSgate(bs_theta, 0) | (q[0], q[1])
        
        return prog


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
    
    def get_action_space(self) -> gym.spaces.Box:
        """Returns the action space (2D or 3D continuous)."""
        if self.tunable_r:
            shape = (2,)  # [r, theta]
        else:
            shape = (2,)  # [phase, theta]
        return spaces.Box(
            low=-1.0, high=1.0, shape=shape, dtype=np.float32
        )
    
    def build_reset_program(self) -> sf.Program:
        """Initializes the loop with high squeezing."""
        prog = sf.Program(2)
        with prog.context as q:
            Sgate(self.max_squeezing) | q[0]
        return prog
    
    def build_step_program(self, action: np.ndarray) -> sf.Program:
        """
        Builds the circuit for one time step.
        
        If tunable_r:
            action: [squeezing_r (normalized), theta_1 (normalized)]
        Else:
            action: [squeezing_phase (normalized), theta_1 (normalized)]
        """
        if self.tunable_r:
            # Denormalize from [-1, 1] to physical ranges
            squeezing_r = np.clip(action[0] * self.max_squeezing, -self.max_squeezing, self.max_squeezing)
            theta_1 = np.clip(action[1] * np.pi/2, 0, np.pi/2)
            squeezing_phase = 0
        else:
            # Fixed squeezing magnitude
            squeezing_r = self.max_squeezing
            theta_1 = np.clip(action[1] * np.pi/2, 0, np.pi/2)
            squeezing_phase = np.clip(action[0] * np.pi, -np.pi, np.pi)

        prog = sf.Program(2)
        with prog.context as q:
            Sgate(squeezing_r, squeezing_phase) | q[1]
            BSgate(theta_1, 0) | (q[0], q[1])
        
        return prog


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
    
    def get_action_space(self) -> gym.spaces.Box:
        """Returns 3D continuous action space."""
        return spaces.Box(
            low=-1.0, high=1.0, shape=(3,), dtype=np.float32
        )
    
    def build_reset_program(self) -> sf.Program:
        """Initialize loop with squeezed state."""
        prog = sf.Program(2)
        with prog.context as q:
            Sgate(self.max_sq_r) | q[0]
        return prog
    
    def build_step_program(self, action: np.ndarray) -> sf.Program:
        """
        Builds the circuit for one time step.
        Action: [r, theta, alpha] (normalized to [-1, 1])
        """
        # Denormalize
        r_val = np.clip(action[0] * self.max_sq_r, -self.max_sq_r, self.max_sq_r)
        theta_val = np.clip(action[1] * np.pi/2, 0, np.pi/2)
        alpha_val = np.clip(action[2] * self.max_disp_alpha, -self.max_disp_alpha, self.max_disp_alpha)

        prog = sf.Program(2)
        with prog.context as q:
            # q[0] = Loop Memory
            # q[1] = Fresh Input
            
            # Prepare Input State
            Sgate(r_val) | q[1]
            
            # Displacement (imaginary axis)
            z = 1j * alpha_val
            r = np.abs(z)
            theta = np.angle(z)
            Dgate(r, theta) | q[1]
            
            # Beamsplitter (VBS)
            BSgate(theta_val, 0) | (q[0], q[1])
            
            # Apply FIXED displacement in measurement arm
            z = 1j * self.fixed_measure_beta
            r = np.abs(z)
            theta = np.angle(z)
            Dgate(r, theta) | q[0]
        
        return prog


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
        self.max_squeezing = max_squeezing  # 8 dB
        self.max_disp = max_disp
    
    def get_action_space(self) -> gym.spaces.Box:
        """Returns 8D continuous action space."""
        return spaces.Box(
            low=-1.0, high=1.0, shape=(8,), dtype=np.float32
        )
    
    def build_reset_program(self) -> sf.Program:
        """Initialize: vacuum with measurement on q[0]."""
        prog = sf.Program(2)
        # Vacuum initialization (implicit)
        return prog
    
    def build_step_program(self, action: np.ndarray) -> sf.Program:
        """
        Builds the 2-mode gadget circuit for one time step.
        Action: [r, phi_sq, theta, phi_bs, alpha_mag, alpha_phi, alpha_mag2, alpha_phi2]
        """
        # Denormalize from [-1, 1] to physical ranges
        r_val = np.clip(action[0] * self.max_squeezing, 0, self.max_squeezing)
        phi_sq_val = np.clip(action[1] * np.pi, -np.pi, np.pi)
        theta_val = np.clip(action[2] * np.pi/2, 0, np.pi/2)
        phi_val = np.clip(action[3] * np.pi, -np.pi, np.pi)
        alpha_mag = np.clip(action[4] * self.max_disp, 0, self.max_disp)
        alpha_phi = np.clip(action[5] * np.pi, -np.pi, np.pi)
        alpha_mag2 = np.clip(action[6] * self.max_disp, 0, self.max_disp)
        alpha_phi2 = np.clip(action[7] * np.pi, -np.pi, np.pi)

        prog = sf.Program(2)
        with prog.context as q:
            # q[0] = Mode A (loop memory)
            # q[1] = Mode B (fresh input)
            
            # Prepare fresh input
            Sgate(r_val, phi_sq_val) | q[1]
            Dgate(alpha_mag, alpha_phi) | q[1]
            
            # Beamsplitter interaction
            BSgate(theta_val, phi_val) | (q[0], q[1])
            
            # Output displacement on mode A
            Dgate(alpha_mag2, alpha_phi2) | q[0]
        
        return prog
