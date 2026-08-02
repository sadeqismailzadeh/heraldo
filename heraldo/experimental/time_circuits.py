"""Concrete implementations of TimeMultiplexedCircuit for time-domain photonic optimization.

Provides multi-mode time-domain multiplexed optical circuit architectures containing
a persistent memory mode (Mode 0) and one or more ancillary modes (Modes 1..N-1).
"""
import numpy as np
import strawberryfields as sf
from strawberryfields.ops import Sgate, Dgate, BSgate, Fock, LossChannel

from heraldo.components.interfaces import TimeMultiplexedCircuit


class BaseTimeDomainGeneral(TimeMultiplexedCircuit):
    """Base class for time-domain multiplexed general circuits.

    Encapsulates common configuration, initial state preparation for loop Mode 0,
    measurement specifications for ancillary modes, and parameter property accessors.

    Args:
        steps (int): Number of time-domain recirculation steps :math:`T`.
        time_invariant (bool, optional): If True, parameter values are identical across all steps.
            Defaults to False.
        clip_size (float, optional): Maximum absolute bound for squeezing and displacement
            magnitude parameters. Defaults to 2.0.
        measure_fock_cutoff (int, optional): Maximum Fock cutoff for photon-number-resolving (PNR)
            detectors on ancillary modes. Defaults to 5.
        num_single_photon (int, optional): Number of ancillary modes initialized with single-photon
            states :math:`|1\\rangle` instead of vacuum :math:`|0\\rangle`. Defaults to 0.
        initial_fock_one (bool, optional): If True, initializes the memory mode (Mode 0) in
            the single-photon state :math:`|1\\rangle` before squeezing. Defaults to False.
        loss_transmissivity (float, optional): Channel transmissivity parameter :math:`\\eta \\in (0, 1]`
            simulating loss on all modes at each step. Defaults to 1.0 (lossless).
        **kwargs: Additional keyword arguments passed to :class:`TimeMultiplexedCircuit`.
    """
    num_modes: int = 2

    def __init__(self, steps: int, time_invariant: bool = False, clip_size: float = 2.0, 
                 measure_fock_cutoff: int = 5, num_single_photon: int = 0,
                 initial_fock_one: bool = False, loss_transmissivity: float = 1.0, **kwargs):
        super().__init__(steps, time_invariant, **kwargs)
        self.clip_size = clip_size
        self.measure_fock_cutoff = measure_fock_cutoff
        self.num_single_photon = num_single_photon
        self.initial_fock_one = initial_fock_one
        self.loss_transmissivity = loss_transmissivity
        
        self._param_names: list[str] = []
        self._bounds: list[tuple[float, float]] = []

    @property
    def per_step_parameter_names(self) -> list[str]:
        """list[str]: Names of optimization parameters applied per time step."""
        return self._param_names

    @property
    def per_step_parameter_bounds(self) -> list[tuple[float, float]]:
        """list[tuple[float, float]]: Parameter bounds (min, max) per time step."""
        return self._bounds

    def get_measurement_specs(self) -> list[tuple[int, int]]:
        """Generates measurement specifications for ancillary modes.

        Returns:
            list[tuple[int, int]]: List of tuples ``(mode_index, max_fock_cutoff)``
            for each measured ancillary mode (Modes 1 to N-1).
        """
        return [(mode, self.measure_fock_cutoff) for mode in range(1, self.num_modes)]

    @property
    def num_initial_parameters(self) -> int:
        """int: Number of parameters used to prepare the initial loop state on Mode 0."""
        return 2

    @property
    def initial_parameter_bounds(self) -> list[tuple[float, float]]:
        """list[tuple[float, float]]: Bounds for the initial state preparation parameters."""
        return [(-self.clip_size, self.clip_size), (-8*np.pi, 8*np.pi)]

    def get_initial_state_ket(self, init_params: np.ndarray, cutoff_dim: int) -> np.ndarray:
        """Generates the initial state vector (ket) for the loop mode (Mode 0).

        Prepares Mode 0 by applying optional single-photon initialization followed by squeezing
        :math:`\\hat{S}(r, \\phi)` with parameters provided in ``init_params``.

        Args:
            init_params (np.ndarray): 1D array containing ``[r, phi]`` initial squeezing parameters.
            cutoff_dim (int): Fock space cutoff dimension for state vector truncation.

        Returns:
            np.ndarray: Complex 1D array representing the initial state vector in Fock space.
        """
        r, phi = init_params[0], init_params[1]

        if abs(r) < 1e-6 and not self.initial_fock_one:
            ket = np.zeros(cutoff_dim, dtype=np.complex128)
            ket[0] = 1.0
            return ket

        prog = sf.Program(1)
        with prog.context as q:
            if self.initial_fock_one:
                Fock(1) | q[0]
            Sgate(r, phi) | q[0]
        
        eng = sf.Engine("fock", backend_options={"cutoff_dim": cutoff_dim})
        result = eng.run(prog)
        return result.state.ket().flatten()


class TwoModeTimeDomainGeneral(BaseTimeDomainGeneral):
    """Two-mode time-domain circuit consisting of a Loop mode (Mode 0) and an Ancilla mode (Mode 1).

    Each time step applies squeezing and displacement gates to the ancillary mode, followed by a 
    beam splitter interaction mixing the persistent loop mode with the prepared ancilla.

    Architecture per step:
        1. Prepare Ancilla (Mode 1) in vacuum :math:`|0\\rangle` or single-photon :math:`|1\\rangle`.
        2. Apply Squeezing gate :math:`\\hat{S}(r_{\\text{sq}}, \\phi_{\\text{sq}})` to Ancilla.
        3. Apply Displacement gate :math:`\\hat{D}(r_{\\text{disp}}, \\phi_{\\text{disp}})` to Ancilla.
        4. Apply Beam Splitter :math:`\\hat{BS}(\\theta, \\phi)` between Loop (Mode 0) and Ancilla (Mode 1).
        5. Apply optional loss channels with transmissivity :math:`\\eta`.

    Args:
        steps (int): Number of time-domain recirculation steps :math:`T`.
        time_invariant (bool, optional): If True, circuit parameters are identical across all steps.
            Defaults to False.
        clip_size (float, optional): Maximum absolute bound for squeezing and displacement magnitudes.
            Defaults to 2.0.
        measure_fock_cutoff (int, optional): PNR detector cutoff dimension for Mode 1. Defaults to 5.
        num_single_photon (int, optional): Number of ancillary modes initialized in :math:`|1\\rangle` (0 or 1).
            Defaults to 0.
        initial_fock_one (bool, optional): If True, initializes Mode 0 in :math:`|1\\rangle`. Defaults to False.
        loss_transmissivity (float, optional): Channel transmissivity :math:`\\eta \\in (0, 1]`. Defaults to 1.0.
        **kwargs: Additional arguments passed to :class:`BaseTimeDomainGeneral`.
    """
    num_modes = 2

    def __init__(self, steps: int, time_invariant: bool = False, clip_size: float = 2.0, 
                 measure_fock_cutoff: int = 5, num_single_photon: int = 0,
                 initial_fock_one: bool = False, loss_transmissivity: float = 1.0, **kwargs):
        super().__init__(
            steps=steps, time_invariant=time_invariant, clip_size=clip_size,
            measure_fock_cutoff=measure_fock_cutoff, num_single_photon=num_single_photon,
            initial_fock_one=initial_fock_one, loss_transmissivity=loss_transmissivity,
            **kwargs
        )
        
        self._param_names = [
            'sq_r', 'sq_phi', 
            'disp_r', 'disp_phi', 
            'bs_theta', 'bs_phi'
        ]

        self._param_names = ['init_sq_r', 'init_sq_phi'] + self._param_names
        
        self._bounds = [
            (-self.clip_size, self.clip_size),  # sq_r
            (-8*np.pi, 8*np.pi),        # sq_phi
            (-self.clip_size, self.clip_size),  # disp_r
            (-8*np.pi, 8*np.pi),        # disp_phi
            (-8*np.pi, 8*np.pi),       # bs_theta
            (-8*np.pi, 8*np.pi)         # bs_phi
        ]

    def run_step(self, state, step_idx: int, step_params: np.ndarray, engine: sf.Engine):
        """Executes a single step of the two-mode general time-domain circuit.

        Args:
            state: Unused legacy state parameter (state preparation is handled via engine context).
            step_idx (int): Current step index :math:`t`.
            step_params (np.ndarray): Array of 6 step parameters:
                ``[sq_r, sq_phi, disp_r, disp_phi, bs_theta, bs_phi]``.
            engine (sf.Engine): Strawberry Fields engine instance configured with Mode 0 input.

        Returns:
            sf.engine.Result: Strawberry Fields execution result containing updated state.
        """
        sq_r, sq_phi, disp_r, disp_phi, bs_theta, bs_phi = step_params
        
        prog = sf.Program(2)
        with prog.context as q:
            if self.num_single_photon >= 1:
                Fock(1) | q[1]
            
            Sgate(sq_r, sq_phi) | q[1]
            Dgate(disp_r, disp_phi) | q[1]
            BSgate(bs_theta, bs_phi) | (q[0], q[1])
            
            if self.loss_transmissivity < 1.0:
                LossChannel(self.loss_transmissivity) | q[0]
                LossChannel(self.loss_transmissivity) | q[1]
            
        return engine.run(prog)


class TwoModeTimeDomainSqueezeOnly(BaseTimeDomainGeneral):
    """Two-mode time-domain circuit with squeezing-only ancilla preparation (no displacement).

    Architecture per step:
        1. Prepare Ancilla (Mode 1) in vacuum :math:`|0\\rangle` or single-photon :math:`|1\\rangle`.
        2. Apply Squeezing gate :math:`\\hat{S}(r_{\\text{sq}}, \\phi_{\\text{sq}})` to Ancilla.
        3. Apply Beam Splitter :math:`\\hat{BS}(\\theta, \\phi)` between Loop (Mode 0) and Ancilla (Mode 1).
        4. Apply optional loss channels with transmissivity :math:`\\eta`.

    Args:
        steps (int): Number of time-domain recirculation steps :math:`T`.
        time_invariant (bool, optional): If True, circuit parameters are identical across all steps.
            Defaults to False.
        clip_size (float, optional): Maximum absolute bound for squeezing magnitudes. Defaults to 2.0.
        measure_fock_cutoff (int, optional): PNR detector cutoff dimension for Mode 1. Defaults to 5.
        num_single_photon (int, optional): Number of ancillary modes initialized in :math:`|1\\rangle`.
            Defaults to 0.
        initial_fock_one (bool, optional): If True, initializes Mode 0 in :math:`|1\\rangle`. Defaults to False.
        loss_transmissivity (float, optional): Channel transmissivity :math:`\\eta \\in (0, 1]`. Defaults to 1.0.
        **kwargs: Additional arguments passed to :class:`BaseTimeDomainGeneral`.
    """
    num_modes = 2

    def __init__(self, steps: int, time_invariant: bool = False, clip_size: float = 2.0, 
                 measure_fock_cutoff: int = 5, num_single_photon: int = 0,
                 initial_fock_one: bool = False, loss_transmissivity: float = 1.0, **kwargs):
        super().__init__(
            steps=steps, time_invariant=time_invariant, clip_size=clip_size,
            measure_fock_cutoff=measure_fock_cutoff, num_single_photon=num_single_photon,
            initial_fock_one=initial_fock_one, loss_transmissivity=loss_transmissivity,
            **kwargs
        )
        
        self._param_names = [
            'sq_r', 'sq_phi', 
            'bs_theta', 'bs_phi'
        ]

        self._param_names = ['init_sq_r', 'init_sq_phi'] + self._param_names
        
        self._bounds = [
            (-self.clip_size, self.clip_size),  # sq_r
            (-8*np.pi, 8*np.pi),        # sq_phi
            (-8*np.pi, 8*np.pi),       # bs_theta
            (-8*np.pi, 8*np.pi)         # bs_phi
        ]

    def run_step(self, state, step_idx: int, step_params: np.ndarray, engine: sf.Engine):
        """Executes a single step of the two-mode squeeze-only time-domain circuit.

        Args:
            state: Unused legacy state parameter.
            step_idx (int): Current step index :math:`t`.
            step_params (np.ndarray): Array of 4 step parameters:
                ``[sq_r, sq_phi, bs_theta, bs_phi]``.
            engine (sf.Engine): Strawberry Fields engine instance configured with Mode 0 input.

        Returns:
            sf.engine.Result: Strawberry Fields execution result containing updated state.
        """
        sq_r, sq_phi, bs_theta, bs_phi = step_params
        
        prog = sf.Program(2)
        with prog.context as q:
            if self.num_single_photon >= 1:
                Fock(1) | q[1]
            
            Sgate(sq_r, sq_phi) | q[1]
            BSgate(bs_theta, bs_phi) | (q[0], q[1])

            if self.loss_transmissivity < 1.0:
                LossChannel(self.loss_transmissivity) | q[0]
                LossChannel(self.loss_transmissivity) | q[1]
            
        return engine.run(prog)


class ThreeModeTimeDomainGeneral(BaseTimeDomainGeneral):
    """Three-mode time-domain circuit consisting of 1 Loop mode (Mode 0) and 2 Ancilla modes (Modes 1, 2).

    Includes independent squeezing and displacement on both ancillae, and three beam splitter interactions.

    Architecture per step:
        1. Prepare Ancillae (Modes 1 & 2) in vacuum or single-photon states.
        2. Apply Squeezing :math:`\\hat{S}_1`, :math:`\\hat{S}_2` to Modes 1 & 2.
        3. Apply Displacement :math:`\\hat{D}_1`, :math:`\\hat{D}_2` to Modes 1 & 2.
        4. Apply Beam Splitter sequence: :math:`\\hat{BS}(0,1)`, :math:`\\hat{BS}(1,2)`, and :math:`\\hat{BS}(0,1)`.
        5. Apply optional loss channels with transmissivity :math:`\\eta`.

    Args:
        steps (int): Number of time-domain recirculation steps :math:`T`.
        time_invariant (bool, optional): If True, circuit parameters are identical across all steps.
            Defaults to False.
        clip_size (float, optional): Maximum absolute bound for squeezing and displacement magnitudes.
            Defaults to 2.0.
        measure_fock_cutoff (int, optional): PNR detector cutoff dimension for Modes 1 & 2. Defaults to 5.
        num_single_photon (int, optional): Number of ancillary modes initialized in :math:`|1\\rangle` (0, 1, or 2).
            Defaults to 0.
        initial_fock_one (bool, optional): If True, initializes Mode 0 in :math:`|1\\rangle`. Defaults to False.
        loss_transmissivity (float, optional): Channel transmissivity :math:`\\eta \\in (0, 1]`. Defaults to 1.0.
        **kwargs: Additional arguments passed to :class:`BaseTimeDomainGeneral`.
    """
    num_modes = 3

    def __init__(self, steps: int, time_invariant: bool = False, clip_size: float = 2.0, 
                 measure_fock_cutoff: int = 5, num_single_photon: int = 0,
                 initial_fock_one: bool = False, loss_transmissivity: float = 1.0, **kwargs):
        super().__init__(
            steps=steps, time_invariant=time_invariant, clip_size=clip_size,
            measure_fock_cutoff=measure_fock_cutoff, num_single_photon=num_single_photon,
            initial_fock_one=initial_fock_one, loss_transmissivity=loss_transmissivity,
            **kwargs
        )
        
        self._param_names = [
            'sq1_r', 'sq1_phi',
            'sq2_r', 'sq2_phi',
            'disp1_r', 'disp1_phi',
            'disp2_r', 'disp2_phi',
            'bs_theta1', 'bs_phi1',
            'bs_theta2', 'bs_phi2',
            'bs_theta3', 'bs_phi3'
        ]

        self._param_names = ['init_sq_r', 'init_sq_phi'] + self._param_names
        
        self._bounds = []
        self._bounds.extend([(-self.clip_size, self.clip_size), (-8*np.pi, 8*np.pi)] * 2)
        self._bounds.extend([(-self.clip_size, self.clip_size), (-8*np.pi, 8*np.pi)] * 2)
        self._bounds.extend([(-8*np.pi, 8*np.pi), (-8*np.pi, 8*np.pi)] * 3)

    def run_step(self, state, step_idx: int, step_params: np.ndarray, engine: sf.Engine):
        """Executes a single step of the three-mode general time-domain circuit.

        Args:
            state: Unused legacy state parameter.
            step_idx (int): Current step index :math:`t`.
            step_params (np.ndarray): Array of 14 step parameters:
                ``[sq1_r, sq1_phi, sq2_r, sq2_phi, disp1_r, disp1_phi, disp2_r, disp2_phi,
                bs_theta1, bs_phi1, bs_theta2, bs_phi2, bs_theta3, bs_phi3]``.
            engine (sf.Engine): Strawberry Fields engine instance configured with Mode 0 input.

        Returns:
            sf.engine.Result: Strawberry Fields execution result containing updated state.
        """
        sq1_r, sq1_phi = step_params[0], step_params[1]
        sq2_r, sq2_phi = step_params[2], step_params[3]
        
        d1_r, d1_phi = step_params[4], step_params[5]
        d2_r, d2_phi = step_params[6], step_params[7]
        
        bs1_th, bs1_ph = step_params[8], step_params[9]
        bs2_th, bs2_ph = step_params[10], step_params[11]
        bs3_th, bs3_ph = step_params[12], step_params[13]
        
        prog = sf.Program(3)
        with prog.context as q:
            if self.num_single_photon >= 1:
                Fock(1) | q[1]
            if self.num_single_photon >= 2:
                Fock(1) | q[2]
            
            Sgate(sq1_r, sq1_phi) | q[1]
            Sgate(sq2_r, sq2_phi) | q[2]
            
            Dgate(d1_r, d1_phi) | q[1]
            Dgate(d2_r, d2_phi) | q[2]
            
            BSgate(bs1_th, bs1_ph) | (q[0], q[1])
            BSgate(bs2_th, bs2_ph) | (q[1], q[2])
            BSgate(bs3_th, bs3_ph) | (q[0], q[1])
            
            if self.loss_transmissivity < 1.0:
                LossChannel(self.loss_transmissivity) | q[0]
                LossChannel(self.loss_transmissivity) | q[1]
                LossChannel(self.loss_transmissivity) | q[2]
            
        return engine.run(prog)


class ThreeModeTimeDomainSqueezeOnly(BaseTimeDomainGeneral):
    """Three-mode time-domain circuit with squeezing-only ancilla preparation (no displacement).

    Architecture per step:
        1. Prepare Ancillae (Modes 1 & 2) in vacuum or single-photon states.
        2. Apply Squeezing :math:`\\hat{S}_1`, :math:`\\hat{S}_2` to Modes 1 & 2.
        3. Apply Beam Splitter sequence: :math:`\\hat{BS}(0,1)`, :math:`\\hat{BS}(1,2)`, and :math:`\\hat{BS}(0,1)`.
        4. Apply optional loss channels with transmissivity :math:`\\eta`.

    Args:
        steps (int): Number of time-domain recirculation steps :math:`T`.
        time_invariant (bool, optional): If True, circuit parameters are identical across all steps.
            Defaults to False.
        clip_size (float, optional): Maximum absolute bound for squeezing magnitudes. Defaults to 2.0.
        measure_fock_cutoff (int, optional): PNR detector cutoff dimension for Modes 1 & 2. Defaults to 5.
        num_single_photon (int, optional): Number of ancillary modes initialized in :math:`|1\\rangle` (0, 1, or 2).
            Defaults to 0.
        initial_fock_one (bool, optional): If True, initializes Mode 0 in :math:`|1\\rangle`. Defaults to False.
        loss_transmissivity (float, optional): Channel transmissivity :math:`\\eta \\in (0, 1]`. Defaults to 1.0.
        **kwargs: Additional arguments passed to :class:`BaseTimeDomainGeneral`.
    """
    num_modes = 3

    def __init__(self, steps: int, time_invariant: bool = False, clip_size: float = 2.0, 
                 measure_fock_cutoff: int = 5, num_single_photon: int = 0,
                 initial_fock_one: bool = False, loss_transmissivity: float = 1.0, **kwargs):
        super().__init__(
            steps=steps, time_invariant=time_invariant, clip_size=clip_size,
            measure_fock_cutoff=measure_fock_cutoff, num_single_photon=num_single_photon,
            initial_fock_one=initial_fock_one, loss_transmissivity=loss_transmissivity,
            **kwargs
        )
        
        self._param_names = [
            'sq1_r', 'sq2_r',
            'sq1_phi', 'sq2_phi',
            'bs_theta1', 'bs_theta2', 'bs_theta3',
            'bs_phi1', 'bs_phi2', 'bs_phi3'
        ]

        self._param_names = ['init_sq_r', 'init_sq_phi'] + self._param_names
        
        self._bounds = []
        self._bounds.extend([(-self.clip_size, self.clip_size)] * 2)
        self._bounds.extend([(-8*np.pi, 8*np.pi)] * 2)
        self._bounds.extend([(-8*np.pi, 8*np.pi)] * 3)
        self._bounds.extend([(-8*np.pi, 8*np.pi)] * 3)

    def run_step(self, state, step_idx: int, step_params: np.ndarray, engine: sf.Engine):
        """Executes a single step of the three-mode squeeze-only time-domain circuit.

        Args:
            state: Unused legacy state parameter.
            step_idx (int): Current step index :math:`t`.
            step_params (np.ndarray): Array of 10 step parameters:
                ``[sq1_r, sq2_r, sq1_phi, sq2_phi, bs_theta1, bs_theta2, bs_theta3, bs_phi1, bs_phi2, bs_phi3]``.
            engine (sf.Engine): Strawberry Fields engine instance configured with Mode 0 input.

        Returns:
            sf.engine.Result: Strawberry Fields execution result containing updated state.
        """
        sq_r = step_params[:2]
        sq_phi = step_params[2:4]
        bs_theta = step_params[4:7]
        bs_phi = step_params[7:]
        
        prog = sf.Program(3)
        with prog.context as q:
            if self.num_single_photon >= 1:
                Fock(1) | q[1]
            if self.num_single_photon >= 2:
                Fock(1) | q[2]
            
            Sgate(sq_r[0], sq_phi[0]) | q[1]
            Sgate(sq_r[1], sq_phi[1]) | q[2]
                
            BSgate(bs_theta[0], bs_phi[0]) | (q[0], q[1])
            BSgate(bs_theta[1], bs_phi[1]) | (q[1], q[2])
            BSgate(bs_theta[2], bs_phi[2]) | (q[0], q[1])

            if self.loss_transmissivity < 1.0:
                LossChannel(self.loss_transmissivity) | q[0]
                LossChannel(self.loss_transmissivity) | q[1]
                LossChannel(self.loss_transmissivity) | q[2]
            
        return engine.run(prog)


class FourModeTimeDomainSqueezeOnly(BaseTimeDomainGeneral):
    """Four-mode time-domain circuit consisting of 1 Loop mode (Mode 0) and 3 Ancilla modes (Modes 1, 2, 3).

    Applies squeezing to ancillae 1, 2, and 3, followed by a 6 beam-splitter nearest-neighbor ladder network.

    Architecture per step:
        1. Prepare Ancillae (Modes 1, 2, 3) in vacuum or single-photon states.
        2. Apply Squeezing :math:`\\hat{S}_1`, :math:`\\hat{S}_2`, :math:`\\hat{S}_3` to Modes 1, 2, 3.
        3. Apply Beam Splitter network across adjacent mode pairs:
           :math:`\\hat{BS}(0,1)`, :math:`\\hat{BS}(2,3)`, :math:`\\hat{BS}(1,2)`,
           :math:`\\hat{BS}(0,1)`, :math:`\\hat{BS}(2,3)`, :math:`\\hat{BS}(1,2)`.
        4. Apply optional loss channels with transmissivity :math:`\\eta`.

    Args:
        steps (int): Number of time-domain recirculation steps :math:`T`.
        time_invariant (bool, optional): If True, circuit parameters are identical across all steps.
            Defaults to False.
        clip_size (float, optional): Maximum absolute bound for squeezing magnitudes. Defaults to 2.0.
        measure_fock_cutoff (int, optional): PNR detector cutoff dimension for Modes 1, 2 & 3. Defaults to 5.
        num_single_photon (int, optional): Number of ancillary modes initialized in :math:`|1\\rangle` (0 to 3).
            Defaults to 0.
        initial_fock_one (bool, optional): If True, initializes Mode 0 in :math:`|1\\rangle`. Defaults to False.
        loss_transmissivity (float, optional): Channel transmissivity :math:`\\eta \\in (0, 1]`. Defaults to 1.0.
        **kwargs: Additional arguments passed to :class:`BaseTimeDomainGeneral`.
    """
    num_modes = 4

    def __init__(self, steps: int, time_invariant: bool = False, clip_size: float = 2.0, 
                 measure_fock_cutoff: int = 5, num_single_photon: int = 0,
                 initial_fock_one: bool = False, loss_transmissivity: float = 1.0, **kwargs):
        super().__init__(
            steps=steps, time_invariant=time_invariant, clip_size=clip_size,
            measure_fock_cutoff=measure_fock_cutoff, num_single_photon=num_single_photon,
            initial_fock_one=initial_fock_one, loss_transmissivity=loss_transmissivity,
            **kwargs
        )
        
        self._param_names = [
            'sq1_r', 'sq2_r', 'sq3_r',
            'sq1_phi', 'sq2_phi', 'sq3_phi',
            'bs_theta1', 'bs_theta2', 'bs_theta3',
            'bs_theta4', 'bs_theta5', 'bs_theta6',
            'bs_phi1', 'bs_phi2', 'bs_phi3',
            'bs_phi4', 'bs_phi5', 'bs_phi6'
        ]

        self._param_names = ['init_sq_r', 'init_sq_phi'] + self._param_names
        
        self._bounds = []
        self._bounds.extend([(-self.clip_size, self.clip_size)] * 3)
        self._bounds.extend([(-8*np.pi, 8*np.pi)] * 3)
        self._bounds.extend([(-8*np.pi, 8*np.pi)] * 6)
        self._bounds.extend([(-8*np.pi, 8*np.pi)] * 6)

    def run_step(self, state, step_idx: int, step_params: np.ndarray, engine: sf.Engine):
        """Executes a single step of the four-mode squeeze-only time-domain circuit.

        Args:
            state: Unused legacy state parameter.
            step_idx (int): Current step index :math:`t`.
            step_params (np.ndarray): Array of 18 step parameters:
                ``[sq1_r, sq2_r, sq3_r, sq1_phi, sq2_phi, sq3_phi,
                bs_theta1, ..., bs_theta6, bs_phi1, ..., bs_phi6]``.
            engine (sf.Engine): Strawberry Fields engine instance configured with Mode 0 input.

        Returns:
            sf.engine.Result: Strawberry Fields execution result containing updated state.
        """
        sq_r = step_params[:3]
        sq_phi = step_params[3:6]
        bs_theta = step_params[6:12]
        bs_phi = step_params[12:]
        
        prog = sf.Program(4)
        with prog.context as q:
            if self.num_single_photon >= 1:
                Fock(1) | q[1]
            if self.num_single_photon >= 2:
                Fock(1) | q[2]
            if self.num_single_photon >= 3:
                Fock(1) | q[3]
            
            Sgate(sq_r[0], sq_phi[0]) | q[1]
            Sgate(sq_r[1], sq_phi[1]) | q[2]
            Sgate(sq_r[2], sq_phi[2]) | q[3]
            
            BSgate(bs_theta[0], bs_phi[0]) | (q[0], q[1])
            BSgate(bs_theta[1], bs_phi[1]) | (q[2], q[3])
            BSgate(bs_theta[2], bs_phi[2]) | (q[1], q[2])

            BSgate(bs_theta[3], bs_phi[3]) | (q[0], q[1])
            BSgate(bs_theta[4], bs_phi[4]) | (q[2], q[3])
            BSgate(bs_theta[5], bs_phi[5]) | (q[1], q[2])

            if self.loss_transmissivity < 1.0:
                LossChannel(self.loss_transmissivity) | q[0]
                LossChannel(self.loss_transmissivity) | q[1]
                LossChannel(self.loss_transmissivity) | q[2]
                LossChannel(self.loss_transmissivity) | q[3]
            
        return engine.run(prog)
