"""Concrete implementations of StaticCircuit for static spatial photonic optimization.

Provides multi-mode static spatial optical circuit architectures containing
an unmeasured output mode (Mode 0) and one or more ancillary modes (Modes 1..N-1).
"""
import numpy as np
import strawberryfields as sf
from strawberryfields.ops import Sgate, Dgate, BSgate, Fock, LossChannel

from heraldo.components.interfaces import StaticCircuit


class BaseStaticCircuit(StaticCircuit):
    """Base class for static spatial circuits.

    Encapsulates common configuration, measurement specifications for ancillary modes,
    and parameter property accessors for static GBS-like spatial devices.

    Args:
        clip_size (float, optional): Maximum absolute bound for squeezing and displacement
            magnitude parameters. Defaults to 2.0.
        measure_fock_cutoff (int, optional): Maximum Fock cutoff for photon-number-resolving (PNR)
            detectors on ancillary modes. Defaults to 5.
        num_single_photon (int, optional): Number of ancillary modes initialized with single-photon
            states :math:`|1\\rangle` instead of vacuum :math:`|0\\rangle`. Defaults to 0.
        loss_transmissivity (float, optional): Channel transmissivity parameter :math:`\\eta \\in (0, 1]`
            simulating loss on all modes. Defaults to 1.0 (lossless).
        **kwargs: Additional keyword arguments.
    """
    num_modes: int = 2

    def __init__(self, clip_size: float = 2.0, measure_fock_cutoff: int = 5,
                 num_single_photon: int = 0, loss_transmissivity: float = 1.0, **kwargs):
        super().__init__()
        self.clip_size = clip_size
        self.measure_fock_cutoff = measure_fock_cutoff
        self.num_single_photon = num_single_photon
        self.loss_transmissivity = loss_transmissivity

        self._param_names: list[str] = []
        self._bounds: list[tuple[float, float]] = []

    @property
    def parameter_names(self) -> list[str]:
        """list[str]: Names of static circuit optimization parameters."""
        return self._param_names

    @property
    def parameter_bounds(self) -> list[tuple[float, float]]:
        """list[tuple[float, float]]: Parameter bounds (min, max) for the static circuit."""
        return self._bounds

    def get_measurement_specs(self) -> list[tuple[int, int]]:
        """Generates measurement specifications for ancillary modes.

        Returns:
            list[tuple[int, int]]: List of tuples ``(mode_index, max_fock_cutoff)``
            for each measured ancillary mode (Modes 1 to N-1).
        """
        return [(mode, self.measure_fock_cutoff) for mode in range(1, self.num_modes)]


class TwoModeStaticGeneral(BaseStaticCircuit):
    """Two-mode static spatial circuit consisting of Output mode (Mode 0) and Ancilla mode (Mode 1).

    Applies squeezing and displacement to both modes, followed by a beam splitter interaction.

    Architecture:
        1. Prepare Ancilla (Mode 1) in vacuum :math:`|0\\rangle` or single-photon :math:`|1\\rangle`.
        2. Apply Squeezing gates :math:`\\hat{S}_0, \\hat{S}_1` to Modes 0 and 1.
        3. Apply Displacement gates :math:`\\hat{D}_0, \\hat{D}_1` to Modes 0 and 1.
        4. Apply Beam Splitter :math:`\\hat{BS}(\\theta, \\phi)` between Mode 0 and Mode 1.
        5. Apply optional loss channels with transmissivity :math:`\\eta`.

    Args:
        clip_size (float, optional): Maximum absolute bound for squeezing/displacement magnitudes. Defaults to 2.0.
        measure_fock_cutoff (int, optional): PNR detector cutoff dimension for Mode 1. Defaults to 5.
        num_single_photon (int, optional): Number of ancillary modes initialized in :math:`|1\\rangle`. Defaults to 0.
        loss_transmissivity (float, optional): Channel transmissivity :math:`\\eta \\in (0, 1]`. Defaults to 1.0.
    """
    num_modes = 2

    def __init__(self, clip_size: float = 2.0, measure_fock_cutoff: int = 5,
                 num_single_photon: int = 0, loss_transmissivity: float = 1.0, **kwargs):
        super().__init__(
            clip_size=clip_size, measure_fock_cutoff=measure_fock_cutoff,
            num_single_photon=num_single_photon, loss_transmissivity=loss_transmissivity,
            **kwargs
        )

        self._param_names = [
            'sq0_r', 'sq0_phi',
            'sq1_r', 'sq1_phi',
            'disp0_r', 'disp0_phi',
            'disp1_r', 'disp1_phi',
            'bs_theta', 'bs_phi'
        ]

        self._bounds = [
            (-self.clip_size, self.clip_size),  # sq0_r
            (-8*np.pi, 8*np.pi),                # sq0_phi
            (-self.clip_size, self.clip_size),  # sq1_r
            (-8*np.pi, 8*np.pi),                # sq1_phi
            (-self.clip_size, self.clip_size),  # disp0_r
            (-8*np.pi, 8*np.pi),                # disp0_phi
            (-self.clip_size, self.clip_size),  # disp1_r
            (-8*np.pi, 8*np.pi),                # disp1_phi
            (-8*np.pi, 8*np.pi),                # bs_theta
            (-8*np.pi, 8*np.pi)                 # bs_phi
        ]

    def run_circuit(self, params: np.ndarray, engine: sf.Engine) -> sf.engine.Result:
        """Executes the two-mode general static spatial circuit.

        Args:
            params (np.ndarray): Array of 10 static parameters:
                ``[sq0_r, sq0_phi, sq1_r, sq1_phi, disp0_r, disp0_phi, disp1_r, disp1_phi, bs_theta, bs_phi]``.
            engine (sf.Engine): Strawberry Fields engine instance.

        Returns:
            sf.engine.Result: Strawberry Fields execution result.
        """
        sq0_r, sq0_phi, sq1_r, sq1_phi, d0_r, d0_phi, d1_r, d1_phi, bs_th, bs_ph = params

        prog = sf.Program(2)
        with prog.context as q:
            if self.num_single_photon >= 1:
                Fock(1) | q[1]

            Sgate(sq0_r, sq0_phi) | q[0]
            Sgate(sq1_r, sq1_phi) | q[1]

            Dgate(d0_r, d0_phi) | q[0]
            Dgate(d1_r, d1_phi) | q[1]

            BSgate(bs_th, bs_ph) | (q[0], q[1])

            if self.loss_transmissivity < 1.0:
                LossChannel(self.loss_transmissivity) | q[0]
                LossChannel(self.loss_transmissivity) | q[1]

        return engine.run(prog)


class TwoModeStaticSqueezeOnly(BaseStaticCircuit):
    """Two-mode static spatial circuit with squeezing-only preparation (no displacement).

    Architecture:
        1. Prepare Ancilla (Mode 1) in vacuum :math:`|0\\rangle` or single-photon :math:`|1\\rangle`.
        2. Apply Squeezing gates :math:`\\hat{S}_0, \\hat{S}_1` to Modes 0 and 1.
        3. Apply Beam Splitter :math:`\\hat{BS}(\\theta, \\phi)` between Mode 0 and Mode 1.
        4. Apply optional loss channels with transmissivity :math:`\\eta`.

    Args:
        clip_size (float, optional): Maximum absolute bound for squeezing magnitudes. Defaults to 2.0.
        measure_fock_cutoff (int, optional): PNR detector cutoff dimension for Mode 1. Defaults to 5.
        num_single_photon (int, optional): Number of ancillary modes initialized in :math:`|1\\rangle`. Defaults to 0.
        loss_transmissivity (float, optional): Channel transmissivity :math:`\\eta \\in (0, 1]`. Defaults to 1.0.
    """
    num_modes = 2

    def __init__(self, clip_size: float = 2.0, measure_fock_cutoff: int = 5,
                 num_single_photon: int = 0, loss_transmissivity: float = 1.0, **kwargs):
        super().__init__(
            clip_size=clip_size, measure_fock_cutoff=measure_fock_cutoff,
            num_single_photon=num_single_photon, loss_transmissivity=loss_transmissivity,
            **kwargs
        )

        self._param_names = [
            'sq0_r', 'sq0_phi',
            'sq1_r', 'sq1_phi',
            'bs_theta', 'bs_phi'
        ]

        self._bounds = [
            (-self.clip_size, self.clip_size),  # sq0_r
            (-8*np.pi, 8*np.pi),                # sq0_phi
            (-self.clip_size, self.clip_size),  # sq1_r
            (-8*np.pi, 8*np.pi),                # sq1_phi
            (-8*np.pi, 8*np.pi),                # bs_theta
            (-8*np.pi, 8*np.pi)                 # bs_phi
        ]

    def run_circuit(self, params: np.ndarray, engine: sf.Engine) -> sf.engine.Result:
        """Executes the two-mode squeeze-only static spatial circuit.

        Args:
            params (np.ndarray): Array of 6 static parameters:
                ``[sq0_r, sq0_phi, sq1_r, sq1_phi, bs_theta, bs_phi]``.
            engine (sf.Engine): Strawberry Fields engine instance.

        Returns:
            sf.engine.Result: Strawberry Fields execution result.
        """
        sq0_r, sq0_phi, sq1_r, sq1_phi, bs_th, bs_ph = params

        prog = sf.Program(2)
        with prog.context as q:
            if self.num_single_photon >= 1:
                Fock(1) | q[1]

            Sgate(sq0_r, sq0_phi) | q[0]
            Sgate(sq1_r, sq1_phi) | q[1]

            BSgate(bs_th, bs_ph) | (q[0], q[1])

            if self.loss_transmissivity < 1.0:
                LossChannel(self.loss_transmissivity) | q[0]
                LossChannel(self.loss_transmissivity) | q[1]

        return engine.run(prog)


class ThreeModeStaticGeneral(BaseStaticCircuit):
    """Three-mode static spatial circuit consisting of Output mode (Mode 0) and 2 Ancilla modes (Modes 1, 2).

    Applies squeezing and displacement to all modes, and a 3 beam-splitter network.

    Architecture:
        1. Prepare Ancillae (Modes 1, 2) in vacuum or single-photon states.
        2. Apply Squeezing :math:`\\hat{S}_0, \\hat{S}_1, \\hat{S}_2` to all modes.
        3. Apply Displacement :math:`\\hat{D}_0, \\hat{D}_1, \\hat{D}_2` to all modes.
        4. Apply Beam Splitter sequence: :math:`\\hat{BS}(0,1)`, :math:`\\hat{BS}(1,2)`, and :math:`\\hat{BS}(0,1)`.
        5. Apply optional loss channels with transmissivity :math:`\\eta`.

    Args:
        clip_size (float, optional): Maximum absolute bound for squeezing/displacement magnitudes. Defaults to 2.0.
        measure_fock_cutoff (int, optional): PNR detector cutoff dimension for Modes 1 & 2. Defaults to 5.
        num_single_photon (int, optional): Number of ancillary modes initialized in :math:`|1\\rangle` (0, 1, or 2). Defaults to 0.
        loss_transmissivity (float, optional): Channel transmissivity :math:`\\eta \\in (0, 1]`. Defaults to 1.0.
    """
    num_modes = 3

    def __init__(self, clip_size: float = 2.0, measure_fock_cutoff: int = 5,
                 num_single_photon: int = 0, loss_transmissivity: float = 1.0, **kwargs):
        super().__init__(
            clip_size=clip_size, measure_fock_cutoff=measure_fock_cutoff,
            num_single_photon=num_single_photon, loss_transmissivity=loss_transmissivity,
            **kwargs
        )

        self._param_names = [
            'sq0_r', 'sq0_phi',
            'sq1_r', 'sq1_phi',
            'sq2_r', 'sq2_phi',
            'disp0_r', 'disp0_phi',
            'disp1_r', 'disp1_phi',
            'disp2_r', 'disp2_phi',
            'bs_theta1', 'bs_phi1',
            'bs_theta2', 'bs_phi2',
            'bs_theta3', 'bs_phi3'
        ]

        self._bounds = []
        self._bounds.extend([(-self.clip_size, self.clip_size), (-8*np.pi, 8*np.pi)] * 3)
        self._bounds.extend([(-self.clip_size, self.clip_size), (-8*np.pi, 8*np.pi)] * 3)
        self._bounds.extend([(-8*np.pi, 8*np.pi), (-8*np.pi, 8*np.pi)] * 3)

    def run_circuit(self, params: np.ndarray, engine: sf.Engine) -> sf.engine.Result:
        """Executes the three-mode general static spatial circuit.

        Args:
            params (np.ndarray): Array of 18 static parameters.
            engine (sf.Engine): Strawberry Fields engine instance.

        Returns:
            sf.engine.Result: Strawberry Fields execution result.
        """
        sq0_r, sq0_phi, sq1_r, sq1_phi, sq2_r, sq2_phi = params[0:6]
        d0_r, d0_phi, d1_r, d1_phi, d2_r, d2_phi = params[6:12]
        bs1_th, bs1_ph, bs2_th, bs2_ph, bs3_th, bs3_ph = params[12:18]

        prog = sf.Program(3)
        with prog.context as q:
            if self.num_single_photon >= 1:
                Fock(1) | q[1]
            if self.num_single_photon >= 2:
                Fock(1) | q[2]

            Sgate(sq0_r, sq0_phi) | q[0]
            Sgate(sq1_r, sq1_phi) | q[1]
            Sgate(sq2_r, sq2_phi) | q[2]

            Dgate(d0_r, d0_phi) | q[0]
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


class ThreeModeStaticSqueezeOnly(BaseStaticCircuit):
    """Three-mode static spatial circuit with squeezing-only preparation (no displacement).

    Architecture:
        1. Prepare Ancillae (Modes 1, 2) in vacuum or single-photon states.
        2. Apply Squeezing :math:`\\hat{S}_0, \\hat{S}_1, \\hat{S}_2` to all modes.
        3. Apply Beam Splitter sequence: :math:`\\hat{BS}(0,1)`, :math:`\\hat{BS}(1,2)`, and :math:`\\hat{BS}(0,1)`.
        4. Apply optional loss channels with transmissivity :math:`\\eta`.

    Args:
        clip_size (float, optional): Maximum absolute bound for squeezing magnitudes. Defaults to 2.0.
        measure_fock_cutoff (int, optional): PNR detector cutoff dimension for Modes 1 & 2. Defaults to 5.
        num_single_photon (int, optional): Number of ancillary modes initialized in :math:`|1\\rangle` (0, 1, or 2). Defaults to 0.
        loss_transmissivity (float, optional): Channel transmissivity :math:`\\eta \\in (0, 1]`. Defaults to 1.0.
    """
    num_modes = 3

    def __init__(self, clip_size: float = 2.0, measure_fock_cutoff: int = 5,
                 num_single_photon: int = 0, loss_transmissivity: float = 1.0, **kwargs):
        super().__init__(
            clip_size=clip_size, measure_fock_cutoff=measure_fock_cutoff,
            num_single_photon=num_single_photon, loss_transmissivity=loss_transmissivity,
            **kwargs
        )

        self._param_names = [
            'sq0_r', 'sq1_r', 'sq2_r',
            'sq0_phi', 'sq1_phi', 'sq2_phi',
            'bs_theta1', 'bs_theta2', 'bs_theta3',
            'bs_phi1', 'bs_phi2', 'bs_phi3'
        ]

        self._bounds = []
        self._bounds.extend([(-self.clip_size, self.clip_size)] * 3)
        self._bounds.extend([(-8*np.pi, 8*np.pi)] * 3)
        self._bounds.extend([(-8*np.pi, 8*np.pi)] * 3)
        self._bounds.extend([(-8*np.pi, 8*np.pi)] * 3)

    def run_circuit(self, params: np.ndarray, engine: sf.Engine) -> sf.engine.Result:
        """Executes the three-mode squeeze-only static spatial circuit.

        Args:
            params (np.ndarray): Array of 12 static parameters.
            engine (sf.Engine): Strawberry Fields engine instance.

        Returns:
            sf.engine.Result: Strawberry Fields execution result.
        """
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

            Sgate(sq_r[0], sq_phi[0]) | q[0]
            Sgate(sq_r[1], sq_phi[1]) | q[1]
            Sgate(sq_r[2], sq_phi[2]) | q[2]

            BSgate(bs_theta[0], bs_phi[0]) | (q[0], q[1])
            BSgate(bs_theta[1], bs_phi[1]) | (q[1], q[2])
            BSgate(bs_theta[2], bs_phi[2]) | (q[0], q[1])

            if self.loss_transmissivity < 1.0:
                LossChannel(self.loss_transmissivity) | q[0]
                LossChannel(self.loss_transmissivity) | q[1]
                LossChannel(self.loss_transmissivity) | q[2]

        return engine.run(prog)


class FourModeStaticSqueezeOnly(BaseStaticCircuit):
    """Four-mode static spatial circuit consisting of Output mode (Mode 0) and 3 Ancilla modes (Modes 1, 2, 3).

    Applies squeezing to all modes, followed by a 6 beam-splitter nearest-neighbor ladder network.

    Architecture:
        1. Prepare Ancillae (Modes 1, 2, 3) in vacuum or single-photon states.
        2. Apply Squeezing :math:`\\hat{S}_0, \\hat{S}_1, \\hat{S}_2, \\hat{S}_3` to all modes.
        3. Apply Beam Splitter network across adjacent mode pairs:
           :math:`\\hat{BS}(0,1)`, :math:`\\hat{BS}(2,3)`, :math:`\\hat{BS}(1,2)`,
           :math:`\\hat{BS}(0,1)`, :math:`\\hat{BS}(2,3)`, :math:`\\hat{BS}(1,2)`.
        4. Apply optional loss channels with transmissivity :math:`\\eta`.

    Args:
        clip_size (float, optional): Maximum absolute bound for squeezing magnitudes. Defaults to 2.0.
        measure_fock_cutoff (int, optional): PNR detector cutoff dimension for Modes 1, 2 & 3. Defaults to 5.
        num_single_photon (int, optional): Number of ancillary modes initialized in :math:`|1\\rangle` (0 to 3). Defaults to 0.
        loss_transmissivity (float, optional): Channel transmissivity :math:`\\eta \\in (0, 1]`. Defaults to 1.0.
    """
    num_modes = 4

    def __init__(self, clip_size: float = 2.0, measure_fock_cutoff: int = 5,
                 num_single_photon: int = 0, loss_transmissivity: float = 1.0, **kwargs):
        super().__init__(
            clip_size=clip_size, measure_fock_cutoff=measure_fock_cutoff,
            num_single_photon=num_single_photon, loss_transmissivity=loss_transmissivity,
            **kwargs
        )

        self._param_names = [
            'sq0_r', 'sq1_r', 'sq2_r', 'sq3_r',
            'sq0_phi', 'sq1_phi', 'sq2_phi', 'sq3_phi',
            'bs_theta1', 'bs_theta2', 'bs_theta3',
            'bs_theta4', 'bs_theta5', 'bs_theta6',
            'bs_phi1', 'bs_phi2', 'bs_phi3',
            'bs_phi4', 'bs_phi5', 'bs_phi6'
        ]

        self._bounds = []
        self._bounds.extend([(-self.clip_size, self.clip_size)] * 4)
        self._bounds.extend([(-8*np.pi, 8*np.pi)] * 4)
        self._bounds.extend([(-8*np.pi, 8*np.pi)] * 6)
        self._bounds.extend([(-8*np.pi, 8*np.pi)] * 6)

    def run_circuit(self, params: np.ndarray, engine: sf.Engine) -> sf.engine.Result:
        """Executes the four-mode squeeze-only static spatial circuit.

        Args:
            params (np.ndarray): Array of 20 static parameters.
            engine (sf.Engine): Strawberry Fields engine instance.

        Returns:
            sf.engine.Result: Strawberry Fields execution result.
        """
        sq_r = params[:4]
        sq_phi = params[4:8]
        bs_theta = params[8:14]
        bs_phi = params[14:]

        prog = sf.Program(4)
        with prog.context as q:
            if self.num_single_photon >= 1:
                Fock(1) | q[1]
            if self.num_single_photon >= 2:
                Fock(1) | q[2]
            if self.num_single_photon >= 3:
                Fock(1) | q[3]

            Sgate(sq_r[0], sq_phi[0]) | q[0]
            Sgate(sq_r[1], sq_phi[1]) | q[1]
            Sgate(sq_r[2], sq_phi[2]) | q[2]
            Sgate(sq_r[3], sq_phi[3]) | q[3]

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
