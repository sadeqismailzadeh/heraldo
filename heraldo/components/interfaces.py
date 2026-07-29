"""Abstract base interfaces for target state generators and time-multiplexed circuits."""

import abc
import numpy as np
import strawberryfields as sf


class TargetGenerator(abc.ABC):
    """Abstract base class for quantum target state generators in the Fock basis.

    Subclasses implement specific non-Gaussian or Gaussian quantum state
    preparations used as target states in photonic circuit optimization routines.
    """

    @abc.abstractmethod
    def get_target_ket(self, cutoff_dim: int) -> np.ndarray:
        """Generates the target state vector (ket) represented in the Fock basis.

        Args:
            cutoff_dim (int): The Fock space truncation cutoff dimension :math:`D`.

        Returns:
            np.ndarray: Complex 1D array of length `cutoff_dim` representing the state vector
            in the Fock basis :math:`\\sum_{n=0}^{D-1} c_n |n\\rangle`.
        """
        pass


class TimeMultiplexedCircuit(abc.ABC):
    """Abstract base class for Time-Domain Multiplexed (TDM) quantum optical circuits.

    Defines the standard operational framework for time-domain multiplexed circuits
    consisting of a persistent loop memory mode (Mode 0) and one or more measured
    ancillary pulse modes (Modes :math:`1 \\dots N-1`). Execution occurs sequentially across
    multiple time steps :math:`t \\in \\{1, \\dots, T\\}`.

    Args:
        steps (int): Total number of time-domain recirculation steps :math:`T`.
        time_invariant (bool, optional): If True, parameters are identical across all steps.
            If False, each step utilizes independent parameters. Defaults to False.
        **kwargs: Additional keyword arguments.
    """

    def __init__(self, steps: int, time_invariant: bool = False, **kwargs):
        self.steps = steps
        self.time_invariant = time_invariant

    @property
    def num_initial_parameters(self) -> int:
        """int: Number of parameters required to initialize the persistent memory state on Mode 0. Defaults to 0."""
        return 0

    @property
    def initial_parameter_bounds(self) -> list[tuple[float, float]]:
        """list[tuple[float, float]]: Parameter bounds ``(min, max)`` for the Mode 0 initialization stage. Defaults to empty list."""
        return []

    def get_initial_state_ket(self, init_params: np.ndarray, cutoff_dim: int) -> np.ndarray:
        """Generates the initial state vector (ket) for Mode 0 prior to time-step execution.

        The default implementation prepares the single-mode vacuum state :math:`|0\\rangle`.

        Args:
            init_params (np.ndarray): 1D array of initialization parameters.
            cutoff_dim (int): The Fock space cutoff dimension.

        Returns:
            np.ndarray: Complex 1D array of shape ``(cutoff_dim,)`` representing the initial state vector.
        """
        ket = np.zeros(cutoff_dim, dtype=np.complex128)
        ket[0] = 1.0
        return ket

    @property
    @abc.abstractmethod
    def per_step_parameter_names(self) -> list[str]:
        """list[str]: Abstract property returning parameter names required for a single time step."""
        pass

    @property
    @abc.abstractmethod
    def per_step_parameter_bounds(self) -> list[tuple[float, float]]:
        """list[tuple[float, float]]: Abstract property returning ``(min, max)`` bounds per time step."""
        pass

    def map_parameters(self, flat_params: np.ndarray) -> np.ndarray:
        """Maps a 1D flat array of step optimization parameters into a 2D matrix of shape ``(steps, n_params)``.

        Handles time-invariant parameter broadcasting (tiling a single parameter set across all steps)
        or time-variant parameter reshaping.

        Args:
            flat_params (np.ndarray): 1D array containing optimization parameters for all steps.

        Returns:
            np.ndarray: 2D array of shape ``(steps, n_params)`` containing mapped step parameters.
        """
        flat_params = np.array(flat_params)

        if self.time_invariant:
            return np.tile(flat_params, (self.steps, 1))
        else:
            return flat_params.reshape(self.steps, -1)

    @abc.abstractmethod
    def run_step(self, state, step_idx: int, step_params: np.ndarray, engine: sf.Engine):
        """Executes the unitary evolution and interaction gates for a single time step :math:`t`.

        Args:
            state: Current quantum state of the system (managed by the optimization runner).
            step_idx (int): Zero-based index of the current time step :math:`t \\in \\{0, \\dots, T-1\\}`.
            step_params (np.ndarray): 1D array of parameter values for step `step_idx`.
            engine (sf.Engine): Strawberry Fields engine instance with initial mode context prepared.

        Returns:
            sf.engine.Result: Strawberry Fields execution result containing the updated quantum state.
        """
        pass

    @abc.abstractmethod
    def get_measurement_specs(self) -> list[tuple[int, int]]:
        """Returns photon-number-resolving (PNR) measurement specifications for ancillary modes.

        Returns:
            list[tuple[int, int]]: List of tuples ``(mode_index, max_fock_cutoff)`` specifying
            the index of each measured ancillary mode and its associated detector cutoff dimension.
        """
        pass
