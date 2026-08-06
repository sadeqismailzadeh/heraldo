"""Abstract base interfaces for target state generators and photonic circuit models."""

import abc
import numpy as np
import strawberryfields as sf


class ObjectiveFunction(abc.ABC):
    """Abstract base class for circuit optimization objective functions.

    Subclasses implement non-linear objective evaluation metrics over candidate
    measurement outcomes, probabilities, and state fidelities.
    """

    def __call__(self, probs: np.ndarray, fidelities: np.ndarray) -> float | np.ndarray:
        """Evaluates the objective loss metric given outcome probabilities and fidelities.

        Args:
            probs (np.ndarray): Array of probabilities :math:`p_k` for surviving output patterns.
            fidelities (np.ndarray): Array of state fidelities :math:`F_k` for surviving output patterns (1D or 2D).

        Returns:
            float or np.ndarray: Calculated objective value(s).
        """
        probs_ext = probs[:, None] if fidelities.ndim == 2 else probs
        res = self._compute(probs_ext, fidelities)
        return float(res) if fidelities.ndim == 1 else res

    @abc.abstractmethod
    def _compute(self, probs: np.ndarray, fidelities: np.ndarray) -> float | np.ndarray:
        """Internal computation method implemented by subclasses.

        Args:
            probs (np.ndarray): Array of probabilities (reshaped to 2D column vector if fidelities is 2D).
            fidelities (np.ndarray): Array of state fidelities (1D or 2D).

        Returns:
            float or np.ndarray: Calculated objective score(s).
        """
        pass


class StaticCircuit(abc.ABC):
    """Abstract base class for static Continuous-Variable (CV) spatial photonic circuits.

    Defines the standard operational framework for static spatial circuits consisting of
    an unmeasured output mode (Mode 0) and one or more measured ancillary modes (Modes 1..N-1).
    """

    @property
    @abc.abstractmethod
    def parameter_names(self) -> list[str]:
        """list[str]: Names of all optimization parameters for the static circuit."""
        pass

    @property
    @abc.abstractmethod
    def parameter_bounds(self) -> list[tuple[float, float]]:
        """list[tuple[float, float]]: Parameter bounds (min, max) for all static circuit parameters."""
        pass

    @abc.abstractmethod
    def run_circuit(self, params: np.ndarray, engine: sf.Engine) -> sf.engine.Result:
        """Executes the static circuit unitary evolution and interactions.

        Args:
            params (np.ndarray): 1D array of optimization parameters.
            engine (sf.Engine): Strawberry Fields engine instance configured for the simulation.

        Returns:
            sf.engine.Result: Strawberry Fields execution result containing updated state.
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

