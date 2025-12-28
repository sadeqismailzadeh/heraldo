import abc
import numpy as np
import strawberryfields as sf

class OptimizableCircuit(abc.ABC):
    """
    Abstract base class for circuits compatible with scipy.optimize.
    """
    
    @property
    @abc.abstractmethod
    def parameter_names(self) -> list[str]:
        """List of parameter names for logging/debugging."""
        pass
        
    @property
    @abc.abstractmethod
    def parameter_bounds(self) -> list[tuple[float, float]]:
        """List of (min, max) bounds for each parameter."""
        pass

    @property
    def num_parameters(self) -> int:
        return len(self.parameter_names)

    @abc.abstractmethod
    def run_circuit(self, params: np.ndarray, engine: sf.Engine) -> sf.backends.BaseState:
        """
        Constructs and runs the circuit on the provided engine.
        Returns the resulting state object.
        """
        pass
    
    @abc.abstractmethod
    def extract_output(self, state: sf.backends.BaseState, post_select_dict: dict) -> tuple[np.ndarray, float]:
        """
        Performs post-selection/projection logic on the output state.
        
        Args:
            state: The Strawberry Fields state object (pure).
            post_select_dict: Dictionary {mode_index: fock_value}
            
        Returns:
            (normalized_ket, probability)
        """
        pass

    @abc.abstractmethod
    def extract_all_outputs(self, state: sf.backends.BaseState, measure_modes: list[int]) -> list[tuple[np.ndarray, float, tuple]]:
        """
        Extracts states for all possible measurement outcomes on specified modes.

        Args:
            state: The Strawberry Fields state object (pure).
            measure_modes: List of mode indices to measure.

        Returns:
            List of tuples: (normalized_ket, probability, outcome_tuple)
        """
        pass
