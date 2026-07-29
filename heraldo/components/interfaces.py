import abc
import numpy as np
import strawberryfields as sf



class TargetGenerator(abc.ABC):
    """Abstract base class for target state generators in the Fock basis.

    Subclasses implement specific non-Gaussian or Gaussian quantum state
    preparations used as targets in circuit optimization.
    """

    @abc.abstractmethod
    def get_target_ket(self, cutoff_dim: int) -> np.ndarray:
        """Generates the target state vector (ket) in the Fock basis.

        Args:
            cutoff_dim (int): The Fock space truncation cutoff dimension.

        Returns:
            np.ndarray: Complex 1D array representing the state vector in Fock space.
        """
        pass


# --- Target Implementations ---

class TimeMultiplexedCircuit(abc.ABC):
    """
    Abstract interface for Time-Domain Multiplexed Quantum Circuits.
    
    These circuits involve sequential processing over 'steps', where parameters
    can either be unique per step (time-variant) or shared (time-invariant).
    """
    
    def __init__(self, steps: int, time_invariant: bool = False, **kwargs):
        self.steps = steps
        self.time_invariant = time_invariant


    @property
    def num_initial_parameters(self) -> int:
        """Number of parameters used for initialization. Default 0."""
        return 0

    @property
    def initial_parameter_bounds(self) -> list[tuple[float, float]]:
        """Bounds for initialization parameters. Default empty."""
        return []

    def get_initial_state_ket(self, init_params: np.ndarray, cutoff_dim: int) -> np.ndarray:
        """
        Generates the initial ket for Mode 0. 
        Default implementation returns Vacuum |0>.
        """
        # Default: Vacuum
        ket = np.zeros(cutoff_dim, dtype=np.complex128)
        ket[0] = 1.0
        return ket

    @property
    @abc.abstractmethod
    def per_step_parameter_names(self) -> list[str]:
        """Names of parameters required for a single time step."""
        pass

    @property
    @abc.abstractmethod
    def per_step_parameter_bounds(self) -> list[tuple[float, float]]:
        """Bounds for parameters in a single time step."""
        pass

    def map_parameters(self, flat_params: np.ndarray) -> np.ndarray:
        """
        Maps flat optimization parameters to a (steps, n_params) matrix.
        
        Uses vectorized operations to avoid loops.
        """
        flat_params = np.array(flat_params)
        
        if self.time_invariant:
            # Broadcast the single set of parameters across all steps
            # Input shape: (n_params,)
            # Output shape: (steps, n_params)
            return np.tile(flat_params, (self.steps, 1))
        else:
            # Reshape the flat array into time steps
            # Input shape: (steps * n_params,)
            # Output shape: (steps, n_params)
            return flat_params.reshape(self.steps, -1)

    @abc.abstractmethod
    def run_step(self, state, step_idx: int, step_params: np.ndarray, engine: sf.Engine):
        """
        Applies the unitary evolution for a single time step.
        
        Args:
            state: The current state (managed by the Runner, usually).
            step_idx: The current time step index.
            step_params: The parameters for this specific step (1D array).
            engine: The StrawberryFields engine.
        """
        pass

    @abc.abstractmethod
    def get_measurement_specs(self) -> list[tuple[int, int]]:
        """
        Returns the measurement specifications.
        
        Returns:
            list[tuple]: List of (measurement_mode_index, max_fock_cutoff)
        """
        pass
