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

    def extract_all_outputs(self, state: sf.backends.BaseState, measure_modes: list[int]) -> list[tuple[np.ndarray, float, tuple]]:
        """
        Extracts states for all possible measurement outcomes on specified modes.

        Args:
            state: The Strawberry Fields state object (pure).
            measure_modes: List of mode indices to measure.

        Returns:
            List of tuples: (normalized_ket, probability, outcome_tuple)
        """
        full_ket = state.ket()
        num_modes = len(full_ket.shape)
        cutoff = full_ket.shape[0]

        # 1. Determine Modes
        all_modes = set(range(num_modes))
        meas_set = set(measure_modes)
        kept_set = all_modes - meas_set

        if len(kept_set) != 1:
            raise ValueError(f"OptimizableCircuit expects exactly one unmeasured mode. Found {len(kept_set)}.")

        kept_mode = list(kept_set)[0]
        sorted_measure_modes = sorted(list(meas_set))

        # 2. Transpose: Put measured modes first (in order), kept mode last
        # shape becomes (cutoff_m1, cutoff_m2, ..., cutoff_kept)
        perm = sorted_measure_modes + [kept_mode]
        transposed_ket = np.transpose(full_ket, axes=perm)

        # 3. Reshape: (N_outcomes, cutoff_kept)
        # Flattens all measurement axes into one batch dimension
        reshaped_ket = transposed_ket.reshape(-1, cutoff)

        # 4. Probabilities: Sum of squared amplitudes
        probs = np.sum(np.abs(reshaped_ket)**2, axis=1)

        # 5. Masking
        valid_indices = np.where(probs > 1e-9)[0]

        if len(valid_indices) == 0:
            return []

        # 6. Normalization
        valid_kets = reshaped_ket[valid_indices]
        valid_probs = probs[valid_indices]
        # Normalize: ket / sqrt(prob) (broadcasting over batch dim)
        normalized_kets = valid_kets / np.sqrt(valid_probs)[:, None]

        # 7. Tuple Recovery
        # The indices correspond to the dimensions of measured modes in order
        dims = [cutoff] * len(sorted_measure_modes)
        # returns tuple of arrays (arr_dim1, arr_dim2, ...)
        unraveled = np.unravel_index(valid_indices, dims)
        # Zip them to get list of tuples
        outcome_tuples = list(zip(*unraveled))

        # 8. Assemble results
        results = []
        for i in range(len(valid_indices)):
            results.append((
                normalized_kets[i],
                float(valid_probs[i]),
                outcome_tuples[i]
            ))

        return results
