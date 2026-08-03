"""Internal utilities and helpers for heraldo."""

from typing import Any, List, Tuple
import numpy as np


def _normalize_outcomes(outcomes: Any, num_meas_modes: int) -> List[Tuple[int, ...]]:
    """Normalizes user input outcome formats into a list of mode outcome tuples."""
    if outcomes is None:
        return []

    if isinstance(outcomes, np.ndarray):
        outcomes = outcomes.tolist()

    if isinstance(outcomes, (int, np.integer)):
        if num_meas_modes != 1:
            raise ValueError(f"Single integer outcome passed, but circuit has {num_meas_modes} measured modes.")
        return [(int(outcomes),)]

    if isinstance(outcomes, tuple):
        if len(outcomes) == num_meas_modes and all(isinstance(x, (int, np.integer)) for x in outcomes):
            return [tuple(int(x) for x in outcomes)]
        elif num_meas_modes == 1 and all(isinstance(x, (int, np.integer)) for x in outcomes):
            return [(int(x),) for x in outcomes]

    if isinstance(outcomes, list):
        normalized = []
        for item in outcomes:
            if isinstance(item, (int, np.integer)):
                if num_meas_modes == 1:
                    normalized.append((int(item),))
                else:
                    raise ValueError(f"Integer item {item} passed in outcomes list, but circuit has {num_meas_modes} measured modes.")
            elif isinstance(item, (tuple, list)):
                if len(item) == num_meas_modes:
                    normalized.append(tuple(int(x) for x in item))
                else:
                    raise ValueError(f"Outcome item {item} has length {len(item)}, expected {num_meas_modes}.")
            else:
                raise TypeError(f"Invalid outcome item type: {type(item)}")
        return normalized

    raise TypeError(f"Invalid outcomes argument type: {type(outcomes)}")