import numpy as np
from heraldo.components.interfaces import TargetGenerator

class CustomFockTarget(TargetGenerator):
    """Custom target generator: a superposition of Fock states with given coefficients."""
    def __init__(self, coeffs=None):
        if coeffs is None:
            coeffs = [1.0, 0.0, 1.0]  # default: |0> + |2>
        self.coeffs = coeffs

    def get_target_ket(self, cutoff_dim: int) -> np.ndarray:
        ket = np.zeros(cutoff_dim, dtype=np.complex128)
        for n, c in enumerate(self.coeffs):
            if n >= cutoff_dim:
                break
            ket[n] = c
        norm = np.linalg.norm(ket)
        if norm > 0:
            ket /= norm
        return ket