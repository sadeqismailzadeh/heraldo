"""Optimization objective functions for circuit evaluation.

Defines non-linear objective function classes used during parameter optimization,
distinguished from physical optical photon loss.
"""

import numpy as np

from heraldo.components.interfaces import ObjectiveFunction


class BeamSearchLoss(ObjectiveFunction):
    r"""Evaluates the multi-outcome beam search pattern discovery loss function.

    Implements the non-linear multi-outcome loss metric (Eqs. 3–5 in paper) that filters
    low-fidelity outcomes and sharpens optimization gradients around high-quality candidate patterns:

    .. math::
        L_{\text{beam}} = \log\left( \sum_k p_k (\tilde{F}_k^2 \Lambda_k)^4 + \delta \right) + \lambda \sum_k p_k (\tilde{F}_k^2 \Lambda_k)^4

    Args:
        epsilon (float, optional): Clipping parameter :math:`\epsilon` defining fidelity bounds. Defaults to 0.02.
        delta (float, optional): Regularization parameter :math:`\delta` to prevent logarithmic divergence. Defaults to 1e-72.
        lam (float, optional): Linear penalty multiplier :math:`\lambda`. Defaults to 1e4.
    """

    def __init__(self, epsilon: float = 2e-2, delta: float = 1e-72, lam: float = 1e4):
        self.epsilon = epsilon
        self.delta = delta
        self.lam = lam

    def __call__(self, probs: np.ndarray, fidelities: np.ndarray) -> float | np.ndarray:
        """Evaluates the beam search loss score.

        Args:
            probs (np.ndarray): Array of probabilities :math:`p_k` for surviving output patterns.
            fidelities (np.ndarray): Array of state fidelities :math:`F_k` for surviving output patterns (1D or 2D).

        Returns:
            float or np.ndarray: Calculated beam search objective value(s).
        """
        probs_ext = probs[:, None] if fidelities.ndim == 2 else probs
        infidelities = np.maximum(1.0 - fidelities, self.epsilon)
        capped_fidelities = np.minimum(fidelities, 1.0 - self.epsilon)

        log_vals = np.log10(infidelities) / np.log10(self.epsilon)
        score = np.sum(probs_ext * (capped_fidelities**2 * log_vals)**4, axis=0)
        res = np.log(score + self.delta) + self.lam * score
        return float(res) if fidelities.ndim == 1 else res


class FixedPatternCappedLoss(ObjectiveFunction):
    r"""Evaluates the fixed-pattern optimization loss function under a capped fidelity regime.

    Implements the capped fidelity objective (Eq. 2 in paper):

    .. math::
        L_{\text{fixed}} = \sum_k \left( \alpha p_k + \min(F_k, F_{\text{cap}}) \right)

    Args:
        f_cap (float, optional): Maximum fidelity cap :math:`F_{\text{cap}}`. Defaults to 0.95.
        alpha (float, optional): Weighting coefficient :math:`\alpha`. Defaults to `len(probs)` if None.
    """

    def __init__(self, f_cap: float = 0.95, alpha: float | None = None):
        self.f_cap = f_cap
        self.alpha = alpha

    def __call__(self, probs: np.ndarray, fidelities: np.ndarray) -> float | np.ndarray:
        """Evaluates the fixed-pattern capped loss score.

        Args:
            probs (np.ndarray): Array of probabilities :math:`p_k` for fixed outcome patterns.
            fidelities (np.ndarray): Array of state fidelities :math:`F_k` for fixed outcome patterns (1D or 2D).

        Returns:
            float or np.ndarray: Calculated objective score(s).
        """
        alpha = float(len(probs)) if self.alpha is None else float(self.alpha)
        probs_ext = probs[:, None] if fidelities.ndim == 2 else probs
        capped_fidelities = np.minimum(fidelities, self.f_cap)
        res = np.sum(alpha * probs_ext + capped_fidelities, axis=0)
        return float(res) if fidelities.ndim == 1 else res


class FixedPatternFreeLoss(ObjectiveFunction):
    r"""Evaluates the fixed-pattern optimization loss function under an uncapped (free) fidelity regime.

    Implements the uncapped objective function (Eq. 2 in paper):

    .. math::
        L_{\text{fixed}} = \sum_k \left( \alpha p_k + F_k \right)

    Args:
        alpha (float, optional): Weighting coefficient :math:`\alpha`. Defaults to `0.1 * len(probs)` if None.
    """

    def __init__(self, alpha: float | None = None):
        self.alpha = alpha

    def __call__(self, probs: np.ndarray, fidelities: np.ndarray) -> float | np.ndarray:
        """Evaluates the fixed-pattern free loss score.

        Args:
            probs (np.ndarray): Array of probabilities :math:`p_k` for fixed outcome patterns.
            fidelities (np.ndarray): Array of state fidelities :math:`F_k` for fixed outcome patterns (1D or 2D).

        Returns:
            float or np.ndarray: Calculated objective score(s).
        """
        alpha = 0.1 * float(len(probs)) if self.alpha is None else float(self.alpha)
        probs_ext = probs[:, None] if fidelities.ndim == 2 else probs
        res = np.sum(alpha * probs_ext + fidelities, axis=0)
        return float(res) if fidelities.ndim == 1 else res


# Pre-instantiated default instances for backwards compatibility and easy usage.
# Because these are instances of ObjectiveFunction, they serialize perfectly via heraldo.factory.
beam_search_loss_fn = BeamSearchLoss()
fixed_pattern_capped_loss_fn = FixedPatternCappedLoss()
fixed_pattern_free_loss_fn = FixedPatternFreeLoss()

default_loss_fn = beam_search_loss_fn
