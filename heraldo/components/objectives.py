"""Optimization objective functions for circuit evaluation.

Defines non-linear objective functions used during parameter optimization,
distinguished from physical optical photon loss.
"""

import numpy as np


def beam_search_loss_fn(probs: np.ndarray, fidelities: np.ndarray, epsilon: float = 2e-2,
                        delta: float = 1e-72, lam: float = 1e4) -> float:
    r"""Evaluates the multi-outcome beam search pattern discovery loss function.

    Implements the non-linear multi-outcome loss metric (Eqs. 3–5 in paper) that filters
    low-fidelity outcomes and sharpens optimization gradients around high-quality candidate patterns:

    .. math::
        L_{\text{beam}} = \log\left( \sum_k p_k (\tilde{F}_k^2 \Lambda_k)^4 + \delta \right) + \lambda \sum_k p_k (\tilde{F}_k^2 \Lambda_k)^4

    Args:
        probs (np.ndarray): Array of probabilities :math:`p_k` for surviving output patterns.
        fidelities (np.ndarray): Array of state fidelities :math:`F_k` for surviving output patterns.
        epsilon (float, optional): Clipping parameter :math:`\epsilon` defining fidelity bounds. Defaults to 0.02.
        delta (float, optional): Regularization parameter :math:`\delta` to prevent logarithmic divergence. Defaults to 1e-72.
        lam (float, optional): Linear penalty multiplier :math:`\lambda`. Defaults to 1e4.

    Returns:
        float: Calculated beam search objective value.
    """
    infidelities = np.maximum(1.0 - fidelities, epsilon)
    capped_fidelities = np.minimum(fidelities, 1.0 - epsilon)

    log_vals = np.log10(infidelities) / np.log10(epsilon)
    score = np.sum(probs * (capped_fidelities**2 * log_vals)**4)
    return float(np.log(score + delta) + lam * score)


default_loss_fn = beam_search_loss_fn


def fixed_pattern_capped_loss_fn(probs: np.ndarray, fidelities: np.ndarray, f_cap: float = 0.95,
                                alpha: float = None) -> float:
    r"""Evaluates the fixed-pattern optimization loss function under a capped fidelity regime.

    Implements the capped fidelity objective (Eq. 2 in paper):

    .. math::
        L_{\text{fixed}} = \sum_k \left( \alpha p_k + \min(F_k, F_{\text{cap}}) \right)

    Args:
        probs (np.ndarray): Array of probabilities :math:`p_k` for fixed outcome patterns.
        fidelities (np.ndarray): Array of state fidelities :math:`F_k` for fixed outcome patterns.
        f_cap (float, optional): Maximum fidelity cap :math:`F_{\text{cap}}`. Defaults to 0.95.
        alpha (float, optional): Weighting coefficient :math:`\alpha`. Defaults to `len(probs)` if None.

    Returns:
        float: Calculated objective score.
    """
    if alpha is None:
        alpha = float(len(probs))
    capped_fidelities = np.minimum(fidelities, f_cap)
    return float(np.sum(alpha * probs + capped_fidelities))


def fixed_pattern_free_loss_fn(probs: np.ndarray, fidelities: np.ndarray,
                              alpha: float = None) -> float:
    r"""Evaluates the fixed-pattern optimization loss function under an uncapped (free) fidelity regime.

    Implements the uncapped objective function (Eq. 2 in paper):

    .. math::
        L_{\text{fixed}} = \sum_k \left( \alpha p_k + F_k \right)

    Args:
        probs (np.ndarray): Array of probabilities :math:`p_k` for fixed outcome patterns.
        fidelities (np.ndarray): Array of state fidelities :math:`F_k` for fixed outcome patterns.
        alpha (float, optional): Weighting coefficient :math:`\alpha`. Defaults to `0.1 * len(probs)` if None.

    Returns:
        float: Calculated objective score.
    """
    if alpha is None:
        alpha = 0.1 * float(len(probs))
    return float(np.sum(alpha * probs + fidelities))