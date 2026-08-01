"""Evaluation runner for static spatial photonic circuits used in static setup evaluations."""

import numpy as np
import strawberryfields as sf

from heraldo.components.interfaces import StaticCircuit
from heraldo.components.runner import beam_search_loss_fn, fixed_pattern_capped_loss_fn


def _process_fixed_patterns(circuit: StaticCircuit, full_ket: np.ndarray,
                           cutoff_dim: int, measurement_patterns):
    """Simulates static spatial circuit outcomes under pre-specified measurement patterns.

    Args:
        circuit (StaticCircuit): Static spatial circuit instance.
        full_ket (np.ndarray): Multi-mode joint state vector in Fock basis.
        cutoff_dim (int): Fock space truncation cutoff dimension.
        measurement_patterns: Array or list of measurement outcome patterns of shape
            ``(n_sequences, n_meas_modes)`` or ``(n_meas_modes,)``.

    Returns:
        tuple or None: Tuple containing ``(kets, probs, outcome_sums, outcomes)``
        if at least one pattern has non-zero probability, or ``None`` if all patterns are impossible.
    """
    meas_specs = circuit.get_measurement_specs()
    meas_modes = [m for m, c in meas_specs]
    meas_cutoffs = [min(c, cutoff_dim) for m, c in meas_specs]

    patterns_arr = np.asarray(measurement_patterns, dtype=int)
    if patterns_arr.ndim == 1:
        patterns_arr = patterns_arr[None, :]

    if patterns_arr.ndim != 2:
        raise ValueError(
            f"measurement_patterns must be 1D or 2D array; got ndim={patterns_arr.ndim}"
        )

    if patterns_arr.shape[1] != len(meas_modes):
        raise ValueError(
            f"Patterns mismatch: patterns have {patterns_arr.shape[1]} measured modes, circuit expects {len(meas_modes)}"
        )

    perm = [0] + meas_modes
    transposed = np.transpose(full_ket, axes=perm)

    indexer = (slice(None),) + tuple(slice(0, c) for c in meas_cutoffs)
    sliced_ket = transposed[indexer]

    adv_index = (slice(None),) + tuple(patterns_arr[:, j] for j in range(patterns_arr.shape[1]))
    projected = sliced_ket[adv_index]
    raw_kets = projected.T

    probs = np.sum(np.abs(raw_kets)**2, axis=1)
    nonzero_mask = probs >= 1e-12

    if not np.any(nonzero_mask):
        return None

    kets = raw_kets[nonzero_mask] / np.sqrt(probs[nonzero_mask])[:, None]
    probs = probs[nonzero_mask]
    outcomes = patterns_arr[nonzero_mask]
    outcome_sums = np.sum(outcomes, axis=1)

    return kets, probs, outcome_sums, outcomes


def _process_beam_search(circuit: StaticCircuit, full_ket: np.ndarray,
                         cutoff_dim: int, beam_width: int):
    """Selects top-K outcome trajectories for static spatial circuit mode 0 output using Beam Search.

    Args:
        circuit (StaticCircuit): Static spatial circuit instance.
        full_ket (np.ndarray): Multi-mode joint state vector in Fock basis.
        cutoff_dim (int): Fock space truncation cutoff dimension.
        beam_width (int): Maximum number of top measurement branches retained.

    Returns:
        tuple or None: Tuple containing ``(kets, probs, outcome_sums, outcomes)``
        if non-zero probability outcomes survive, or ``None`` otherwise.
    """
    meas_specs = circuit.get_measurement_specs()
    meas_modes = [m for m, c in meas_specs]
    meas_cutoffs = [min(c, cutoff_dim) for m, c in meas_specs]

    perm = [0] + meas_modes
    transposed = np.transpose(full_ket, axes=perm)

    indexer = (slice(None),) + tuple(slice(0, c) for c in meas_cutoffs)
    sliced_ket = transposed[indexer]

    probs_tensor = np.sum(np.abs(sliced_ket)**2, axis=0)

    flat_P = probs_tensor.flatten()
    k = min(beam_width, flat_P.size)
    top_indices = np.argpartition(flat_P, -k)[-k:]
    top_indices = top_indices[np.argsort(-flat_P[top_indices])]

    outcomes_unraveled = np.unravel_index(top_indices, probs_tensor.shape)
    selected_probs = probs_tensor[outcomes_unraveled]

    adv_index = (slice(None),) + outcomes_unraveled
    projected = sliced_ket[adv_index]
    raw_kets = projected.T

    norms = np.linalg.norm(raw_kets, axis=1)
    mask = (norms > 1e-9) & (selected_probs > 1e-12)

    if not np.any(mask):
        return None

    kets = raw_kets[mask] / norms[mask][:, None]
    probs = selected_probs[mask]
    outcomes = np.stack(outcomes_unraveled, axis=1)[mask]
    outcome_sums = np.sum(outcomes, axis=1)

    return kets, probs, outcome_sums, outcomes


def _compute_fidelities_and_loss(kets: np.ndarray, probs: np.ndarray,
                                 outcome_sums: np.ndarray, target_kets: list[np.ndarray],
                                 truncation_error: float, penalty_strength: float,
                                 loss_fn=None):
    """Computes target state fidelities and loss metric for projected output states.

    Args:
        kets (np.ndarray): 2D array of state kets for active trajectories.
        probs (np.ndarray): 1D array of trajectory probabilities.
        outcome_sums (np.ndarray): 1D array of total detected photon count per trajectory.
        target_kets (list[np.ndarray]): List of candidate target state vectors in Fock space.
        truncation_error (float): Fock state truncation error.
        penalty_strength (float): Penalty coefficient for truncation errors.
        loss_fn (callable, optional): Objective loss evaluation function.

    Returns:
        tuple: A tuple containing ``(loss, expected_fidelity, fidelities, best_target_indices, mask_nonzero)``.
    """
    if loss_fn is None:
        loss_fn = beam_search_loss_fn

    mask_nonzero = outcome_sums > 0

    expected_fidelity = 0.0
    fidelities = np.array([])
    best_target_indices = np.array([])

    if np.any(mask_nonzero):
        final_kets = kets[mask_nonzero]
        final_probs = probs[mask_nonzero]

        targets_arr = np.array(target_kets)
        prod = np.conj(final_kets[:, None, :]) * targets_arr[None, :, :]

        fft_vals = np.fft.fft(prod, n=256, axis=-1)

        pairwise_fidelities = np.max(np.abs(fft_vals)**2, axis=-1)

        fidelities = np.max(pairwise_fidelities, axis=1)
        best_target_indices = np.argmax(pairwise_fidelities, axis=1)

        expected_fidelity = loss_fn(final_probs, fidelities)

    loss = -1 * expected_fidelity + (penalty_strength * truncation_error)

    return loss, expected_fidelity, fidelities, best_target_indices, mask_nonzero


def _format_branch_details(mask_nonzero: np.ndarray, probs: np.ndarray,
                           fidelities: np.ndarray, best_target_indices: np.ndarray,
                           outcomes: np.ndarray) -> list[dict]:
    """Formats trajectory and outcome metadata dictionaries.

    Args:
        mask_nonzero (np.ndarray): Boolean mask indicating branches with non-zero photon counts.
        probs (np.ndarray): Array of probabilities for active branches.
        fidelities (np.ndarray): Array of state fidelities for active branches.
        best_target_indices (np.ndarray): Array of best-matching target indices per branch.
        outcomes (np.ndarray): Array of photon detection outcome tuples for measured modes.

    Returns:
        list[dict]: List of metadata dictionaries.
    """
    branch_details = []

    if np.any(mask_nonzero):
        final_probs = probs[mask_nonzero]
        final_outcomes = outcomes[mask_nonzero]

        for i in range(len(final_probs)):
            outcome = tuple(final_outcomes[i].tolist())

            branch_details.append({
                "outcome": outcome,
                "prob": float(final_probs[i]),
                "fidelity": float(fidelities[i]),
                "target_idx": int(best_target_indices[i]),
            })

    return branch_details


def evaluate_circuit(params: np.ndarray,
                     circuit: StaticCircuit,
                     target_kets: list[np.ndarray],
                     cutoff_dim: int,
                     beam_width: int = 5,
                     penalty_strength: float = 10.0,
                     measurement_patterns=None,
                     return_details: bool = False,
                     loss_fn=None):
    """Evaluates a static spatial photonic circuit against target state generators.

    Executes a single static circuit run and evaluates Mode 0 output state under
    Beam Search outcome selection or fixed measurement outcome filtering.

    Args:
        params (np.ndarray): 1D array of static circuit parameters.
        circuit (StaticCircuit): Static spatial circuit model to evaluate.
        target_kets (list[np.ndarray]): Target state vectors in Fock basis.
        cutoff_dim (int): Fock space truncation cutoff dimension.
        beam_width (int, optional): Maximum number of outcome patterns retained. Defaults to 5.
        penalty_strength (float, optional): Penalty coefficient applied to Fock truncation errors. Defaults to 10.0.
        measurement_patterns (optional): Pre-specified fixed measurement patterns. Defaults to None.
        return_details (bool, optional): If True, returns evaluation metrics and trajectory metadata. Defaults to False.
        loss_fn (callable, optional): Custom objective loss evaluation function.

    Returns:
        float or dict: Objective loss float if `return_details` is False, or dictionary of evaluation results.
    """
    engine = sf.Engine("fock", backend_options={"cutoff_dim": cutoff_dim})
    result = circuit.run_circuit(params, engine)
    full_ket = result.state.ket()

    flat_ket = full_ket.flatten()
    norm_sq = np.real(np.vdot(flat_ket, flat_ket))
    truncation_error = np.abs(1.0 - norm_sq)

    use_fixed_patterns = (measurement_patterns is not None)

    if use_fixed_patterns:
        res = _process_fixed_patterns(circuit, full_ket, cutoff_dim, measurement_patterns)
    else:
        res = _process_beam_search(circuit, full_ket, cutoff_dim, beam_width)

    if res is None:
        return 100.0

    kets, probs, outcome_sums, outcomes = res

    if loss_fn is None:
        loss_fn = fixed_pattern_capped_loss_fn if use_fixed_patterns else beam_search_loss_fn

    loss, expected_fidelity, fidelities, best_target_indices, mask_nonzero = _compute_fidelities_and_loss(
        kets, probs, outcome_sums, target_kets, truncation_error, penalty_strength, loss_fn=loss_fn
    )

    if not return_details:
        return loss

    branch_details = _format_branch_details(
        mask_nonzero, probs, fidelities, best_target_indices, outcomes
    )

    return {
        "loss": loss,
        "expected_fidelity": float(expected_fidelity),
        "branches": branch_details,
        "total_probability": float(np.sum(probs[mask_nonzero])) if np.any(mask_nonzero) else 0.0
    }