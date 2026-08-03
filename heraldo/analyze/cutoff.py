"""Cutoff evaluation and truncation analysis utilities for photonic circuits."""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import strawberryfields as sf

from heraldo.analyze.saver import reconstruct_objects
from heraldo.analyze.plotter import _normalize_outcomes
from heraldo.analyze.rotations import get_target_display_names
from heraldo.components.runner import _process_fixed_patterns, _compute_fidelities_and_loss
from heraldo.factory import create_from_config


def analyze_cutoff(
    results: Dict[str, Any],
    low_cutoff: int = 30,
    high_cutoff: int = 50,
    outcomes: Optional[Union[int, Tuple[int, ...], List[Union[int, Tuple[int, ...]]]]] = None,
    targets: Optional[Union[Any, List[Any]]] = None,
    print_summary: bool = True,
) -> Dict[str, Any]:
    """Evaluates circuit performance across two Fock cutoff dimensions to quantify truncation error.

    Runs the static circuit at `low_cutoff` and `high_cutoff`, computing state fidelities,
    infidelities :math:`1 - F`, absolute truncation error, and log discrepancy
    :math:`\\log_{10}(1 - F_{\\text{high}}) - \\log_{10}(1 - F_{\\text{low}})` for each outcome pattern.

    Reuses `_process_fixed_patterns` and `_compute_fidelities_and_loss` from `heraldo.components.runner`.

    Works for fixed measurement patterns automatically. For beam search optimization runs, specific
    outcome patterns MUST be specified via the `outcomes` parameter.

    Args:
        results (dict): Optimization result dictionary returned by `BasinHoppingRunner.run()`
            or loaded via `load_results()`.
        low_cutoff (int, optional): Lower Fock space truncation cutoff dimension. Defaults to 30.
        high_cutoff (int, optional): Higher Fock space truncation cutoff dimension. Defaults to 50.
        outcomes (int, tuple, or list, optional): Measurement outcome pattern(s) to analyze.
            If None, retrieved from fixed measurement patterns stored in `results`.
            Mandatory if beam search optimization was used.
        targets (TargetGenerator or list, optional): Target state generator(s).
            If None, retrieved or reconstructed from `results`.
        print_summary (bool, optional): Whether to print a formatted summary report. Defaults to True.

    Returns:
        dict: Evaluation dictionary containing per-outcome metrics, truncation errors,
            and log discrepancies, as well as overall maximum discrepancy summary metrics.

    Raises:
        TypeError: If `results` is not a dictionary.
        ValueError: If circuit/target/parameter information is missing, or if `outcomes` is None
            for a beam search run.
    """
    if not isinstance(results, dict):
        raise TypeError(f"Expected results to be a dictionary, got {type(results).__name__}")

    circuit = results.get("circuit")
    if circuit is None:
        reconstructed = reconstruct_objects(results)
        circuit = reconstructed.get("circuit")

    if circuit is None and "circuit_config" in results and results["circuit_config"]:
        circuit = create_from_config(results["circuit_config"])

    if circuit is None:
        raise ValueError("Circuit configuration or object missing from results. Cannot execute circuit.")

    x_params = results.get("x")
    if x_params is None:
        raise ValueError("Optimized parameter vector 'x' missing from results.")

    runner_cfg = results.get("runner_config", {})

    # Resolve targets
    if targets is None:
        targets = results.get("targets")
        if targets is None:
            reconstructed = reconstruct_objects(results)
            targets = reconstructed.get("targets")

    if targets is None and "target_configs" in results and results["target_configs"]:
        tc = results["target_configs"]
        if isinstance(tc, list):
            targets = [create_from_config(item) for item in tc]
        else:
            targets = [create_from_config(tc)]

    if targets is None or (isinstance(targets, list) and len(targets) == 0):
        raise ValueError("Target generator(s) missing from results and not provided.")

    if not isinstance(targets, list):
        targets = [targets]

    target_objs = []
    for t in targets:
        if isinstance(t, dict):
            target_objs.append(create_from_config(t))
        else:
            target_objs.append(t)
    targets = target_objs

    # Resolve outcomes
    meas_patterns = runner_cfg.get("measurement_patterns") if runner_cfg else None
    if meas_patterns is None:
        meas_patterns = results.get("measurement_patterns")

    is_beam_search = (meas_patterns is None)

    if is_beam_search and outcomes is None:
        raise ValueError(
            "For beam search optimization, specific outcome(s) must be specified via the 'outcomes' parameter "
            "(e.g., outcomes=4 or outcomes=[4, 5] or outcomes=[(4,)])."
        )

    meas_specs = circuit.get_measurement_specs()
    num_meas_modes = len(meas_specs)

    if outcomes is None:
        outcomes_list = [tuple(int(val) for val in pat) for pat in meas_patterns]
    else:
        outcomes_list = _normalize_outcomes(outcomes, num_meas_modes)

    def _eval_cutoff_pass(cutoff_dim: int):
        target_kets = [t.get_target_ket(cutoff_dim) for t in targets]
        engine = sf.Engine("fock", backend_options={"cutoff_dim": cutoff_dim})
        eval_res = circuit.run_circuit(np.asarray(x_params), engine)
        full_ket = eval_res.state.ket()

        flat_ket = full_ket.flatten()
        norm_sq = np.real(np.vdot(flat_ket, flat_ket))
        trunc_err = np.abs(1.0 - norm_sq)

        proc_res = _process_fixed_patterns(circuit, full_ket, cutoff_dim, outcomes_list)
        outcomes_dict = {}

        if proc_res is not None:
            kets, probs, outcome_sums, active_outcomes = proc_res
            _, _, fidelities, best_target_indices, mask_nonzero = _compute_fidelities_and_loss(
                kets, probs, np.ones_like(outcome_sums), target_kets, trunc_err, penalty_strength=0.0
            )

            final_outcomes = active_outcomes[mask_nonzero]
            final_probs = probs[mask_nonzero]

            for i in range(len(final_probs)):
                out_tuple = tuple(final_outcomes[i].tolist())
                outcomes_dict[out_tuple] = {
                    "prob": float(final_probs[i]),
                    "fidelity": float(fidelities[i]),
                    "target_idx": int(best_target_indices[i]),
                }

        return outcomes_dict, float(trunc_err)

    outcomes_dict_low, trunc_error_low = _eval_cutoff_pass(low_cutoff)
    outcomes_dict_high, trunc_error_high = _eval_cutoff_pass(high_cutoff)

    target_names = get_target_display_names(targets)
    analysis_list = []

    max_abs_error = -1.0
    worst_abs_outcome = None

    max_log_disc = -float('inf')
    worst_log_outcome = None

    for outcome_tuple in outcomes_list:
        res_low = outcomes_dict_low.get(outcome_tuple, {})
        res_high = outcomes_dict_high.get(outcome_tuple, {})

        prob_low = res_low.get("prob", 0.0)
        fid_low = res_low.get("fidelity", 0.0)
        target_idx_low = res_low.get("target_idx", 0)

        prob_high = res_high.get("prob", 0.0)
        fid_high = res_high.get("fidelity", 0.0)

        I_low = 1.0 - fid_low
        I_high = 1.0 - fid_high

        abs_error = abs(I_high - I_low)

        if I_high > 1e-30 and I_low > 1e-30:
            log_disc = float(np.log10(I_high) - np.log10(I_low))
        else:
            log_disc = None

        t_name = target_names[target_idx_low] if target_idx_low < len(target_names) else f"Target_{target_idx_low}"

        item = {
            "outcome": outcome_tuple,
            "target_idx": target_idx_low,
            "target_name": t_name,
            "prob_low": prob_low,
            "prob_high": prob_high,
            "fidelity_low": fid_low,
            "fidelity_high": fid_high,
            "infidelity_low": I_low,
            "infidelity_high": I_high,
            "abs_error": abs_error,
            "log_discrepancy": log_disc,
        }
        analysis_list.append(item)

        if abs_error > max_abs_error:
            max_abs_error = abs_error
            worst_abs_outcome = outcome_tuple

        if log_disc is not None and log_disc > max_log_disc:
            max_log_disc = log_disc
            worst_log_outcome = outcome_tuple

    if max_log_disc == -float('inf'):
        max_log_disc = None

    output_dict = {
        "low_cutoff": low_cutoff,
        "high_cutoff": high_cutoff,
        "truncation_error_low": trunc_error_low,
        "truncation_error_high": trunc_error_high,
        "outcomes": outcomes_list,
        "analysis": analysis_list,
        "max_abs_error": max_abs_error if max_abs_error >= 0 else 0.0,
        "max_log_discrepancy": max_log_disc,
        "worst_outcome_abs_error": worst_abs_outcome,
        "worst_outcome_log_discrepancy": worst_log_outcome,
    }

    if print_summary:
        width = 96
        separator = "=" * width
        sub_separator = "-" * width

        print("\n" + separator)
        print(f"{f'CUTOFF TRUNCATION ANALYSIS RESULTS ({low_cutoff} vs {high_cutoff})':^{width}}")
        print(separator)

        header = f"  {'Outcome':<14} {'Target Name':<22} {f'1-F ({low_cutoff})':<14} {f'1-F ({high_cutoff})':<14} {'Abs. Error':<14} {'Log Disc.':<10}"
        print(header)
        print(sub_separator)

        for res in analysis_list:
            outcome_str = ", ".join(str(v) for v in res["outcome"])
            out_disp = f"n=({outcome_str})" if len(res["outcome"]) > 1 else f"n={outcome_str}"
            t_name = res["target_name"]
            i_low_str = f"{res['infidelity_low']:.2e}"
            i_high_str = f"{res['infidelity_high']:.2e}"
            abs_err_str = f"{res['abs_error']:.2e}"

            log_d = res['log_discrepancy']
            log_str = f"{log_d:+.2f}" if log_d is not None else "N/A"

            print(f"  {out_disp:<14} {t_name:<22} {i_low_str:<14} {i_high_str:<14} {abs_err_str:<14} {log_str:<10}")

        print(sub_separator)
        print(f"  Max Absolute Error            : {max_abs_error:.6e}" + (f" (Outcome: n={worst_abs_outcome})" if worst_abs_outcome else ""))
        if max_log_disc is not None:
            print(f"  Max Log Discrepancy           : {max_log_disc:+.6f}" + (f" (Outcome: n={worst_log_outcome})" if worst_log_outcome else ""))
        print("  Pre-Measurement Joint State Truncation Error (1 - ||ket||²):")
        print(f"    • Low cutoff ({low_cutoff})   : {trunc_error_low:.2e}")
        print(f"    • High cutoff ({high_cutoff})  : {trunc_error_high:.2e}")
        print(separator + "\n")

    return output_dict
