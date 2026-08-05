"""Loss analysis utilities for quantum states and circuit outcomes under photon loss."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import copy
import numpy as np
import strawberryfields as sf

from heraldo._internal import _normalize_outcomes
from heraldo.analyze.rotations import get_target_display_names
from heraldo.serialization import create_from_config, reconstruct_objects, to_config


def _compute_max_fidelity_dm(rho: np.ndarray, target_ket: np.ndarray, n_fft: int = 256) -> float:
    """Computes max fidelity F = max_phi <target_phi | R(phi) rho R^dagger(phi) | target_phi> using FFT.

    Args:
        rho (np.ndarray): Density matrix of the output state (shape cutoff_dim x cutoff_dim).
        target_ket (np.ndarray): Target state vector in Fock basis.
        n_fft (int, optional): Resolution for phase rotation optimization. Defaults to 256.

    Returns:
        float: Maximum achievable fidelity in [0, 1].
    """
    target_ket = np.asarray(target_ket, dtype=np.complex128).flatten()
    D = len(target_ket)
    coeffs = np.zeros(n_fft, dtype=np.complex128)
    target_conj = np.conj(target_ket)

    coeffs[0] = np.sum(np.diagonal(rho) * np.abs(target_ket)**2)

    for delta in range(1, D):
        rho_diag = np.diagonal(rho, offset=delta)
        term = np.sum(rho_diag * target_ket[delta:] * target_conj[:-delta])
        coeffs[delta] = term
        coeffs[-delta] = np.conj(term)

    vals = np.fft.ifft(coeffs) * n_fft
    return float(np.clip(np.max(np.real(vals)), 0.0, 1.0))


def analyze_loss(
    results: Dict[str, Any],
    transmissivities: Optional[List[float]] = None,
    outcomes: Optional[Union[int, Tuple[int, ...], List[Union[int, Tuple[int, ...]]]]] = None,
    cutoff_dim: Optional[int] = None,
    print_summary: bool = True,
) -> Dict[str, Any]:
    """Analyzes the performance of optimized static spatial circuits under photon loss.

    Re-evaluates the circuit across specified loss transmissivity levels eta in (0, 1],
    computing the conditional success probability P and state fidelity F for each outcome pattern
    using density matrix simulation.

    Args:
        results (dict): Optimization result dictionary returned by `BasinHoppingRunner.run()`
            or loaded via `load_results()`.
        transmissivities (list[float], optional): List of channel transmissivity values eta in (0, 1].
            Defaults to [1.0, 0.99, 0.90] (corresponding to 0%, 1%, and 10% loss).
        outcomes (int, tuple, or list, optional): Measurement outcome pattern(s) to analyze.
            If None, retrieved from runner configuration.
        cutoff_dim (int, optional): Fock space truncation cutoff dimension.
            If None, defaults to 15 for 3-mode and higher circuits, and 30 for 2-mode circuits.
        print_summary (bool, optional): Whether to print a formatted summary report. Defaults to True.

    Returns:
        dict: Summary dictionary containing transmissivity values, analyzed outcomes, and detailed metrics.
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
    if cutoff_dim is None:
        num_modes = getattr(circuit, "num_modes", None)
        if num_modes is None and hasattr(circuit, "get_measurement_specs"):
            num_modes = len(circuit.get_measurement_specs()) + 1
        if num_modes is not None and num_modes >= 3:
            cutoff_dim = 15
        else:
            cutoff_dim = 30

    n_fft = 256

    if transmissivities is None:
        transmissivities = [1.0, 0.99, 0.90]

    # Resolve targets
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
        raise ValueError("Target generator(s) missing from results.")

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

    if circuit is not None:
        num_meas_modes = len(circuit.get_measurement_specs())
    else:
        temp_circ = create_from_config(results["circuit_config"])
        num_meas_modes = len(temp_circ.get_measurement_specs())

    if outcomes is None:
        if meas_patterns is not None:
            outcomes_list = _normalize_outcomes(meas_patterns, num_meas_modes)
        else:
            branches = results.get("branches", [])
            outcomes_list = [b["outcome"] for b in branches if "outcome" in b]

            if not outcomes_list:
                raise ValueError(
                    "No measurement outcomes specified or found in results. "
                    "Please specify outcomes (e.g., outcomes=[4, 5] or outcomes=[(4,)])."
                )
    else:
        outcomes_list = _normalize_outcomes(outcomes, num_meas_modes)

    target_kets = [t.get_target_ket(cutoff_dim) for t in targets]
    target_names = get_target_display_names(targets)

    results_by_outcome = {out: {"results_by_transmissivity": {}} for out in outcomes_list}
    sorted_transmissivities = sorted(transmissivities, reverse=True)
    ideal_target_indices = {}

    for eta in sorted_transmissivities:
        if "circuit_config" in results and results["circuit_config"]:
            cfg = copy.deepcopy(results["circuit_config"])
            cfg.setdefault("params", {})["loss_transmissivity"] = float(eta)
            eval_circuit = create_from_config(cfg)
        elif circuit is not None:
            try:
                cfg = to_config(circuit)
                cfg.setdefault("params", {})["loss_transmissivity"] = float(eta)
                eval_circuit = create_from_config(cfg)
            except Exception:
                eval_circuit = copy.deepcopy(circuit)
                setattr(eval_circuit, "loss_transmissivity", float(eta))
        else:
            raise ValueError("Cannot instantiate circuit for loss simulation.")

        meas_specs = [(m, min(c, cutoff_dim)) for m, c in eval_circuit.get_measurement_specs()]
        n_modes = eval_circuit.num_modes

        engine = sf.Engine("fock", backend_options={"cutoff_dim": cutoff_dim})
        eval_res = eval_circuit.run_circuit(np.asarray(x_params), engine)
        full_dm = eval_res.state.dm()

        for outcome_tuple in outcomes_list:
            indexer = [slice(None)] * (2 * n_modes)
            for i, (m_idx, _) in enumerate(meas_specs):
                val = outcome_tuple[i]
                indexer[2 * m_idx] = val
                indexer[2 * m_idx + 1] = val

            projected_dm = full_dm[tuple(indexer)]
            trace_prob = float(np.real(np.trace(projected_dm)))

            if trace_prob > 1e-12:
                normalized_dm = projected_dm / trace_prob
                all_fids = [_compute_max_fidelity_dm(normalized_dm, t_ket, n_fft=n_fft) for t_ket in target_kets]

                if outcome_tuple not in ideal_target_indices:
                    ideal_target_indices[outcome_tuple] = int(np.argmax(all_fids))

                best_t_idx = ideal_target_indices[outcome_tuple]
                fid = float(all_fids[best_t_idx])
            else:
                if outcome_tuple not in ideal_target_indices:
                    ideal_target_indices[outcome_tuple] = 0
                best_t_idx = ideal_target_indices[outcome_tuple]
                fid = 0.0

            results_by_outcome[outcome_tuple]["results_by_transmissivity"][eta] = {
                "prob": trace_prob,
                "fidelity": fid,
            }

    analysis_list = []
    for outcome_tuple in outcomes_list:
        t_idx = ideal_target_indices[outcome_tuple]
        t_name = target_names[t_idx] if t_idx < len(target_names) else f"Target_{t_idx}"
        analysis_list.append({
            "outcome": outcome_tuple,
            "target_idx": t_idx,
            "target_name": t_name,
            "results_by_transmissivity": results_by_outcome[outcome_tuple]["results_by_transmissivity"],
        })

    output_dict = {
        "transmissivities": transmissivities,
        "outcomes": outcomes_list,
        "analysis": analysis_list,
    }

    if print_summary:
        header_labels = []
        for eta in transmissivities:
            if abs(eta - 1.0) < 1e-6:
                header_labels.append("100.0% (Ideal)")
            else:
                loss_pct = (1.0 - eta) * 100
                if loss_pct >= 1.0:
                    header_labels.append(f"{eta*100:.1f}% ({loss_pct:.0f}% Loss)")
                else:
                    header_labels.append(f"{eta*100:.1f}% ({loss_pct:.1f}% Loss)")

        col_width = 22
        outcome_col_width = 14
        target_col_width = 24

        total_width = outcome_col_width + target_col_width + len(transmissivities) * col_width
        separator = "=" * total_width
        sub_separator = "-" * total_width

        print("\n" + separator)
        print(f"{'PHOTON LOSS ANALYSIS RESULTS':^{total_width}}")
        print(separator)

        eta_summary_str = ", ".join(header_labels)
        print(f"  Transmissivities Evaluated : {eta_summary_str}")
        print(sub_separator)

        h1 = f"  {'Outcome':<{outcome_col_width-2}} {'Target Name':<{target_col_width}}"
        for lbl in header_labels:
            h1 += f"{lbl:^{col_width}}"
        print(h1)

        h2 = f"  {' ':<{outcome_col_width-2}} {' ':<{target_col_width}}"
        for _ in transmissivities:
            h2 += f"  {'Prob':<9} {'Fidelity':<9} "
        print(h2)
        print(sub_separator)

        for item in analysis_list:
            out_tuple = item["outcome"]
            outcome_str = ", ".join(str(v) for v in out_tuple)
            out_disp = f"n=({outcome_str})" if len(out_tuple) > 1 else f"n={outcome_str}"
            t_name = item["target_name"]

            row_str = f"  {out_disp:<{outcome_col_width-2}} {t_name:<{target_col_width}}"
            for eta in transmissivities:
                res_eta = item["results_by_transmissivity"][eta]
                p = res_eta["prob"]
                f = res_eta["fidelity"]
                p_str = f"{p:.2%}"
                f_str = f"{f:.4f}"
                row_str += f"  {p_str:<9} {f_str:<9} "
            print(row_str)

        print(separator + "\n")

    return output_dict
