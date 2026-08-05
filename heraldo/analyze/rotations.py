"""Rotation analysis utilities for quantum states and circuit outcomes."""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import strawberryfields as sf

from heraldo._internal import _normalize_outcomes
from heraldo.components.targets import (
    BinomialCodeTarget, CatTarget, CoreGKPTarget, CubicPhaseTarget, CubicResourceTarget, SqueezedCatTarget
)
from heraldo.serialization import create_from_config, reconstruct_objects


def compute_angular_range(angles_deg: List[float]) -> float:
    """Computes the shortest angular interval containing all angles on a circle (in degrees).

    Args:
        angles_deg (list[float]): List of angles in degrees.

    Returns:
        float: Angular span/range in degrees [0, 360).
    """
    if not angles_deg or len(angles_deg) <= 1:
        return 0.0

    normalized = [a % 360.0 for a in angles_deg]
    if max(normalized) - min(normalized) < 1e-9:
        return 0.0

    normalized.sort()
    gaps = []
    n = len(normalized)
    for i in range(n):
        gap = (normalized[(i + 1) % n] - normalized[i]) % 360.0
        gaps.append(gap)

    max_gap = max(gaps)
    if max_gap < 1e-9:
        return 0.0

    return 360.0 - max_gap


def get_target_display_names(targets: List[Any]) -> List[str]:
    """Generates user-friendly display names for target state generators.

    Args:
        targets (list): List of TargetGenerator instances or config dicts.

    Returns:
        list[str]: Display names for each target generator.
    """
    def _fmt_val(val: Any) -> str:
        if isinstance(val, (float, np.floating)):
            if abs(val) < 1e-9:
                val = 0.0
            s = f"{val:.2f}".rstrip('0').rstrip('.')
            return s if s != "" else "0"
        return str(val)

    target_names = []
    for t in targets:
        if isinstance(t, CoreGKPTarget):
            target_names.append(f"GKP_n{t.n_max}_mu{t.mu}")
        elif isinstance(t, SqueezedCatTarget):
            target_names.append(f"SqCat_a{_fmt_val(t.alpha)}_r{_fmt_val(t.r)}_p{t.p}")
        elif isinstance(t, CatTarget):
            target_names.append(f"Cat_a{_fmt_val(t.alpha)}_p{t.p}")
        elif isinstance(t, BinomialCodeTarget):
            target_names.append(f"Binomial_N{t.N}_S{t.S}_mu{t.mu}")
        elif isinstance(t, CubicPhaseTarget):
            target_names.append(f"CubicPhase_g{_fmt_val(t.gamma)}_r{_fmt_val(t.r)}")
        elif isinstance(t, CubicResourceTarget):
            target_names.append(f"CubicResource_a{_fmt_val(t.a)}")
        elif isinstance(t, dict) and "class_name" in t:
            cls_name = t["class_name"]
            params = t.get("params", {})
            if params:
                param_strs = [f"{k}{_fmt_val(v)}" for k, v in params.items()]
                target_names.append(f"{cls_name}_{'_'.join(param_strs)}")
            else:
                target_names.append(cls_name)
        elif hasattr(t, "__class__"):
            target_names.append(t.__class__.__name__)
        else:
            target_names.append(str(t))
    return target_names


def analyze_rotations(
    results: Dict[str, Any],
    outcomes: Optional[Union[int, Tuple[int, ...], List[Union[int, Tuple[int, ...]]]]] = None,
    targets: Optional[Union[Any, List[Any]]] = None,
    cutoff_dim: Optional[int] = None,
    n_fft: int = 256,
    print_summary: bool = True,
) -> List[Dict[str, Any]]:
    """Analyzes the optimal phase space rotation of output states relative to target states.

    Computes the maximum achievable fidelity over phase rotations R(phi) = exp(-i phi n)
    for specified or all circuit measurement outcomes, identifying the best-matching target state
    and optimal rotation angle.

    If beam search optimization was used, the `outcomes` argument MUST be provided.
    If fixed pattern optimization was used and `outcomes` is None, all fixed measurement patterns
    are analyzed by default.

    Args:
        results (dict): Optimization result dictionary returned by `BasinHoppingRunner.run()`
            or loaded via `load_results()`.
        outcomes (int, tuple, or list, optional): Measurement outcome pattern(s) to analyze.
            Mandatory if beam search optimization was used.
        targets (TargetGenerator or list, optional): Target state generator(s).
            If None, retrieved or reconstructed from `results`.
        cutoff_dim (int, optional): Fock space truncation cutoff dimension.
            If None, retrieved from runner configuration or defaults to 30.
        n_fft (int, optional): Angle search resolution for FFT rotation optimization. Defaults to 256.
        print_summary (bool, optional): Whether to print a formatted summary report. Defaults to True.

    Returns:
        list[dict]: List of dictionaries containing detailed rotation analysis per outcome.

    Raises:
        TypeError: If `results` is not a dictionary.
        ValueError: If required circuit or target information is missing, or if `outcomes`
            is None for beam search optimization.
    """
    if not isinstance(results, dict):
        raise TypeError(f"Expected results to be a dictionary, got {type(results).__name__}")

    circuit = results.get("circuit")
    if circuit is None:
        reconstructed = reconstruct_objects(results)
        circuit = reconstructed.get("circuit")

    if circuit is None:
        raise ValueError("Circuit configuration or object missing from results. Cannot execute circuit.")

    x_params = results.get("x")
    if x_params is None:
        raise ValueError("Optimized parameter vector 'x' missing from results.")

    runner_cfg = results.get("runner_config", {})
    meas_patterns = runner_cfg.get("measurement_patterns") if runner_cfg else None
    is_beam_search = (meas_patterns is None)

    if is_beam_search and outcomes is None:
        raise ValueError(
            "For beam search optimization, specific outcome(s) must be specified via the 'outcomes' parameter "
            "(e.g., outcomes=4 or outcomes=[(4,)] or outcomes=[4, 5])."
        )

    # Resolve target generators
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

    meas_specs = circuit.get_measurement_specs()
    num_meas_modes = len(meas_specs)

    if outcomes is None:
        outcomes_list = _normalize_outcomes(meas_patterns, num_meas_modes)
    else:
        outcomes_list = _normalize_outcomes(outcomes, num_meas_modes)

    if cutoff_dim is None:
        cutoff_dim = runner_cfg.get("cutoff_dim", 30) if runner_cfg else 30

    target_kets = [t.get_target_ket(cutoff_dim) for t in targets]
    target_names = get_target_display_names(targets)

    engine = sf.Engine("fock", backend_options={"cutoff_dim": cutoff_dim})
    eval_res = circuit.run_circuit(np.asarray(x_params), engine)
    full_ket = eval_res.state.ket()

    meas_modes = [m for m, c in meas_specs]
    meas_cutoffs = [min(c, cutoff_dim) for m, c in meas_specs]

    perm = [0] + meas_modes
    transposed = np.transpose(full_ket, axes=perm)
    indexer = (slice(None),) + tuple(slice(0, c) for c in meas_cutoffs)
    sliced_ket = transposed[indexer]

    branches = results.get("branches", [])
    branch_map = {b.get("outcome"): b for b in branches if "outcome" in b}

    analysis_results = []
    all_angles = []
    target_angles: Dict[int, List[float]] = {}

    for outcome_tuple in outcomes_list:
        for mode_idx, (val, c_dim) in enumerate(zip(outcome_tuple, meas_cutoffs)):
            if val >= c_dim or val < 0:
                raise ValueError(
                    f"Outcome value {val} for measured mode index {mode_idx} exceeds cutoff dimension {c_dim}."
                )

        adv_index = (slice(None),) + outcome_tuple
        mode0_ket_raw = sliced_ket[adv_index]
        prob = np.vdot(mode0_ket_raw, mode0_ket_raw).real

        if prob < 1e-15:
            print(f"Warning: Outcome {outcome_tuple} has near-zero probability ({prob:.2e}).")
            mode0_ket = mode0_ket_raw
        else:
            mode0_ket = mode0_ket_raw / np.sqrt(prob)

        # FFT Rotation Optimization
        prod = np.conj(mode0_ket) * np.array(target_kets)
        fft_vals = np.fft.fft(prod, n=n_fft, axis=-1)
        fidelities = np.abs(fft_vals)**2
        max_fid_per_target = np.max(fidelities, axis=-1)

        best_t_idx = int(np.argmax(max_fid_per_target))
        best_fid = float(max_fid_per_target[best_t_idx])
        best_k = int(np.argmax(fidelities[best_t_idx]))

        angle_rad = 2 * np.pi * best_k / n_fft
        if angle_rad > np.pi:
            angle_rad -= 2 * np.pi
        angle_deg = float(np.degrees(angle_rad))

        t_name = target_names[best_t_idx] if best_t_idx < len(target_names) else f"Target_{best_t_idx}"

        branch_info = branch_map.get(outcome_tuple)
        recorded_prob = branch_info.get("prob", prob) if branch_info else prob

        item = {
            "outcome": outcome_tuple,
            "prob": float(recorded_prob),
            "fidelity": float(best_fid),
            "target_idx": best_t_idx,
            "target_name": t_name,
            "angle_rad": float(angle_rad),
            "angle_deg": angle_deg,
        }
        analysis_results.append(item)
        all_angles.append(angle_deg)
        target_angles.setdefault(best_t_idx, []).append(angle_deg)

    if print_summary:
        width = 88
        separator = "=" * width
        sub_separator = "-" * width

        print("\n" + separator)
        print(f"{'ROTATION ANALYSIS RESULTS':^{width}}")
        print(separator)

        header = f"  {'Outcome':<14} {'Probability':<14} {'Target Name':<24} {'Fidelity':<12} {'Angle (rad)':<13} {'Angle (deg)':<11}"
        print(header)
        print(sub_separator)

        for res in analysis_results:
            outcome_str = ", ".join(str(v) for v in res["outcome"])
            out_disp = f"n=({outcome_str})" if len(res["outcome"]) > 1 else f"n={outcome_str}"
            p_str = f"{res['prob']:.2%}"
            fid_str = f"{res['fidelity']:.4f}"
            rad_str = f"{res['angle_rad']:+.4f}"
            deg_str = f"{res['angle_deg']:+.1f}°"
            t_name = res["target_name"]

            print(f"  {out_disp:<14} {p_str:<14} {t_name:<24} {fid_str:<12} {rad_str:<13} {deg_str:<11}")

        print(sub_separator)
        overall_range = compute_angular_range(all_angles)
        print(f"  Overall Rotation Range (All Outcomes) : {overall_range:.2f}° ({np.radians(overall_range):.4f} rad)")

        for idx in sorted(target_angles.keys()):
            t_angs = target_angles[idx]
            t_name = target_names[idx] if idx < len(target_names) else f"Target_{idx}"
            t_range = compute_angular_range(t_angs)
            count_str = f"({len(t_angs)} outcome)" if len(t_angs) == 1 else f"({len(t_angs)} outcomes)"
            print(f"  Range for {t_name:<24} : {t_range:.2f}° ({np.radians(t_range):.4f} rad) {count_str}")

        print(separator + "\n")

    return analysis_results
