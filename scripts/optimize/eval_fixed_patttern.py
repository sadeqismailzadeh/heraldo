"""
Evaluation and Analysis Suite for Optimized Static Spatial Photonic Circuits.

Provides analysis routines, density matrix noise/loss evaluations, phase rotation optimization,
high-resolution publication-quality Wigner plotting, cutoff truncation fidelity benchmarks,
and LaTeX summary report generation for static spatial circuits.
"""

import argparse
import copy
import glob
import itertools
import json
import os
from pathlib import Path
import pickle
import platform
import re

import heraldo
import matplotlib.pyplot as plt
import numpy as np
import strawberryfields as sf

import heraldo.components.targets as target_module
import heraldo.components.static_circuits as circuit_module
from heraldo.components.targets import (
    BinomialCodeTarget, CatTarget, CoreGKPTarget, CubicPhaseTarget, SqueezedCatTarget
)
from heraldo.factory import create_from_config
from heraldo.components.interfaces import StaticCircuit
from heraldo.components.static_runner import (
    evaluate_circuit
)

from heraldo.components.objectives import (
    beam_search_loss_fn,
    fixed_pattern_capped_loss_fn, fixed_pattern_free_loss_fn
)
from heraldo.utils import (
    db_to_r, fidelity_max_rotation, fidelity_pure_state, windows_to_wsl_path
)


def sanitize_config_paths(config):
    """
    Recursively fix file paths in configuration dictionaries to work across WSL/Windows.
    """
    if isinstance(config, dict):
        new_config = config.copy()
        if config.get('class_name') == 'CoreGKPTarget' and 'params' in config:
            params = config['params'].copy()
            if 'csv_path' in params:
                path_str = params['csv_path']
                if platform.system() == "Windows" and path_str.startswith("/mnt/"):
                    parts = path_str.split('/')
                    if len(parts) > 2:
                        drive_letter = parts[2]
                        rest_of_path = "/".join(parts[3:])
                        new_path = f"{drive_letter.upper()}:/{rest_of_path}"
                        params['csv_path'] = new_path
                        print(f"Sanitized WSL path: {path_str} -> {new_path}")
            new_config['params'] = params
            return new_config

        for k, v in new_config.items():
            new_config[k] = sanitize_config_paths(v)
        return new_config

    elif isinstance(config, list):
        return [sanitize_config_paths(item) for item in config]

    return config


def prepare_measurement_patterns(patterns):
    """Normalizes measurement pattern lists into a list of outcome tuples."""
    if patterns is None:
        return None

    normalized = []
    p_list = patterns.tolist() if isinstance(patterns, np.ndarray) else patterns

    for item in p_list:
        flat = np.array(item, dtype=int).flatten().tolist()
        normalized.append(tuple(flat))

    return normalized


def _compute_max_fidelity_dm(rho, target_ket, n_fft=256):
    """
    Computes max fidelity F = max_phi <target_phi | rho | target_phi>
    using FFT for rotation optimization.
    """
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
    return float(np.max(np.real(vals)))


def evaluate_static_circuit_dm(params, circuit: StaticCircuit, target_kets, cutoff_dim,
                               beam_width=100, penalty_strength=0.0, prob_power=1.0,
                               measurement_patterns=None, loss_fn=None):
    """
    Evaluates the static circuit using Density Matrices to support loss/noise.

    Args:
        params: Flat parameter vector.
        circuit: StaticCircuit instance.
        target_kets: List of target kets (pure states).
        cutoff_dim: Fock cutoff.
        beam_width: Number of branches to keep.
        penalty_strength: (Unused in this evaluator, kept for signature compatibility)
        prob_power: (Unused in this evaluator, kept for signature compatibility)
        measurement_patterns: Optional list or array of fixed outcome patterns.
        loss_fn: Loss function to aggregate fidelity across branches.

    Returns:
        dict: Results containing 'branches', 'expected_fidelity', and 'total_probability'.
    """
    meas_specs = [(m, min(c, cutoff_dim)) for m, c in circuit.get_measurement_specs()]
    meas_modes = [m for m, c in meas_specs]
    n_modes = circuit.num_modes

    eng = sf.Engine("fock", backend_options={"cutoff_dim": cutoff_dim})
    result = circuit.run_circuit(params, eng)
    full_dm = result.state.dm()

    use_fixed_patterns = measurement_patterns is not None

    if use_fixed_patterns:
        loop_outcomes = prepare_measurement_patterns(measurement_patterns)
    else:
        ranges = [range(c) for _, c in meas_specs]
        loop_outcomes = list(itertools.product(*ranges))

    candidates = []
    for outcomes in loop_outcomes:
        indexer = [slice(None)] * (2 * n_modes)
        for i, (m_idx, _) in enumerate(meas_specs):
            val = outcomes[i]
            indexer[2 * m_idx] = val
            indexer[2 * m_idx + 1] = val

        projected_dm = full_dm[tuple(indexer)]
        trace_prob = np.real(np.trace(projected_dm))

        if trace_prob > 1e-12:
            new_dm = projected_dm / trace_prob
            candidates.append({
                'dm': new_dm,
                'prob': trace_prob,
                'outcome': outcomes
            })

    if not use_fixed_patterns:
        candidates.sort(key=lambda x: x['prob'], reverse=True)
        active_branches = candidates[:beam_width]
    else:
        active_branches = candidates

    results = []
    total_captured_prob = 0.0

    for branch in active_branches:
        rho = branch['dm']
        prob = branch['prob']
        total_captured_prob += prob

        max_fid = 0.0
        best_target_idx = 0

        for t_i, t_ket in enumerate(target_kets):
            fid = _compute_max_fidelity_dm(rho, t_ket)
            if fid > max_fid:
                max_fid = fid
                best_target_idx = t_i

        results.append({
            'outcome': branch['outcome'],
            'prob': prob,
            'fidelity': max_fid,
            'target_idx': best_target_idx
        })

    if results:
        if loss_fn is None:
            loss_fn = fixed_pattern_free_loss_fn if use_fixed_patterns else beam_search_loss_fn

        probs_arr = np.array([b['prob'] for b in results])
        fids_arr = np.array([b['fidelity'] for b in results])
        expected_fidelity = loss_fn(probs_arr, fids_arr)
    else:
        expected_fidelity = 0.0

    return {
        "branches": results,
        "expected_fidelity": expected_fidelity,
        "total_probability": total_captured_prob
    }


def run_deterministic_path(circuit: StaticCircuit, params: np.ndarray, measurement_outcome, cutoff_dim: int):
    """
    Evaluates a static spatial circuit for a specific measurement outcome.

    Args:
        circuit: The StaticCircuit instance
        params: Flat parameter array for the static circuit
        measurement_outcome: Outcome tuple/array across measured modes, e.g. (1, 3) or (4,)
        cutoff_dim: Fock truncation dimension

    Returns:
        dict with 'final_probability' and 'final_state_ket' if valid, or None if zero prob.
    """
    meas_specs = [(m, min(c, cutoff_dim)) for m, c in circuit.get_measurement_specs()]
    meas_modes = [m for m, c in meas_specs]
    meas_cutoffs = [c for m, c in meas_specs]

    eng = sf.Engine("fock", backend_options={"cutoff_dim": cutoff_dim})
    res = circuit.run_circuit(params, eng)
    full_ket = res.state.ket()

    perm = [0] + meas_modes
    transposed_ket = np.transpose(full_ket, axes=perm)

    outcome_tuple = tuple(np.array(measurement_outcome, dtype=int).flatten())

    if len(outcome_tuple) != len(meas_modes):
        return None

    for val, limit in zip(outcome_tuple, meas_cutoffs):
        if val >= limit:
            return None

    indexer = (slice(None),) + outcome_tuple
    proj_ket = transposed_ket[indexer]

    prob = np.real(np.vdot(proj_ket, proj_ket))
    if prob < 1e-12:
        return None

    norm_ket = proj_ket / np.sqrt(prob)
    return {
        "final_probability": float(prob),
        "final_state_ket": norm_ket
    }


def plot_wigner_print_quality(ket, filename="state_plot.png", title="State", cutoff_dim=50,
                              grid_size=400, x_limit=6, ax_wigner=None, ax_fock=None):
    """
    Generates a high-resolution Wigner function and Fock distribution plot suitable for publication/print.
    """
    norm = np.linalg.norm(ket)
    if abs(norm - 1.0) > 1e-6:
        ket = ket / norm

    prog = sf.Program(1)
    with prog.context as q:
        from strawberryfields.ops import Ket
        Ket(ket) | q[0]
    eng = sf.Engine("fock", backend_options={"cutoff_dim": cutoff_dim})
    result = eng.run(prog)
    state = result.state

    xvec = np.linspace(-x_limit, x_limit, grid_size)
    pvec = np.linspace(-x_limit, x_limit, grid_size)
    W = state.wigner(mode=0, xvec=xvec, pvec=pvec)

    probs = state.all_fock_probs(cutoff=cutoff_dim)

    plt.rcParams.update({'font.size': 14, 'font.family': 'sans-serif'})
    custom_axes = (ax_wigner is not None) and (ax_fock is not None)

    if custom_axes:
        ax1, ax2 = ax_wigner, ax_fock
        fig = ax1.figure
    else:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7), dpi=300)

    X, P = np.meshgrid(xvec, pvec)
    lim = np.max(np.abs(W))
    c = ax1.pcolormesh(X, P, W, cmap='RdBu', shading='auto', vmin=-lim, vmax=lim, rasterized=True)

    cbar = fig.colorbar(c, ax=ax1, label='W(x, p)', pad=0.02)
    cbar.ax.tick_params(labelsize=12)

    ax1.set_title(f"Wigner Function: {title}", fontsize=16, pad=15)
    ax1.set_xlabel("x (Position)", fontsize=14)
    ax1.set_ylabel("p (Momentum)", fontsize=14)
    ax1.set_aspect('equal')
    ax1.axhline(0, color='gray', linestyle=':', alpha=0.5, linewidth=1)
    ax1.axvline(0, color='gray', linestyle=':', alpha=0.5, linewidth=1)

    display_cutoff = min(cutoff_dim, 60)
    indices = np.arange(display_cutoff)
    ax2.bar(indices, probs[:display_cutoff], color='#2c7bb6', alpha=0.8, edgecolor='black', width=0.7)

    ax2.set_title("Fock State Probabilities", fontsize=16, pad=15)
    ax2.set_xlabel("Fock Number |n>", fontsize=14)
    ax2.set_ylabel("Probability", fontsize=14)

    step = 5 if display_cutoff > 20 else 1
    ax2.set_xticks(np.arange(0, display_cutoff, step))
    ax2.grid(axis='y', linestyle='--', alpha=0.3)
    ax2.set_xlim(-0.5, display_cutoff - 0.5)
    ax2.set_ylim(0, max(probs) * 1.1)

    if not custom_axes:
        plt.tight_layout()
        save_path = Path(filename).resolve()
        plt.savefig(save_path, bbox_inches='tight', dpi=100)
        plt.close(fig)
        print(f"High-quality plot saved to: {save_path}")


def load_optimization_run(results_dir: Path, selection: str = "best"):
    """
    Unified loader for optimization results.
    """
    target_file = None

    if (results_dir / "results.pkl").exists():
        target_file = results_dir / "results.pkl"
    elif selection == "best":
        best_dir = results_dir / "best"
        best_files = sorted(best_dir.glob("best_run_*.pkl"))
        if best_files:
            target_file = best_files[-1]
        else:
            target_file = best_dir / "best_run_0001.pkl"
    elif selection == "latest":
        runs = sorted(results_dir.glob("run_*.pkl"))
        if runs:
            target_file = runs[-1]
    else:
        try:
            run_num = int(selection)
            target_file = results_dir / f"run_{run_num:04d}.pkl"
        except ValueError:
            print(f"Error: Invalid selection '{selection}'. Use 'best', 'latest', or a number.")
            return None

    if not target_file or not target_file.exists():
        print(f"Error: Result file not found at {target_file}")
        return None

    print(f"Loading data from: {target_file}")
    with open(target_file, "rb") as f:
        data = pickle.load(f)

    if isinstance(data, dict) and "meta" in data and "res" in data:
        res = data["res"]
        meta = data["meta"]
        if "circuit_config" in meta:
            res["circuit_config"] = meta["circuit_config"]
        if "target_configs" in meta:
            res["target_configs"] = meta["target_configs"]
        if "measurement_patterns" in meta:
            res["measurement_patterns"] = meta["measurement_patterns"]
    else:
        res = data

    return {
        "best_res": res,
        "x": res.get("x")
    }


def _reshape_outcome_flat(outcome_flat, circuit: StaticCircuit):
    """
    Normalize stored branch outcome into a per-mode tuple for static circuits,
    e.g. (4,) for 2 modes or (1, 3) for 3 modes.
    """
    if outcome_flat is None:
        return None

    meas_specs = circuit.get_measurement_specs()
    n_meas_modes = len(meas_specs)

    arr = np.array(outcome_flat, dtype=int).flatten()

    if arr.size == n_meas_modes:
        return tuple(int(x) for x in arr)
    elif arr.size % n_meas_modes == 0:
        return tuple(int(x) for x in arr[:n_meas_modes])

    raise ValueError(
        f"Outcome length {arr.size} incompatible with static circuit "
        f"(meas_modes={n_meas_modes})"
    )


def _outcome_to_str(reshaped_pattern) -> str:
    """Flattens a reshaped outcome pattern into an underscore-separated string."""
    flat_outcomes = np.array(reshaped_pattern, dtype=int).flatten()
    return "_".join(map(str, flat_outcomes))


def save_all_fixed_pattern_wigners(circuit, flat_x, measurement_patterns, cutoff, results_dir, combine_plots=False):
    """
    Iterates over all saved fixed measurement patterns, evaluates them deterministically,
    and saves their high-quality Wigner plots.
    """
    if measurement_patterns is None or len(measurement_patterns) == 0:
        print("No fixed measurement patterns provided to save.")
        return

    print(f"\n=== Evaluating Wigner plots for {len(measurement_patterns)} fixed patterns ===")

    valid_results = []
    for i, pattern in enumerate(measurement_patterns):
        try:
            reshaped_pattern = _reshape_outcome_flat(pattern, circuit)
        except Exception as e:
            print(f"Failed to reshape pattern {pattern}: {e}")
            continue

        print(f"Evaluating pattern {i+1}/{len(measurement_patterns)}: {reshaped_pattern}")
        res = run_deterministic_path(circuit, flat_x, reshaped_pattern, cutoff)

        if res is None:
            print(f"  -> Path {reshaped_pattern} is not physically possible (zero probability). Skipping.")
            continue

        valid_results.append({
            'reshaped_pattern': reshaped_pattern,
            'ket': res['final_state_ket'],
            'prob': res['final_probability']
        })

    if not valid_results:
        print("No valid patterns found to plot.")
        return

    if not combine_plots:
        print(f"\n=== Saving {len(valid_results)} individual Wigner plots ===")
        for res_dict in valid_results:
            reshaped_pattern = res_dict['reshaped_pattern']
            ket = res_dict['ket']
            prob = res_dict['prob']

            outcome_str = _outcome_to_str(reshaped_pattern)
            filename = results_dir / f"wigner_hq_{outcome_str}.png"
            title = f"Outcome {reshaped_pattern} (P={prob:.2e})"

            plot_wigner_print_quality(ket, filename=filename, title=title, cutoff_dim=cutoff)

    else:
        print(f"\n=== Saving {len(valid_results)} Wigner plots into a single combined image ===")
        n_plots = len(valid_results)

        fig, axes = plt.subplots(nrows=n_plots, ncols=2, figsize=(16, 7 * n_plots), dpi=150)
        if n_plots == 1:
            axes = np.array([axes])

        for i, res_dict in enumerate(valid_results):
            reshaped_pattern = res_dict['reshaped_pattern']
            ket = res_dict['ket']
            prob = res_dict['prob']

            ax_wigner, ax_fock = axes[i]
            title = f"Outcome {reshaped_pattern} (P={prob:.2e})"

            plot_wigner_print_quality(
                ket,
                title=title,
                cutoff_dim=cutoff,
                ax_wigner=ax_wigner,
                ax_fock=ax_fock
            )

        plt.tight_layout()
        combined_filename = results_dir / "wigner_hq_combined_all_patterns.png"
        plt.savefig(combined_filename, bbox_inches='tight', dpi=150)
        plt.close(fig)
        print(f"Combined high-quality plot saved to: {combined_filename}")


def compute_angular_range(angles_deg):
    """Computes the shortest interval containing all angles on a circle."""
    if not angles_deg:
        return 0.0
    if len(angles_deg) == 1:
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


def _compute_ket_target_fidelities(ket, target_kets, n_fft=256):
    """Computes max fidelity over phase and targets using FFT."""
    prod = np.conj(ket) * np.array(target_kets)
    fft_vals = np.fft.fft(prod, n=n_fft, axis=-1)
    fidelities = np.abs(fft_vals)**2
    max_fid_per_target = np.max(fidelities, axis=-1)
    best_t_idx = int(np.argmax(max_fid_per_target))
    best_fid = float(max_fid_per_target[best_t_idx])
    best_k = int(np.argmax(fidelities[best_t_idx]))
    return best_fid, best_t_idx, best_k, fidelities


def _iter_opt_folders(results_base_dir: Path, circuit_module, require_targets: bool = False):
    """
    Generator yielding (results_dir, circuit, targets, flat_x, stored_patterns, best_res)
    for all valid opt_* and job_* directories under results_base_dir.
    """
    base_dir = Path(results_base_dir)
    opt_folders = [p for p in base_dir.rglob("*") if p.is_dir() and (p.name.startswith("opt_") or p.name.startswith("job_"))]
    if not opt_folders:
        print(f"No 'opt_' or 'job_' folders found in {base_dir}")
        return

    for results_dir in opt_folders:
        try:
            best = load_optimization_run(results_dir, selection="best")
            if not best:
                continue
            best_res = best.get('best_res', {})
            stored_patterns = best_res.get('measurement_patterns')
            flat_x = best.get('x', best_res.get('x'))
            if flat_x is None:
                continue

            circuit_config = sanitize_config_paths(best_res.get('circuit_config'))
            if not circuit_config:
                continue
            circuit = create_from_config(circuit_config, circuit_module)

            target_configs = sanitize_config_paths(best_res.get('target_configs'))
            targets = []
            if target_configs:
                targets = [create_from_config(cfg, target_module) for cfg in target_configs]

            if require_targets and not targets:
                continue

            yield results_dir, circuit, targets, flat_x, stored_patterns, best_res
        except Exception as e:
            print(f"  [Error] Failed to process {results_dir.name}: {e}")


def evaluate_and_report_rotations(circuit, flat_x, measurement_patterns, targets, cutoff, results_dir):
    """
    Evaluates the optimal rotation angle for fixed measurement patterns with respect to the best target.
    Saves the results in a report file in the results directory.
    """
    if measurement_patterns is None or len(measurement_patterns) == 0 or targets is None or len(targets) == 0:
        print("Skipping rotation report: Missing measurement patterns or targets.")
        return

    print(f"\n=== Evaluating rotations for {len(measurement_patterns)} fixed patterns ===")

    target_kets = [t.get_target_ket(cutoff) for t in targets]
    target_names = get_target_display_names(targets)

    report_lines = [
        f"{'Pattern':<20} | {'Prob':<10} | {'Best Target':<15} | {'Fidelity':<10} | {'Angle (rad)':<12} | {'Angle (deg)':<12}",
        "-" * 92
    ]

    ket_report_lines = [
        "==================================================",
        " RAW STATE KET REPORT",
        "==================================================",
        ""
    ]

    n_fft = 256
    valid_count = 0
    all_angles = []
    target_angles = {}

    for i, pattern in enumerate(measurement_patterns):
        try:
            reshaped_pattern = _reshape_outcome_flat(pattern, circuit)
        except Exception:
            continue

        res = run_deterministic_path(circuit, flat_x, reshaped_pattern, cutoff)
        if res is None:
            continue

        ket = res['final_state_ket']
        prob = res['final_probability']
        pattern_str = _outcome_to_str(reshaped_pattern)

        ket_report_lines.append(f"Pattern: {pattern_str} (Prob: {prob:.4e})")
        ket_report_lines.append("-" * 50)
        for n, amp in enumerate(ket):
            ket_report_lines.append(f"  |{n:>2}> : {amp.real:>6.3f} {amp.imag:>+6.3f}j")
        ket_report_lines.append("\n")

        best_fid, best_t_idx, best_k, _ = _compute_ket_target_fidelities(ket, target_kets, n_fft=n_fft)

        angle_rad = 2 * np.pi * best_k / n_fft
        if angle_rad > np.pi:
            angle_rad -= 2 * np.pi
        angle_deg = np.degrees(angle_rad)

        t_name = target_names[best_t_idx] if best_t_idx < len(target_names) else f"Target_{best_t_idx}"

        report_lines.append(f"{pattern_str:<20} | {prob:<10.2e} | {t_name:<15} | {best_fid:<10.4f} | {angle_rad:<12.4f} | {angle_deg:<12.1f}")
        all_angles.append(angle_deg)
        target_angles.setdefault(best_t_idx, []).append(angle_deg)
        valid_count += 1

    if valid_count > 0:
        overall_range = compute_angular_range(all_angles)

        summary_lines = [
            "",
            "=" * 92,
            " ROTATION RANGE SUMMARY",
            "=" * 92,
            f"Overall Rotation Range (All Patterns): {overall_range:.2f}° ({np.radians(overall_range):.4f} rad)"
        ]

        for idx in sorted(target_angles.keys()):
            t_angs = target_angles[idx]
            t_name = target_names[idx] if idx < len(target_names) else f"Target_{idx}"
            t_range = compute_angular_range(t_angs)
            summary_lines.append(f"Range for {t_name:<20} ({len(t_angs)} states): {t_range:.2f}° ({np.radians(t_range):.4f} rad)")
        summary_lines.append("=" * 92)

        report_lines.extend(summary_lines)
        print("\n".join(summary_lines))

        report_path = results_dir / "rotation_report.txt"
        with open(report_path, "w") as f:
            f.write("\n".join(report_lines))
        print(f"Saved rotation report to: {report_path}")

        ket_report_path = results_dir / "ket_report.txt"
        with open(ket_report_path, "w") as f:
            f.write("\n".join(ket_report_lines))
        print(f"Saved ket report to:      {ket_report_path}")
    else:
        print("No valid patterns found to evaluate rotations.")


def generate_wigners_for_all_opt_folders(results_base_dir: Path, circuit_module, cutoff: int = 30, combine_plots: bool = False):
    """
    Iterates through all 'opt_*' and 'job_*' folders in a given base directory and saves
    fixed pattern Wigner figures for each valid optimization result.
    """
    for results_dir, circuit, _, flat_x, stored_patterns, _ in _iter_opt_folders(results_base_dir, circuit_module):
        if stored_patterns is None or len(stored_patterns) == 0:
            print(f"  [Skip] No measurement_patterns found in {results_dir.name}.")
            continue
        save_all_fixed_pattern_wigners(circuit, np.asarray(flat_x), stored_patterns, cutoff, results_dir, combine_plots=combine_plots)
        print(f"  [Success] Processed Wigner figures for {results_dir.name}.")


def report_rotations_for_all_opt_folders(results_base_dir: Path, circuit_module, cutoff: int = 30):
    """
    Iterates through all 'opt_*' and 'job_*' folders in a given base directory and generates
    rotation reports for each valid optimization result.
    """
    for results_dir, circuit, targets, flat_x, stored_patterns, _ in _iter_opt_folders(results_base_dir, circuit_module, require_targets=True):
        if stored_patterns is None or len(stored_patterns) == 0:
            print(f"  [Skip] No measurement_patterns found in {results_dir.name}.")
            continue
        evaluate_and_report_rotations(circuit, np.asarray(flat_x), stored_patterns, targets, cutoff, results_dir)
        print(f"  [Success] Processed rotation reports for {results_dir.name}.")


def format_prob(p_val):
    """Helper to format probability percentages matching paper style."""
    pct = p_val * 100
    if pct >= 1.0:
        return f"{pct:.1f}\\%"
    else:
        return f"{pct:.2f}\\%"


def format_outcome_latex(reshaped_pattern, target):
    """Helper to format LaTeX representation of outcome state."""
    flat_outcomes = np.array(reshaped_pattern, dtype=int).flatten().tolist()
    outcome_str = ", ".join(map(str, flat_outcomes))

    if len(flat_outcomes) == 1 and hasattr(target, 'mu') and hasattr(target, 'n_max'):
        return f"({flat_outcomes[0]}) \\to \\ket{{{target.mu}_{{A{target.n_max}}}}}"
    elif len(flat_outcomes) == 1:
        return f"({flat_outcomes[0]})"
    else:
        return f"({outcome_str})"


def evaluate_loss_influence(results_base_dir: Path, circuit_module):
    """
    Re-runs the optimized static circuits across different photon loss levels (ideal, 1%, and 10% loss)
    for all configurations in the given base directory.
    Outputs a LaTeX table showing the success probability (P) and state fidelity (F).
    """
    print(f"\n=== Starting Loss Influence Evaluation ===")
    collected_data = []

    for results_dir, circuit, targets, flat_x, stored_patterns, best_res in _iter_opt_folders(results_base_dir, circuit_module, require_targets=True):
        if stored_patterns is None or len(stored_patterns) == 0:
            continue

        circuit_config = sanitize_config_paths(best_res.get('circuit_config'))
        meas_specs = circuit.get_measurement_specs()
        meas_modes = [m for m, c in meas_specs]
        n_modes = max(meas_modes) + 1 if meas_modes else 1

        cutoff = 15 if n_modes == 3 else 30
        target_kets = [t.get_target_ket(cutoff) for t in targets]

        target_names = []
        for t in targets:
            if isinstance(t, CoreGKPTarget):
                target_names.append(f"{{GKP core}} $\\ket{{{t.mu}_{{A{t.n_max}}}}}$" if n_modes == 3 else f"{{GKP core}} $\\mu={t.mu}$")
            elif isinstance(t, SqueezedCatTarget):
                target_names.append(f"{{SqCat}} $\\alpha={t.alpha}$")
            elif isinstance(t, CatTarget):
                target_names.append(f"{{Cat}} $\\alpha={t.alpha}$")
            else:
                target_names.append(f"{{{t.__class__.__name__.replace('Target', '')}}}")

        strategy = "Multiplex"
        folder_lower = results_dir.name.lower()
        if "harvest" in folder_lower:
            strategy = "Harvest"
        elif "single" in folder_lower:
            strategy = "Single"

        runs_results = {}
        for loss_val in [1.0, 0.99, 0.90]:
            current_config = copy.deepcopy(circuit_config)
            current_config.setdefault('params', {})['loss_transmissivity'] = loss_val
            eval_circuit = create_from_config(current_config, circuit_module)

            runs_results[loss_val] = evaluate_static_circuit_dm(
                np.asarray(flat_x), eval_circuit, target_kets, cutoff,
                beam_width=100, penalty_strength=0.0, prob_power=1.0,
                measurement_patterns=stored_patterns,
                loss_fn=fixed_pattern_free_loss_fn
            )

        branches_ideal = {b['outcome']: b for b in runs_results[1.0]['branches']}
        branches_1 = {b['outcome']: b for b in runs_results[0.99]['branches']}
        branches_10 = {b['outcome']: b for b in runs_results[0.90]['branches']}

        for pattern in stored_patterns:
            try:
                reshaped_pattern = _reshape_outcome_flat(pattern, circuit)
            except Exception:
                continue

            b_ideal = branches_ideal.get(reshaped_pattern)
            if not b_ideal:
                continue
            b_1 = branches_1.get(reshaped_pattern)
            b_10 = branches_10.get(reshaped_pattern)

            t_idx = b_ideal.get('target_idx', 0)
            t_name = target_names[t_idx] if t_idx < len(target_names) else target_names[0]
            t_obj = targets[t_idx] if t_idx < len(targets) else targets[0]

            collected_data.append({
                'folder': results_dir.name,
                'target': t_name,
                'modes': n_modes,
                'strategy': strategy,
                'outcome_latex': format_outcome_latex(reshaped_pattern, t_obj),
                'p_ideal': b_ideal['prob'],
                'f_ideal': b_ideal['fidelity'],
                'p_1': b_1['prob'] if b_1 else 0.0,
                'f_1': b_1['fidelity'] if b_1 else 0.0,
                'p_10': b_10['prob'] if b_10 else 0.0,
                'f_10': b_10['fidelity'] if b_10 else 0.0
            })

    if not collected_data:
        print("No valid data collected to print a loss report.")
        return

    def get_sort_key(row):
        target_val = 0 if "GKP" in row['target'] else 1
        nums = [int(s) for s in re.findall(r'\d+', row['outcome_latex'])]
        out_val = nums[0] if nums else 0
        return (row['modes'], target_val, row['target'], row['strategy'], out_val)

    collected_data.sort(key=get_sort_key)

    latex_lines = [
        "\\begin{table*}[t]",
        "\\centering",
        "\\caption{Impact of photon loss on the performance of optimized multi-outcome static circuits. We compare the success probability ($P$) and state fidelity $\\mathcal{F}$ across three loss conditions: ideal, 1\\% loss, and 10\\% loss. Photon loss is simulated by placing fictitious beam splitters on both the ancillary modes and the heralded output mode prior to detection.}",
        "\\label{tab:loss_analysis}",
        "\\begin{tabular}{l c l p{2.5cm} c @{\\hspace{1.5em}} c @{\\hspace{3em}} c @{\\hspace{1.5em}} c @{\\hspace{3em}} c @{\\hspace{1.5em}} c}",
        "    \\toprule",
        "    & & & & \\multicolumn{2}{c}{\\hspace{-1.5em}ideal} & \\multicolumn{2}{c}{\\hspace{-1.5em}1\\% Loss} & \\multicolumn{2}{c}{10\\% Loss} \\\\",
        "    \\cmidrule(l{0em}r{2em}){5-6} \\cmidrule(l{0em}r{2em}){7-8} \\cmidrule(l{0em}r{0em}){9-10}",
        "    {Target} & {Modes} & {Strategy} & {Outcome} $\\mathbf{n}$ & $P$ & $\\mathcal{F}$ & $P$ & $\\mathcal{F}$ & $P$ & $\\mathcal{F}$ \\\\ ",
        "    \\midrule"
    ]

    prev_target, prev_modes, prev_strategy = None, None, None
    for idx, row in enumerate(collected_data):
        is_same = (row['target'] == prev_target and row['modes'] == prev_modes and row['strategy'] == prev_strategy)
        t_col = row['target'] if not is_same else ""
        m_col = str(row['modes']) if not is_same else ""
        s_col = row['strategy'] if not is_same else ""

        if idx > 0:
            if row['target'] != prev_target or row['modes'] != prev_modes:
                latex_lines.append("    \\midrule")
            elif row['strategy'] != prev_strategy:
                latex_lines.append("    \\addlinespace")

        latex_lines.append(
            f" {t_col} & {m_col} & {s_col} & {row['outcome_latex']} & "
            f"{format_prob(row['p_ideal'])} & {row['f_ideal']:.2f} & "
            f"{format_prob(row['p_1'])} & {row['f_1']:.2f} & "
            f"{format_prob(row['p_10'])} & {row['f_10']:.2f} \\\\"
        )
        prev_target, prev_modes, prev_strategy = row['target'], row['modes'], row['strategy']

    latex_lines.extend(["    \\bottomrule", "\\end{tabular}", "\\end{table*}"])
    latex_output = "\n".join(latex_lines)

    report_path = Path(results_base_dir) / "loss_influence_report.tex"
    with open(report_path, "w") as f:
        f.write(latex_output)

    print(f"\nSaved loss influence LaTeX report to: {report_path}")
    print("\n--- GENERATED LATEX TABLE (MARKDOWN COMPATIBLE) ---")
    print("```latex")
    print(latex_output)
    print("```")
    print("------------------------------\n")


def evaluate_cutoff_fidelity(results_base_dir: Path, circuit_module, low_cutoff: int = 30, high_cutoff: int = 50):
    """
    Evaluates the target fidelity for states generated with low_cutoff and high_cutoff 
    for all fixed measurement patterns across all opt_* and job_* folders.
    """
    print(f"\n=== Starting Cutoff Fidelity Evaluation ({low_cutoff} vs {high_cutoff}) ===")

    report_lines = [
        f"Cutoff Fidelity Report: {low_cutoff} vs {high_cutoff}",
        "=" * 125,
        f"{'Folder':<40} | {'Pattern':<15} | {'1-F_'+str(low_cutoff):<10} | {'1-F_'+str(high_cutoff):<10} | "
        f"{'Abs. Error':<10} | {'error / ( 1-F_' + str(low_cutoff) + ')':<18} | {'Log Disc.':<10}",
        "-" * 125
    ]

    max_error, worst_pattern, worst_folder = -1.0, None, None
    max_rel_dev, worst_rel_pattern, worst_rel_folder = -1.0, None, None
    max_log_disc, worst_log_pattern, worst_log_folder = -float('inf'), None, None

    for results_dir, circuit, targets, flat_x, stored_patterns, _ in _iter_opt_folders(results_base_dir, circuit_module, require_targets=True):
        if stored_patterns is None or len(stored_patterns) == 0:
            continue

        target_kets_low = np.array([t.get_target_ket(low_cutoff) for t in targets])
        target_kets_high = np.array([t.get_target_ket(high_cutoff) for t in targets])

        for pattern in stored_patterns:
            try:
                reshaped_pattern = _reshape_outcome_flat(pattern, circuit)
            except Exception:
                continue

            res_low = run_deterministic_path(circuit, np.asarray(flat_x), reshaped_pattern, low_cutoff)
            res_high = run_deterministic_path(circuit, np.asarray(flat_x), reshaped_pattern, high_cutoff)

            if res_low is None or res_high is None:
                continue

            F_low, _, _, _ = _compute_ket_target_fidelities(res_low['final_state_ket'], target_kets_low)
            F_high, _, _, _ = _compute_ket_target_fidelities(res_high['final_state_ket'], target_kets_high)

            I_low = 1.0 - F_low
            I_high = 1.0 - F_high
            error = abs(I_high - I_low)
            rel_deviation = (error / I_low) if I_low > 1e-18 else (0.0 if error < 1e-18 else float('inf'))

            pattern_str = _outcome_to_str(reshaped_pattern)
            rel_dev_str = f"{rel_deviation:.2e}" if rel_deviation != float('inf') else "inf"

            if I_high > 1e-30 and I_low > 1e-30:
                log_discrepancy = np.log10(I_high) - np.log10(I_low)
                log_disc_str = f"{log_discrepancy:+.2f}"
            else:
                log_disc_str, log_discrepancy = "N/A", -float('inf')

            report_lines.append(
                f"{results_dir.name:<40} | {pattern_str:<15} | {I_low:<10.2e} | {I_high:<10.2e} | "
                f"{error:<10.2e} | {rel_dev_str:<18} | {log_disc_str:<10}"
            )

            if error > max_error:
                max_error, worst_pattern, worst_folder = error, pattern_str, results_dir.name
            if rel_deviation != float('inf') and rel_deviation > max_rel_dev:
                max_rel_dev, worst_rel_pattern, worst_rel_folder = rel_deviation, pattern_str, results_dir.name
            if log_discrepancy > max_log_disc:
                max_log_disc, worst_log_pattern, worst_log_folder = log_discrepancy, pattern_str, results_dir.name

    report_lines.append("=" * 125)
    report_lines.append(f"MAXIMUM TRUNCATION ERROR: {max_error:.6e}")
    if worst_folder:
        report_lines.append(f"Found in Folder: {worst_folder}\nWith Pattern: {worst_pattern}")
    report_lines.append("-" * 125)
    report_lines.append(f"MAXIMUM RELATIVE ERROR (error / ( 1-F_{low_cutoff})): {max_rel_dev:.6e}")
    if worst_rel_folder:
        report_lines.append(f"Found in Folder: {worst_rel_folder}\nWith Pattern: {worst_rel_pattern}")
    report_lines.append("-" * 125)
    report_lines.append(f"MAXIMUM LOG DISCREPANCY: {max_log_disc:.6f}")
    if worst_log_folder:
        report_lines.append(f"Found in Folder: {worst_log_folder}\nWith Pattern: {worst_log_pattern}")

    report_path = Path(results_base_dir) / "cutoff_infidelity_report.txt"
    with open(report_path, "w") as f:
        f.write("\n".join(report_lines))

    print(f"\nSaved cutoff infidelity report to: {report_path}")
    print(f"Maximum Truncation Error: {max_error:.6e} (Folder: {worst_folder}, Pattern: {worst_pattern})")
    print(f"Maximum Relative Error (error / ( 1-F_{low_cutoff})): {max_rel_dev:.6e} (Folder: {worst_rel_folder}, Pattern: {worst_rel_pattern})")
    print(f"Maximum Log Discrepancy: {max_log_disc:.6f} (Folder: {worst_log_folder}, Pattern: {worst_log_pattern})")


def save_density_matrices_for_all_opt_folders(results_base_dir: Path, circuit_module, cutoff: int = 30):
    """
    Iterates through all 'opt_*' and 'job_*' folders in a given base directory,
    evaluates all fixed measurement patterns, and saves their density matrices as .npy files.
    """
    for results_dir, circuit, _, flat_x, stored_patterns, _ in _iter_opt_folders(results_base_dir, circuit_module):
        if stored_patterns is None or len(stored_patterns) == 0:
            print(f"  [Skip] No measurement_patterns found in {results_dir.name}.")
            continue
        for pattern in stored_patterns:
            try:
                reshaped_pattern = _reshape_outcome_flat(pattern, circuit)
            except Exception as e:
                print(f"  Failed to reshape pattern {pattern}: {e}")
                continue
            res = run_deterministic_path(circuit, np.asarray(flat_x), reshaped_pattern, cutoff)
            if res is None or res.get('final_state_ket') is None:
                continue
            dm = np.outer(res['final_state_ket'], np.conj(res['final_state_ket']))
            outcome_str = _outcome_to_str(reshaped_pattern)
            save_path = results_dir / f"state_dm_{outcome_str}.npy"
            np.save(save_path, dm)
            print(f"  Saved DM for pattern {reshaped_pattern} to: {save_path.name}")
        print(f"  [Success] Processed density matrices for {results_dir.name}.")


def get_target_display_names(targets: list) -> list[str]:
    """Generates display names for target generators."""
    target_names = []
    for t in targets:
        if isinstance(t, CoreGKPTarget):
            target_names.append(f"GKP_n{t.n_max}_mu{t.mu}")
        elif isinstance(t, SqueezedCatTarget):
            target_names.append(f"SqCat_a{t.alpha}_r{t.r}_p{t.p}")
        elif isinstance(t, CatTarget):
            target_names.append(f"Cat_a{t.alpha}_p{t.p}")
        elif isinstance(t, BinomialCodeTarget):
            target_names.append(f"Binomial_N{t.N}_S{t.S}_mu{t.mu}")
        elif isinstance(t, CubicPhaseTarget):
            target_names.append(f"CubicPhase_g{t.gamma}_r{t.r}")
        else:
            target_names.append(t.__class__.__name__)
    return target_names


def main():
    cutoff = 30
    visualize_results_path = windows_to_wsl_path(r"E:\Quantum\code\results\sweeps_static_fid099_20260801T190934Z")
    all_results_path = windows_to_wsl_path(r"E:\Quantum\code\results\sweeps_static_fid099_20260801T190934Z")
    loss_results_path = windows_to_wsl_path(r"E:\Quantum\code\results\sweeps_static_fid099_20260801T190934Z")

    if all_results_path:
        generate_wigners_for_all_opt_folders(all_results_path, circuit_module, cutoff=cutoff)
        report_rotations_for_all_opt_folders(all_results_path, circuit_module, cutoff=cutoff)
        evaluate_cutoff_fidelity(all_results_path, circuit_module, low_cutoff=cutoff, high_cutoff=50)

    if visualize_results_path:
        save_density_matrices_for_all_opt_folders(visualize_results_path, circuit_module, cutoff=cutoff)

    if loss_results_path:
        evaluate_loss_influence(loss_results_path, circuit_module)


if __name__ == "__main__":
    main()
