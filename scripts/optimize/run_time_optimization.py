"""
Script to run single-configuration time-domain optimization for loop-based
photonic continuous-variable (CV) state-preparation circuits using Basin-Hopping.
"""

import os
import sys
import json
import time
import pickle
import shutil
import warnings
import itertools
from pathlib import Path
from datetime import datetime
import numpy as np

# --- Set thread limits for NumPy/OpenBLAS/MKL before importing libraries ---
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

warnings.filterwarnings("ignore", category=RuntimeWarning, module="scipy.optimize")

import heraldo.components.circuits as circuit_module
import heraldo.components.targets as target_module
from heraldo.components.interfaces import TimeMultiplexedCircuit
from heraldo.components.runner import (
    BasinHoppingRunner, beam_search_loss_fn, fixed_pattern_capped_loss_fn, fixed_pattern_free_loss_fn
)
from heraldo.components.targets import (
    TargetGenerator, CoreGKPTarget, SqueezedCatTarget, CatTarget,
    BinomialCodeTarget, CubicPhaseTarget, TrisqueezedTarget, QuadsqueezedTarget
)
from heraldo.utils import db_to_r, compute_ng_scores
from heraldo.factory import create_from_config


def prepare_measurement_patterns(patterns):
    """
    Convert user-friendly list/tuple measurement patterns into a NumPy array structure
    expected by the optimizer runner.
    """
    if patterns is None:
        return None

    if isinstance(patterns, np.ndarray):
        if patterns.ndim == 2:
            return patterns[None, ...]
        return patterns

    try:
        patterns_np = np.array(patterns, dtype=int)
        if patterns_np.ndim == 2:
            patterns_np = patterns_np[None, ...]
        return patterns_np
    except Exception:
        # Fallback for ragged or non-uniform pattern shapes
        return patterns


def generate_measurement_patterns(circuit: TimeMultiplexedCircuit, 
                                  min_total_photons: int = None, 
                                  max_total_photons: int = None, 
                                  exact_total: int = None,
                                  max_per_mode: int = None):
    """
    Generate all possible measurement outcome patterns for a given circuit under photon count constraints.

    Args:
        circuit: TimeMultiplexedCircuit instance.
        min_total_photons: Minimum total photon sum across all measured modes/steps.
        max_total_photons: Maximum total photon sum across all measured modes/steps.
        exact_total: Exact photon sum constraint (overrides min/max if set).
        max_per_mode: Max photons allowed in a single measurement outcome.

    Returns:
        List of pattern sequences, where each pattern is a list of tuples per step.
    """
    if exact_total is not None:
        min_total_photons = exact_total
        max_total_photons = exact_total

    if min_total_photons is None:
        min_total_photons = 0
    if max_total_photons is None:
        max_total_photons = float('inf')

    meas_specs = circuit.get_measurement_specs()
    num_measured_modes = len(meas_specs)
    cutoffs = [cutoff for _, cutoff in meas_specs]

    if max_per_mode is None:
        max_per_mode_list = [c - 1 for c in cutoffs]
    else:
        max_per_mode_list = [max_per_mode] * num_measured_modes

    single_step_outcomes = []
    ranges = [range(0, min(max_per_mode_list[i] + 1, cutoffs[i])) for i in range(num_measured_modes)]

    for combo in itertools.product(*ranges):
        single_step_outcomes.append(combo)

    all_patterns = []
    for step_combo in itertools.product(single_step_outcomes, repeat=circuit.steps):
        total_photons = sum(sum(step) for step in step_combo)
        if min_total_photons <= total_photons <= max_total_photons:
            all_patterns.append(list(step_combo))

    return all_patterns


def print_targets(targets, cutoff_dim, tolerance=1e-6):
    """Prints target state labels and non-zero Fock basis coefficients."""
    for i, target in enumerate(targets):
        if isinstance(target, CoreGKPTarget):
            target_name = f"GKP_n{target.n_max}_mu{target.mu}"
        elif isinstance(target, SqueezedCatTarget):
            target_name = f"SqCat_a{target.alpha}_r{target.r}_p{target.p}"
        elif isinstance(target, CatTarget):
            target_name = f"Cat_a{target.alpha}_p{target.p}"
        elif isinstance(target, BinomialCodeTarget):
            target_name = f"Binomial_N{target.N}_S{target.S}_mu{target.mu}"
        elif isinstance(target, CubicPhaseTarget):
            target_name = f"CubicPhase_g{target.gamma}_r{target.r}"
        else:
            target_name = target.__class__.__name__

        print(f"\nTarget {i+1}: {target_name}")
        print("-" * (len(target_name) + 10))

        ket_state = target.get_target_ket(cutoff_dim)
        for n, val in enumerate(ket_state):
            if np.abs(val) > tolerance:
                out_val = val.real if np.abs(val.imag) < 1e-8 else val
                print(f"  |{n}>: {out_val:.6f}")
        print("")


def save_experiment_details(results_dir: Path, circuit_config: dict, target_configs: list, patterns=None, beam_width=None):
    """Saves a readable summary of the experiment setup to experiment_details.txt."""
    lines = []
    lines.append("=" * 80)
    lines.append(" EXPERIMENT CONFIGURATION DETAILS")
    lines.append("=" * 80)
    lines.append(f"Generated at: {datetime.utcnow().isoformat()}Z")
    lines.append("")

    lines.append("--- CIRCUIT CONFIGURATION ---")
    lines.append(f"Class: {circuit_config.get('class_name', 'Unknown')}")
    params = circuit_config.get('params', {})
    for k, v in params.items():
        lines.append(f"  {k:<25}: {v}")
    lines.append("")

    lines.append("--- TARGET CONFIGURATIONS ---")
    lines.append(f"Total Targets: {len(target_configs)}")
    for i, cfg in enumerate(target_configs):
        lines.append(f"Target {i+1}: {cfg.get('class_name', 'Unknown')}")
        t_params = cfg.get('params', {})
        for pk, pv in t_params.items():
            lines.append(f"  - {pk:<23}: {pv}")
        lines.append("")

    lines.append("--- MEASUREMENT STRATEGY ---")
    if patterns is not None:
        n_p = len(patterns)
        lines.append(f"Fixed Measurement Patterns ({n_p} total):")
        p_to_print = patterns.tolist() if hasattr(patterns, 'tolist') else patterns
        for i, p in enumerate(p_to_print):
            if i < 100:
                lines.append(f"  {i+1:3d}: {p}")
            else:
                lines.append(f"  ... and {n_p - 100} more patterns.")
                break
    else:
        lines.append(f"Beam Search Discovery Mode (Beam Width = {beam_width})")
    lines.append("")
    lines.append("=" * 80)

    content = "\n".join(lines)
    with open(results_dir / "experiment_details.txt", "w") as f:
        f.write(content)
    return content


def format_branches_report(branches, target_names, success_threshold):
    """Formats branch probability and fidelity statistics into a readable report."""
    lines = []
    lines.append("-" * 80)
    lines.append(f"{'Outcome':<20} {'Prob':<10} {'Fidelity':<10} {'1-Fid':<10} {'Best Target':<15}")
    lines.append("-" * 80)

    sorted_branches = sorted(branches, key=lambda x: x['prob'], reverse=True)
    total_prob = 0.0
    for b in sorted_branches:
        total_prob += b['prob']
        outcome_str = str(b['outcome'])
        t_idx = b.get('target_idx', 0)
        tgt_name = target_names[t_idx] if t_idx < len(target_names) else f"Target_{t_idx}"
        lines.append(f"{outcome_str:<20} {b['prob']:<10.4f} {b['fidelity']:<10.4f} {(1-b['fidelity']):<10.1e} {tgt_name:<15}")

    lines.append(f"\nTotal Probability captured: {total_prob:.5f}")

    # Target Distribution Analysis
    lines.append("-" * 60)
    lines.append(f"Target Distribution Analysis (Fidelity > {success_threshold}):")
    lines.append(f"{'Rank':<5} {'Target Name':<20} {'Tot. Prob':<10} {'Outcomes (Top 3)'}")
    lines.append("-" * 60)

    target_stats = {}
    for b in branches:
        if b['fidelity'] > success_threshold:
            idx = b.get('target_idx', 0)
            if idx not in target_stats:
                target_stats[idx] = {'prob': 0.0, 'outcomes': []}
            target_stats[idx]['prob'] += b['prob']
            target_stats[idx]['outcomes'].append((b['outcome'], b['prob']))

    sorted_targets = sorted(target_stats.items(), key=lambda x: x[1]['prob'], reverse=True)

    if not sorted_targets:
        lines.append("No branches met the success threshold.")
    else:
        for rank, (idx, stats) in enumerate(sorted_targets):
            stats['outcomes'].sort(key=lambda x: x[1], reverse=True)
            top_outcomes = [str(o[0]) for o in stats['outcomes'][:3]]
            outcome_str = ", ".join(top_outcomes)
            if len(stats['outcomes']) > 3:
                outcome_str += ", ..."
            t_name = target_names[idx] if idx < len(target_names) else f"Target_{idx}"
            lines.append(f"{rank+1:<5} {t_name:<20} {stats['prob']:<10.4f} {outcome_str}")

    return "\n".join(lines)


def main():
    # =========================================================================
    # 1. HYPERPARAMETERS & SIMULATION SETTINGS
    # =========================================================================
    CUTOFF_DIM = 30          # Fock space cutoff dimension
    STEPS = 1                # Depth / time steps of the circuit
    BEAM_WIDTH = 200         # Beam width (max branches tracked during search)
    TIME_INVARIANT = False   # True = identical params across time steps
    MEASURE_CUTOFF = CUTOFF_DIM  # Max photon number cutoff for ancilla measurement
    SUCCESS_THRESHOLD = 0.97     # Fidelity threshold for considering a branch successful
    
    N_GENERATIONS = 200      # Basin-Hopping iterations per run
    N_RUNS = 1               # Number of global optimization attempts
    NUM_PROCESSES = 4        # Parallel process count for optimizer runner

    squeezing = db_to_r(12)  # 12 dB squeezing converted to squeezing parameter r
    csv_path_abs = str(Path(__file__).resolve().parent.parent.parent / "data" / "GKP_core_coefficients.csv")

    print("=" * 80)
    print(" TIME-DOMAIN CIRCUIT OPTIMIZATION (SINGLE RUN) ")
    print("=" * 80)
    print(f"Steps: {STEPS} | Beam Width: {BEAM_WIDTH} | Cutoff Dim: {CUTOFF_DIM} | Initial Squeezing r: {squeezing:.4f}\n")

    # =========================================================================
    # 2. TARGET STATE CONFIGURATIONS (PRESETS)
    # =========================================================================
    # Select which target state(s) to optimize for.

    # Preset A: Cubic Phase State
    preset_cubic = [
        {'class_name': 'CubicPhaseTarget', 'params': {'gamma': -0.2, 'r': -0.7, 'alpha': 1.25}}
    ]

    # Preset B: Core GKP State (mu=0, n_max=4)
    preset_gkp_mu0 = [
        {'class_name': 'CoreGKPTarget', 'params': {'csv_path': csv_path_abs, 'n_max': 4, 'delta_db': 10, 'mu': 0}}
    ]

    # Preset C: Core GKP State (mu=1, n_max=4)
    preset_gkp_mu1 = [
        {'class_name': 'CoreGKPTarget', 'params': {'csv_path': csv_path_abs, 'n_max': 4, 'delta_db': 10, 'mu': 1}}
    ]

    # Preset D: Squeezed Cat States (Even & Odd superpositions)
    preset_sq_cat = [
        {'class_name': 'SqueezedCatTarget', 'params': {'alpha': np.sqrt(6), 'r': 0.5, 'p': 0}},
        {'class_name': 'SqueezedCatTarget', 'params': {'alpha': np.sqrt(6), 'r': 0.5, 'p': 1}}
    ]

    # Preset E: Binomial Code State
    preset_binomial = [
        {'class_name': 'BinomialCodeTarget', 'params': {'N': 2, 'S': 2, 'mu': 0}}
    ]

    # ---> ACTIVE TARGET SELECTION <---
    active_target_configs = preset_sq_cat

    # Instantiate Target Generators
    targets = [create_from_config(cfg, target_module) for cfg in active_target_configs]
    print(f"Loaded {len(targets)} target generator(s):")
    print_targets(targets, CUTOFF_DIM)

    # =========================================================================
    # 3. CIRCUIT CONFIGURATIONS (PRESETS)
    # =========================================================================

    # Preset 1: 3-Mode Squeeze-Only Circuit
    preset_circuit_3m_squeeze = {
        'class_name': 'ThreeModeTimeDomainSqueezeOnly',
        'params': {
            'steps': STEPS,
            'time_invariant': TIME_INVARIANT,
            'clip_size': squeezing,
            'measure_fock_cutoff': MEASURE_CUTOFF,
            'num_single_photon': 0,
            'train_initial_state': True,
            'initial_r': squeezing
        }
    }

    # Preset 2: 2-Mode Squeeze-Only Circuit
    preset_circuit_2m_squeeze = {
        'class_name': 'TwoModeTimeDomainSqueezeOnly',
        'params': {
            'steps': STEPS,
            'time_invariant': TIME_INVARIANT,
            'clip_size': squeezing,
            'measure_fock_cutoff': MEASURE_CUTOFF,
            'num_single_photon': 0,
            'train_initial_state': True,
            'initial_r': squeezing,
            'initial_fock_one': False
        }
    }

    # Preset 3: 3-Mode Time-Domain General
    preset_circuit_3m_General = {
        'class_name': 'ThreeModeTimeDomainGeneral',
        'params': {
            'steps': STEPS,
            'time_invariant': TIME_INVARIANT,
            'clip_size': squeezing,
            'measure_fock_cutoff': MEASURE_CUTOFF,
            'train_initial_state': True,
            'initial_r': squeezing
        }
    }

    # ---> ACTIVE CIRCUIT SELECTION <---
    active_circuit_config = preset_circuit_2m_squeeze

    circuit = create_from_config(active_circuit_config, circuit_module)

    # =========================================================================
    # 4. MEASUREMENT PATTERN SELECTION
    # =========================================================================
    # Options:
    #   patterns = None                     -> Unconstrained Beam Search Pattern Discovery
    #   patterns = [[(o1, o2)], ...]        -> Evaluate fixed set of heralding patterns

    # patterns = [[(0, 6)], [(2, 6)], [(4, 6)], [(8, 6)], [(10, 6)]]
    patterns = [[(4,)],[(5,)],]

    # To enable unconstrained pattern discovery via beam search, uncomment below:
    # patterns = None

    # =========================================================================
    # 5. OUTPUT DIRECTORY SETUP
    # =========================================================================
    c_name = active_circuit_config.get('class_name', '')
    if "General" in c_name: 
        c_tag = "General"
    elif "TwoMode" in c_name and "SqueezeOnly" in c_name: 
        c_tag = "Sq2"
    elif "ThreeMode" in c_name and "SqueezeOnly" in c_name: 
        c_tag = "Sq3"
    elif "FourMode" in c_name and "SqueezeOnly" in c_name: 
        c_tag = "Sq4"
    else: 
        c_tag = "Circ"

    t_tag = "Tgt"
    if active_target_configs and isinstance(active_target_configs, list) and len(active_target_configs) > 0:
        t_first = active_target_configs[0].get('class_name', '')
        if "GKP" in t_first: t_tag = "GKP"
        elif "SqueezedCat" in t_first: t_tag = "SqCat"
        elif "Cat" in t_first: t_tag = "Cat"
        elif "Binomial" in t_first: t_tag = "Bin"
        elif "Cubic" in t_first: t_tag = "Cub"

    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    folder_name = f"opt_{c_tag}_{t_tag}_{timestamp}"
    
    results_dir = Path(__file__).resolve().parent.parent.parent / "results" / folder_name
    results_dir.mkdir(parents=True, exist_ok=True)
    print(f"Results will be written to: {results_dir}\n")

    # Save experiment details
    details_txt = save_experiment_details(
        results_dir, 
        active_circuit_config, 
        active_target_configs, 
        patterns=patterns, 
        beam_width=BEAM_WIDTH
    )
    print(details_txt)

    prepared_patterns = prepare_measurement_patterns(patterns)

    # Calculate non-Gaussianity scores for target states
    print("\nNon-Gaussianity scores for targets:")
    for i, target in enumerate(targets):
        ket = target.get_target_ket(CUTOFF_DIM)
        ng_score = compute_ng_scores([ket], CUTOFF_DIM)[0]
        print(f"  Target {i+1}: {ng_score:.4f}")

    # =========================================================================
    # 6. RUNNER INITIALIZATION & OPTIMIZATION LOOP
    # =========================================================================
    # Select loss function based on measurement strategy and regime
    if prepared_patterns is None:
        active_loss_fn = beam_search_loss_fn
    else:
        active_loss_fn = fixed_pattern_capped_loss_fn  # or fixed_pattern_free_loss_fn

    runner = BasinHoppingRunner(
        num_processes=NUM_PROCESSES,
        circuit=circuit,
        target_gens=targets,
        cutoff_dim=CUTOFF_DIM,
        beam_width=BEAM_WIDTH,
        penalty_strength=1.0,
        measurement_patterns=prepared_patterns,
        loss_fn=active_loss_fn
    )

    results_list = []
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
            target_names.append("UnknownTarget")

    print(f"\nStarting {N_RUNS} optimization run(s) ({N_GENERATIONS} Basin-Hopping iterations per run)...")

    for run_idx in range(N_RUNS):
        print(f"\n--- Optimization Run {run_idx+1}/{N_RUNS} ---")
        try:
            res = runner.run(n_generations=N_GENERATIONS, prob_power=1.0)
            
            branches = res.get('branches', [])
            branches.sort(key=lambda x: x['prob'], reverse=True)

            expected_fidelity = sum(b['prob'] * b['fidelity'] for b in branches)
            success_prob = sum(b['prob'] for b in branches if b['fidelity'] >= SUCCESS_THRESHOLD)

            res['expected_fidelity'] = expected_fidelity
            res['success_prob'] = success_prob
            res['run_index'] = run_idx + 1

            print(f"  * Loss:               {res['loss']:.5f}")
            print(f"  * Expected Fidelity:  {expected_fidelity:.5f}")
            print(f"  * Success Prob (>{SUCCESS_THRESHOLD:.2f}): {success_prob:.2%}")
            print(f"  * Total Beam Prob:    {res.get('total_probability', 0.0):.5f}")
            
            print(f"\n  {'Outcome':<15} {'Prob':<10} {'Fidelity':<10} {'Best Target'}")
            print("  " + "-" * 50)
            for b in branches[:10]:  # Print top 10 branches
                tgt_name = target_names[b['target_idx']] if b['target_idx'] < len(target_names) else f"T{b['target_idx']}"
                print(f"  {str(b['outcome']):<15} {b['prob']:<10.4f} {b['fidelity']:<10.4f} {tgt_name}")

            results_list.append(res)

            # Save per-run artifacts
            run_meta = {
                "run_index": run_idx + 1,
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "prob_power": 1.0,
                "expected_fidelity": float(expected_fidelity),
                "success_prob": float(success_prob),
                "circuit_config": active_circuit_config,
                "target_configs": active_target_configs,
                "measurement_patterns": patterns
            }
            
            run_file = results_dir / f"run_{run_idx+1:04d}.pkl"
            with open(run_file, "wb") as f:
                pickle.dump({"meta": run_meta, "res": res}, f)

            summary = {
                "run_index": run_idx + 1,
                "expected_fidelity": float(expected_fidelity),
                "success_prob": float(success_prob),
                "total_probability": float(res.get("total_probability", 0.0))
            }
            with open(results_dir / f"run_{run_idx+1:04d}_summary.json", "w") as f:
                json.dump(summary, f, indent=2)

            branches_report = format_branches_report(branches, target_names, SUCCESS_THRESHOLD)
            with open(results_dir / f"run_{run_idx+1:04d}_branches.txt", "w") as f:
                f.write(branches_report)

        except Exception as exc:
            print(f"Run {run_idx+1} failed with exception: {exc}")

    if not results_list:
        print("Optimization failed to produce valid results.")
        return

    # =========================================================================
    # 7. SAVE BEST RESULT & SUMMARY REPORTS
    # =========================================================================
    results_list.sort(key=lambda x: x['success_prob'], reverse=True)
    best_res = results_list[0]

    with open(results_dir / "sorted_runs.txt", "w") as f:
        f.write(f"{'Run':<5} {'Success Prob':<15} {'Exp. Fidelity':<15}\n")
        f.write("-" * 40 + "\n")
        for r in results_list:
            f.write(f"{r['run_index']:<5} {r['success_prob']:<15.5f} {r['expected_fidelity']:<15.5f}\n")

    mapped_params = circuit.map_parameters(best_res['x'])
    param_names = circuit.per_step_parameter_names

    best_dir = results_dir / "best"
    best_dir.mkdir(exist_ok=True)

    best_run_idx = best_res['run_index']
    src_pkl = results_dir / f"run_{best_run_idx:04d}.pkl"
    if src_pkl.exists():
        shutil.copy(src_pkl, best_dir / f"best_run_{best_run_idx:04d}.pkl")

    np.save(best_dir / "best_x.npy", best_res['x'])
    np.savez(best_dir / "mapped_params.npz", mapped_params=mapped_params)

    schedule = {
        "param_names": param_names,
        "mapped_params": mapped_params.tolist() if hasattr(mapped_params, "tolist") else [[float(v) for v in row] for row in mapped_params],
        "circuit_config": active_circuit_config,
        "target_configs": active_target_configs
    }
    with open(best_dir / "schedule.json", "w") as f:
        json.dump(schedule, f, indent=2)

    branches_report = format_branches_report(best_res['branches'], target_names, SUCCESS_THRESHOLD)
    with open(best_dir / "best_branches.txt", "w") as f:
        f.write(branches_report)

    # Print Final Summary
    print("\n" + "=" * 80)
    print(" OPTIMIZATION SUMMARY (BEST RESULT) ")
    print("=" * 80)
    print(f"Best Run Index:    {best_res['run_index']}")
    print(f"Final Loss:        {best_res['loss']:.5f}")
    print(f"Expected Fidelity: {best_res['expected_fidelity']:.5f}")
    print(f"Success Prob:      {best_res['success_prob']:.2%}")
    print(f"Total Beam Prob:   {best_res.get('total_probability', 0.0):.5f}")
    print("-" * 80)
    print("\nOptimized Parameter Schedule:")
    header = f"{'Step':<6} | " + " | ".join([f"{name:<12}" for name in param_names])
    print(header)
    print("-" * len(header))

    for t in range(STEPS):
        vals = mapped_params[t]
        val_strs = [f"{v:12.6f}" for v in vals]
        print(f"{t:<6} | " + " | ".join(val_strs))

    print("-" * 80)
    print("\nDetailed Outcome Breakdown:")
    print(branches_report)
    print("=" * 80)
    print(f"\nAll outputs successfully saved to: {results_dir}\n")


if __name__ == "__main__":
    main()
