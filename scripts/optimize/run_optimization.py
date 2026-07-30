"""Single-Configuration Time-Domain Photonic Circuit Optimizer.

This script executes global parameter optimization for single configurations of
continuous-variable (CV) photonic quantum state engineering circuits using the
Basin-Hopping algorithm with local L-BFGS-B / Nelder-Mead minimizers.

Overview & Workflow
-------------------
The script supports two primary operational regimes for state heralding:

1. **Phase 1: Unconstrained Pattern Discovery (Beam Search)**
   Set ``patterns = None`` to enable Beam Search pattern discovery. The runner
   dynamically explores the tree of photon-number-resolving (PNR) detection
   outcomes across ancillary modes, tracking the top ``beam_width`` most probable
   trajectories and evaluating state quality using the non-linear score metric
   :func:`~heraldo.components.runner.beam_search_loss_fn`.

2. **Phase 2: Fixed-Pattern Optimization & Refinement**
   Set ``patterns = [[(n1, n2, ...)], ...]`` to optimize circuit parameters
   for pre-determined PNR detection sequences. This phase supports:
   - **Resource Multiplexing**: Optimizing a single circuit to generate distinct
     target states across different detection events (e.g., even cat on n=4, odd cat on n=5).
   - **Single-Target Probability Harvesting**: Aggregating degenerate measurement
     outcomes that all herald the same target state (e.g., harvesting (1,3), (3,1), and (2,2)
     for GKP logical zero).

Generated Artifacts & Output Directory Structure
------------------------------------------------
Optimizations automatically export results to an output folder named:
``results/opt_<CircuitTag>_<TargetTag>_<Timestamp>/``

The directory contains:
- ``experiment_details.txt``: Plain-text log of circuit, target, and measurement settings.
- ``run_0001.pkl``: Pickle file containing metadata and optimization results.
- ``run_0001_summary.json``: JSON summary of run statistics (fidelity, total probability).
- ``run_0001_branches.txt``: Formatted table of outcome probabilities and target fidelities.
- ``sorted_runs.txt``: Ranked summary of optimization attempts.
- ``best/`` subfolder (legacy output structure mirroring single-run outputs):
  - ``best_x.npy``: Raw 1D NumPy array of optimal circuit parameters.
  - ``mapped_params.npz``: 2D array of per-step control parameters.
  - ``schedule.json``: Human-readable parameter schedule mapped by gate name.
  - ``best_branches.txt``: Outcome details for the run.
  - ``best_run_0001.pkl``: Copy of the result pickle.

.. note::
   This script runs a single optimization pass (``N_RUNS = 1``). Saving the output into
   a ``best/`` subfolder is a legacy output convention maintained for compatibility
   with downstream evaluation scripts (e.g., ``eval_time_optimized.py``).

Execution
---------
Run directly from the repository root:

.. code-block:: bash

    python scripts/optimize/run_time_optimization.py
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
from typing import List, Dict, Tuple, Any, Optional, Union
import numpy as np

# --- Set thread limits for NumPy/OpenBLAS/MKL before importing heavy backend libraries ---
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
    BinomialCodeTarget, CubicPhaseTarget,
)
from heraldo.utils import db_to_r
from heraldo.factory import create_from_config


def prepare_measurement_patterns(patterns: Optional[Union[List, np.ndarray]]) -> Optional[np.ndarray]:
    """Converts user-specified measurement patterns into a normalized NumPy array.

    Args:
        patterns (list or np.ndarray, optional): List or array of measurement pattern sequences.
            Each sequence specifies a list of photon-count tuples per time step,
            e.g., ``[[(4,)], [(5,)]]`` for a 1-step 2-mode circuit or ``[[(1, 3)]]`` for 3-mode.

    Returns:
        np.ndarray or None: Dense 3D NumPy array of shape ``(n_sequences, steps, n_meas_modes)``,
        or ``None`` if ``patterns`` is None (triggering Beam Search discovery mode).
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
        # Fallback for non-standard pattern shapes
        return patterns


def generate_measurement_patterns(circuit: TimeMultiplexedCircuit, 
                                  min_total_photons: Optional[int] = None, 
                                  max_total_photons: Optional[int] = None, 
                                  exact_total: Optional[int] = None,
                                  max_per_mode: Optional[int] = None) -> List[List[Tuple[int, ...]]]:
    """Combinatorially generates all valid measurement patterns subject to photon count constraints.

    Args:
        circuit (TimeMultiplexedCircuit): Target circuit model to generate patterns for.
        min_total_photons (int, optional): Minimum sum of detected photons across all modes and steps.
        max_total_photons (int, optional): Maximum sum of detected photons across all modes and steps.
        exact_total (int, optional): Exact sum of detected photons required (overrides min/max if specified).
        max_per_mode (int, optional): Upper bound on photon number measured by a single detector.

    Returns:
        List[List[Tuple[int, ...]]]: List of valid outcome patterns, where each pattern is a
        list containing one tuple per step with photon numbers for each measured mode.
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


def print_targets(targets: List[TargetGenerator], cutoff_dim: int, tolerance: float = 1e-6) -> None:
    """Prints diagnostic labels and significant Fock-basis expansion coefficients for target states.

    Args:
        targets (List[TargetGenerator]): List of initialized target generator instances.
        cutoff_dim (int): Fock space truncation dimension used to expand the target ket.
        tolerance (float, optional): Absolute amplitude threshold below which coefficients are omitted.
            Defaults to 1e-6.
    """
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


def save_experiment_details(results_dir: Path, 
                            circuit_config: Dict[str, Any], 
                            target_configs: List[Dict[str, Any]], 
                            patterns: Optional[Union[List, np.ndarray]] = None, 
                            beam_width: Optional[int] = None) -> str:
    """Exports a comprehensive plain-text summary of the optimization configuration.

    Args:
        results_dir (Path): Output directory where ``experiment_details.txt`` will be saved.
        circuit_config (dict): Configuration dictionary defining the circuit class and parameters.
        target_configs (list[dict]): Configuration dictionaries defining the target state generators.
        patterns (list or np.ndarray, optional): Fixed measurement patterns array, or None if Beam Search.
        beam_width (int, optional): Trajectory beam width if running in Beam Search mode.

    Returns:
        str: Formatted text string containing the experiment details.
    """
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


def format_branches_report(branches: List[Dict[str, Any]], 
                           target_names: List[str], 
                           success_threshold: float) -> str:
    """Formats branch probability, state fidelity, and target mapping into a plain-text table.

    Args:
        branches (List[dict]): List of dictionary records returned by the circuit runner.
            Each dictionary contains keys ``"outcome"``, ``"prob"``, ``"fidelity"``, and ``"target_idx"``.
        target_names (List[str]): List of human-readable target labels corresponding to target indices.
        success_threshold (float): Minimum state fidelity threshold used to define successful heralding events.

    Returns:
        str: Formatted multi-line text report summarizing branch performance and target distribution.
    """
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


def main() -> None:
    """Executes single-configuration time-domain circuit optimization."""
    # =========================================================================
    # STAGE 1: HYPERPARAMETERS & SIMULATION SETTINGS
    # =========================================================================
    CUTOFF_DIM = 30          # Fock space truncation dimension
    STEPS = 1                # Recirculation depth / spatial stages (steps=1 matches paper model)
    BEAM_WIDTH = 200         # Beam search width (max branches tracked during discovery)
    TIME_INVARIANT = False   # True = identical gate parameters across steps; False = step-dependent
    MEASURE_CUTOFF = CUTOFF_DIM  # PNR detector cutoff dimension for ancillary modes
    SUCCESS_THRESHOLD = 0.97     # Minimum fidelity threshold defining a successful outcome
    
    N_GENERATIONS = 200      # Basin-Hopping global iterations per run
    N_RUNS = 1               # Single-run execution (N_RUNS = 1; 'best/' folder export is a legacy structure)
    NUM_PROCESSES = 4        # Parallel worker processes for BasinHoppingRunner

    squeezing = db_to_r(12)  # Convert 12 dB physical squeezing to squeezing parameter r (~1.38)
    csv_path_abs = str(Path(__file__).resolve().parent.parent.parent / "data" / "GKP_core_coefficients.csv")

    print("=" * 80)
    print(" TIME-DOMAIN CIRCUIT OPTIMIZATION (SINGLE RUN) ")
    print("=" * 80)
    print(f"Steps: {STEPS} | Beam Width: {BEAM_WIDTH} | Cutoff Dim: {CUTOFF_DIM} | Initial Squeezing r: {squeezing:.4f}\n")

    # =========================================================================
    # STAGE 2: TARGET STATE CONFIGURATIONS (PRESETS)
    # =========================================================================
    # Select which target quantum state(s) to optimize for:

    # Preset A: Displaced Cubic Phase State
    preset_cubic = [
        {'class_name': 'CubicPhaseTarget', 'params': {'gamma': -0.2, 'r': -0.7, 'alpha': 1.25}}
    ]

    # Preset B: Core GKP State (Logical 0, n_max=4 stellar rank, 10 dB envelope)
    preset_gkp_mu0 = [
        {'class_name': 'CoreGKPTarget', 'params': {'csv_path': csv_path_abs, 'n_max': 4, 'delta_db': 10, 'mu': 0}}
    ]

    # Preset C: Core GKP State (Logical 1, n_max=4 stellar rank, 10 dB envelope)
    preset_gkp_mu1 = [
        {'class_name': 'CoreGKPTarget', 'params': {'csv_path': csv_path_abs, 'n_max': 4, 'delta_db': 10, 'mu': 1}}
    ]

    # Preset D: Squeezed Schrödinger Cat States (Even & Odd superpositions)
    preset_sq_cat = [
        {'class_name': 'SqueezedCatTarget', 'params': {'alpha': np.sqrt(6), 'r': 0.5, 'p': 0}},
        {'class_name': 'SqueezedCatTarget', 'params': {'alpha': np.sqrt(6), 'r': 0.5, 'p': 1}}
    ]

    # Preset E: Binomial Quantum Code State (N=2, S=2, Logical 0)
    preset_binomial = [
        {'class_name': 'BinomialCodeTarget', 'params': {'N': 2, 'S': 2, 'mu': 0}}
    ]

    # ---> ACTIVE TARGET SELECTION <---
    active_target_configs = preset_sq_cat

    # Instantiate Target Generators using the factory module
    targets = [create_from_config(cfg, target_module) for cfg in active_target_configs]
    print(f"Loaded {len(targets)} target generator(s):")
    print_targets(targets, CUTOFF_DIM)

    # =========================================================================
    # STAGE 3: CIRCUIT ARCHITECTURE CONFIGURATIONS (PRESETS)
    # =========================================================================

    # Preset 1: 3-Mode Squeeze-Only Circuit (1 Loop Mode + 2 Ancillae)
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

    # Preset 2: 2-Mode Squeeze-Only Circuit (1 Loop Mode + 1 Ancilla)
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

    # Preset 3: 3-Mode General Circuit (Includes displacement gates on ancillae)
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
    # STAGE 4: MEASUREMENT PATTERN & OPTIMIZATION MODE SELECTION
    # =========================================================================
    # Options:
    #   patterns = None                     -> Phase 1: Beam Search Pattern Discovery Mode
    #   patterns = [[(n1, n2)], ...]        -> Phase 2: Fixed-Pattern Optimization / Refinement

    patterns = [[(4,)], [(5,)]]

    # To run Beam Search Pattern Discovery instead, set patterns to None:
    # patterns = None

    # =========================================================================
    # STAGE 5: RESULTS DIRECTORY SETUP & EXPERIMENT LOGGING
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

    # Save plain-text configuration metadata
    details_txt = save_experiment_details(
        results_dir, 
        active_circuit_config, 
        active_target_configs, 
        patterns=patterns, 
        beam_width=BEAM_WIDTH
    )
    print(details_txt)

    prepared_patterns = prepare_measurement_patterns(patterns)

    # =========================================================================
    # STAGE 6: LOSS FUNCTION SELECTION & RUNNER INITIALIZATION
    # =========================================================================
    # Choose appropriate objective loss function based on measurement mode
    if prepared_patterns is None:
        active_loss_fn = beam_search_loss_fn
    else:
        active_loss_fn = fixed_pattern_capped_loss_fn  # Capped regime (f_cap=0.95)

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

    # =========================================================================
    # STAGE 7: OPTIMIZATION RUN EXECUTION
    # =========================================================================
    print(f"\nStarting optimization run ({N_GENERATIONS} Basin-Hopping iterations)...")

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
            for b in branches[:10]:  # Display top 10 outcome branches
                tgt_name = target_names[b['target_idx']] if b['target_idx'] < len(target_names) else f"T{b['target_idx']}"
                print(f"  {str(b['outcome']):<15} {b['prob']:<10.4f} {b['fidelity']:<10.4f} {tgt_name}")

            results_list.append(res)

            # Export run metadata and results
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
    # STAGE 8: RESULTS EXPORT & LEGACY 'BEST' SUBFOLDER POPULATION
    # =========================================================================
    # Note: Saving results into the 'best/' subfolder is a legacy convention maintained
    # for compatibility with evaluation scripts (e.g., eval_time_optimized.py).
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

    # Print Final Summary & Parameter Table
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
