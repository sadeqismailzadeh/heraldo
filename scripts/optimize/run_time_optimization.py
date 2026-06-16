"""
Script to run time-domain Beam Search optimization for a loop-based gadget.
"""


import os
import re
import glob
import platform
import multiprocessing as mp
from pathlib import Path

# --- CRITICAL: Set thread limits BEFORE importing other libraries ---
print("--- Configuring thread limits for NumPy/OpenBLAS/MKL ---")
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

import numpy as np
import time
import shutil
from pathlib import Path
from sklearn.cluster import KMeans
import itertools
import warnings


# 3. Suppress SciPy numerical overflow warnings during local search/polishing
# These occur when the optimizer hits a "cliff" in the loss landscape.
warnings.filterwarnings("ignore", category=RuntimeWarning, module="scipy.optimize")


# Imports
import quantum_agent.optimization.time_circuits as circuit_module
import quantum_agent.components.targets as target_module
from quantum_agent.optimization.time_circuits import *
from quantum_agent.optimization.time_runner import  BasinHoppingRunner
from quantum_agent.components.targets import *
from quantum_agent.utils import *
from quantum_agent.factory import create_from_config

import json
import pickle
from datetime import datetime


def prepare_measurement_patterns(patterns):
    """
    Convert user-friendly list/tuple measurement patterns into a NumPy array
    when possible for faster evaluation.

    Accepted inputs:
      - list of sequences: [[(o11, o12), (o21, o22), ...], ...]
      - single sequence:   [(o11, o12), (o21, o22), ...]
      - np.ndarray with shape (n_seq, steps, n_meas_modes)
      - np.ndarray with shape (steps, n_meas_modes)

    Returns:
      - np.ndarray if conversion is possible
      - original input (list/tuple) if patterns are ragged
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
        # Ragged or irregular structure → keep Python lists (slow path)
        return patterns



def generate_measurement_patterns(circuit: TimeMultiplexedCircuit, min_total_photons: int = None, 
                                  max_total_photons: int = None, exact_total: int = None,
                                  max_per_mode: int = None):
    """
    Generate all possible measurement patterns for a given circuit with constraints.
    
    Args:
        circuit: TimeMultiplexedCircuit instance
        min_total_photons: Minimum total photon count across all measurements
        max_total_photons: Maximum total photon count across all measurements
        exact_total: Exact total photon count (overrides min/max if specified)
        max_per_mode: Maximum photons per measurement outcome (defaults to cutoff - 1)
    
    Returns:
        List of patterns, where each pattern is a list of tuples (for each step)
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
    
    # Get cutoffs for each measured mode
    cutoffs = [cutoff for _, cutoff in meas_specs]
    
    # Determine maximum per mode
    if max_per_mode is None:
        max_per_mode_list = [c - 1 for c in cutoffs]
    else:
        max_per_mode_list = [max_per_mode] * num_measured_modes
    
    # Generate all possible outcomes for one step
    single_step_outcomes = []
    ranges = []
    for i in range(num_measured_modes):
        ranges.append(range(0, min(max_per_mode_list[i] + 1, cutoffs[i])))
    
    # Generate cartesian product of all ranges
    for combo in itertools.product(*ranges):
        single_step_outcomes.append(combo)
    
    # Now generate all patterns for all steps
    all_patterns = []
    
    # Generate all combinations of steps
    for step_combo in itertools.product(single_step_outcomes, repeat=circuit.steps):
        # Calculate total photons
        total_photons = sum(sum(step) for step in step_combo)
        
        # Check if total is within bounds
        if min_total_photons <= total_photons <= max_total_photons:
            all_patterns.append(list(step_combo))
    
    return all_patterns



def filter_zero_slot_arrays(targets: list[TargetGenerator], cutoff_dim, tolerance=1e-6):
    """
    Filters a list of arrays, keeping only those where exactly one element is zero
    (within a specified tolerance).

    Args:
        arrays (list of numpy arrays): The input list of arrays.
        tolerance (float): The tolerance value for comparing floating-point numbers to zero.

    Returns:
        list of numpy arrays: A new list containing only the arrays that meet the criteria.
    """
    filtered_arrays = []
    for target in targets:
        if np.count_nonzero(target.get_target_ket(cutoff_dim) > tolerance) > 1:
            filtered_arrays.append(target)
    return filtered_arrays

def print_targets(targets, cutoff_dim, tolerance=1e-6):
    """Print target names and ket states for each target."""
    
    # Iterate over each target
    for i, target in enumerate(targets):
        # Determine the target name
        if isinstance(target, CoreGKPTarget):
            target_name = f"GKP_n{target.n_max}_mu{target.mu}"
        elif isinstance(target, SqueezedCatTarget):
            target_name = f"Cat_a{target.alpha}_p{target.p}"
        elif isinstance(target, BinomialCodeTarget):
            target_name = f"Binomial_N{target.N}_S{target.S}_mu{target.mu}"
        else:
            target_name = "UnknownTarget"
        
        # Print target name
        print(f"\nTarget {i+1}: {target_name}")
        print("-" * len(target_name))
        
        # Get and print ket states
        ket_state = target.get_target_ket(cutoff_dim)
        for n, val in enumerate(ket_state):
            if np.abs(val) > tolerance:
                # Print real part if imaginary is negligible, otherwise show complex
                out_val = val.real if np.abs(val.imag) < 1e-8 else val
                print(f"  |{n}>: {out_val:.6f}")
        
        print("\n")

def save_experiment_details(results_dir: Path, circuit_config: dict, target_configs: list, patterns=None, beam_width=None):
    """Saves a pretty-printed summary of the experiment configuration to a text file."""
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
        lines.append(f"Target {i}: {cfg.get('class_name', 'Unknown')}")
        t_params = cfg.get('params', {})
        for pk, pv in t_params.items():
            lines.append(f"  - {pk:<23}: {pv}")
        lines.append("")

    lines.append("--- BRANCHING STRATEGY ---")
    if patterns is not None:
        n_p = len(patterns)
        lines.append(f"Fixed Measurement Patterns ({n_p} total):")
        # Convert to list if it's a numpy array for easier printing
        p_to_print = patterns.tolist() if hasattr(patterns, 'tolist') else patterns
        for i, p in enumerate(p_to_print):
            if i < 100:  # Limit printing for very large sets
                lines.append(f"  {i:3}: {p}")
            else:
                lines.append(f"  ... and {n_p - 100} more patterns.")
                break
    else:
        lines.append(f"Beam Search Width: {beam_width}")
    lines.append("")
    
    lines.append("=" * 80)
    
    content = "\n".join(lines)
    with open(results_dir / "experiment_details.txt", "w") as f:
        f.write(content)
    return content


def format_branches_report(branches, target_names, success_threshold):
    """Returns a formatted string of branch statistics."""
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

    # Target Analysis
    lines.append("-" * 60)
    lines.append(f"Target Distribution Analysis (Success > {success_threshold}):")
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
    # --- Configuration ---
    CUTOFF_DIM = 30          # Simulation cutoff
    STEPS = 1                # Time steps (depth of the circuit)
    BEAM_WIDTH = 200          # Number of branches to keep
    TIME_INVARIANT = False   # False = different params per step
    MEASURE_CUTOFF = CUTOFF_DIM       # Max Fock state to measure on Ancilla (0, 1)
    SUCCESS_THRESHOLD = 1 - 3e-2
    
    # --- Execution ---
    n_generations = 20       # Number of hops per global search
    niter = 1                # Number of global searches

    # Setup
    print("--- Setting up Time-Domain Optimization ---")
    print(f"Steps: {STEPS}, Beam Width: {BEAM_WIDTH}, Cutoff: {CUTOFF_DIM}")

    squeezing = db_to_r(12)
    print(f"squeezing r = {squeezing}")

    # -------------------------------------------------------------------------
    # 1. Define Target Configs
    # -------------------------------------------------------------------------

    # 1.1 Squeezed Cat
    target_configs1 = [
        {'class_name': 'SqueezedCatTarget', 'params': {'alpha': 3, 'r': squeezing, 'p': 0}},
        {'class_name': 'SqueezedCatTarget', 'params': {'alpha': 3, 'r': squeezing, 'p': 1}}
    ]

    # # 1.2 Cat
    # target_configs_cat = [
    #     {'class_name': 'CatTarget', 'params': {'alpha': 2, 'p': 0}},
    #     {'class_name': 'CatTarget', 'params': {'alpha': 2, 'p': 1}}
    # ]

    target_configs_cat2 = [
        {'class_name': 'SqueezedCatTarget', 'params': {'alpha': np.sqrt(6), 'r': 0.5, 'p': 0}},
        {'class_name': 'SqueezedCatTarget', 'params': {'alpha': np.sqrt(6), 'r': 0.5, 'p': 1}}
    ]

    # target_configs_cat3 = [
    #     {'class_name': 'SqueezedCatTarget', 'params': {'alpha': np.sqrt(8), 'r': 0.5, 'p': 0}},
    #     {'class_name': 'SqueezedCatTarget', 'params': {'alpha': np.sqrt(8), 'r': 0.5, 'p': 1}}
    # ]


    # 1.6 Cubic Phase Target
    cubic_phase_config = [
        {'class_name': 'CubicPhaseTarget', 'params': {'gamma': -0.2, 'r': -0.7, 'alpha': 1.25}}
    ]

    # 1.7 Tri-squeezed and Quad-squeezed Targets
    higher_order_squeezed_configs = [
        {'class_name': 'TrisqueezedTarget', 'params': {}},
        {'class_name': 'QuadsqueezedTarget', 'params': {}}
    ]

    # 1.3 GKP
    csv_path_abs = Path(__file__).resolve().parent.parent.parent / "data" / "GKP_core_coefficients.csv"
    gkp_target_configs = [
        {'class_name': 'CoreGKPTarget', 'params': {'csv_path': str(csv_path_abs), 'n_max': n, 'delta_db': 10, 'mu': m}}
        # for n in [8, 12] for m in [0]
        for n in [4, 6, 8, 10, 12] for m in [0, 1]
    ]

    # 1.4 Binomial
    binomial_target_configs_all = []
    max_fock_n = 14
    for S in range(2, max_fock_n):
        for N in range(2, max_fock_n):
            if (N + 1) * (S + 1) <= max_fock_n:
                binomial_target_configs_all.append({'class_name': 'BinomialCodeTarget', 'params': {'N': N, 'S': S, 'mu': 0}})
                binomial_target_configs_all.append({'class_name': 'BinomialCodeTarget', 'params': {'N': N, 'S': S, 'mu': 1}})
    
    # 1.5 Single Core GKP
    target_config_3 = [{'class_name': 'CoreGKPTarget', 'params': {'csv_path': str(csv_path_abs), 'n_max': 4, 'delta_db': 10, 'mu': 0}}]
    
    # 1.6 Cubic
    target_config_cubic = [{'class_name': 'CubicResourceTarget', 'params': {'a': 0.61}}]

    # --- Select Active Target Configs ---
    
    # Here we select which group we want to use. 
    # NOTE: Binomial needs filtering, handled below.
    
    active_target_configs =  target_config_3
    # active_target_configs = target_configs1
    # active_target_configs = binomial_target_configs_all # Needs filtering below

    # --- Instantiation and Filtering ---
    
    targets = []
    final_target_configs = []

    # Binomial special case for filtering:
    # We instantiate all, filter instances, and keep corresponding configs
    if active_target_configs == binomial_target_configs_all:
        temp_targets = [create_from_config(cfg, target_module) for cfg in active_target_configs]
        for t, cfg in zip(temp_targets, active_target_configs):
             # filter_zero_slot_arrays logic inlined or called (returns new list)
             # Easier to inline check here:
             if np.count_nonzero(t.get_target_ket(CUTOFF_DIM) > 1e-6) > 1:
                 targets.append(t)
                 final_target_configs.append(cfg)
    else:
        # Standard instantiation
        for cfg in active_target_configs:
            targets.append(create_from_config(cfg, target_module))
        final_target_configs = active_target_configs

    # Overwrite the 'targets' variable used later
    print(f"Optimizing for {len(targets)} targets.")
    print_targets(targets, CUTOFF_DIM)

    # -------------------------------------------------------------------------
    # 2. Define Circuit Configs
    # -------------------------------------------------------------------------

    circuit_config_gadget = {
        'class_name': 'TwoModeTimeDomainGadget',
        'params': {
            'steps': STEPS,
            'time_invariant': TIME_INVARIANT,
            'clip_size': squeezing,
            'measure_fock_cutoff': MEASURE_CUTOFF,
            'train_initial_state': True,
            'initial_r': squeezing
        }
    }

    circuit_config_gadget_2 = {
        'class_name': 'ThreeModeTimeDomainGadget',
        'params': {
            'steps': STEPS,
            'time_invariant': TIME_INVARIANT,
            'clip_size': squeezing,
            'measure_fock_cutoff': MEASURE_CUTOFF,
            'train_initial_state': True,
            'initial_r': squeezing
        }
    }


    circuit_config_1 = {
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

    circuit_config_2 = {
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

    circuit_config_3 = {
        'class_name': 'FourModeTimeDomainSqueezeOnly',
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

    # --- Select Active Circuit ---
    active_circuit_config = circuit_config_2
    
    # --- Results Directory Setup ---
    # Generate short tags for folder name based on active configs
    c_name = active_circuit_config.get('class_name', '')
    if "Gadget" in c_name: 
        c_tag = "Gadget"
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
    print(f"Saving per-run results to: {results_dir}")

    # Instantiate Circuit
    circuit = create_from_config(active_circuit_config, circuit_module)
    
    # -------------------------------------------------------------------------
    # Patterns and Scores
    # -------------------------------------------------------------------------
    patterns = None

    # patterns = generate_measurement_patterns(circuit, exact_total=3) + \
    #  generate_measurement_patterns(circuit, exact_total=4) + \
    #      generate_measurement_patterns(circuit, exact_total=6) + \
    #       generate_measurement_patterns(circuit, exact_total=8) + \
    #        generate_measurement_patterns(circuit, exact_total=10) 
   
    # patterns = generate_measurement_patterns(circuit, exact_total=4) 

            
    # patterns = [(2,4)]
    # patterns = [[(4,)]]

    # patterns = [[(2,2)],[(1,3)],]
    # patterns = [[(3,)], [(4,)], [(6,)], [(8,)], [(10,)]] 
    # patterns = [[(1, 3)], [(0, 4)], [(1, 4)], [(0, 5)], [(2, 2)], [(2, 3)], [(3, 2)],]
    # patterns = [[(0,6)], [(2,6)], [(4,6)], [(8,6)], [(10,6)]]
    # patterns = [[(2,2)], [(3,3)], [(4,4)],[(5,5)], [(6,6)]]
    # patterns = [[(2,2)], [(4,4)], [(6,6)]]

    # patterns = [[(2,4)]]

    # patterns = [[(1,)]]
    patterns = [[(1,3)], [(3,1)],]
    # patterns = [[(0,5)], [(5,0)]]

    # patterns = [[(1,2)], [(2,1)]]
    # patterns = None
    print(f"Measurement patterns: {patterns}")

    # Save pretty-printed experiment details
    details_txt = save_experiment_details(
        results_dir, 
        active_circuit_config, 
        final_target_configs, 
        patterns=patterns, 
        beam_width=BEAM_WIDTH
    )
    print("\nExperiment Configuration Summary:")
    print(details_txt)

    patterns = prepare_measurement_patterns(patterns)
    
    
    print("\nNon-Gaussianity scores for targets:")
    for i, target in enumerate(targets):
        ket = target.get_target_ket(CUTOFF_DIM)
        ng_score = compute_ng_scores([ket], CUTOFF_DIM)[0]
        print(f"  Target {i+1}: {ng_score:.4f}")

    # 3. Runner
    runner = BasinHoppingRunner(
        num_processes=4,
        circuit=circuit,
        target_gens=targets,
        cutoff_dim=CUTOFF_DIM,
        beam_width=BEAM_WIDTH,
        penalty_strength=1,
        success_threshold=SUCCESS_THRESHOLD,
        success_weight=0.0,
        ng_weight=0.,
        ng_threshold=0,
        sigma0=1,
        photon_dist_weight=0,
        max_photon_dist=CUTOFF_DIM,
        measurement_patterns=patterns,
        popsize=15
    )
    


    suc_pb_ls = []
    hpx = []
    results_ls = []

    print(f"Starting {niter} global optimization runs (each with {n_generations} hops)...")

    target_names = []
    for t in targets:
        if isinstance(t, CoreGKPTarget):
            target_names.append(f"GKP_n{t.n_max}_mu{t.mu}")
        elif isinstance(t, SqueezedCatTarget):
            target_names.append(f"Sq_cat_a{t.alpha}_r{t.r}_p{t.p}")
        elif isinstance(t, CatTarget):
            target_names.append(f"Cat_a{t.alpha}_p{t.p}")
        elif isinstance(t, BinomialCodeTarget):
            target_names.append(f"Binomial_N{t.N}_S{t.S}_mu{t.mu}")
        elif isinstance(t, CubicPhaseTarget):
            target_names.append(f"CubicPh_g{t.gamma}_r{t.r}")
        elif isinstance(t, TrisqueezedTarget):
            target_names.append("TriSq")
        elif isinstance(t, QuadsqueezedTarget):
            target_names.append("QuadSq")
        else:
            target_names.append("UnknownTarget")

    prob_powers=[1, 0.7, 0.5, 0.3, 0.2]
    for e in range(niter):
        print(f"Global explore {e+1}/{niter}")
        try:
            # prob_power = np.random.uniform(0.01, 1)

            # Sample prob_power from log distribution between 0.01 and 1.0
            prob_power = 10 ** np.random.uniform(-2, 1)
            # prob_power = prob_powers[e]
            prob_power = 1

            print(f"prob_power = {prob_power:.5f}")
            
            res = runner.run(n_generations=n_generations, prob_power=prob_power)
            
            # Recalculate expected fidelity from branches
            # (TimeDomainRunner objective is -ExpFid + Penalty, but we want pure ExpFid for stats)
            branches = res.get('branches', [])
            # Sort branches by probability (descending)
            branches.sort(key=lambda x: x['prob'], reverse=True)

            expected_fidelity = sum(b['prob'] * b['fidelity'] for b in branches)
            
            # Inject back into result dict for later use
            res['expected_fidelity'] = expected_fidelity

            success_prob = sum(b['prob'] for b in branches if b['fidelity'] > SUCCESS_THRESHOLD)
            
            # Inject back into result dict
            res['success_prob'] = success_prob
            res['run_index'] = e + 1

            print(f"  -> Final Expected Fidelity: {expected_fidelity:.5f}")
            print(f"  -> Success Prob (> {SUCCESS_THRESHOLD}): {success_prob:.5f}")
            print(f"  -> Total Beam Prob: {res.get('total_probability', 0.0):.5f}")
            print(f"  {'Outcome':<15} {'Prob':<10} {'Fidelity':<10} {'Best Target'}")
            
            for b in branches:
                #  if b['prob'] > 0.001:
                     tgt_name = target_names[b['target_idx']] if b['target_idx'] < len(target_names) else f"T{b['target_idx']}"
                     print(f"  {str(b['outcome']):<15} {b['prob']:<10.4f} {b['fidelity']:<10.4f} {tgt_name}")
            print("")

            suc_pb_ls.append(success_prob)
            hpx.append(res['x'])
            results_ls.append(res)

            # --- Save per-run result (pickle + small JSON summary) ---
            try:
                run_meta = {
                    "run_index": e+1,
                    "timestamp": datetime.utcnow().isoformat() + "Z",
                    "prob_power": prob_power,
                    "expected_fidelity": float(expected_fidelity),
                    "success_prob": float(success_prob),
                    "circuit_config": active_circuit_config,   # <--- Save Circuit Recipe
                    "target_configs": final_target_configs,     # <--- Save Target Recipes
                    "measurement_patterns": patterns
                }
                run_file = results_dir / f"run_{e+1:04d}.pkl"
                with open(run_file, "wb") as f:
                    # store both meta and raw res for later inspection
                    pickle.dump({"meta": run_meta, "res": res}, f)

                summary = {
                    "run_index": e+1,
                    "expected_fidelity": float(expected_fidelity),
                    "success_prob": float(success_prob),
                    "total_probability": float(res.get("total_probability", 0.0))
                }
                with open(results_dir / f"run_{e+1:04d}_summary.json", "w") as f:
                    json.dump(summary, f, indent=2)

                # Save pretty-printed branch details for this run
                branches_report = format_branches_report(branches, target_names, SUCCESS_THRESHOLD)
                with open(results_dir / f"run_{e+1:04d}_branches.txt", "w") as f:
                    f.write(branches_report)

                print(f"run {e+1} saved to {results_dir}")
            except Exception as save_exc:
                print(f"  Warning: failed to save run {e+1} result: {save_exc}")
            
        except Exception as exc:
            print(f"Run {e+1} failed: {exc}")

    # Convert to arrays
    suc_pb_ls = np.array(suc_pb_ls)
    hpx = np.array(hpx)  # Array of flat parameters

    # Filter NaNs
    valid_mask = ~np.isnan(suc_pb_ls)
    suc_pb_ls = suc_pb_ls[valid_mask]
    
    # Filter results list as well (hpx might be ragged if filtering happens, but usually fixed size)
    # We just rebuild results_ls based on mask
    results_ls = [r for i, r in enumerate(results_ls) if valid_mask[i]]
    if len(hpx) > 0:
        hpx = hpx[valid_mask]

    if len(suc_pb_ls) == 0:
        print("All runs failed.")
        return

    # Sort results list by success probability
    results_ls.sort(key=lambda x: x['success_prob'], reverse=True)

    # Save sorted list of runs
    with open(results_dir / "sorted_runs.txt", "w") as f:
        f.write(f"{'Run':<5} {'Success Prob':<15} {'Exp. Fidelity':<15}\n")
        f.write("-" * 40 + "\n")
        for r in results_ls:
            f.write(f"{r['run_index']:<5} {r['success_prob']:<15.5f} {r['expected_fidelity']:<15.5f}\n")

    # Save detailed sorted list of runs
    with open(results_dir / "detailed_sorted_runs.txt", "w") as f:
        for r in results_ls:
            f.write(f"Run {r['run_index']} (Success Prob: {r['success_prob']:.5f}, Exp. Fidelity: {r['expected_fidelity']:.5f})\n")
            f.write(f"All Branches with Fidelity > {SUCCESS_THRESHOLD} (Sorted by Prob):\n")
            f.write(f"{'Outcome':<20} {'Prob':<10} {'Fidelity':<10} {'1-Fid':<10} {'Best Target':<15}\n")
            
            branches = r.get('branches', [])
            good_branches = [b for b in branches if b['fidelity'] > SUCCESS_THRESHOLD]
            good_branches.sort(key=lambda x: x['prob'], reverse=True)
            
            if not good_branches:
                f.write("  No branches met the success threshold.\n")
            else:
                for b in good_branches:
                    outcome_str = str(b['outcome'])
                    t_idx = b.get('target_idx', 0)
                    tgt_name = target_names[t_idx] if t_idx < len(target_names) else f"Target_{t_idx}"
                    f.write(f"{outcome_str:<20} {b['prob']:<10.4f} {b['fidelity']:<10.4f} {(1-b['fidelity']):<10.1e} {tgt_name:<15}\n")
            f.write("\n" + "=" * 80 + "\n\n")

    best_res = results_ls[0]
    best_success_prob = best_res['success_prob']

    # --- Report ---
    print("\n" + "="*60)
    print(f" Time-Domain Optimization Results (Best of {niter}) ")
    print("="*60)
    print(f"Final Loss:          {best_res['loss']:.5f}")
    print(f"Expected Fidelity:   {best_res['expected_fidelity']:.5f}")
    print(f"Total Beam Prob:     {best_res.get('total_probability', 0.0):.5f}") 
    print(f"Success Prob (> {SUCCESS_THRESHOLD}): {best_success_prob:.5f}")
    print(f"Duration:            {best_res['duration']:.2f}s")
    print("-" * 60)
    
    # Map flat parameters to (Steps, Params) matrix
    mapped_params = circuit.map_parameters(best_res['x'])
    param_names = circuit.per_step_parameter_names

    # --- Save best result and parameters ---
    try:
        best_dir = results_dir / "best"
        best_dir.mkdir(exist_ok=True)

        # Copy best run files
        best_run_idx = best_res['run_index']
        print(f"Best run index: {best_run_idx}")
        
        src_pkl = results_dir / f"run_{best_run_idx:04d}.pkl"
        src_branches = results_dir / f"run_{best_run_idx:04d}_branches.txt"
        src_summary = results_dir / f"run_{best_run_idx:04d}_summary.json"

        if src_pkl.exists():
            shutil.copy(src_pkl, best_dir / f"best_run_{best_run_idx:04d}.pkl")
        if src_branches.exists():
            shutil.copy(src_branches, best_dir / f"best_run_{best_run_idx:04d}_branches.txt")
        if src_summary.exists():
            shutil.copy(src_summary, best_dir / f"best_run_{best_run_idx:04d}_summary.json")

        # save flat vector
        np.save(best_dir / "best_x.npy", best_res['x'])

        # save mapped params as npz (and JSON human-readable schedule)
        np.savez(best_dir / "mapped_params.npz", mapped_params=mapped_params)
        schedule = {
            "param_names": param_names,
            "mapped_params": mapped_params.tolist() if hasattr(mapped_params, "tolist") else [[float(v) for v in row] for row in mapped_params],
            "circuit_config": active_circuit_config,
            "target_configs": final_target_configs
        }
        with open(best_dir / "schedule.json", "w") as f:
            json.dump(schedule, f, indent=2)

        print(f"Saved best result to {best_dir}")
    except Exception as save_best_exc:
        print(f"Warning: failed to save best result: {save_best_exc}")

    print("Optimized Parameters Schedule:")
    header = f"{'Step':<6} | " + " | ".join([f"{name:<10}" for name in param_names])
    print(header)
    print("-" * len(header))

    for t in range(STEPS):
        row_str = f"{t:<6} | "
        vals = mapped_params[t]
        val_strs = [f"{v:10.4f}" for v in vals]
        row_str += " | ".join(val_strs)
        print(row_str)

    print("-" * 60)
    print("\nBest Run Detailed Branch Report:")
    print(format_branches_report(best_res['branches'], target_names, SUCCESS_THRESHOLD))
    print("="*60)

if __name__ == "__main__":
    main()
