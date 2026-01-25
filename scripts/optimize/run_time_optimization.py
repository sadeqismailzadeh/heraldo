"""
Script to run time-domain Beam Search optimization for a loop-based gadget.
"""


from ast import pattern
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

import operator
import numpy as np
import time
from pathlib import Path
from sklearn.cluster import KMeans
import itertools

# Imports
from quantum_agent.optimization.time_circuits import *
from quantum_agent.optimization.time_runner import CMAESOptimizationRunner, DifferentialEvolutionRunner, TimeDomainRunner
from quantum_agent.components.targets import *
from quantum_agent.utils import *

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

def main():
    # --- Configuration ---
    CUTOFF_DIM = 30          # Simulation cutoff
    STEPS = 1                # Time steps (depth of the circuit)
    BEAM_WIDTH = 100          # Number of branches to keep
    TIME_INVARIANT = False   # False = different params per step
    MEASURE_CUTOFF = CUTOFF_DIM       # Max Fock state to measure on Ancilla (0, 1)
    SUCCESS_THRESHOLD = 1 - 1e-2
    
    OPTIMIZER_METHOD = "CMA" # Options: "CMA", "DE", "BASIN"

    # Setup
    print("--- Setting up Time-Domain Optimization ---")
    print(f"Steps: {STEPS}, Beam Width: {BEAM_WIDTH}, Cutoff: {CUTOFF_DIM}")

    # Results directory (per run-session)
    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    results_dir = Path(__file__).resolve().parent.parent.parent / "results" / f"cma_run_{timestamp}"
    results_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving per-run results to: {results_dir}")


    squeezing = db_to_r(12)
    print(f"squeezing r = {squeezing}")
    # 1. Target
    targets1 = [
        SqueezedCatTarget(alpha=3, r=1.38, p=0),
        SqueezedCatTarget(alpha=3, r=1.38, p=1)
    ]

    target_cat = [
        CatTarget(alpha=2, p=0),
        CatTarget(alpha=2, p=1)
    ]
    gkp_targets = [CoreGKPTarget(csv_path=Path(__file__).resolve().parent.parent.parent / "data" / "GKP_core_coefficients.csv", 
                            n_max=n, delta_db=10, mu=m)
            for n in [4, 6, 8, 10, 12] for m in [1]]
    

        # Generate all Binomial Codes with max Fock state <= 12
    binomial_targets = []
    max_fock_n = 14
    for S in range(1, max_fock_n):
        for N in range(2, max_fock_n):
            if (N + 1) * (S + 1) <= max_fock_n:
                binomial_targets.append(BinomialCodeTarget(N=N, S=S, mu=0))
                binomial_targets.append(BinomialCodeTarget(N=N, S=S, mu=1))
    binomial_targets = filter_zero_slot_arrays(binomial_targets, CUTOFF_DIM)

    csv_path =  Path(__file__).resolve().parent.parent.parent / "data" / "GKP_core_coefficients.csv"
    target3=CoreGKPTarget(csv_path=csv_path, 
                          n_max=4, 
                          delta_db=10, 
                          mu=0)
    
    target_cubic = CubicResourceTarget(a=0.61)
    

    

    # 2. Circuit
    circuit_gadget = TwoModeTimeDomainGadget(
        steps=STEPS,
        time_invariant=TIME_INVARIANT,
        clip_size=1,
        measure_fock_cutoff=MEASURE_CUTOFF,
        train_initial_state=False, 
        initial_r=1,
    )


    circuit1 = TwoModeTimeDomainSqueezeOnly(steps=STEPS,
                                            time_invariant=TIME_INVARIANT,
                                            clip_size=squeezing,
                                            measure_fock_cutoff=MEASURE_CUTOFF,
                                            num_single_photon=0,
                                            train_initial_state=True, 
                                            initial_r=squeezing )
    

    circuit2 = ThreeModeTimeDomainSqueezeOnly(steps=STEPS,
                                        time_invariant=TIME_INVARIANT,
                                        clip_size=squeezing,
                                        measure_fock_cutoff=MEASURE_CUTOFF,
                                        num_single_photon=0,
                                        train_initial_state=True, 
                                        initial_r=squeezing )

    circuit3 = FourModeTimeDomainSqueezeOnly(steps=STEPS,
                                    time_invariant=TIME_INVARIANT,
                                    clip_size=squeezing,
                                    measure_fock_cutoff=MEASURE_CUTOFF,
                                    num_single_photon=0,
                                    train_initial_state=True, 
                                    initial_r=squeezing )





    
    circuit = circuit2
    targets = gkp_targets
    print(f"Optimizing for {len(targets)} targets.")
    print_targets(targets, CUTOFF_DIM)
    # patterns = generate_measurement_patterns(circuit, exact_total=4)
   
    # pattern = [[(1,), (3,)], [(2,), (2,)], [(3,), (1,)]] 
    # patterns = [(2,2,4)]
    # patterns = [(4,4)]
    # patterns = [[(1,3)]]
    patterns = None
    print(patterns)
    patterns = prepare_measurement_patterns(patterns)
    print(patterns)
    
    
    print("\nNon-Gaussianity scores for targets:")
    for i, target in enumerate(targets):
        ket = target.get_target_ket(CUTOFF_DIM)
        ng_score = compute_ng_scores([ket], CUTOFF_DIM)[0]
        print(f"  Target {i+1}: {ng_score:.4f}")

    # 3. Runner
    runner_map = {
        "CMA": CMAESOptimizationRunner,
        "DE": DifferentialEvolutionRunner,
        "BASIN": TimeDomainRunner
    }
    
    UnifiedRunner = runner_map.get(OPTIMIZER_METHOD)
    if UnifiedRunner is None:
        raise ValueError(f"Unknown optimizer method: {OPTIMIZER_METHOD}")

    runner = UnifiedRunner(
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
    
    # --- Execution ---
    n_generations = 1000       # Number of hops per global search
    niter = 100      # Number of global searches

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
        else:
            target_names.append("UnknownTarget")

    for e in range(niter):
        print(f"Global explore {e+1}/{niter}")
        try:
            # prob_power = np.random.uniform(0.01, 1)

            # Sample prob_power from log distribution between 0.01 and 1.0
            prob_power = 10 ** np.random.uniform(-2, 1)
            # prob_power = 1
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

    # Clustering logic
    if len(suc_pb_ls) > 1:
        try:
            res_kmeans = KMeans(n_clusters=2, n_init='auto').fit(suc_pb_ls.reshape(-1, 1))
            mean0 = np.mean(suc_pb_ls[np.where(res_kmeans.labels_ == 0)])
            mean1 = np.mean(suc_pb_ls[np.where(res_kmeans.labels_ == 1)])

            if np.abs(mean0 - mean1) < 0.01:
                print("Clusters indistinguishable, keeping all.")
            else:
                drop = 1 if mean0 > mean1 else 0
                print(f"Mean cluster 0: {mean0:.4f}, Mean cluster 1: {mean1:.4f}. Dropping cluster {drop}.")
                suc_pb_ls[np.where(res_kmeans.labels_ == drop)] = 0.0
        except Exception as e:
            print(f"KMeans filtering skipped: {e}")

    # Select best based on Success Probability (filtered by clusters)
    success_probs = np.array([r['success_prob'] for r in results_ls])
    
    # Zero out success probs for runs dropped by clustering (where exp_fid_ls was set to 0.0)
    success_probs[suc_pb_ls == 0.0] = -1.0
    
    index, value = max(enumerate(success_probs), key=operator.itemgetter(1))
    
    best_res = results_ls[index]
    best_success_prob = sum(b['prob'] for b in best_res['branches'] if b['fidelity'] > SUCCESS_THRESHOLD)

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

        with open(best_dir / "best_result.pkl", "wb") as f:
            pickle.dump(best_res, f)

        # save flat vector
        np.save(best_dir / "best_x.npy", best_res['x'])

        # save mapped params as npz (and JSON human-readable schedule)
        np.savez(best_dir / "mapped_params.npz", mapped_params=mapped_params)
        schedule = {
            "param_names": param_names,
            "mapped_params": mapped_params.tolist() if hasattr(mapped_params, "tolist") else [[float(v) for v in row] for row in mapped_params]
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
    print("Dominant Outcome Branches (Sorted by Prob):")
    print(f"{'Outcome':<15} {'Prob':<10} {'Fidelity':<10} {'Best Target':<15}")
    print("-" * 60)
    
    sorted_branches = sorted(best_res['branches'], key=lambda x: x['prob'], reverse=True)
    
    for b in sorted_branches:
        # if b['prob'] > 0.001:
            outcome_str = str(b['outcome'])
            # Time runner usually has single target index 0
            tgt_name = target_names[b['target_idx']] if b['target_idx'] < len(target_names) else "Target"
            print(f"{outcome_str:<15} {b['prob']:<10.4f} {b['fidelity']:<10.4f} {tgt_name:<15}")

    # --- Target Analysis ---
    print("-" * 60)
    print("Target Distribution Analysis (Aggregated Success):")
    print(f"{'Rank':<5} {'Target Name':<20} {'Tot. Prob':<10} {'Outcomes (Top 3)'}")
    print("-" * 60)

    target_stats = {} 
    for b in best_res['branches']:
        if b['fidelity'] > SUCCESS_THRESHOLD:
            idx = b['target_idx']
            if idx not in target_stats:
                target_stats[idx] = {'prob': 0.0, 'outcomes': []}
            target_stats[idx]['prob'] += b['prob']
            target_stats[idx]['outcomes'].append((b['outcome'], b['prob']))

    sorted_targets = sorted(target_stats.items(), key=lambda x: x[1]['prob'], reverse=True)

    if not sorted_targets:
        print("No branches met the success threshold.")
    
    for rank, (idx, stats) in enumerate(sorted_targets):
        stats['outcomes'].sort(key=lambda x: x[1], reverse=True)
        top_outcomes = [str(o[0]) for o in stats['outcomes'][:3]]
        outcome_str = ", ".join(top_outcomes)
        if len(stats['outcomes']) > 3:
            outcome_str += ", ..."
        
        t_name = target_names[idx] if idx < len(target_names) else f"Target_{idx}"
        print(f"{rank+1:<5} {t_name:<20} {stats['prob']:<10.4f} {outcome_str}")

    print("="*60)

if __name__ == "__main__":
    main()
