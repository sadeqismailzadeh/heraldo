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

import operator
import numpy as np
import time
from pathlib import Path
from sklearn.cluster import KMeans

# Imports
from quantum_agent.optimization.time_circuits import *
from quantum_agent.optimization.time_runner import CMAESOptimizationRunner
from quantum_agent.components.targets import *



def filter_zero_slot_arrays(targets, cutoff_dim, tolerance=1e-6):
    """
    Filters a list of arrays, keeping only those where exactly one slot is zero
    (within a specified tolerance).

    Args:
        arrays (list of numpy arrays): The input list of arrays.
        tolerance (float): The tolerance value for comparing floating-point numbers to zero.

    Returns:
        list of numpy arrays: A new list containing only the arrays that meet the criteria.
    """
    filtered_arrays = []
    for target in targets:
        if np.count_nonzero(target.get_target_ket(cutoff_dim) < tolerance) == 1:
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
    CUTOFF_DIM = 50          # Simulation cutoff
    STEPS = 2                # Time steps (depth of the circuit)
    BEAM_WIDTH = 100          # Number of branches to keep
    TIME_INVARIANT = False   # False = different params per step
    MEASURE_CUTOFF = 16       # Max Fock state to measure on Ancilla (0, 1)
    SUCCESS_THRESHOLD = 0.99
    
    # Setup
    print("--- Setting up Time-Domain Optimization ---")
    print(f"Steps: {STEPS}, Beam Width: {BEAM_WIDTH}, Cutoff: {CUTOFF_DIM}")

    # 1. Target
    targets1 = [
        SqueezedCatTarget(alpha=3, r=1.38, p=0),
        SqueezedCatTarget(alpha=3, r=1.38, p=1)
    ]
    gkp_targets = [CoreGKPTarget(csv_path=Path(__file__).resolve().parent.parent.parent / "data" / "GKP_core_coefficients.csv", 
                            n_max=n, delta_db=10.4, mu=m)
            for n in [4, 6, 8, 10, 12] for m in [1]]
    

    

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
                                            clip_size=1.38,
                                            measure_fock_cutoff=MEASURE_CUTOFF,
                                            num_single_photon=1,
                                            train_initial_state=False, 
                                            initial_r=1.38 )
    

    circuit2 = ThreeModeTimeDomainSqueezeOnly(steps=STEPS,
                                        time_invariant=TIME_INVARIANT,
                                        clip_size=1,
                                        measure_fock_cutoff=MEASURE_CUTOFF,
                                        train_initial_state=False, 
                                        initial_r=1 )



    # Generate all Binomial Codes with max Fock state <= 12
    binomial_targets = []
    max_fock_n = 12
    for S in range(1, max_fock_n):
        for N in range(1, max_fock_n):
            if (N + 1) * (S + 1) <= max_fock_n:
                binomial_targets.append(BinomialCodeTarget(N=N, S=S, mu=0))
                binomial_targets.append(BinomialCodeTarget(N=N, S=S, mu=1))
    
    binomial_targets = filter_zero_slot_arrays(binomial_targets, CUTOFF_DIM)
    circuit = circuit1
    targets = gkp_targets 
    print(f"Optimizing for {len(targets)} targets.")
    print_targets(targets)
    # 3. Runner
    runner = CMAESOptimizationRunner(
        num_processes=4,
        circuit=circuit,
        target_gens=targets,
        cutoff_dim=CUTOFF_DIM,
        beam_width=BEAM_WIDTH,
        penalty_strength=10.0,
        success_threshold = SUCCESS_THRESHOLD,
        success_weight = 1000.0,
        ng_weight = 10.0,
        ng_threshold = 0.1,
    )
    
    # --- Execution ---
    n_generations = 100       # Number of hops per global search
    niter = 30      # Number of global searches

    exp_fid_ls = []
    hpx = []
    results_ls = []

    print(f"Starting {niter} global optimization runs (each with {n_generations} hops)...")

    target_names = []
    for t in targets:
        if isinstance(t, CoreGKPTarget):
            target_names.append(f"GKP_n{t.n_max}_mu{t.mu}")
        elif isinstance(t, SqueezedCatTarget):
            target_names.append(f"Cat_a{t.alpha}_p{t.p}")
        elif isinstance(t, BinomialCodeTarget):
            target_names.append(f"Binomial_N{t.N}_S{t.S}_mu{t.mu}")
        else:
            target_names.append("UnknownTarget")

    for e in range(niter):
        print(f"Global explore {e+1}/{niter}")
        try:
            res = runner.run(n_generations=n_generations)
            
            # Recalculate expected fidelity from branches
            # (TimeDomainRunner objective is -ExpFid + Penalty, but we want pure ExpFid for stats)
            branches = res.get('branches', [])
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
                 if b['prob'] > 0.001:
                     tgt_name = target_names[b['target_idx']] if b['target_idx'] < len(target_names) else f"T{b['target_idx']}"
                     print(f"  {str(b['outcome']):<15} {b['prob']:<10.4f} {b['fidelity']:<10.4f} {tgt_name}")
            print("")

            exp_fid_ls.append(expected_fidelity)
            hpx.append(res['x'])
            results_ls.append(res)
            
        except Exception as exc:
            print(f"Run {e+1} failed: {exc}")

    # Convert to arrays
    exp_fid_ls = np.array(exp_fid_ls)
    hpx = np.array(hpx)  # Array of flat parameters
    
    # Filter NaNs
    valid_mask = ~np.isnan(exp_fid_ls)
    exp_fid_ls = exp_fid_ls[valid_mask]
    
    # Filter results list as well (hpx might be ragged if filtering happens, but usually fixed size)
    # We just rebuild results_ls based on mask
    results_ls = [r for i, r in enumerate(results_ls) if valid_mask[i]]
    if len(hpx) > 0:
        hpx = hpx[valid_mask]

    if len(exp_fid_ls) == 0:
        print("All runs failed.")
        return

    # Clustering logic
    if len(exp_fid_ls) > 1:
        try:
            res_kmeans = KMeans(n_clusters=2, n_init='auto').fit(exp_fid_ls.reshape(-1, 1))
            mean0 = np.mean(exp_fid_ls[np.where(res_kmeans.labels_ == 0)])
            mean1 = np.mean(exp_fid_ls[np.where(res_kmeans.labels_ == 1)])

            if np.abs(mean0 - mean1) < 0.01:
                print("Clusters indistinguishable, keeping all.")
            else:
                drop = 1 if mean0 > mean1 else 0
                print(f"Mean cluster 0: {mean0:.4f}, Mean cluster 1: {mean1:.4f}. Dropping cluster {drop}.")
                exp_fid_ls[np.where(res_kmeans.labels_ == drop)] = 0.0
        except Exception as e:
            print(f"KMeans filtering skipped: {e}")

    # Select best based on Success Probability (filtered by clusters)
    success_probs = np.array([r['success_prob'] for r in results_ls])
    
    # Zero out success probs for runs dropped by clustering (where exp_fid_ls was set to 0.0)
    success_probs[exp_fid_ls == 0.0] = -1.0
    
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
    print("Dominant Outcome Branches (>1% Prob):")
    print(f"{'Outcome':<15} {'Prob':<10} {'Fidelity':<10} {'Best Target':<15}")
    print("-" * 60)
    
    for b in best_res['branches']:
        if b['prob'] > 0.01:
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
