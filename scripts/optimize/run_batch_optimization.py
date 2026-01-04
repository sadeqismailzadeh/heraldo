"""
Script to run batch optimization maximizing Expected Fidelity across multiple targets.
"""
import operator
import numpy as np
from sklearn.cluster import KMeans
from pathlib import Path

# 2. Imports
from quantum_agent.optimization.circuits import ThreeModeSqueezeOnly, TwoModeGadget, TwoModeSqueezeOnly
from quantum_agent.optimization.batch_runner import BatchOptimizationRunner
from quantum_agent.components.targets import *

def main():
    # --- Configuration ---
    CUTOFF_DIM = 20
    MEASURE_MODES = [1, 2]  # Measure mode 0, leaving state on mode 1
    SUCCESS_THRESHOLD = 0.98
    
    # Setup
    print("--- Setting up Batch Optimization ---")
    
    # 1. Circuit
    circuit1 = TwoModeSqueezeOnly(clip_size=1.38)

    circuit2 = ThreeModeSqueezeOnly(clip_size=1)
    circuit3 = TwoModeGadget(clip_size=1.38)
    
    # 2. Targets (List)
    # The optimizer will reward the circuit if the output is close to EITHER of these
    targets = [
        SqueezedCatTarget(alpha=3.0, r=1.38, p =0),
        SqueezedCatTarget(alpha=3.0, r=1.38, p =1),
    ]

    gkp_targets = [CoreGKPTarget(csv_path=Path(__file__).resolve().parent.parent.parent / "data" / "GKP_core_coefficients.csv", 
                                n_max=n, delta_db=10.4, mu=m)
                for n in [4, 6, 8, 10, 12] for m in [0, 1]]

    targets = gkp_targets
    circuit = circuit2
    
    # 3. Runner
    runner = BatchOptimizationRunner(
        circuit=circuit,
        target_gens=targets,
        cutoff_dim=CUTOFF_DIM,
        measure_modes=MEASURE_MODES,
        penalty_strength=10.0,
        success_threshold=SUCCESS_THRESHOLD,
        success_weight=20.0,
        max_post_select = 12
    )
    
    # --- Execution ---
    print(f"Optimizing for {len(targets)} targets simultaneousy.")
    print(f"Objective: Maximize Expected Fidelity (Sum of Prob * MaxFidelity)")

    nhp = 30       # Number of hops per global search
    niter = 20     # Number of global searches

    exp_fid_ls = []
    hpx = []
    results_ls = []

    print(f"Starting {niter} global optimization runs (each with {nhp} hops)...")

    target_names = []
    for i, t in enumerate(targets):
        if hasattr(t, "n_max") and hasattr(t, "mu"):
            target_names.append(f"GKP_n{t.n_max}_mu{t.mu}")
        elif hasattr(t, "p"):
            target_names.append(f"Cat_p{t.p}")
        else:
            target_names.append(f"Target_{i}")

    for e in range(niter):
        print(f"Global explore {e+1}/{niter}")
        res = runner.run(n_iter=nhp, method="SLSQP")
        
        # Calculate success probability
        success_prob = sum(b['prob'] for b in res['branches'] if b['fidelity'] > SUCCESS_THRESHOLD)

        print(f"  -> Final Expected Fidelity: {res['expected_fidelity']:.5f}")
        print(f"  -> Success Prob (> {SUCCESS_THRESHOLD}): {success_prob:.5f}")
        print(f"  {'Outcome':<10} {'Prob':<10} {'Fidelity':<10} {'Best Target'}")
        
        # Only print branches with significant probability to avoid spam
        for b in res['branches']:
             if b['prob'] > 0.002:
                 tgt_name = target_names[b['target_idx']] if b['target_idx'] < len(target_names) else f"T{b['target_idx']}"
                 print(f"  {str(b['outcome']):<10} {b['prob']:<10.4f} {b['fidelity']:<10.4f} {tgt_name}")
        print("")

        exp_fid_ls.append(res['expected_fidelity'])
        hpx.append(res['x'])
        results_ls.append(res)

    # Convert to arrays
    exp_fid_ls = np.array(exp_fid_ls)
    hpx = np.array(hpx)
    
    # Filter NaNs if any
    valid_mask = ~np.isnan(exp_fid_ls)
    exp_fid_ls = exp_fid_ls[valid_mask]
    hpx = hpx[valid_mask]
    # Filter results list as well
    results_ls = [r for i, r in enumerate(results_ls) if valid_mask[i]]

    if len(exp_fid_ls) == 0:
        print("All runs failed.")
        return

    # Clustering logic to remove sub-optimal fidelities
    if len(exp_fid_ls) > 1:
        try:
            res_kmeans = KMeans(n_clusters=2, n_init='auto').fit(exp_fid_ls.reshape(-1, 1))
            mean0 = np.mean(exp_fid_ls[np.where(res_kmeans.labels_ == 0)])
            mean1 = np.mean(exp_fid_ls[np.where(res_kmeans.labels_ == 1)])

            if np.abs(mean0 - mean1) < 0.01:
                print("Clusters indistinguishable, keeping all.")
            else:
                if mean0 > mean1:
                    drop = 1
                else:
                    drop = 0
                
                print(f"Mean cluster 0: {mean0:.4f}, Mean cluster 1: {mean1:.4f}. Dropping cluster {drop}.")
                # Zero out expected fidelities of the lower cluster to ignore them
                exp_fid_ls[np.where(res_kmeans.labels_ == drop)] = 0.0
        except Exception as e:
            print(f"KMeans filtering skipped: {e}")

    # Select best result based on Expected Fidelity
    index, value = max(enumerate(exp_fid_ls), key=operator.itemgetter(1))
    
    best_x = hpx[index]
    best_res = results_ls[index]
    
    best_success_prob = sum(b['prob'] for b in best_res['branches'] if b['fidelity'] > SUCCESS_THRESHOLD)

    # --- Report ---
    print("\n" + "="*60)
    print(f" Batch Optimization Results (Best of {niter}) ")
    print("="*60)
    print(f"Final Loss:          {best_res['loss']:.5f}")
    print(f"Expected Fidelity:   {best_res['expected_fidelity']:.5f}")
    print(f"Success Prob (> {SUCCESS_THRESHOLD}): {best_success_prob:.5f}")
    print(f"Duration:            {best_res['duration']:.2f}s")
    print("-" * 60)
    print("Best Parameters:")
    for name, val in zip(circuit.parameter_names, best_res['x']):
        print(f"  {name:<12}: {val:.4f}")
        
    print("-" * 60)
    print("Dominant Outcome Branches (>1% Prob):")
    print(f"{'Outcome':<10} {'Prob':<10} {'Fidelity':<10} {'Best Target':<15}")
    print("-" * 60)
    
    # Dynamic target naming for large lists
    get_tgt_name = lambda idx: target_names[idx] if idx < len(target_names) else f"Target_{idx}"

    for b in best_res['branches']:
        if b['prob'] > 0.01:
            outcome_str = str(b['outcome'])
            tgt_name = get_tgt_name(b['target_idx'])
            print(f"{outcome_str:<10} {b['prob']:<10.4f} {b['fidelity']:<10.4f} {tgt_name:<15}")

    # --- Target Analysis ---
    print("-" * 60)
    print("Target Distribution Analysis (Aggregated Success):")
    print(f"{'Rank':<5} {'Target Name':<20} {'Tot. Prob':<10} {'Outcomes (Top 3)'}")
    print("-" * 60)

    # Aggregate stats per target
    target_stats = {} # idx -> {'prob': float, 'outcomes': list of (outcome, prob)}
    
    for b in best_res['branches']:
        # Only count successful branches towards the target's score
        if b['fidelity'] > SUCCESS_THRESHOLD:
            idx = b['target_idx']
            if idx not in target_stats:
                target_stats[idx] = {'prob': 0.0, 'outcomes': []}
            
            target_stats[idx]['prob'] += b['prob']
            target_stats[idx]['outcomes'].append((b['outcome'], b['prob']))

    # Sort targets by total probability mass
    sorted_targets = sorted(target_stats.items(), key=lambda x: x[1]['prob'], reverse=True)

    for rank, (idx, stats) in enumerate(sorted_targets):
        # Sort outcomes for this target by probability
        stats['outcomes'].sort(key=lambda x: x[1], reverse=True)
        top_outcomes = [str(o[0]) for o in stats['outcomes'][:3]]
        outcome_str = ", ".join(top_outcomes)
        if len(stats['outcomes']) > 3:
            outcome_str += ", ..."
            
        print(f"{rank+1:<5} {get_tgt_name(idx):<20} {stats['prob']:<10.4f} {outcome_str}")

    print("="*60)

if __name__ == "__main__":
    main()
