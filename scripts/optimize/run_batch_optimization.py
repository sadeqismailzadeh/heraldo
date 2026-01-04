"""
Script to run batch optimization maximizing Expected Fidelity across multiple targets.
"""
import operator
import numpy as np
from sklearn.cluster import KMeans

# 1. Apply Performance Patches
from quantum_agent.patches.sf_operations_no_cache import disable_fock_caching
disable_fock_caching()

from quantum_agent.patches.beamsplitter_patch import patch_beamsplitter
patch_beamsplitter()

# 2. Imports
from quantum_agent.optimization.circuits import ThreeModeSqueezeOnly, TwoModeGadget, TwoModeSqueezeOnly
from quantum_agent.optimization.batch_runner import BatchOptimizationRunner
from quantum_agent.components.targets import CubicResourceTarget, SqueezedCatTarget

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
    
    # 3. Runner
    runner = BatchOptimizationRunner(
        circuit=circuit2,
        target_gens=targets,
        cutoff_dim=CUTOFF_DIM,
        measure_modes=MEASURE_MODES,
        penalty_strength=10.0,
        success_threshold=SUCCESS_THRESHOLD,
        success_weight=20.0
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

    target_names = ["Cat0", "Cat1"]

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
    
    for b in best_res['branches']:
        if b['prob'] > 0.01:
            outcome_str = str(b['outcome'])
            tgt_name = target_names[b['target_idx']] if b['target_idx'] < len(target_names) else f"T{b['target_idx']}"
            print(f"{outcome_str:<10} {b['prob']:<10.4f} {b['fidelity']:<10.4f} {tgt_name:<15}")
    print("="*60)

if __name__ == "__main__":
    main()
