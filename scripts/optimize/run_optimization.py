"""
Script to run deterministic optimization for the 2-mode gadget.
"""
import os
import operator
import numpy as np
from sklearn.cluster import KMeans
from pathlib import Path

# 1. Apply Performance Patches (Crucial for speed)
# We must do this before any SF operations
from quantum_agent.patches.sf_operations_no_cache import disable_fock_caching
disable_fock_caching()

from quantum_agent.patches.beamsplitter_patch import patch_beamsplitter
patch_beamsplitter()

# 2. Imports
from quantum_agent.optimization.circuits import ThreeModeGadget, ThreeModeSqueezeOnly, TwoModeGadget, TwoModeSqueezeOnly
from quantum_agent.optimization.runner import OptimizationRunner
from quantum_agent.components.targets import *

def main():
    # --- Configuration ---
    CUTOFF_DIM = 20
    TARGET_A = 0.3          # Cubic phase parameter
    POST_SELECT_VAL = 8     # |2>
    
    # Setup
    print("--- Setting up Optimization ---")
    
    # 1. Circuit
    circuit1 = TwoModeGadget(clip_size=1.0)

    circuit2 = TwoModeSqueezeOnly(clip_size=1.38)

    circuit3 = ThreeModeGadget(clip_size=1)

    circuit4 = ThreeModeSqueezeOnly(clip_size=1)
    
    # 2. Target (Cubic Phase Resource State)
    target = CubicResourceTarget(a=TARGET_A)

    target1 = SqueezedCatTarget(
        alpha=3, 
        r=1.38,
        p=0
    )

    csv_path =  Path(__file__).resolve().parent.parent.parent / "data" / "GKP_core_coefficients.csv"
    target3=CoreGKPTarget(csv_path=csv_path, 
                          n_max=12, 
                          delta_db=10.4, 
                          mu=0)
    
    # 3. Runner
    # We want to measure mode 0 and find Fock state 2.
    # The dictionary maps {mode_index: fock_value}
    post_select = {0: POST_SELECT_VAL} 
    post_select_3mode = {0: 5, 1: 7}
    
    current_circuit = circuit1
    runner = OptimizationRunner(
        circuit=circuit4,
        target_gen=target1,
        cutoff_dim=CUTOFF_DIM,
        post_select_dict=post_select_3mode,
        alpha_prob=0.1,      # Weight for probability in loss
        penalty_strength=10.0
    )
    
    # --- Execution ---
    print(f"Target: Cubic Resource (a={TARGET_A})")
    print(f"Post-selection: Measure Mode 0 -> |{POST_SELECT_VAL}>")

    # Parameters matching two_mode.py
    nhp = 50       # Number of hops per global search
    niter = 30     # Number of global searches

    fid_ls = []
    prob_ls = []
    hpx = []

    print(f"Starting {niter} global optimization runs (each with {nhp} hops)...")

    for e in range(niter):
        print(f"Global explore {e+1}/{niter}")
        res = runner.run(n_iter=nhp, method="SLSQP")
        
        print("final fid {}, prob {}".format(res['fidelity'], res['probability']))

        fid_ls.append(res['fidelity'])
        prob_ls.append(res['probability'])
        hpx.append(res['x'])

    # Convert to arrays
    fid_ls = np.array(fid_ls)
    prob_ls = np.array(prob_ls)
    hpx = np.array(hpx)
    
    # Filter NaNs if any
    valid_mask = ~np.isnan(fid_ls)
    fid_ls = fid_ls[valid_mask]
    prob_ls = prob_ls[valid_mask]
    hpx = hpx[valid_mask]

    if len(fid_ls) == 0:
        print("All runs failed.")
        return

    # Clustering logic to remove sub-optimal fidelities (from two_mode.py)
    if len(fid_ls) > 1:
        try:
            res_kmeans = KMeans(n_clusters=2, n_init='auto').fit(fid_ls.reshape(-1, 1))
            mean0 = np.mean(fid_ls[np.where(res_kmeans.labels_ == 0)])
            mean1 = np.mean(fid_ls[np.where(res_kmeans.labels_ == 1)])

            if np.abs(mean0 - mean1) < 0.01:
                print("Clusters indistinguishable, keeping all.")
            else:
                if mean0 > mean1:
                    drop = 1
                else:
                    drop = 0
                
                print(f"Mean cluster 0: {mean0:.4f}, Mean cluster 1: {mean1:.4f}. Dropping cluster {drop}.")
                # Zero out probabilities of the lower fidelity cluster
                prob_ls[np.where(res_kmeans.labels_ == drop)] = 0
        except Exception as e:
            print(f"KMeans filtering skipped: {e}")

    # Select best result based on probability
    index, value = max(enumerate(prob_ls), key=operator.itemgetter(1))
    
    best_x = hpx[index]
    best_fid = fid_ls[index]
    best_prob = prob_ls[index]
    
    # --- Report ---
    print("\n" + "="*40)
    print(f" Optimization Results (Best of {niter}) ")
    print("="*40)
    print(f"Fidelity:    {best_fid:.6f}")
    print(f"Probability: {best_prob:.6f}")
    print("-" * 40)
    print("Best Parameters:")
    for name, val in zip(current_circuit.parameter_names, best_x):
        print(f"  {name:<12}: {val:.4f}")
    print("="*40)

if __name__ == "__main__":
    main()
