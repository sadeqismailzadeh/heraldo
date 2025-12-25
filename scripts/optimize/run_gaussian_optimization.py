"""
Script to run fast Gaussian-based optimization for the 2-mode gadget.
"""
import os
import operator
import numpy as np
from pathlib import Path
from sklearn.cluster import KMeans

# 1. Imports
from quantum_agent.optimization.circuits import TwoModeGadget, ThreeModeGadget
from quantum_agent.optimization.gaussian_runner import GaussianOptimizationRunner
from quantum_agent.components.targets import CubicResourceTarget, CoreGKPTarget

def main():
    # --- Configuration ---
    # With Gaussian backend, we can afford higher cutoffs for fidelity checking
    CUTOFF_DIM = 25 
    TARGET_A = 0.3          
    POST_SELECT_VAL = 4     # |2> (in mode 0)
    
    # Setup
    print("--- Setting up Gaussian Optimization ---")
    
    # 1. Circuit
    # Note: Ensure the circuit only uses Gaussian gates (S, D, BS, R). 
    # TwoModeGadget meets this criteria.
    circuit = TwoModeGadget(clip_size=1.0)
    
    # 2. Target
    target = CubicResourceTarget(a=TARGET_A)
    
    # 3. Runner
    # We want to measure mode 0 and find Fock state 4.
    post_select = {0: POST_SELECT_VAL}
    
    runner = GaussianOptimizationRunner(
        circuit=circuit,
        target_gen=target,
        cutoff_dim=CUTOFF_DIM,
        post_select_dict=post_select,
        alpha_prob=0.1 # Weighting for probability
    )
    
    # --- Execution ---
    print(f"Target: Cubic Resource (a={TARGET_A})")
    print(f"Post-selection: Measure Mode 0 -> |{POST_SELECT_VAL}>")
    print(f"Backend: Strawberry Fields 'gaussian' + TheWalrus Hafnians")

    nhp = 50       # Hops
    niter = 30     # Global runs

    fid_ls = []
    prob_ls = []
    hpx = []

    print(f"Starting {niter} global optimization runs (each with {nhp} hops)...")

    for e in range(niter):
        print(f"Global explore {e+1}/{niter}")
        res = runner.run(n_iter=nhp, method="SLSQP")
        
        print("  -> Final Fid: {:.4f}, Prob: {:.4f}".format(res['fidelity'], res['probability']))

        fid_ls.append(res['fidelity'])
        prob_ls.append(res['probability'])
        hpx.append(res['x'])

    # Convert to arrays
    fid_ls = np.array(fid_ls)
    prob_ls = np.array(prob_ls)
    hpx = np.array(hpx)
    
    # Basic filtering of results
    if len(fid_ls) > 0:
        # Select best result based on a metric (e.g., Fid * Prob or just Fid)
        # Here we maximize Fidelity primarily, but you can filter by prob threshold
        
        # Simple selection: Best Fidelity
        index, value = max(enumerate(fid_ls), key=operator.itemgetter(1))
        
        best_x = hpx[index]
        best_fid = fid_ls[index]
        best_prob = prob_ls[index]
        
        # --- Report ---
        print("\n" + "="*40)
        print(f" Gaussian Optimization Results (Best of {niter}) ")
        print("="*40)
        print(f"Fidelity:    {best_fid:.6f}")
        print(f"Probability: {best_prob:.6f}")
        print("-" * 40)
        print("Best Parameters:")
        for name, val in zip(circuit.parameter_names, best_x):
            print(f"  {name:<12}: {val:.4f}")
        print("="*40)

if __name__ == "__main__":
    main()
