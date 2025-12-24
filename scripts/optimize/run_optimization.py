"""
Script to run deterministic optimization for the 2-mode gadget.
"""
import os
import numpy as np

# 1. Apply Performance Patches (Crucial for speed)
# We must do this before any SF operations
from quantum_agent.patches.sf_operations_no_cache import disable_fock_caching
disable_fock_caching()

from quantum_agent.patches.beamsplitter_patch import patch_beamsplitter
patch_beamsplitter()

# 2. Imports
from quantum_agent.optimization.circuits import TwoModeGadget
from quantum_agent.optimization.runner import OptimizationRunner
from quantum_agent.components.targets import CubicResourceTarget

def main():
    # --- Configuration ---
    CUTOFF_DIM = 15
    TARGET_A = 0.3          # Cubic phase parameter
    POST_SELECT_VAL = 2     # |2>
    
    # Setup
    print("--- Setting up Optimization ---")
    
    # 1. Circuit
    circuit = TwoModeGadget(clip_size=1.0)
    
    # 2. Target (Cubic Phase Resource State)
    target = CubicResourceTarget(a=TARGET_A)
    
    # 3. Runner
    # We want to measure mode 0 and find Fock state 2.
    # The dictionary maps {mode_index: fock_value}
    post_select = {0: POST_SELECT_VAL} 
    
    runner = OptimizationRunner(
        circuit=circuit,
        target_gen=target,
        cutoff_dim=CUTOFF_DIM,
        post_select_dict=post_select,
        alpha_prob=1.0,      # Weight for probability in loss
        penalty_strength=10.0
    )
    
    # --- Execution ---
    print(f"Target: Cubic Resource (a={TARGET_A})")
    print(f"Post-selection: Measure Mode 0 -> |{POST_SELECT_VAL}>")
    
    results = runner.run(n_iter=20, method="L-BFGS-B") # L-BFGS-B handles bounds well
    
    # --- Report ---
    print("\n" + "="*40)
    print(" Optimization Results ")
    print("="*40)
    print(f"Fidelity:    {results['fidelity']:.6f}")
    print(f"Probability: {results['probability']:.6f}")
    print(f"Loss:        {results['loss']:.6f}")
    print(f"Time:        {results['duration']:.2f}s")
    print("-" * 40)
    print("Best Parameters:")
    for name, val in zip(circuit.parameter_names, results['x']):
        print(f"  {name:<12}: {val:.4f}")
    print("="*40)

if __name__ == "__main__":
    main()