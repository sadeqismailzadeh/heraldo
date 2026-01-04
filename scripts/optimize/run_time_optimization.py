"""
Script to run time-domain Beam Search optimization for a loop-based gadget.
"""
import numpy as np
import time
from pathlib import Path

# Imports
from quantum_agent.optimization.time_circuits import TwoModeTimeDomainGadget
from quantum_agent.optimization.time_runner import TimeDomainRunner
from quantum_agent.components.targets import SqueezedCatTarget, CoreGKPTarget

def main():
    # --- Configuration ---
    CUTOFF_DIM = 20          # Simulation cutoff
    STEPS = 4                # Time steps (depth of the circuit)
    BEAM_WIDTH = 8           # Number of branches to keep
    TIME_INVARIANT = False   # False = different params per step
    MEASURE_CUTOFF = 2       # Max Fock state to measure on Ancilla (0, 1)
    
    # Global Optimization settings
    N_GLOBAL_ITER = 5        # Number of global basin hopping runs
    N_HOPS = 20              # Steps per basin hopping
    
    print("--- Setting up Time-Domain Optimization ---")
    print(f"Steps: {STEPS}, Beam Width: {BEAM_WIDTH}, Cutoff: {CUTOFF_DIM}")

    # 1. Target
    # Using Squeezed Cat for quick testing. 
    # For GKP, ensure 'GKP_core_coefficients.csv' is available.
    target = SqueezedCatTarget(alpha=2.5, r=1.0)
    # target = CoreGKPTarget(n_max=4, delta_db=10.0, mu=0)
    
    print(f"Target: {target.__class__.__name__}")

    # 2. Circuit
    circuit = TwoModeTimeDomainGadget(
        steps=STEPS,
        time_invariant=TIME_INVARIANT,
        clip_size=1.5,
        measure_fock_cutoff=MEASURE_CUTOFF
    )
    
    # 3. Runner
    runner = TimeDomainRunner(
        circuit=circuit,
        target_gen=target,
        cutoff_dim=CUTOFF_DIM,
        beam_width=BEAM_WIDTH,
        penalty_strength=10.0
    )
    
    # --- Execution ---
    best_loss = float('inf')
    best_result = None

    print(f"\nStarting {N_GLOBAL_ITER} global optimization runs...")

    for i in range(N_GLOBAL_ITER):
        print(f"\n--- Run {i+1}/{N_GLOBAL_ITER} ---")
        
        try:
            res = runner.run(n_iter=N_HOPS, method="SLSQP")
            
            print(f"  -> Loss: {res['loss']:.5f}")
            print(f"  -> Duration: {res['duration']:.2f}s")
            
            if res['loss'] < best_loss:
                best_loss = res['loss']
                best_result = res
                
        except Exception as e:
            print(f"Run {i+1} failed: {e}")

    # --- Report ---
    if best_result is None:
        print("Optimization failed.")
        return

    print("\n" + "="*60)
    print(f" Time-Domain Optimization Results (Best of {N_GLOBAL_ITER}) ")
    print("="*60)
    print(f"Final Loss:          {best_result['loss']:.5f}")
    
    # Calculate explicit Fidelity from loss (approximate if penalties are low)
    # Loss = -F + Penalty. If converged, Penalty ~ 0, so F ~ -Loss
    # (Note: exact fidelity isn't returned in the dict by default to save overhead, 
    # but -loss is a lower bound estimate).
    print(f"Approx Fidelity:     {-best_result['loss']:.5f}") 
    print(f"Duration:            {best_result['duration']:.2f}s")
    print("-" * 60)
    
    # Map flat parameters to (Steps, Params) matrix
    mapped_params = circuit.map_parameters(best_result['x'])
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

    print("="*60)

if __name__ == "__main__":
    main()