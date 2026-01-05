"""
Script to run time-domain Beam Search optimization for a loop-based gadget.
"""
import operator
import numpy as np
import time
from pathlib import Path
from sklearn.cluster import KMeans

# Imports
from quantum_agent.optimization.time_circuits import *
from quantum_agent.optimization.cma_runner import CMAESOptimizationRunner
from quantum_agent.components.targets import SqueezedCatTarget, CoreGKPTarget

def main():
    # --- Configuration ---
    CUTOFF_DIM = 30          # Simulation cutoff
    STEPS = 4                # Time steps (depth of the circuit)
    BEAM_WIDTH = 100          # Number of branches to keep
    TIME_INVARIANT = False   # False = different params per step
    MEASURE_CUTOFF = 10       # Max Fock state to measure on Ancilla (0, 1)
    SUCCESS_THRESHOLD = 0.98
    
    # Setup
    print("--- Setting up Time-Domain Optimization ---")
    print(f"Steps: {STEPS}, Beam Width: {BEAM_WIDTH}, Cutoff: {CUTOFF_DIM}")

    # 1. Target
    targets1 = [
        SqueezedCatTarget(alpha=2.5, r=1.0, p=0),
        SqueezedCatTarget(alpha=2.5, r=1.0, p=1)
    ]
    gkp_targets = [CoreGKPTarget(csv_path=Path(__file__).resolve().parent.parent.parent / "data" / "GKP_core_coefficients.csv", 
                            n_max=n, delta_db=10.4, mu=m)
            for n in [4, 6, 8, 10, 12] for m in [0, 1]]
    
    targets = gkp_targets
    
    print(f"Optimizing for {len(targets)} targets.")

    # 2. Circuit
    circuit = TwoModeTimeDomainGadget(
        steps=STEPS,
        time_invariant=TIME_INVARIANT,
        clip_size=1,
        measure_fock_cutoff=MEASURE_CUTOFF
    )


    circuit1 = TwoModeTimeDomainSqueezeOnly(steps=STEPS,
                                            time_invariant=TIME_INVARIANT,
                                            clip_size=1,
                                            measure_fock_cutoff=MEASURE_CUTOFF)
    
    # 3. Runner
    runner = CMAESOptimizationRunner(
        num_processes=4,
        circuit=circuit1,
        target_gens=targets,
        cutoff_dim=CUTOFF_DIM,
        beam_width=BEAM_WIDTH,
        penalty_strength=10.0,
        success_threshold = 0.98,
        success_weight = 20.0
    )
    
    # --- Execution ---
    nhp = 20       # Number of hops per global search
    niter = 1      # Number of global searches

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
        try:
            res = runner.run(n_generations=50)
            
            # Recalculate expected fidelity from branches
            # (TimeDomainRunner objective is -ExpFid + Penalty, but we want pure ExpFid for stats)
            branches = res.get('branches', [])
            expected_fidelity = sum(b['prob'] * b['fidelity'] for b in branches)
            
            # Inject back into result dict for later use
            res['expected_fidelity'] = expected_fidelity

            success_prob = sum(b['prob'] for b in branches if b['fidelity'] > SUCCESS_THRESHOLD)

            print(f"  -> Final Expected Fidelity: {expected_fidelity:.5f}")
            print(f"  -> Success Prob (> {SUCCESS_THRESHOLD}): {success_prob:.5f}")
            print(f"  {'Outcome':<15} {'Prob':<10} {'Fidelity':<10} {'Best Target'}")
            
            for b in branches:
                 if b['prob'] > 0.002:
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

    # Select best
    index, value = max(enumerate(exp_fid_ls), key=operator.itemgetter(1))
    
    best_res = results_ls[index]
    best_success_prob = sum(b['prob'] for b in best_res['branches'] if b['fidelity'] > SUCCESS_THRESHOLD)

    # --- Report ---
    print("\n" + "="*60)
    print(f" Time-Domain Optimization Results (Best of {niter}) ")
    print("="*60)
    print(f"Final Loss:          {best_res['loss']:.5f}")
    print(f"Expected Fidelity:   {best_res['expected_fidelity']:.5f}")
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
