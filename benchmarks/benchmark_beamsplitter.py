import scipy.integrate
if not hasattr(scipy.integrate, 'simps'):
    scipy.integrate.simps = scipy.integrate.simpson

import numpy as np
import time
import strawberryfields as sf
from strawberryfields.ops import Dgate, BSgate
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from heraldo.patches.beamsplitter_patch import patch_beamsplitter, revert_beamsplitter_patch


def run_dynamic_benchmark(truncs=[10, 15, 20, 30, 40, 50, 60], 
                          n_modes_list=[2], 
                          num_random_runs=10):
    """
    Benchmarks performance where every run uses a DIFFERENT random theta/phi.
    This simulates a variational loop where parameters change constantly.
    """
    print("--- Running Dynamic Parameter Benchmark ---")
    results = []

    # 1. Run ORIGINAL (Unpatched)
    print("\nBenchmarking ORIGINAL beamsplitter...")
    revert_beamsplitter_patch()

    for n_modes in n_modes_list:
        for trunc in truncs:
            np.random.seed(42)
            
            eng = sf.Engine("fock", backend_options={"cutoff_dim": trunc})
            timings = []

            for _ in range(num_random_runs):
                theta = np.random.uniform(0, 2 * np.pi)
                phi = np.random.uniform(0, 2 * np.pi)

                prog = sf.Program(n_modes)
                with prog.context as q:
                    for i in range(n_modes):
                        Dgate(0.3 * (i + 1), 0) | q[i]
                    
                    BSgate(theta, phi) | (q[0], q[1])
                    
                    if n_modes > 2:
                        BSgate(theta / 2, phi / 2) | (q[1], q[2])

                start = time.time()
                eng.run(prog)
                end = time.time()
                timings.append(end - start)

            avg_time = np.mean(timings)
            print(f"Original: Modes={n_modes}, Cutoff={trunc} -> {avg_time:.6f}s (avg of {num_random_runs} random angles)")
            
            results.append({
                'n_modes': n_modes,
                'trunc': trunc,
                'original': avg_time
            })

    # 2. Run OPTIMIZED (Patched)
    print("\nBenchmarking OPTIMIZED beamsplitter...")
    patch_beamsplitter()

    print("Running JIT warmup...")
    prog_warm = sf.Program(2)
    with prog_warm.context as q:
        BSgate(0.1, 0.1) | (q[0], q[1])
    sf.Engine("fock", backend_options={"cutoff_dim": 5}).run(prog_warm)
    print("Warmup complete.")

    for i, res in enumerate(results):
        n_modes = res['n_modes']
        trunc = res['trunc']
        
        np.random.seed(42)
        
        eng = sf.Engine("fock", backend_options={"cutoff_dim": trunc})
        timings = []

        for _ in range(num_random_runs):
            theta = np.random.uniform(0, 2 * np.pi)
            phi = np.random.uniform(0, 2 * np.pi)

            prog = sf.Program(n_modes)
            with prog.context as q:
                for j in range(n_modes):
                    Dgate(0.3 * (j + 1), 0) | q[j]
                
                BSgate(theta, phi) | (q[0], q[1])
                
                if n_modes > 2:
                    BSgate(theta / 2, phi / 2) | (q[1], q[2])

            start = time.time()
            eng.run(prog)
            end = time.time()
            timings.append(end - start)

        avg_time = np.mean(timings)
        results[i]['optimized'] = avg_time
        print(f"Optimized: Modes={n_modes}, Cutoff={trunc} -> {avg_time:.6f}s")

    # 3. Summary
    print("\n" + "="*80)
    print(f"{'Modes':<6} {'Cutoff':<8} {'Original (s)':<15} {'Optimized (s)':<15} {'Speedup':<10}")
    print("-" * 80)
    
    for res in results:
        t_orig = res['original']
        t_opt = res['optimized']
        speedup = t_orig / t_opt if t_opt > 0 else 0
        
        print(f"{res['n_modes']:<6} {res['trunc']:<8} {t_orig:<15.6f} {t_opt:<15.6f} {speedup:.2f}x")
    print("="*80)
    
    revert_beamsplitter_patch()


if __name__ == "__main__":
    run_dynamic_benchmark()
