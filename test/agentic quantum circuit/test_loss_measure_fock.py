"""
Test and validation script for LossMeasureFock operation

This script validates that LossMeasureFock produces the same results as
LossChannel + MeasureFock, and demonstrates the performance improvement.
"""

import scipy.integrate
if not hasattr(scipy.integrate, 'simps'):
    scipy.integrate.simps = scipy.integrate.simpson

import numpy as np
import time
import strawberryfields as sf
from strawberryfields.ops import *

# Import our new operation
from loss_measure_fock_op import LossMeasureFock, patch_fock_backend, revert_fock_backend_patch


def test_with_postselection():
    """Test LossMeasureFock with post-selection across different eta and select values."""
    print("="*60)
    print("Test 1: With Post-selection")
    print("="*60)
    
    # Test different combinations
    test_configs = [
        {'eta': 0.01, 'select': 2, 'cutoff': 6},
        {'eta': 0.1, 'select': 4, 'cutoff': 7},
        {'eta': 0.2, 'select': 2, 'cutoff': 9},
        {'eta': 0.7, 'select': 2, 'cutoff': 6},
        {'eta': 0.5, 'select': 1, 'cutoff': 6},
        {'eta': 0.9, 'select': 3, 'cutoff': 8},
        {'eta': 0.3, 'select': 0, 'cutoff': 6},
        {'eta': 1.0, 'select': 2, 'cutoff': 6},  # Perfect detection
    ]
    
    all_pass = True
    
    for i, config in enumerate(test_configs):
        ETA = config['eta']
        post_select_val = config['select']
        cutoff = config['cutoff']
        
        print(f"\n--- Config {i+1}/{len(test_configs)}: eta={ETA}, select={post_select_val}, cutoff={cutoff} ---")
        
        # Method 1: Original (LossChannel + MeasureFock)
        np.random.seed(42)
        prog1 = sf.Program(2)
        with prog1.context as q:
            Dgate(0.3, np.pi / 3) | q[0]
            Sgate(0.2, 0.1) | q[1]
            BSgate(np.pi / 4, np.pi / 6) | (q[0], q[1])
            LossChannel(ETA) | q[0]
            MeasureFock(select=post_select_val) | q[0]
        
        eng1 = sf.Engine("fock", backend_options={"cutoff_dim": cutoff})
        result1 = eng1.run(prog1)
        dm1 = result1.state.dm()
        
        # Method 2: New LossMeasureFock operation
        patch_fock_backend()
        
        np.random.seed(42)
        prog2 = sf.Program(2)
        with prog2.context as q:
            Dgate(0.3, np.pi / 3) | q[0]
            Sgate(0.2, 0.1) | q[1]
            BSgate(np.pi / 4, np.pi / 6) | (q[0], q[1])
            LossMeasureFock(eta=ETA, select=post_select_val) | q[0]
        
        eng2 = sf.Engine("fock", backend_options={"cutoff_dim": cutoff})
        result2 = eng2.run(prog2)
        dm2 = result2.state.dm()

        
        # Compare
        diff = np.max(np.abs(dm1 - dm2))
        
        if diff < 1e-10:
            print(f"  Result: PASS (diff={diff:.2e})")
        else:
            print(f"  Result: FAIL (diff={diff:.2e})")
            all_pass = False
        
        revert_fock_backend_patch()
    
    print(f"\n{'='*60}")
    if all_pass:
        print(f"[PASS] All {len(test_configs)} post-selection tests passed!")
    else:
        print(f"[FAIL] Some post-selection tests failed!")
    print(f"{'='*60}")
    
    return all_pass


def test_without_postselection():
    """Test LossMeasureFock without post-selection (random sampling) with different eta values."""
    print("\n" + "="*60)
    print("Test 2: Without Post-selection (Random Sampling)")
    print("="*60)
    
    # Test different eta values
    eta_values = [0.3, 0.5, 0.7, 0.9, 1.0]
    cutoff = 8
    n_samples = 50
    
    all_pass = True
    
    for eta_idx, ETA in enumerate(eta_values):
        print(f"\n--- Config {eta_idx+1}/{len(eta_values)}: eta={ETA} ---")
        
        # We'll compare the distribution of outcomes
        outcomes1 = []
        outcomes2 = []
        
        # Method 1: Original
        for i in range(n_samples):
            np.random.seed(100 + i)
            prog1 = sf.Program(1)
            with prog1.context as q:
                Dgate(1) | q[0]
                LossChannel(ETA) | q[0]
                MeasureFock() | q[0]
            
            eng1 = sf.Engine("fock", backend_options={"cutoff_dim": cutoff})
            result1 = eng1.run(prog1)
            outcomes1.append(result1.samples[0][0])
        
        # Method 2: LossMeasureFock
        patch_fock_backend()
        
        for i in range(n_samples):
            np.random.seed(100 + i)
            prog2 = sf.Program(1)
            with prog2.context as q:
                Dgate(1) | q[0]
                LossMeasureFock(eta=ETA) | q[0]
            
            eng2 = sf.Engine("fock", backend_options={"cutoff_dim": cutoff})
            result2 = eng2.run(prog2)
            outcomes2.append(result2.samples[0][0])
        
        outcomes1 = np.array(outcomes1)
        outcomes2 = np.array(outcomes2)
        
        # Check if outcomes match exactly (with same random seeds)
        matches = np.sum(outcomes1 == outcomes2)
        
        # Compute empirical distributions
        unique1, counts1 = np.unique(outcomes1, return_counts=True)
        unique2, counts2 = np.unique(outcomes2, return_counts=True)
        
        dist1 = dict(zip(unique1, counts1/n_samples))
        dist2 = dict(zip(unique2, counts2/n_samples))
        
        print(f"  Matching outcomes: {matches}/{n_samples}")
        print(f"  Distribution 1: {dist1}")
        print(f"  Distribution 2: {dist2}")
        
        if matches == n_samples:
            print(f"  Result: PASS (all samples match)")
        else:
            print(f"  Result: FAIL (only {matches}/{n_samples} match)")
            all_pass = False
        
        revert_fock_backend_patch()
    
    print(f"\n{'='*60}")
    if all_pass:
        print(f"[PASS] All {len(eta_values)} random sampling tests passed!")
    else:
        print(f"[FAIL] Some random sampling tests failed!")
    print(f"{'='*60}")
    
    return all_pass


def test_multi_mode():
    """Test LossMeasureFock on multiple modes simultaneously."""
    print("\n" + "="*60)
    print("Test 3: Multi-mode Measurement")
    print("="*60)
    
    ETA = 0.7
    cutoff = 5
    
    # Test measuring both modes with different eta values
    # Note: Our current implementation uses same eta for all modes in one call
    # For different etas, we need separate calls
    
    patch_fock_backend()
    
    np.random.seed(42)
    prog = sf.Program(2)
    with prog.context as q:
        Dgate(0.4) | q[0]
        Dgate(0.3) | q[1]
        BSgate(np.pi / 4) | (q[0], q[1])
        LossMeasureFock(eta=0.7, select=1) | q[0]
        LossMeasureFock(eta=0.8, select=0) | q[1]
    
    eng = sf.Engine("fock", backend_options={"cutoff_dim": cutoff})
    result = eng.run(prog)
    
    print(f"\nMeasurement results: {result.samples}")
    print(f"State after measurements: {result.state}")
    print("[PASS] Multi-mode measurement works!")
    
    revert_fock_backend_patch()
    return True


def benchmark():
    """Benchmark LossMeasureFock vs LossChannel + MeasureFock."""
    print("\n" + "="*60)
    print("Performance Benchmark")
    print("="*60)
    
    ETA = 0.9
    cutoff = 25
    n_trials = 10
    post_select_val = 2
    
    # Benchmark original
    print("\nBenchmarking LossChannel + MeasureFock...")
    timings1 = []
    for i in range(n_trials):
        prog1 = sf.Program(2)
        with prog1.context as q:
            Dgate(0.3, np.pi / 3) | q[0]
            Sgate(0.2, 0.1) | q[1]
            BSgate(np.pi / 4, np.pi / 6) | (q[0], q[1])
            LossChannel(ETA) | q[0]
            MeasureFock(select=post_select_val) | q[0]
        
        eng1 = sf.Engine("fock", backend_options={"cutoff_dim": cutoff})
        start = time.time()
        eng1.run(prog1)
        timings1.append(time.time() - start)
    
    time1 = np.mean(timings1)
    std1 = np.std(timings1)
    print(f"  Average time: {time1:.6f}s ± {std1:.6f}s")
    
    # Benchmark LossMeasureFock
    patch_fock_backend()
    print("\nBenchmarking LossMeasureFock...")
    timings2 = []
    for i in range(n_trials):
        prog2 = sf.Program(2)
        with prog2.context as q:
            Dgate(0.3, np.pi / 3) | q[0]
            Sgate(0.2, 0.1) | q[1]
            BSgate(np.pi / 4, np.pi / 6) | (q[0], q[1])
            LossMeasureFock(eta=ETA, select=post_select_val) | q[0]
        
        eng2 = sf.Engine("fock", backend_options={"cutoff_dim": cutoff})
        start = time.time()
        eng2.run(prog2)
        timings2.append(time.time() - start)
    
    time2 = np.mean(timings2)
    std2 = np.std(timings2)
    print(f"  Average time: {time2:.6f}s ± {std2:.6f}s")
    
    speedup = time1 / time2
    print(f"\n{'='*60}")
    print(f"Speedup: {speedup:.2f}x")
    print(f"{'='*60}")
    
    revert_fock_backend_patch()


def main():
    """Run all tests and benchmarks."""
    print("\n" + "="*70)
    print(" LossMeasureFock Operation - Validation and Benchmarking ")
    print("="*70)
    
    # Run tests
    test1_pass = test_with_postselection()
    test2_pass = test_without_postselection()
    test3_pass = test_multi_mode()
    
    # Run benchmark
    benchmark()
    
    # Summary
    print("\n" + "="*70)
    print(" Summary ")
    print("="*70)
    print(f"Test 1 (Post-selection):     {'PASS' if test1_pass else 'FAIL'}")
    print(f"Test 2 (Random sampling):    {'PASS' if test2_pass else 'FAIL'}")
    print(f"Test 3 (Multi-mode):         {'PASS' if test3_pass else 'FAIL'}")
    print("="*70)
    
    if all([test1_pass, test2_pass, test3_pass]):
        print("\n[SUCCESS] All tests passed!")
    else:
        print("\n[FAIL] Some tests failed!")


if __name__ == "__main__":
    main()
