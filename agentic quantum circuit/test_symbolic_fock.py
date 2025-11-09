"""
Test if Fock backend supports symbolic parameters (FreeParameters).
If this works, we can create the program ONCE and just update parameter values.
"""

import scipy.integrate
if not hasattr(scipy.integrate, 'simps'):
    scipy.integrate.simps = scipy.integrate.simpson

import numpy as np
import time
import strawberryfields as sf
from strawberryfields.ops import *

# Import optimized loss channel
from loss_measure_fock_optimized_v2 import LossMeasureFock, patch_fock_backend
patch_fock_backend()

cutoff = 25
n_iterations = 100
eta = 0.99

print("="*80)
print(" Testing Symbolic Parameters with Fock Backend ")
print("="*80)

# Create program ONCE with symbolic parameters
print("\n[1/2] Creating program with symbolic parameters...")
prog = sf.Program(2)

# Define symbolic parameters
r_squeeze, phi_squeeze, theta_bs = prog.params("r_squeeze", "phi_squeeze", "theta_bs")

with prog.context as q:
    Sgate(r_squeeze, phi_squeeze) | q[1]
    BSgate(theta_bs, 0) | (q[0], q[1])
    LossMeasureFock(eta=eta) | q[0]
    BSgate(np.pi/2, 0) | (q[0], q[1])

print("  Program created successfully with symbolic params")
print(f"  Free parameters: {list(prog.free_params.keys())}")

# Create engine ONCE
eng = sf.Engine("fock", backend_options={"cutoff_dim": cutoff})
print("  Engine created")

# Test running with different parameter values
print("\n[2/2] Running program multiple times with different parameter values...")

timings = []
for i in range(n_iterations + 1):
    # Generate random parameter values
    np.random.seed(1000 + i)
    r_val = np.random.uniform(0.3, 1.0)
    phi_val = np.random.uniform(0, 2*np.pi)
    theta_val = np.random.uniform(0, np.pi/2)
    
    # Bind parameters to values
    args = {
        "r_squeeze": r_val,
        "phi_squeeze": phi_val,
        "theta_bs": theta_val
    }
    
    start = time.time()
    
    # Run with bound parameters
    result = eng.run(prog, args=args)
    dm = result.state.reduced_dm(modes=[0])
    
    elapsed = time.time() - start
    
    if i > 0:  # Skip warmup
        timings.append(elapsed)

time_symbolic = np.mean(timings)
std_symbolic = np.std(timings)
fps_symbolic = 1.0 / time_symbolic

print(f"  Time per iteration: {time_symbolic*1000:.2f} +/- {std_symbolic*1000:.2f} ms")
print(f"  FPS: {fps_symbolic:.1f}")

# Compare with baseline (fresh program each time)
print("\n[BASELINE] Fresh program each iteration...")
timings_baseline = []
for i in range(n_iterations + 1):
    np.random.seed(1000 + i)
    r_val = np.random.uniform(0.3, 1.0)
    phi_val = np.random.uniform(0, 2*np.pi)
    theta_val = np.random.uniform(0, np.pi/2)
    
    start = time.time()
    
    # Create fresh engine
    eng_fresh = sf.Engine("fock", backend_options={"cutoff_dim": cutoff})
    
    # Create fresh program
    prog_fresh = sf.Program(2)
    with prog_fresh.context as q:
        Sgate(r_val, phi_val) | q[1]
        BSgate(theta_val, 0) | (q[0], q[1])
        LossMeasureFock(eta=eta) | q[0]
        BSgate(np.pi/2, 0) | (q[0], q[1])
    
    result = eng_fresh.run(prog_fresh)
    dm = result.state.reduced_dm(modes=[0])
    
    elapsed = time.time() - start
    
    if i > 0:
        timings_baseline.append(elapsed)

time_baseline = np.mean(timings_baseline)
std_baseline = np.std(timings_baseline)
fps_baseline = 1.0 / time_baseline

print(f"  Time per iteration: {time_baseline*1000:.2f} +/- {std_baseline*1000:.2f} ms")
print(f"  FPS: {fps_baseline:.1f}")

# Results
print("\n" + "="*80)
print(" RESULTS ")
print("="*80)
print(f"\nBaseline (fresh program + engine): {time_baseline*1000:.2f} ms → {fps_baseline:.1f} FPS")
print(f"Symbolic params (reuse program):   {time_symbolic*1000:.2f} ms → {fps_symbolic:.1f} FPS")
print(f"\nSpeedup: {time_baseline/time_symbolic:.2f}x")
print(f"FPS improvement: {fps_symbolic/fps_baseline:.2f}x")

if time_symbolic < time_baseline:
    print("\n✅ SUCCESS! Symbolic parameters are FASTER!")
    print("\nThis approach should be used in quantum_circuit_env.py:")
    print("  1. Create program with symbolic params in __init__()")
    print("  2. Create engine once in __init__() or reset()")
    print("  3. In step(), just call eng.run(prog, args={...})")
else:
    print("\n⚠️ Symbolic parameters are not faster in this test")
    print("Need to investigate further...")

# Estimate training impact
print("\n" + "="*80)
print(" TRAINING IMPACT ESTIMATE ")
print("="*80)
training_fps_current = 11
training_fps_no_loss = 130

expected_training_fps = training_fps_current * (fps_symbolic / fps_baseline)
print(f"\nCurrent training FPS: {training_fps_current}")
print(f"Expected training FPS: ~{expected_training_fps:.0f}")
print(f"Target (no loss): {training_fps_no_loss}")
print(f"\nRemaining overhead after fix: {training_fps_no_loss/expected_training_fps:.1f}x")

print("\n" + "="*80)
