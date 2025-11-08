"""
Test optimization improvement for random sampling with large cutoff
"""

import scipy.integrate
if not hasattr(scipy.integrate, 'simps'):
    scipy.integrate.simps = scipy.integrate.simpson

import numpy as np
import time
import strawberryfields as sf
from strawberryfields.ops import *
from strawberryfields import ops

print("="*80)
print(" Comparing Optimized Loss vs LossMeasureFock for Random Sampling with Large Cutoff ")
print("="*80)

# Test configuration
eta = 0.5
cutoff = 25
n_trials = 50
n_modes = 2

print(f"\nTest Configuration:")
print(f"  eta = {eta}")
print(f"  cutoff = {cutoff}")
print(f"  n_modes = {n_modes}")
print(f"  select = None (random sampling)")
print(f"  n_trials = {n_trials}\n")

# No Loss (Baseline for measuring pure overhead)
print("[0/3] Benchmarking NO LOSS (just MeasureFock)...")
from loss_channel_patch_v2 import revert_patch as revert_loss
revert_loss()
is_pure = []
timings_no_loss = []
eng = sf.Engine("fock", backend_options={"cutoff_dim": cutoff})
for trial in range(n_trials + 1):  # +1 for warmup
    # Random gate parameters for each trial
    np.random.seed(1000 + trial)
    r_squeeze = np.random.uniform(0.3, 1.0)
    phi_squeeze = np.random.uniform(0, 2*np.pi)
    theta_bs = np.random.uniform(0, np.pi/2)
    
    prog = sf.Program(n_modes)
    with prog.context as q:
        Sgate(r_squeeze, phi_squeeze) | q[1]
        BSgate(theta_bs, 0) | (q[0], q[1])
        # No loss channel!
        MeasureFock() | q[0]
        BSgate(np.pi/2, 0) | (q[0], q[1])
    
    
    start = time.time()
    state=eng.run(prog).state
    elapsed = time.time() - start
    
    if trial > 0:  # Skip first trial (warmup)
        timings_no_loss.append(elapsed)

time_no_loss = np.mean(timings_no_loss)
std_no_loss = np.std(timings_no_loss)
print(f"is pure = {state.is_pure}")
print(f"  Time: {time_no_loss*1000:.2f} +/- {std_no_loss*1000:.2f} ms (excluding first trial)\n")

# Original
print("[1/3] Benchmarking ORIGINAL Loss channel...")
revert_loss()

timings_orig = []
eng = sf.Engine("fock", backend_options={"cutoff_dim": cutoff})
for trial in range(n_trials + 1):  # +1 for warmup
    # Random gate parameters for each trial
    np.random.seed(1000 + trial)
    r_squeeze = np.random.uniform(0.3, 1.0)
    phi_squeeze = np.random.uniform(0, 2*np.pi)
    theta_bs = np.random.uniform(0, np.pi/2)
    
    prog = sf.Program(n_modes)
    with prog.context as q:
        Sgate(r_squeeze, phi_squeeze) | q[1]
        BSgate(theta_bs, 0) | (q[0], q[1])
        LossChannel(eta) | q[0]
        MeasureFock() | q[0]
        BSgate(np.pi/2, 0) | (q[0], q[1])


    start = time.time()
    state=eng.run(prog).state
    elapsed = time.time() - start
    
    if trial > 0:  # Skip first trial (warmup)
        timings_orig.append(elapsed)

time_orig = np.mean(timings_orig)
std_orig = np.std(timings_orig)
print(f"is pure = {state.is_pure}")
print(f"  Time: {time_orig*1000:.2f} +/- {std_orig*1000:.2f} ms (excluding first trial)\n")

# Optimized Loss
print("[2/3] Benchmarking OPTIMIZED Loss channel...")
from loss_channel_patch_v2 import patch_loss_channel
patch_loss_channel()

# Warmup JIT
print("  JIT warmup...")
warmup = sf.Program(2)
with warmup.context as q:
    LossChannel(0.5) | q[0]
eng_warmup = sf.Engine("fock", backend_options={"cutoff_dim": 5})
eng_warmup.run(warmup)
print("  Warmup done")

timings_opt_loss = []
eng = sf.Engine("fock", backend_options={"cutoff_dim": cutoff})
for trial in range(n_trials + 1):  # +1 for warmup
    # Random gate parameters for each trial
    np.random.seed(1000 + trial)
    r_squeeze = np.random.uniform(0.3, 1.0)
    phi_squeeze = np.random.uniform(0, 2*np.pi)
    theta_bs = np.random.uniform(0, np.pi/2)
    
    prog = sf.Program(n_modes)
    with prog.context as q:
        Sgate(r_squeeze, phi_squeeze) | q[1]
        Sgate(r_squeeze, phi_squeeze) | q[1]
        BSgate(theta_bs, 0) | (q[0], q[1])
        LossChannel(eta) | q[0]
        BSgate(np.pi/2, 0) | (q[0], q[1])

    
    start = time.time()
    state=eng.run(prog).state
    elapsed = time.time() - start
    
    if trial > 0:  # Skip first trial
        timings_opt_loss.append(elapsed)

time_opt_loss = np.mean(timings_opt_loss)
std_opt_loss = np.std(timings_opt_loss)
print(f"is pure = {state.is_pure}")

print(f"  Time: {time_opt_loss*1000:.2f} +/- {std_opt_loss*1000:.2f} ms (excluding first trial)\n")

# LossMeasureFock
print("[3/3] Benchmarking LossMeasureFock ...")
from loss_measure_fock_optimized_v2 import (
    LossMeasureFock,
    patch_fock_backend,
    revert_fock_backend_patch
)
patch_fock_backend()

# Warmup JIT
print("  JIT warmup...")
warmup2 = sf.Program(1)
with warmup2.context as q:
    Dgate(0.1) | q[0]
    LossMeasureFock(eta=0.5) | q[0]
eng_warmup2 = sf.Engine("fock", backend_options={"cutoff_dim": 5})
eng_warmup2.run(warmup2)
print("  Warmup done")

timings_lmf = []
eng = sf.Engine("fock", backend_options={"cutoff_dim": cutoff})
for trial in range(n_trials + 1):  # +1 for warmup
    # Random gate parameters for each trial
    np.random.seed(1000 + trial)
    r_squeeze = np.random.uniform(0.3, 1.0)
    phi_squeeze = np.random.uniform(0, 2*np.pi)
    theta_bs = np.random.uniform(0, np.pi/2)
    
    prog = sf.Program(n_modes)
    with prog.context as q:
        Sgate(r_squeeze, phi_squeeze) | q[1]
        BSgate(theta_bs, 0) | (q[0], q[1])
        LossMeasureFock(eta=eta) | q[0]
        BSgate(np.pi/2, 0) | (q[0], q[1])

    start = time.time()
    eng.run(prog)
    elapsed = time.time() - start
    
    if trial > 0:  # Skip first trial
        timings_lmf.append(elapsed)

time_lmf = np.mean(timings_lmf)
std_lmf = np.std(timings_lmf)
print(f"  Time: {time_lmf*1000:.2f} +/- {std_lmf*1000:.2f} ms (excluding first trial)\n")

revert_fock_backend_patch()

# Summary
print("="*80)
print(" RESULTS ")
print("="*80)
print(f"\n{'Method':<30} {'Time (ms)':<15} {'Speedup':<12} {'Overhead':<10}")
print("-" * 85)
print(f"{'No Loss (baseline)':<30} {time_no_loss*1000:>10.2f}     {'-':<12} {'-':<10}")
print(f"{'Original Loss channel':<30} {time_orig*1000:>10.2f}     {'1.00x':<12} {time_orig/time_no_loss:>6.2f}x")
print(f"{'Optimized Loss channel':<30} {time_opt_loss*1000:>10.2f}     {time_orig/time_opt_loss:>5.2f}x      {time_opt_loss/time_no_loss:>6.2f}x")
print(f"{'LossMeasureFock':<30} {time_lmf*1000:>10.2f}     {time_orig/time_lmf:>5.2f}x      {time_lmf/time_no_loss:>6.2f}x")

print("\n" + "="*80)
print(" KEY INSIGHTS ")
print("="*80)
print(f"Loss Channel Overhead (Original): {(time_orig - time_no_loss)*1000:.2f} ms ({(time_orig/time_no_loss - 1)*100:.1f}% increase)")
print(f"Loss Channel Overhead (LossMeasureFock): {(time_lmf - time_no_loss)*1000:.2f} ms ({(time_lmf/time_no_loss - 1)*100:.1f}% increase)")
print(f"\nOverhead Reduction: {(time_orig - time_no_loss)/(time_lmf - time_no_loss):.2f}x")

print("\n" + "="*80)
print(" IMPROVEMENT ")
print("="*80)
print(f"LossMeasureFock vs Optimized Loss: {time_opt_loss/time_lmf:.2f}x faster")
print(f"LossMeasureFock vs Original:       {time_orig/time_lmf:.2f}x faster")

print("="*80)
