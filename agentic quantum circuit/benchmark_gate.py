"""
Benchmark comparing speed of Sgate, BSgate, Dgate, and MeasureFock by increasing cutoff in 2 mode circuit
"""

import scipy.integrate
if not hasattr(scipy.integrate, 'simps'):
    scipy.integrate.simps = scipy.integrate.simpson

import numpy as np
import time
import strawberryfields as sf
from strawberryfields.ops import *
from strawberryfields import ops
from monitored_loss_measure_fock_patch import MonitoredLossMeasureFock, patch_fock_backend, decode_measurement_result
patch_fock_backend()


print("="*80)
print(" Benchmark: Sgate vs BSgate vs Dgate vs MeasureFock with Varying Cutoff ")
print("="*80)

# Test configuration
cutoffs = [5, 10, 15, 20, 25, 30, 40, 50]
n_trials = 20
n_modes = 2

print(f"\nTest Configuration:")
print(f"  cutoffs = {cutoffs}")
print(f"  n_modes = {n_modes}")
print(f"  n_trials = {n_trials}\n")

# Storage for results
results = {
    'Sgate': {'cutoffs': [], 'times': [], 'stds': []},
    'BSgate': {'cutoffs': [], 'times': [], 'stds': []},
    'Dgate': {'cutoffs': [], 'times': [], 'stds': []},
    'MeasureFock': {'cutoffs': [], 'times': [], 'stds': []}
}

for cutoff in cutoffs:
    print(f"\nTesting with cutoff = {cutoff}")
    print("-" * 60)
    
    timings_sgate = []
    timings_bsgate = []
    timings_dgate = []
    timings_mf = []
    
    eng = sf.Engine("fock", backend_options={"cutoff_dim": cutoff})
    
    for trial in range(n_trials + 1):  # +1 for warmup
        # Random gate parameters for each trial
        np.random.seed(1000 + trial)
        r_squeeze = np.random.uniform(0.3, 1.0)
        phi_squeeze = np.random.uniform(0, 2*np.pi)
        theta_bs = np.random.uniform(0, np.pi/2)
        alpha = np.random.uniform(0.5, 2.0)
        phi_d = np.random.uniform(0, 2*np.pi)
        
        # Test Sgate
        prog = sf.Program(n_modes)
        with prog.context as q:
            Sgate(r_squeeze, phi_squeeze) | q[1]
        
        start = time.time()
        state = eng.run(prog).state
        elapsed_sgate = time.time() - start
        if trial > 0:
            timings_sgate.append(elapsed_sgate)
        
        # Test BSgate
        prog = sf.Program(n_modes)
        with prog.context as q:
            BSgate(theta_bs, 0) | (q[0], q[1])
        
        start = time.time()
        state = eng.run(prog).state
        elapsed_bsgate = time.time() - start
        if trial > 0:
            timings_bsgate.append(elapsed_bsgate)
        
        # Test Dgate
        prog = sf.Program(n_modes)
        with prog.context as q:
            Dgate(alpha, phi_d) | q[0]
        
        start = time.time()
        state = eng.run(prog).state
        elapsed_dgate = time.time() - start
        if trial > 0:
            timings_dgate.append(elapsed_dgate)
        
        # Test MeasureFock
        prog = sf.Program(n_modes)
        with prog.context as q:
            MonitoredLossMeasureFock(1) | q[0]
        
        start = time.time()
        state = eng.run(prog).state
        elapsed_mf = time.time() - start
        if trial > 0:
            timings_mf.append(elapsed_mf)
    
    # Calculate statistics
    time_sgate = np.mean(timings_sgate)
    std_sgate = np.std(timings_sgate)
    results['Sgate']['cutoffs'].append(cutoff)
    results['Sgate']['times'].append(time_sgate)
    results['Sgate']['stds'].append(std_sgate)
    
    time_bsgate = np.mean(timings_bsgate)
    std_bsgate = np.std(timings_bsgate)
    results['BSgate']['cutoffs'].append(cutoff)
    results['BSgate']['times'].append(time_bsgate)
    results['BSgate']['stds'].append(std_bsgate)
    
    time_dgate = np.mean(timings_dgate)
    std_dgate = np.std(timings_dgate)
    results['Dgate']['cutoffs'].append(cutoff)
    results['Dgate']['times'].append(time_dgate)
    results['Dgate']['stds'].append(std_dgate)
    
    time_mf = np.mean(timings_mf)
    std_mf = np.std(timings_mf)
    results['MeasureFock']['cutoffs'].append(cutoff)
    results['MeasureFock']['times'].append(time_mf)
    results['MeasureFock']['stds'].append(std_mf)
    
    print(f"  Sgate:       {time_sgate*1000:.2f} +/- {std_sgate*1000:.2f} ms")
    print(f"  BSgate:      {time_bsgate*1000:.2f} +/- {std_bsgate*1000:.2f} ms")
    print(f"  Dgate:       {time_dgate*1000:.2f} +/- {std_dgate*1000:.2f} ms")
    print(f"  MeasureFock: {time_mf*1000:.2f} +/- {std_mf*1000:.2f} ms")

# Summary table
print("\n" + "="*80)
print(" SUMMARY: Gate Execution Times (ms) ")
print("="*80)
print(f"\n{'Cutoff':<10} {'Sgate':<20} {'BSgate':<20} {'Dgate':<20} {'MeasureFock':<20}")
print("-" * 90)

for i, cutoff in enumerate(cutoffs):
    sgate_str = f"{results['Sgate']['times'][i]*1000:.2f} ± {results['Sgate']['stds'][i]*1000:.2f}"
    bsgate_str = f"{results['BSgate']['times'][i]*1000:.2f} ± {results['BSgate']['stds'][i]*1000:.2f}"
    dgate_str = f"{results['Dgate']['times'][i]*1000:.2f} ± {results['Dgate']['stds'][i]*1000:.2f}"
    mf_str = f"{results['MeasureFock']['times'][i]*1000:.2f} ± {results['MeasureFock']['stds'][i]*1000:.2f}"
    print(f"{cutoff:<10} {sgate_str:<20} {bsgate_str:<20} {dgate_str:<20} {mf_str:<20}")

print("\n" + "="*80)
print(" Benchmark Complete ")
print("="*80)

