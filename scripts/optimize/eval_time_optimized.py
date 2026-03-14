# Added imports for file loading, plotting and CLI handling; removed duplicate imports
import argparse
import glob
import json
import pickle
import numpy as np
import os
from pathlib import Path
import matplotlib.pyplot as plt
import platform
import itertools

import quantum_agent
import strawberryfields as sf
from strawberryfields.ops import Ket, DensityMatrix

# Optimization and Component modules
import quantum_agent.optimization.time_circuits as circuit_module
import quantum_agent.components.targets as target_module
from quantum_agent.optimization.time_circuits import *
from quantum_agent.optimization.time_interfaces import TimeMultiplexedCircuit
from quantum_agent.optimization.time_runner import evaluate_time_domain_circuit
from quantum_agent.components.targets import *
from quantum_agent.utils import *
from quantum_agent.factory import create_from_config


def print_optimized_parameters(circuit, flat_x, mapped_params=None):
    """
    Pretty prints the optimized parameter schedule.
    
    Args:
        circuit: The TimeMultiplexedCircuit instance
        flat_x: Flat parameter vector
        mapped_params: Pre-mapped parameters (optional, will compute if None)
    """
    # Map parameters if not provided
    if mapped_params is None:
        mapped_params = circuit.map_parameters(np.asarray(flat_x))
    
    param_names = circuit.per_step_parameter_names
    
    print("\n" + "="*70)
    print(" OPTIMIZED PARAMETERS SCHEDULE ")
    print("="*70)
    
    # Calculate column widths
    step_width = 6
    param_width = 14
    
    # Header
    header_parts = [f"{'Step':>{step_width}}"]
    for name in param_names:
        display_name = name[:param_width].center(param_width)
        header_parts.append(display_name)
    
    separator = "-" * (step_width + 2) + "-" * ((param_width + 2) * len(param_names))
    
    print(" | ".join(header_parts))
    print(separator)
    
    # Handle both 2D arrays (steps x params) and edge cases
    if hasattr(mapped_params, 'shape') and len(mapped_params.shape) >= 2:
        for step_idx in range(mapped_params.shape[0]):
            row_parts = [f"{step_idx:>{step_width}}"]
            for param_idx, val in enumerate(mapped_params[step_idx]):
                row_parts.append(f"{val:>{param_width}.6f}")
            print(" | ".join(row_parts))
    else:
        # Fallback for 1D or irregular structures
        print(f"Parameters: {mapped_params}")
    
    print("="*70)


def print_config_info(circuit_config, target_configs):
    """Pretty prints loaded configuration for inspection."""
    print("\n" + "="*50)
    print(" EXPERIMENT CONFIGURATION ")
    print("="*50)
    
    # Circuit
    if circuit_config:
        print(f"\nCircuit: {circuit_config.get('class_name', 'Unknown')}")
        if 'params' in circuit_config:
            for k, v in circuit_config['params'].items():
                print(f"  • {k:<15} : {v}")
    else:
        print("\nCircuit: None")
            
    # Targets
    if not target_configs:
        print("\nTargets: None")
    else:
        print(f"\nTargets ({len(target_configs)}):")
        for i, t_cfg in enumerate(target_configs):
            name = t_cfg.get('class_name', 'Unknown')
            print(f"  [{i}] {name}")
            if 'params' in t_cfg:
                for k, v in t_cfg['params'].items():
                    print(f"      - {k:<13} : {v}")
    print("="*50 + "\n")


def sanitize_config_paths(config):
    """
    Recursively fix file paths in configuration dictionaries to work across WSL/Windows.
    """
    if isinstance(config, dict):
        new_config = config.copy()
        # Specific fix for CoreGKPTarget csv_path
        if config.get('class_name') == 'CoreGKPTarget' and 'params' in config:
            params = config['params'].copy()
            if 'csv_path' in params:
                path_str = params['csv_path']
                # Detect if we are on Windows but the path is WSL
                if platform.system() == "Windows" and path_str.startswith("/mnt/"):
                    # Convert /mnt/e/folder -> E:/folder
                    parts = path_str.split('/')
                    if len(parts) > 2:
                        drive_letter = parts[2] # 'e'
                        rest_of_path = "/".join(parts[3:])
                        new_path = f"{drive_letter.upper()}:/{rest_of_path}"
                        params['csv_path'] = new_path
                        print(f"Sanitized WSL path: {path_str} -> {new_path}")
            new_config['params'] = params
            return new_config
        
        # General recursion for other keys
        for k, v in new_config.items():
            new_config[k] = sanitize_config_paths(v)
        return new_config
    
    elif isinstance(config, list):
        return [sanitize_config_paths(item) for item in config]
    
    return config

def get_all_optimization_results(circuit: TimeMultiplexedCircuit, flat_params: np.ndarray, targets: list, cutoff_dim: int, beam_width: int = 100, measurement_patterns=None):
    """
    Runs the circuit evaluation with the provided parameters to generate full branch details.
    
    Args:
        circuit: The circuit instance.
        flat_params: The optimized parameters.
        targets: List of TargetGenerator instances.
        cutoff_dim: Simulation cutoff.
        beam_width: Beam width for search (higher = more branches captured).
        measurement_patterns: Optional list/array of fixed measurement patterns to evaluate.
        
    Returns:
        dict: The result dictionary containing 'loss', 'branches', 'expected_fidelity', etc.
    """
    target_kets = [t.get_target_ket(cutoff_dim) for t in targets]
    
    # Run evaluation with details enabled
    # We set penalty/weights to 0/defaults as we are analyzing physical outcomes
    return evaluate_time_domain_circuit(
        flat_params,
        circuit,
        target_kets,
        cutoff_dim,
        beam_width=beam_width,
        penalty_strength=0.0,
        success_threshold=0.99, # Used for internal loss calculation
        measurement_patterns=measurement_patterns,
        return_details=True
    )




def _compute_max_fidelity_dm(rho, target_ket, n_fft=256):
    """
    Computes max fidelity F = max_phi <target_phi | rho | target_phi>
    using FFT for rotation optimization.
    """
    D = len(target_ket)
    coeffs = np.zeros(n_fft, dtype=np.complex128)
    
    # Precompute conjugated target
    target_conj = np.conj(target_ket)
    
    # Delta = 0: C_0 = sum_n rho_nn |tau_n|^2
    # diag(rho) gives main diagonal elements rho_{nn}
    coeffs[0] = np.sum(np.diagonal(rho) * np.abs(target_ket)**2)
    
    # Delta > 0: C_delta = sum_n rho_{n, n+delta} * tau_{n+delta} * tau_n^*
    # Corresponds to e^{i * delta * phi} terms in the Fourier series of F(phi)
    for delta in range(1, D):
        # Upper diagonal of rho at offset delta: elements rho_{n, n+delta}
        rho_diag = np.diagonal(rho, offset=delta)
        
        # Target terms matching indices: tau_{n+delta} * tau_n^*
        # target_ket[delta:] corresponds to tau_{delta}, tau_{delta+1}... (which is tau_{n+delta})
        # target_conj[:-delta] corresponds to tau_0^*, tau_1^*... (which is tau_n^*)
        term = np.sum(rho_diag * target_ket[delta:] * target_conj[:-delta])
        
        # Assign C_delta to index delta
        coeffs[delta] = term
        # Assign C_{-delta} to index -delta (conjugate symmetry for real result)
        coeffs[-delta] = np.conj(term)
        
    # IFFT to compute Fourier series sum C_k e^{i k phi}
    # Multiply by n_fft because ifft includes 1/N scaling
    vals = np.fft.ifft(coeffs) * n_fft
    
    # Fidelity is strictly real; take max over sampled phases
    return float(np.max(np.real(vals)))


def evaluate_time_domain_circuit_dm(flat_params, circuit, target_kets, cutoff_dim, beam_width, 
                                    penalty_strength, prob_power, measurement_patterns=None):
    """
    Evaluates the circuit using Density Matrices to support loss/noise, using Beam Search.
    
    Args:
        flat_params: Flat parameter vector.
        circuit: TimeMultiplexedCircuit instance.
        target_kets: List of target kets (pure states).
        cutoff_dim: Fock cutoff.
        beam_width: Number of branches to keep.
        penalty_strength: (Unused in this evaluator, kept for signature compatibility)
        prob_power: (Unused in this evaluator, kept for signature compatibility)
        measurement_patterns: Optional list or array of fixed outcome patterns.
                              If provided, beam search is disabled and only these paths are evaluated.
                              Shape: (n_sequences, steps, n_meas_modes).
    
    Returns:
        dict: Results containing 'branches', 'expected_fidelity', etc.
    """
    
    # 0. Setup Parameters
    n_init = circuit.num_initial_parameters
    if n_init > 0:
        init_params = flat_params[:n_init]
        step_params = flat_params[n_init:]
    else:
        init_params = np.array([])
        step_params = flat_params

    mapped_params = circuit.map_parameters(step_params)
    # Cap measurement cutoffs by simulation cutoff_dim to prevent indexing errors
    meas_specs = [(m, min(c, cutoff_dim)) for m, c in circuit.get_measurement_specs()]
    
    # Identify modes: Max measured mode index + 1 (assuming Loop is 0)
    meas_modes = [m for m, c in meas_specs]
    n_modes = max(meas_modes) + 1 if meas_modes else 1
    
    # 1. Initialize State (Loop Mode 0)
    initial_ket = circuit.get_initial_state_ket(init_params, cutoff_dim)
    # Convert to Density Matrix: |psi><psi|
    initial_dm = np.outer(initial_ket, np.conj(initial_ket))
    
    use_fixed_patterns = measurement_patterns is not None
    patterns_arr = None

    if use_fixed_patterns:
        # Normalize patterns to numpy array (n_patterns, steps, n_modes)
        try:
            patterns_arr = np.array(measurement_patterns, dtype=int)
        except Exception:
            # Fallback for ragged lists or try basic conversion
            patterns_arr = np.array(list(measurement_patterns), dtype=int)
            
        if patterns_arr.ndim == 2:
            patterns_arr = patterns_arr[None, ...]
        if patterns_arr.ndim != 3:
             raise ValueError("measurement_patterns must be shape (n_seq, steps, n_modes) or (steps, n_modes)")
        
        n_patterns = patterns_arr.shape[0]
        # Initialize branches for each pattern
        active_branches = []
        for i in range(n_patterns):
             active_branches.append({
                'dm': initial_dm.copy(), # Copy initial DM for each path
                'prob': 1.0,
                'outcomes': [],
                'pattern_idx': i
             })
    else:
        # Branch structure: {'dm': np.ndarray, 'prob': float, 'outcomes': list of tuples}
        active_branches = [{
            'dm': initial_dm,
            'prob': 1.0,
            'outcomes': []
        }]
    
    # 2. Time Steps
    for step in range(circuit.steps):
        step_p = mapped_params[step]
        candidates = []
        
        # Prepare outcome combinations for this step (only used if NOT fixed patterns)
        outcome_combos = []
        if not use_fixed_patterns:
            ranges = [range(c) for _, c in meas_specs]
            outcome_combos = list(itertools.product(*ranges))
        
        for branch in active_branches:
            parent_dm = branch['dm']
            parent_prob = branch['prob']
            
            # Run Circuit Step
            eng = sf.Engine("fock", backend_options={"cutoff_dim": cutoff_dim})
            prog = sf.Program(n_modes)
            with prog.context as q:
                DensityMatrix(parent_dm) | q[0]
            
            eng.run(prog) 

            # This runs the unitary + loss channels (if configured in circuit)
            result = circuit.run_step(None, step, step_p, eng)
            
            # Extract full density matrix
            # Shape is interleaved: (Ket0, Bra0, Ket1, Bra1, ...)
            full_dm = result.state.dm()
            
            # Determine which outcomes to process for this branch
            if use_fixed_patterns:
                p_idx = branch['pattern_idx']
                # Extract outcome for this step: shape (n_modes,)
                target_outcome = tuple(patterns_arr[p_idx, step, :])
                loop_outcomes = [target_outcome]
            else:
                loop_outcomes = outcome_combos

            # 3. Measurement Projection & Branching
            for outcomes in loop_outcomes:
                # outcomes is tuple (n_m1, n_m2...)
                
                # Construct slicer for the tensor
                indexer = [slice(None)] * (2 * n_modes)
                
                for i, (m_idx, _) in enumerate(meas_specs):
                    val = outcomes[i]
                    # Strawberry Fields dm() indices are (Ket0, Bra0, Ket1, Bra1, ...)
                    # Mode k corresponds to indices 2*k and 2*k+1
                    
                    # Fix Ket index for mode m_idx
                    indexer[2 * m_idx] = val
                    # Fix Bra index for mode m_idx
                    indexer[2 * m_idx + 1] = val
                
                # Perform slice
                # The remaining axes correspond to unmeasured modes (Mode 0)
                # Shape becomes (D, D) for the loop mode
                projected_dm = full_dm[tuple(indexer)]
                
                # Calculate probability (Trace of the unnormalized DM block)
                trace_prob = np.real(np.trace(projected_dm))
                
                if trace_prob > 1e-12:
                    # Renormalize
                    new_dm = projected_dm / trace_prob
                    
                    # Update chain probability
                    new_chain_prob = parent_prob * trace_prob
                    
                    # Record
                    cand = {
                        'dm': new_dm,
                        'prob': new_chain_prob,
                        'outcomes': branch['outcomes'] + [outcomes]
                    }
                    if use_fixed_patterns:
                        cand['pattern_idx'] = branch.get('pattern_idx')
                        
                    candidates.append(cand)
        
        # 4. Pruning or Update
        if use_fixed_patterns:
            # In fixed pattern mode, we don't beam search; we just keep the surviving paths.
            active_branches = candidates
        else:
            # Beam Search Pruning
            candidates.sort(key=lambda x: x['prob'], reverse=True)
            active_branches = candidates[:beam_width]
        
        # Check if dead
        if not active_branches:
            break

    # 5. Finalize Results
    results = []
    expected_fidelity = 0.0
    total_captured_prob = 0.0
    
    for branch in active_branches:
        rho = branch['dm']
        prob = branch['prob']
        total_captured_prob += prob
        
        # Calculate Fidelity against targets (maximized over phase using FFT)
        max_fid = 0.0
        best_target_idx = 0
        
        for t_i, t_ket in enumerate(target_kets):
            fid = _compute_max_fidelity_dm(rho, t_ket)
            
            if fid > max_fid:
                max_fid = fid
                best_target_idx = t_i
        
        expected_fidelity += prob * max_fid
        
        # Reshape outcomes to tuple of tuples
        res_outcomes = tuple(branch['outcomes'])
        
        results.append({
            'outcome': res_outcomes,
            'prob': prob,
            'fidelity': max_fid,
            'target_idx': best_target_idx
        })

    return {
        "branches": results,
        "expected_fidelity": expected_fidelity,
        "total_probability": total_captured_prob
    }


def print_optimization_statistics(result: dict, success_threshold: float = 0.99, target_names: list = None):
    """
    Calculates and prints probabilities, fidelities, and aggregated success statistics.
    
    Args:
        result: The dictionary returned by evaluate_time_domain_circuit or loaded from pickle.
                Must contain a 'branches' key.
        success_threshold: Fidelity threshold to consider a branch "successful".
        target_names: Optional list of names corresponding to target indices.
    """
    branches = result.get('branches', [])
    if not branches:
        print("No branch details found in the provided results.")
        return

    # Generate generic target names if not provided
    if target_names is None:
        max_idx = max((b.get('target_idx', 0) for b in branches), default=0)
        target_names = [f"Target_{i}" for i in range(max_idx + 1)]

    # 1. Print Schedule of Parameters (if x is present, this is usually handled elsewhere, 
    # but we focus on outcomes here).

    print("-" * 80)
    print("Dominant Outcome Branches (Sorted by Prob):")
    print(f"{'Outcome':<20} {'Prob':<10} {'Fidelity':<10} {'1-Fid':<10} {'Best Target':<15}")
    print("-" * 80)
    
    sorted_branches = sorted(branches, key=lambda x: x['prob'], reverse=True)
    
    total_prob = 0.0
    for b in sorted_branches:
        total_prob += b['prob']
        # Filter very small probabilities for display cleanliness if list is huge
        if b['prob'] > 1e-4:
            outcome_str = str(b['outcome'])
            t_idx = b.get('target_idx', 0)
            tgt_name = target_names[t_idx] if t_idx < len(target_names) else f"Target_{t_idx}"
            print(f"{outcome_str:<20} {b['prob']:<10.4f} {b['fidelity']:<10.4f} {(1-b['fidelity']):<10.1e} {tgt_name:<15}")

    print(f"\nTotal Probability captured: {total_prob:.5f}")

    # 2. Target Distribution Analysis
    print("-" * 60)
    print(f"Target Distribution Analysis (Success Threshold > {success_threshold}):")
    print(f"{'Rank':<5} {'Target Name':<20} {'Tot. Prob':<10} {'Outcomes (Top 3)'}")
    print("-" * 60)

    target_stats = {} 
    
    # Calculate global success probability
    global_success_prob = 0.0

    for b in branches:
        if b['fidelity'] > success_threshold:
            global_success_prob += b['prob']
            idx = b.get('target_idx', 0)
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

    print("-" * 60)
    print(f"Global Success Probability: {global_success_prob:.5f}")

    # 3. Successful Branches Detail
    print("-" * 80)
    print(f"All Branches with Fidelity > {success_threshold} (Sorted by Prob):")
    print(f"{'Outcome':<20} {'Prob':<10} {'Fidelity':<10} {'1-Fid':<10} {'Best Target':<15}")
    print("-" * 80)

    successful_branches = [b for b in branches if b['fidelity'] > success_threshold]
    successful_branches.sort(key=lambda x: x['prob'], reverse=True)

    if not successful_branches:
        print("No branches met the success threshold.")
    else:
        for b in successful_branches:
            outcome_str = str(b['outcome'])
            t_idx = b.get('target_idx', 0)
            tgt_name = target_names[t_idx] if t_idx < len(target_names) else f"Target_{t_idx}"
            print(f"{outcome_str:<20} {b['prob']:<10.4f} {b['fidelity']:<10.4f} {(1-b['fidelity']):<10.1e} {tgt_name:<15}")

    print("=" * 60)


def run_deterministic_path(circuit: TimeMultiplexedCircuit, flat_params: np.ndarray, measurement_outcomes: tuple, cutoff_dim):
    """
    Evaluates a time-domain circuit for a specific path of measurements.
    
    Args:
        circuit: The TimeMultiplexedCircuit instance
        flat_params: Flat parameter array for the circuit
        measurement_outcomes: Tuple of measurement outcomes (one per step)
        
    Returns:
        A dictionary with final probability and final state ket if the path is valid,
        or None if the path is not physically possible.
    """
    # Split initial parameters if needed
    n_init = circuit.num_initial_parameters
    if n_init > 0:
        init_params = flat_params[:n_init]
        step_params = flat_params[n_init:]
    else:
        init_params = np.array([])
        step_params = flat_params

    mapped_params = circuit.map_parameters(step_params)
    
    # Get measurement specs (modes and cutoffs)
    # Cap measurement cutoffs by simulation cutoff_dim to prevent indexing errors
    meas_specs = [(m, min(c, cutoff_dim)) for m, c in circuit.get_measurement_specs()]
    meas_modes = [m for m, c in meas_specs]
    meas_cutoffs = [c for m, c in meas_specs]
    perm = [0] + meas_modes
    
    # Initialize the state
    initial_ket = circuit.get_initial_state_ket(init_params, cutoff_dim)  # Use a reasonable cutoff
    current_ket = initial_ket.copy()
    
    # Store probability at each step
    probabilities = []
    
    for step in range(circuit.steps):
        step_params = mapped_params[step]
        
        # Run the circuit step
        eng = sf.Engine("fock", backend_options={"cutoff_dim": cutoff_dim})
        
        prog_prep = sf.Program(len(meas_modes) + 1)
        with prog_prep.context as q:
            Ket(current_ket) | q[0]
        
        eng.run(prog_prep)
        
        result = circuit.run_step(None, step, step_params, eng)
        full_ket = result.state.ket()
    
        # Transpose to [loop, ancilla1, ancilla2,...]
        transposed_ket = np.transpose(full_ket, axes=perm)
        
        # Slice measured modes
        slices = [slice(None)] + [slice(0, c) for c in meas_cutoffs]
        sliced_ket = transposed_ket[tuple(slices)]
        
        # Calculate probabilities and check if the desired measurement is possible
        probs_tensor = np.sum(np.abs(sliced_ket)**2, axis=0)
        
        # Get the probability of the specified outcome at this step
        current_outcome_index = tuple([int(outcome) for outcome in measurement_outcomes[step]])
        
        if len(meas_specs) == 1:
            # Single ancilla case
            prob = probs_tensor[current_outcome_index[0]]
        else:
            # Multiple ancillas: index into the tensor
            prob = probs_tensor[current_outcome_index]
            
        probabilities.append(prob)
        
        # Project onto the measurement outcome (discard other possibilities)
        if len(meas_specs) == 1:
            # Single mode projection
            proj_ket = sliced_ket[:, current_outcome_index[0]]
        else:
            # Multi-mode projection: take the specific tensor element
            indices = [slice(None)] + [int(outcome) for outcome in measurement_outcomes[step]]
            proj_ket = sliced_ket[tuple(indices)]
            
        # Normalize and store as new state
        norm = np.linalg.norm(proj_ket)
        
        if abs(norm) < 1e-9:
            return None  # Path is not physically possible
            
        current_ket = proj_ket / norm
        
    final_probability = np.prod(probabilities)
    
    result = {
        "final_probability": final_probability,
        "final_state_ket": current_ket
    }
    
    return result

# Replaced the example main() with a full evaluator:
# - loads latest (or user-specified) results/cma_run_* directory
# - loads best_x / best_result
# - allows selecting a branch by index or specifying a measurement tuple
# - runs deterministic post-selection via run_deterministic_path()
# - visualizes Wigner + Fock probs (re-using demo_target plotting style)
def plot_ket_wigner(ket, title="State", cutoff_dim=40, grid_size=200, x_limit=5):
    """Plot Wigner function and Fock probabilities for a single-mode ket using Strawberry Fields state tools."""
    # Ensure normalization
    norm = np.linalg.norm(ket)
    if abs(norm - 1.0) > 1e-6:
        ket = ket / norm

    prog = sf.Program(1)
    with prog.context as q:
        Ket(ket) | q[0]
    eng = sf.Engine("fock", backend_options={"cutoff_dim": cutoff_dim})
    result = eng.run(prog)
    state = result.state

    xvec = np.linspace(-x_limit, x_limit, grid_size)
    pvec = np.linspace(-x_limit, x_limit, grid_size)
    W = state.wigner(mode=0, xvec=xvec, pvec=pvec)

    probs = state.all_fock_probs(cutoff=cutoff_dim)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    X, P = np.meshgrid(xvec, pvec)
    lim = np.max(np.abs(W))
    c = ax1.pcolormesh(X, P, W, cmap='RdBu', shading='auto', vmin=-lim, vmax=lim)
    fig.colorbar(c, ax=ax1, label='W(x, p)')
    ax1.set_title(f"Wigner Function ({title})")
    ax1.set_xlabel("x (Position)")
    ax1.set_ylabel("p (Momentum)")
    ax1.set_aspect('equal')
    ax1.axhline(0, color='black', linestyle='--', alpha=0.3)
    ax1.axvline(0, color='black', linestyle='--', alpha=0.3)

    display_cutoff = min(cutoff_dim, 60)
    ax2.bar(range(display_cutoff), probs[:display_cutoff], color='teal', alpha=0.7, edgecolor='black')
    ax2.set_title("Fock State Probabilities")
    ax2.set_xlabel("Fock Number |n>")
    ax2.set_ylabel("Probability")
    ax2.set_xticks(range(display_cutoff))

    plt.tight_layout()
    plt.show()


def plot_wigner_print_quality(ket, filename="state_plot.png", title="State", cutoff_dim=50, grid_size=400, x_limit=6):
    """
    Generates a high-resolution Wigner function and Fock distribution plot suitable for publication/print.
    Saves the figure to disk.
    """
    # Ensure normalization
    norm = np.linalg.norm(ket)
    if abs(norm - 1.0) > 1e-6:
        ket = ket / norm

    # Run simple engine to get state object
    prog = sf.Program(1)
    with prog.context as q:
        Ket(ket) | q[0]
    eng = sf.Engine("fock", backend_options={"cutoff_dim": cutoff_dim})
    result = eng.run(prog)
    state = result.state

    # 1. Calculate Wigner
    xvec = np.linspace(-x_limit, x_limit, grid_size)
    pvec = np.linspace(-x_limit, x_limit, grid_size)
    W = state.wigner(mode=0, xvec=xvec, pvec=pvec)

    # 2. Calculate Fock Probs
    probs = state.all_fock_probs(cutoff=cutoff_dim)
    
    # 3. Setup High-Res Plot
    # Use standard settings for cleanliness
    plt.rcParams.update({'font.size': 14, 'font.family': 'sans-serif'})
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7), dpi=300)
    
    # Plot Wigner
    X, P = np.meshgrid(xvec, pvec)
    lim = np.max(np.abs(W))
    # Use RdBu_r so red is positive, blue is negative (standard in some papers) or RdBu
    c = ax1.pcolormesh(X, P, W, cmap='RdBu', shading='auto', vmin=-lim, vmax=lim, rasterized=True)
    
    cbar = fig.colorbar(c, ax=ax1, label='W(x, p)', pad=0.02)
    cbar.ax.tick_params(labelsize=12)
    
    ax1.set_title(f"Wigner Function: {title}", fontsize=16, pad=15)
    ax1.set_xlabel("x (Position)", fontsize=14)
    ax1.set_ylabel("p (Momentum)", fontsize=14)
    ax1.set_aspect('equal')
    # Subtle guidelines
    ax1.axhline(0, color='gray', linestyle=':', alpha=0.5, linewidth=1)
    ax1.axvline(0, color='gray', linestyle=':', alpha=0.5, linewidth=1)

    # Plot Fock
    display_cutoff = min(cutoff_dim, 60)
    indices = np.arange(display_cutoff)
    ax2.bar(indices, probs[:display_cutoff], color='#2c7bb6', alpha=0.8, edgecolor='black', width=0.7)
    
    ax2.set_title("Fock State Probabilities", fontsize=16, pad=15)
    ax2.set_xlabel("Fock Number |n>", fontsize=14)
    ax2.set_ylabel("Probability", fontsize=14)
    
    # Clean up x-axis ticks
    step = 5 if display_cutoff > 20 else 1
    ax2.set_xticks(np.arange(0, display_cutoff, step))
    ax2.grid(axis='y', linestyle='--', alpha=0.3)
    ax2.set_xlim(-0.5, display_cutoff - 0.5)
    ax2.set_ylim(0, max(probs) * 1.1)

    plt.tight_layout()
    
    save_path = Path(filename).resolve()
    plt.savefig(save_path, bbox_inches='tight', dpi=100)
    plt.close(fig)
    print(f"High-quality plot saved to: {save_path}")


def _find_latest_results_dir(base_dir: Path):
    # Modified to match "opt_*" instead of "opt_run_*" to support new naming tags
    # and sort by modification time to ensure we get the actual latest run.
    matches = [p for p in base_dir.glob("opt_*") if p.is_dir()]
    
    if not matches:
        return None
        
    # Return the directory with the most recent modification time
    return max(matches, key=lambda p: p.stat().st_mtime)


def load_optimization_run(results_dir: Path, selection: str = "best"):
    """
    Unified loader for optimization results.
    
    Args:
        results_dir: The opt_XXXX directory.
        selection: 
            - "best": loads the highest numbered best_run_XXXX.pkl from the 'best' subdirectory
            - "latest": loads the highest numbered run_XXXX.pkl from the results directory
            - integer string (e.g. "5"): loads run_0005.pkl from the results directory
    """
    target_file = None

    if selection == "best":
        best_dir = results_dir / "best"
        best_files = sorted(best_dir.glob("best_run_*.pkl"))
        if best_files:
            target_file = best_files[-1]
        else:
            # Fallback or specific file check
            target_file = best_dir / "best_run_0001.pkl"
    elif selection == "latest":
        runs = sorted(results_dir.glob("run_*.pkl"))
        if runs:
            target_file = runs[-1]
    else:
        # Try to parse as specific run number
        try:
            run_num = int(selection)
            target_file = results_dir / f"run_{run_num:04d}.pkl"
        except ValueError:
            print(f"Error: Invalid selection '{selection}'. Use 'best', 'latest', or a number.")
            return None

    if not target_file or not target_file.exists():
        print(f"Error: Result file not found at {target_file}")
        return None

    print(f"Loading data from: {target_file}")
    with open(target_file, "rb") as f:
        data = pickle.load(f)

    # Structure check and normalization: {"meta": ..., "res": ...}
    if isinstance(data, dict) and "meta" in data and "res" in data:
        res = data["res"]
        meta = data["meta"]
        # Inject recipes from meta into res if they are missing
        if "circuit_config" in meta:
            res["circuit_config"] = meta["circuit_config"]
        if "target_configs" in meta:
            res["target_configs"] = meta["target_configs"]
        if "measurement_patterns" in meta:
            res["measurement_patterns"] = meta["measurement_patterns"]
    else:
        # Fallback for older or flat structures
        res = data

    return {
        "best_res": res,
        "x": res.get("x")
    }


def load_params_from_json(json_path: str) -> np.ndarray:
    """
    Loads flat parameter vector from a JSON file.
    Can handle a raw list or a dict with keys 'flat_params', 'x', or 'params'.
    """
    path = Path(json_path)
    if not path.exists():
        print(f"Error: JSON parameter file not found at {path}")
        return None
    
    try:
        with open(path, 'r') as f:
            data = json.load(f)
            
        if isinstance(data, list):
            return np.array(data)
        elif isinstance(data, dict):
            for key in ['flat_params', 'x', 'params']:
                if key in data:
                    return np.array(data[key])
            print(f"Error: JSON dict does not contain expected keys (flat_params, x, params). Found: {list(data.keys())}")
            return None
        else:
            print("Error: JSON root must be a list or dict.")
            return None
            
    except Exception as e:
        print(f"Error loading JSON parameters: {e}")
        return None


def _reshape_outcome_flat(outcome_flat, circuit: TimeMultiplexedCircuit):
    """
    Normalize and reshape stored branch outcomes into per-step tuples:
      ((o_step0_mode0, ...), (o_step1_mode0, ...), ...)
    Supports tuples, lists, and numpy arrays.
    """
    if outcome_flat is None:
        return None

    meas_specs = circuit.get_measurement_specs()
    n_meas_modes = len(meas_specs)
    steps = circuit.steps

    # --- Normalize numpy arrays → Python lists ---
    if isinstance(outcome_flat, np.ndarray):
        outcome_flat = outcome_flat.tolist()

    # Case 1: already per-step, e.g. [[a,b], [c,d]]
    if (
        isinstance(outcome_flat, (list, tuple))
        and len(outcome_flat) == steps
    ):
        # unwrap numpy arrays inside tuple/list
        reshaped = []
        for step in outcome_flat:
            if isinstance(step, np.ndarray):
                step = step.tolist()
            reshaped.append(tuple(int(x) for x in step))
        return tuple(reshaped)

    # Case 2: flat list [a,b,c,d]
    outcome_list = list(outcome_flat)
    expected = steps * n_meas_modes

    if len(outcome_list) != expected:
        raise ValueError(
            f"Outcome length {len(outcome_list)} incompatible with circuit "
            f"(steps={steps}, meas_modes={n_meas_modes})"
        )

    reshaped = []
    for s in range(steps):
        start = s * n_meas_modes
        reshaped.append(
            tuple(int(x) for x in outcome_list[start:start + n_meas_modes])
        )

    return tuple(reshaped)


def save_all_fixed_pattern_wigners(circuit, flat_x, measurement_patterns, cutoff, results_dir):
    """
    Iterates over all saved fixed measurement patterns, evaluates them deterministically,
    and saves their high-quality Wigner plots.
    """
    if measurement_patterns is None or len(measurement_patterns) == 0:
        print("No fixed measurement patterns provided to save.")
        return

    print(f"\n=== Saving Wigner plots for {len(measurement_patterns)} fixed patterns ===")
    
    for i, pattern in enumerate(measurement_patterns):
        try:
            # Attempt to reshape the pattern to the required per-step tuple format
            reshaped_pattern = _reshape_outcome_flat(pattern, circuit)
        except Exception as e:
            print(f"Failed to reshape pattern {pattern}: {e}")
            continue
            
        print(f"Evaluating pattern {i+1}/{len(measurement_patterns)}: {reshaped_pattern}")
        res = run_deterministic_path(circuit, flat_x, reshaped_pattern, cutoff)
        
        if res is None:
            print(f"  -> Path {reshaped_pattern} is not physically possible (zero probability). Skipping.")
            continue
            
        ket = res['final_state_ket']
        prob = res['final_probability']
        
        # Format filename based on measurement sequence
        flat_outcomes = []
        for step_out in reshaped_pattern:
            flat_outcomes.extend(step_out)
        outcome_str = "_".join(map(str, flat_outcomes))
        
        filename = results_dir / f"wigner_hq_{outcome_str}.png"
        title = f"Outcome {reshaped_pattern} (P={prob:.2e})"
        
        plot_wigner_print_quality(ket, filename=filename, title=title, cutoff_dim=cutoff)


def main():
    # Configuration - set these variables directly instead of using command-line arguments
    results_path = None  # Set to specific path if desired
    results_path = Path(__file__).resolve().parent.parent.parent / "results" / "opt_Sq2_GKP_20260205T140644Z"  # Set to specific path if desired
    run_selection = "latest" # "best", "latest", or a run number string like "5"
    params_json_path = None # Optional: Path to JSON file containing parameter vector (overrides results)
    # params_json_path =  Path(__file__).resolve().parent.parent.parent / "results" / "manual" / "optimized_params.json"
    branch_index = 0  # Index of branch to visualize from best_result['branches']
    measurement = "1,3"  # Explicit measurement tuple, e.g., "3,1" or "3,1;2,0" (semicolon separated)
    cutoff = 30  # Cutoff dimension for visualization
    recalc_statistics = True # If True, will print the full branch table and aggregated targets
    FORCE_BEAM_SEARCH = False # If True, ignores stored fixed patterns and re-runs Beam Search
    SAVE_ALL_FIXED_PATTERNS = True # If True, generates and saves Wigner plots for all fixed patterns
    
    LOSS_TRANSMISSIVITY = 1 # Set < 1.0 to enable Density Matrix simulation with loss
    USE_DM_EVAL = LOSS_TRANSMISSIVITY < 1.0

    # Find results directory
    base = Path(__file__).resolve().parent.parent.parent / "results"
    
    if results_path:
        results_dir = Path(results_path)
    else:
        results_dir = _find_latest_results_dir(base)
    
    if results_dir is None or not results_dir.exists():
        print(f"No results directory found at {base}. Run the optimization first and make sure results exist.")
        return
    
    print(f"Using results dir: {results_dir}")
    
    best = load_optimization_run(results_dir, selection=run_selection)

    if not best:
        return

    # Extract Results
    best_res = best.get('best_res', {})
    
    # Avoid using 'or' with numpy arrays (truth value is ambiguous).
    flat_x = best.get('x')
    if flat_x is None:
        flat_x = best_res.get('x')

    # Optional: Override parameters from JSON file
    if params_json_path:
        print(f"Attempting to load parameters from JSON: {params_json_path}")
        json_x = load_params_from_json(params_json_path)
        if json_x is not None:
            print(f"Successfully loaded {len(json_x)} parameters from JSON. Overriding result parameters.")
            flat_x = json_x
    
    if flat_x is None:
        print("Could not find flat parameter vector (best_x) and no JSON parameters provided.")
        return

    # -------------------------------------------------------------------------
    # 0. Reconstruct Experiment from Configs (The "Recipe")
    # -------------------------------------------------------------------------
    circuit_config = best_res.get('circuit_config')
    target_configs = best_res.get('target_configs')

    # === APPLY SANITIZER HERE ===
    circuit_config = sanitize_config_paths(circuit_config)
    target_configs = sanitize_config_paths(target_configs)
    # ============================

    # csv_path_abs = Path(__file__).resolve().parent.parent.parent / "data" / "GKP_core_coefficients.csv"
    # target_configs = [
    #     {'class_name': 'CoreGKPTarget', 'params': {'csv_path': str(csv_path_abs), 'n_max': n, 'delta_db': 10, 'mu': m}}
    #     # for n in [8, 12] for m in [0]
    #     for n in [4, 6, 8, 10, 12] for m in [0]
    # ]


    # Inject Loss Parameter if configured
    if circuit_config and 'params' in circuit_config:
        circuit_config['params']['loss_transmissivity'] = LOSS_TRANSMISSIVITY

    print_config_info(circuit_config, target_configs)

    if not circuit_config:
        print("Error: Result file does not contain 'circuit_config'. Cannot reconstruct circuit.")
        return

    print(f"Reconstructing Circuit: {circuit_config.get('class_name')}...")
    circuit = create_from_config(circuit_config, circuit_module)
    
    targets = []
    if target_configs:
        print(f"Reconstructing {len(target_configs)} Targets...")
        targets = [create_from_config(cfg, target_module) for cfg in target_configs]
    else:
        print("Warning: No 'target_configs' found in result. Target analysis will be limited.")

    # Generate display names for targets
    target_names = []
    for t in targets:
        if isinstance(t, CoreGKPTarget):
            target_names.append(f"GKP_n{t.n_max}_mu{t.mu}")
        elif isinstance(t, SqueezedCatTarget):
            target_names.append(f"SqCat_a{t.alpha}_r{t.r}_p{t.p}")
        elif isinstance(t, CatTarget):
            target_names.append(f"Cat_a{t.alpha}_p{t.p}")
        elif isinstance(t, BinomialCodeTarget):
            target_names.append(f"Binomial_N{t.N}_S{t.S}_mu{t.mu}")
        else:
            target_names.append(f"Target")

    # -------------------------------------------------------------------------
    # 0. Print Optimized Parameters Schedule
    # -------------------------------------------------------------------------
    print_optimized_parameters(circuit, flat_x, best.get('mapped_params'))

    # -------------------------------------------------------------------------
    # 1. Print Full Statistics (Requested Feature)
    # -------------------------------------------------------------------------
    if recalc_statistics:
        print("\n=== Optimization Statistics ===")
        
        # If we have targets defined and want to re-run to ensure we capture all branches 
        # (e.g. if we want to change beam width), we can use get_all_optimization_results.
        # Otherwise, we use the stored results.
        
        result_to_analyze = best_res
        
        if targets:
            print("Re-evaluating circuit to ensure fresh branch data...")
            
            # Determine evaluation mode (Fixed Patterns vs Beam Search)
            stored_patterns = best_res.get('measurement_patterns')
            eval_patterns = None
            
            if not FORCE_BEAM_SEARCH and stored_patterns is not None:
                print(" -> Using Original Fixed Patterns for analysis.")
                eval_patterns = stored_patterns
            else:
                print(f" -> Using Beam Search (Width=100) for analysis.")
                eval_patterns = None

            if USE_DM_EVAL:
                print(f"Evaluating with Density Matrices (Loss T={LOSS_TRANSMISSIVITY})...")
                target_kets = [t.get_target_ket(cutoff) for t in targets]
                result_to_analyze = evaluate_time_domain_circuit_dm(
                    np.asarray(flat_x),
                    circuit,
                    target_kets,
                    cutoff,
                    beam_width=100,
                    penalty_strength=0.0,
                    prob_power=1.0,
                    measurement_patterns=eval_patterns
                )
            else:
                # Example: re-run with potentially higher beam width
                result_to_analyze = get_all_optimization_results(
                    circuit, np.asarray(flat_x), targets, cutoff, beam_width=100, measurement_patterns=eval_patterns
                )
        
        if result_to_analyze:
            print_optimization_statistics(
                result_to_analyze, 
                success_threshold=1 - 2e-2, 
                target_names=target_names if target_names else None
            )
        else:
            print("No result dictionary available to analyze.")
            
    # -------------------------------------------------------------------------
    # 2. Save all fixed patterns Wigners (If enabled)
    # -------------------------------------------------------------------------
    if SAVE_ALL_FIXED_PATTERNS and not USE_DM_EVAL:
        stored_patterns = best_res.get('measurement_patterns')
        if stored_patterns is not None:
            save_all_fixed_pattern_wigners(circuit, np.asarray(flat_x), stored_patterns, cutoff, results_dir)
        else:
            print("\nSAVE_ALL_FIXED_PATTERNS is True, but no fixed measurement patterns were found in the results.")

    # -------------------------------------------------------------------------
    # 3. Visualize Specific Outcome
    # -------------------------------------------------------------------------
    if USE_DM_EVAL:
        print("\n=== Single Branch Visualization ===")
        print("Skipping visualization: Wigner plotting for Mixed States (Density Matrices) is not yet implemented in this script.")
        return

    print("\n=== Single Branch Visualization ===")

    # Determine measurement outcomes to evaluate
    measurement_outcomes = None
    if measurement:
        # parse formats like "3,1" or "3,1;2,0" (semicolon between steps)
        steps_raw = measurement.split(";")
        parsed = []
        for s in steps_raw:
            parts = [int(x.strip()) for x in s.split(",") if x.strip() != ""]
            parsed.append(tuple(parts))
        measurement_outcomes = tuple(parsed)
    else:
        # try to pick a branch outcome from best_result
        if best_res and 'branches' in best_res and len(best_res['branches']) > 0:
            branches = best_res['branches']
            # Sort so branch_index 0 is the highest prob one
            branches.sort(key=lambda x: x['prob'], reverse=True)
            
            idx = min(branch_index, len(branches) - 1)
            branch = branches[idx]
            outcome_raw = branch.get('outcome')
            print(f"Auto-selected branch rank {idx}: Outcome {outcome_raw} (Prob: {branch['prob']:.4f})")
            
            if outcome_raw is None:
                print("Selected branch has no explicit 'outcome' stored. Try specifying a measurement.")
            else:
                try:
                    measurement_outcomes = _reshape_outcome_flat(outcome_raw, circuit)
                except Exception as e:
                    print(f"Failed to reshape branch outcome: {e}")
                    measurement_outcomes = None

    if measurement_outcomes is None:
        print("No measurement outcomes selected. Specify a measurement or check stored best_result['branches'].")
        return

    print(f"Evaluating measurement outcomes (per step): {measurement_outcomes}")
    # Run deterministic path
    res = run_deterministic_path(circuit, np.asarray(flat_x), measurement_outcomes, cutoff)
    if res is None:
        print("The specified measurement path is not physically possible (zero probability).")
        return

    print(f"Final probability: {res['final_probability']:.6e}")
    ket = res['final_state_ket']
    # print("Final ket (truncated):")
    # print(ket[:min(len(ket), 20)])

    # --- SAVE DENSITY MATRIX FOR PLOTTING ---
    print("\nSaving density matrix...")
    
    # Convert ket to DM if necessary
    if 'final_state_ket' in res:
        ket = res['final_state_ket']
        # Outer product |psi><psi|
        dm = np.outer(ket, np.conj(ket))
    elif 'final_state_dm' in res:
        dm = res['final_state_dm']
    else:
        # Fallback if manual DM evaluation was run
        dm = None
        print("No final state found to save.")

    if dm is not None:
        # Create a filename based on the measurement outcomes
        # flattens ((1,), (3,)) -> "1_3" or ((1,3),) -> "1_3"
        flat_outcomes = []
        for step_out in measurement_outcomes:
            flat_outcomes.extend(step_out)
        
        outcome_str = "_".join(map(str, flat_outcomes))
        filename = f"state_dm_{outcome_str}.npy"
        
        save_path = results_dir / filename
        np.save(save_path, dm)
        print(f"Density matrix saved to: {save_path}")
        print(f"Run this script again with different measurements to generate comparison files.")

    # Visualize
    # plot_ket_wigner(ket, title=f"postselect {measurement_outcomes}", cutoff_dim=cutoff)
    
    # Save High Quality Plot
    # plot_wigner_print_quality(ket, filename="optimized_state_hq.png", title=f"Outcome {measurement_outcomes}", cutoff_dim=cutoff)

if __name__ == "__main__":
    main()
