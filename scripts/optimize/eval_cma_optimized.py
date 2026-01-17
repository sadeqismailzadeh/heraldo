import numpy as np
import quantum_agent
import strawberryfields as sf
from strawberryfields.ops import Ket
from quantum_agent.optimization.time_circuits import *
from quantum_agent.optimization.time_runner import evaluate_time_domain_circuit
from quantum_agent.components.targets import *
from pathlib import Path

import numpy as np
import strawberryfields as sf
from quantum_agent.optimization.time_interfaces import TimeMultiplexedCircuit
from quantum_agent.optimization.time_runner import evaluate_time_domain_circuit



def evaluate_time_domain_circuit(flat_params, circuit, target_kets, cutoff_dim, beam_width, penalty_strength, success_threshold=0.99, success_weight=5.0, ng_weight=0.0, ng_threshold=0.1):
    """
    Evaluates the circuit using Beam Search and returns the loss.
    """
    # 0. Setup
     # 1. SPLIT PARAMETERS (Minimal Change)
    n_init = circuit.num_initial_parameters
    if n_init > 0:
        init_params = flat_params[:n_init]
        step_params = flat_params[n_init:]
    else:
        init_params = np.array([])
        step_params = flat_params

    # Map only the step parameters
    mapped_params = circuit.map_parameters(step_params)
    meas_specs = circuit.get_measurement_specs()
    
    # 2. INITIALIZE BEAM (Modified)
    # Get the custom initial ket (Vacuum or Squeezed)
    initial_ket = circuit.get_initial_state_ket(init_params, cutoff_dim)

    # Initialize Beam with this ket
    active_kets = np.zeros((1, cutoff_dim), dtype=np.complex128)
    active_kets[0] = initial_ket 
    active_probs = np.array([1.0])
    active_outcome_sums = np.zeros(1, dtype=int)
    
    # Precompute slicing info
    meas_modes = [m for m, c in meas_specs]
    meas_cutoffs = [c for m, c in meas_specs]
    perm = [0] + meas_modes # Loop mode first, then measured

    total_truncation_error = 0.0
    
    # 1. Time Loop (Step 0 to T-1)
    for step in range(circuit.steps):
        step_params = mapped_params[step]
        
        # Containers for vectorized processing
        branch_raw_probs = [] # List of flattened probability arrays
        branch_tensors = []   # List of sliced kets
        
        # --- Phase A: Evolution ---
        for idx in range(len(active_kets)):
            parent_ket = active_kets[idx]
            
            eng = sf.Engine("fock", backend_options={"cutoff_dim": cutoff_dim})
            
            prog_prep = sf.Program(len(meas_modes) + 1)
            with prog_prep.context as q:
                Ket(parent_ket) | q[0]
            eng.run(prog_prep)
            
            result = circuit.run_step(None, step, step_params, eng)
            full_ket = result.state.ket()
            
            # Truncation check
            flat_ket = full_ket.flatten()
            norm_sq = np.real(np.vdot(flat_ket, flat_ket))
            total_truncation_error += np.abs(1.0 - norm_sq) * active_probs[idx]

            # Transpose: (D_loop, D_m1, D_m2, ...)
            transposed_ket = np.transpose(full_ket, axes=perm)
            
            # Slice measured modes
            slices = [slice(None)] + [slice(0, c) for c in meas_cutoffs]
            sliced_ket = transposed_ket[tuple(slices)]
            
            branch_tensors.append(sliced_ket)
            
            # Calculate Probabilities: Sum over Loop (axis 0)
            # Shape (c1, c2, ...) -> flatten
            probs_tensor = np.sum(np.abs(sliced_ket)**2, axis=0)
            branch_raw_probs.append(probs_tensor.flatten())

        # --- Phase B: Vectorized Virtual Branching ---
        
        # P_parent: (N_branches,)
        P_parent = active_probs
        
        # P_raw: (N_branches, meas_cutoff)
        P_raw = np.stack(branch_raw_probs) 
        
        # Joint Probability: P(parent) * P(outcome|parent)
        # Broadcasting: (N, 1) * (N, M) -> (N, M)
        P_total = P_parent[:, None] * P_raw
        
        # --- Phase C: Vectorized Pruning ---
        flat_P = P_total.flatten()
        k = min(beam_width, flat_P.size)
        top_indices = np.argpartition(flat_P, -k)[-k:]
        
        # Resolve indices
        parent_indices, outcome_indices_flat = np.unravel_index(top_indices, P_total.shape)
        outcomes_unraveled = np.unravel_index(outcome_indices_flat, tuple(meas_cutoffs))
        
        # --- Phase D: Vectorized Realization ---
        
        # Stack tensors: (N_parents, cutoff, meas_cutoff)
        tensor_stack = np.stack(branch_tensors)
        
        # Fancy indexing: [parent_idx, :, o1_idx, o2_idx...]
        indexer = (parent_indices, slice(None)) + outcomes_unraveled
        raw_new_kets = tensor_stack[indexer]
        
        # Normalize
        norms = np.linalg.norm(raw_new_kets, axis=1)
        selected_probs = P_total[parent_indices, outcome_indices_flat]
        
        # Calculate step outcome sums for the top K candidates
        if len(outcomes_unraveled) > 0:
            step_outcome_sums = np.sum(np.stack(outcomes_unraveled), axis=0)
        else:
            step_outcome_sums = np.zeros(len(parent_indices), dtype=int)
            
        candidate_outcome_sums = active_outcome_sums[parent_indices] + step_outcome_sums

        # Filter
        mask = (norms > 1e-9) & (selected_probs > 1e-12)
        
        if not np.any(mask):
            return 100.0
        
        # Update Active Beam
        active_kets = raw_new_kets[mask] / norms[mask][:, None]
        active_probs = selected_probs[mask]
        active_outcome_sums = candidate_outcome_sums[mask]

    # --- Final Objective (Vectorized) ---
    
    total_prob = np.sum(active_probs)

    # Filter out zero-outcome branches for reward calculation
    mask_nonzero = active_outcome_sums > 1
    
    expected_fidelity = 0.0
    soft_success_prob = 0.0
    ng_loss = 0.0

    if np.any(mask_nonzero):
        final_kets = active_kets[mask_nonzero]
        final_probs = active_probs[mask_nonzero]
        
        # Batched Fidelity Max Rotation
        # max_phi |<target | e^{i n phi} | state>|^2
        # Equivalent to max |FFT(conj(target) * state)|^2 along Fock axis
        
        # Product: (N_branches, 1, D) * (1, N_targets, D) -> (N_branches, N_targets, D)
        targets_arr = np.array(target_kets)
        prod = np.conj(final_kets[:, None, :]) * targets_arr[None, :, :]
        
        # FFT (n=256 for phase resolution)
        fft_vals = np.fft.fft(prod, n=256, axis=-1)
        
        # Max over phase (axis -1) -> (N_branches, N_targets)
        pairwise_fidelities = np.max(np.abs(fft_vals)**2, axis=-1)

        # Best fidelity across all targets -> (N_branches,)
        fidelities = np.max(pairwise_fidelities, axis=1)

        # Logarithmic Reward
        min_infidel=3e-2
        infidelities = np.maximum(1.0 - fidelities, min_infidel)
        log_vals = np.log10(infidelities)  /  np.log10(min_infidel)
        capped_fidlities=np.maximum(fidelities, 1-min_infidel)
        expected_fidelity = np.sum((final_probs**0.5)
                                   * (capped_fidlities**2  *log_vals)**4)

    # Penalties
    # 1. Total Probability Loss (indicates truncation or dropped branches)
    loss_prob = np.abs(1.0 - total_prob)
    # return  -expected_fidelity - (success_weight * (soft_success_prob+ 4*soft_success_prob2)) + (penalty_strength * loss_prob) + (penalty_strength * total_truncation_error) + (ng_weight * ng_loss)
    return     -expected_fidelity - (success_weight*(soft_success_prob)) \
             + (penalty_strength * total_truncation_error) + (ng_weight * ng_loss)  \


def run_time_circuit_evaluation(circuit, flat_params, measurement_outcomes):
    """
    Evaluates a time-domain circuit with fixed parameters and measurement outcomes.
    
    Args:
        circuit (TimeMultiplexedCircuit): Circuit instance to evaluate.
        flat_params (np.ndarray): Flat array of parameters for the circuit.
        measurement_outcomes (list): List of integers representing measurement outcomes per step.
        
    Returns:
        dict: Dictionary containing final probability and state ket if path is realized, otherwise None.
    """
    n_init = circuit.num_initial_parameters
    if n_init > 0:
        init_params = flat_params[:n_init]
        step_params = flat_params[n_init:]
    else:
        init_params = np.array([])
        step_params = flat_params

    # Map flat parameters to step-specific parameters
    mapped_params = circuit.map_parameters(step_params)
    meas_specs = circuit.get_measurement_specs()
    
    # Get measurement specs from the circuit
    meas_specs = circuit.get_measurement_specs()
    num_meas_modes = len(meas_specs)

    # Initialize state: vacuum state for loop mode (mode 0)
    cutoff_dim = 35  # Example cutoff dimension, can be adjusted
    initial_ket = circuit.get_initial_state_ket(init_params, cutoff_dim)

    # Precompute slicing info
    meas_modes = [m for m, c in meas_specs]
    meas_cutoffs = [c for m, c in meas_specs]
    perm = [0] + meas_modes # Loop mode first, then measured
    
    # Step-by-step execution with deterministic path
    state = initial_ket.copy()
    
    for step in range(circuit.steps):
        # Get parameters for this step
        step_params = mapped_params[step]
        
        # Initialize engine and run the circuit step
        eng = sf.Engine("fock", backend_options={"cutoff_dim": cutoff_dim})
        prog = sf.Program(len(meas_specs) + 1)
        
        with prog.context as q:
            # Set initial state for loop mode (mode 0)
            Ket(state) | q[0]
        
        eng.run(prog)
        
        result = circuit.run_step(None, step, step_params, eng)
        full_ket = result.state.ket()
        
        # Truncation check
        flat_ket = full_ket.flatten()
        norm_sq = np.real(np.vdot(flat_ket, flat_ket))

        # Transpose: (D_loop, D_m1, D_m2, ...)
        transposed_ket = np.transpose(full_ket, axes=perm)
        
        # Slice out measured modes based on cutoffs
        sliced_kets = []
        for i, (mode_idx, cutoff) in enumerate(meas_specs):
            slices = [slice(None)] * len(transposed_ket.shape)
            slices[1 + i] = slice(0, cutoff)
            sliced_ket = transposed_ket[tuple(slices)]
            sliced_kets.append(sliced_ket)
        
        # Project onto the specified measurement outcomes
        projected_kets = []
        for i in range(len(meas_specs)):
            mode_idx, _ = meas_specs[i]
            outcome = measurement_outcomes[step * num_meas_modes + i]
            
            # Extract the relevant part of the ket and project
            selected_ket = sliced_kets[i][..., outcome]
            projected_kets.append(selected_ket)
        
        # Combine projected kets (assuming only one measured mode per step for simplicity)
        # if len(projected_kets) == 1:
        state = projected_kets[0].flatten()
        # else:
        #     raise ValueError("Multiple measurement outcomes not supported in this evaluation.")
    
    # Calculate final probability
    norm_sq = np.real(np.vdot(state, state))
    final_prob = float(norm_sq)
    
    return {
        "final_probability": final_prob,
        "final_state_ket": state
    }

if __name__ == "__main__":

    
    circuit2 = ThreeModeTimeDomainSqueezeOnly(steps=1,
                                    time_invariant=False,
                                    clip_size=1,
                                    measure_fock_cutoff=30,
                                    num_single_photon=0,
                                    train_initial_state=False, 
                                    initial_r=1 )
    flat_params = np.array([0.5, 0.0,
                            1.0, 0.0,
                            0.785, 0.0,
                            0.785, 0.0, 
                            0.785, 0.0])  # Example parameters
    measurement_outcomes = [1, 1]  # Example outcomes
    

    # circuit1 = TwoModeTimeDomainSqueezeOnly(steps=1,
    #                                         time_invariant=False,
    #                                         clip_size=1,
    #                                         measure_fock_cutoff=30,
    #                                         num_single_photon=0,
    #                                         train_initial_state=False, 
    #                                         initial_r=1 )

    # flat_params = np.array([0.5, 0.0, 
    #                     0.785, 0.0])  # Example parameters
    # measurement_outcomes = [1]  # Example outcomes
    
    result = run_time_circuit_evaluation(circuit2, flat_params, measurement_outcomes)
    
    if result:
        print(f"Final Probability: {result['final_probability']:.4f}")
        print("Final State Ket:")
        print(result['final_state_ket'])
    else:
        print("Measurement path not realized.")

