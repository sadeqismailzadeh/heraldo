"""
Optimization runner for time-domain multiplexed circuits using Beam Search.
"""
import time
import numpy as np
from scipy.optimize import basinhopping, differential_evolution, dual_annealing
from scipy.special import expit
from scipy.stats import wasserstein_distance
import strawberryfields as sf
from strawberryfields.ops import Ket
import scipy.sparse as sp
from functools import lru_cache

from quantum_agent.optimization.time_interfaces import TimeMultiplexedCircuit
from quantum_agent.components.targets import TargetGenerator
from quantum_agent.utils import *


import multiprocessing
from functools import partial

try:
    import cma
except ImportError:
    raise ImportError("CMA-ES runner requires the 'cma' package. Please install it via 'pip install cma'.")


def _compute_photon_moments(ket, max_moment= 10):
    """
    Compute photon number moments <n^k> for k=1 to max_moment.

    Args:
        ket: State vector in Fock basis
        max_moment: Maximum moment to compute, e.g., 10 for <n> to <n^10>

    Returns:
        Array of length max_moment with moments
    """
    cutoff = len(ket)
    probs = np.abs(ket) ** 2
    support = np.arange(cutoff)

    # Compute moments <n^k> = Σ_n n^k * P(n)
    moments = np.array([np.sum((support ** k) * probs) for k in range(1, max_moment + 1)])
    return moments

def _compute_moment_similarity(moments_target, moments_state):
    """
    Compute similarity between two sets of photon number moments.
    The similarity is based on the inverse of the mean squared error (MSE)
    of moments normalized by the mean photon number (<n>), to ensure scale-invariance.

    Args:
        moments_target (np.ndarray): Array of target moments [<n>, <n^2>, ...].
        moments_state (np.ndarray): Array of state moments [<n>, <n^2>, ...].

    Returns:
        float: Similarity score, where higher is better. Ranges from (0, 1].
    """
    eps = 1e-12

    # Ensure moments are numpy arrays
    moments_target = np.asarray(moments_target)
    moments_state = np.asarray(moments_state)

    # Normalize by the first moment (<n>) to make the comparison scale-invariant.
    # This focuses on the shape of the distribution rather than its mean.
    if moments_target[0] > eps and moments_state[0] > eps:
        moments_target_norm = moments_target / moments_target[0]
        moments_state_norm = moments_state / moments_state[0]

        # Compute similarity as exp(-MSE) of the normalized moments.
        mse = np.mean((moments_target_norm - moments_state_norm) ** 2)
        similarity = np.exp(-mse)
    else:
        # If either state has close to zero photons, they are not similar
        # to a target that is expected to have photons (or vice-versa).
        similarity = 0.0

    return similarity

def evaluate_time_domain_circuit(flat_params, circuit: TimeMultiplexedCircuit, target_kets, cutoff_dim, beam_width,
                                  penalty_strength, measurement_patterns=None, success_threshold=0.99,
                                  success_weight=5.0, ng_weight=0.0, ng_threshold=0.1, prob_power=1,
                                  photon_dist_weight=0.0, max_photon_dist=10, photon_dist_metric='dot_product',
                                  target_photon_moments=None, vacuum_excluded_weight=0.0, return_details: bool = False):
    """
    Evaluates the circuit using either Beam Search or fixed measurement patterns.
    
    Args:
        flat_params: Flat array of parameters
        circuit: TimeMultiplexedCircuit instance
        target_kets: List of target state vectors
        cutoff_dim: Fock space cutoff dimension
        beam_width: Beam width for beam search (ignored if measurement_patterns is provided)
        penalty_strength: Penalty strength for truncation errors
        measurement_patterns: Optional list of fixed measurement patterns.
            Format options:
            - None: Use beam search with given beam_width
            - List of tuples: Each tuple contains outcomes for all measured modes in that step
              [(step1_outcome1, step1_outcome2, ...), (step2_outcome1, step2_outcome2, ...), ...]
            - List of lists: Multiple complete measurement sequences to average over
              [[(step1_outcome1, step1_outcome2, ...), (step2_outcome1, ...), ...],  # Sequence 1
               [(step1_outcome1, step1_outcome2, ...), (step2_outcome1, ...), ...]]  # Sequence 2
        success_threshold: Fidelity threshold for success
        success_weight: Weight for success reward
        ng_weight: Weight for non-Gaussianity penalty
        ng_threshold: Threshold for non-Gaussianity
        prob_power: Power for probability weighting
        photon_dist_weight: Weight for photon moment similarity term
        max_photon_dist: Maximum photon moment to consider for similarity calculation (e.g., 10 for <n> to <n^10>)
        photon_dist_metric: (Ignored) Type of distance metric. Moment-based similarity is always used.
        target_photon_moments: Precomputed photon moments for target states.
                            If None, will compute from target_kets.
        vacuum_excluded_weight: Weight for Vacuum-Excluded SSD penalty (Sum of Squared Differences ignoring n=0).
    
    Returns:
        Loss value
    """
    # 0. Setup
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
    
    # Precompute target photon moments if not provided
    if photon_dist_weight > 1e-6 and target_photon_moments is None:
        target_photon_moments = [
            _compute_photon_moments(ket, max_photon_dist)
            for ket in target_kets
        ]
    
    
    # 1. INITIALIZE
    # Get the custom initial ket (Vacuum or Squeezed)
    initial_ket = circuit.get_initial_state_ket(init_params, cutoff_dim)

    # Check if we're using fixed patterns
    use_fixed_patterns = (measurement_patterns is not None)
    
    
    if use_fixed_patterns:
        # Require dense numpy arrays for fixed-pattern mode for speed and determinism.
        # Accepted shapes:
        #  - (n_sequences, steps, n_meas_modes)
        #  - (steps, n_meas_modes)  -> interpreted as a single sequence
        if not isinstance(measurement_patterns, np.ndarray):
            try:
                patterns_arr = np.array(measurement_patterns, dtype=int)
            except Exception:
                raise ValueError(
                    "measurement_patterns must be a dense numpy.ndarray of shape "
                    "(n_sequences, steps, n_meas_modes) or (steps, n_meas_modes). "
                    "Ragged lists / list-of-tuples are not supported anymore."
                )
        else:
            patterns_arr = measurement_patterns

        # Normalize to (n_sequences, steps, n_meas_modes)
        if patterns_arr.ndim == 2:
            patterns_arr = patterns_arr[None, ...]
        if patterns_arr.ndim != 3:
            raise ValueError(
                f"measurement_patterns must be 2D or 3D numpy array; got array with ndim={patterns_arr.ndim}"
            )

        n_sequences = int(patterns_arr.shape[0])
        meas_modes = [m for m, c in meas_specs]
        meas_cutoffs = [c for m, c in meas_specs]
        perm = [0] + meas_modes  # Loop mode first, then measured

        # Validate shape
        if patterns_arr.shape[1] != circuit.steps:
            raise ValueError(
                f"Patterns (steps) mismatch: patterns have {patterns_arr.shape[1]} steps, circuit expects {circuit.steps}"
            )
        if patterns_arr.shape[2] != len(meas_modes):
            raise ValueError(
                f"Patterns (measured modes) mismatch: patterns have {patterns_arr.shape[2]} measured modes, circuit expects {len(meas_modes)}"
            )

        # Initialize all sequences with the same initial state
        current_kets = np.tile(initial_ket, (n_sequences, 1))  # Shape: (n_sequences, cutoff_dim)
        sequence_probs = np.ones(n_sequences)
        sequence_active = np.ones(n_sequences, dtype=bool)  # Track which sequences are still possible
        total_truncation_error = 0.0

        # Vectorized step processing
        for step in range(circuit.steps):
            step_params = mapped_params[step]
            step_outcomes = patterns_arr[:, step, :]  # Shape: (n_sequences, n_meas_modes)
            
            # Skip sequences that are already impossible
            if not np.any(sequence_active):
                break
                
            active_indices = np.where(sequence_active)[0]
            n_active = len(active_indices)
            
            # Prepare all active kets in a single batch
            batch_kets = current_kets[active_indices]  # Shape: (n_active, cutoff_dim)
            batch_outcomes = step_outcomes[active_indices]  # Shape: (n_active, n_meas_modes)
            
            # Process each active ket (we still need to run engine separately for each,
            # but we can vectorize the projection and probability calculation)
            next_kets = np.zeros_like(batch_kets, dtype=np.complex128)
            step_probs = np.zeros(n_active)
            step_truncation_errors = np.zeros(n_active)
            
            # Run circuit step for each active ket, *but cache runs for identical parent kets*
            # to avoid repeating expensive Engine runs when multiple sequences share the same parent.
            full_kets = []
            ket_cache = {}  # key -> (full_ket, truncation_error)
            for i in range(n_active):
                ket = batch_kets[i]

                # Create a numerically-stable key for the ket by rounding and using its bytes.
                # Rounding helps avoid tiny floating point differences producing different keys.
                key = np.round(ket, decimals=8).tobytes()

                if key in ket_cache:
                    full_ket, trunc_err = ket_cache[key]
                else:
                    # Run circuit step once for this unique parent ket.
                    eng = sf.Engine("fock", backend_options={"cutoff_dim": cutoff_dim})
                    prog_prep = sf.Program(len(meas_modes) + 1)
                    with prog_prep.context as q:
                        Ket(ket) | q[0]
                    eng.run(prog_prep)

                    result = circuit.run_step(None, step, step_params, eng)
                    full_ket = result.state.ket()

                    # Truncation check
                    flat_ket = full_ket.flatten()
                    norm_sq = np.real(np.vdot(flat_ket, flat_ket))
                    trunc_err = np.abs(1.0 - norm_sq)

                    # Cache for reuse within this step
                    ket_cache[key] = (full_ket, trunc_err)

                # Populate arrays for vectorized processing (one entry per active sequence)
                step_truncation_errors[i] = trunc_err
                full_kets.append(full_ket)

            # Stack into array: (n_active, D_loop, D_m1, D_m2, ...)
            full_kets_arr = np.stack(full_kets, axis=0)

            # Batch transpose: prepend batch axis (0) and shift original perm by +1
            batch_perm = (0,) + tuple(p + 1 for p in perm)
            transposed = np.transpose(full_kets_arr, axes=batch_perm)

            # Slice measured modes: (n_active, cutoff_dim, c1, c2, ...)
            indexer = (slice(None), slice(None)) + tuple(slice(0, c) for c in meas_cutoffs)
            sliced_batch = transposed[indexer]

            # Vectorized projection onto provided outcomes:
            # advanced indexing (batch_idx, loop_dim, outcome1, outcome2, ...)
            batch_idx = np.arange(n_active)
            adv_index = (batch_idx, ) + (slice(None),) + tuple(batch_outcomes[:, j] for j in range(batch_outcomes.shape[1]))
            projected_batch = sliced_batch[adv_index]  # Shape: (n_active, cutoff_dim)

            # Probabilities and normalization (vectorized)
            prob_outcomes = np.sum(np.abs(projected_batch) ** 2, axis=1)
            nonzero_mask = prob_outcomes >= 1e-12

            # Update next_kets and step_probs vectorized
            if np.any(nonzero_mask):
                next_kets[nonzero_mask] = projected_batch[nonzero_mask] / np.sqrt(prob_outcomes[nonzero_mask])[:, None]
                step_probs[nonzero_mask] = prob_outcomes[nonzero_mask]
            # sequences with zero probability remain zero (already set)
            
            # Update sequences
            for idx, active_idx in enumerate(active_indices):
                if step_probs[idx] < 1e-12:
                    sequence_active[active_idx] = False
                    sequence_probs[active_idx] = 0.0
                else:
                    current_kets[active_idx] = next_kets[idx]
                    sequence_probs[active_idx] *= step_probs[idx]
                    total_truncation_error += step_truncation_errors[idx] * sequence_probs[active_idx]
        
        # Filter out impossible sequences
        possible_mask = sequence_active & (sequence_probs > 1e-12)
        
        if not np.any(possible_mask):
            return 100.0
        
        # Extract active states and probabilities
        active_kets = current_kets[possible_mask]
        active_probs = sequence_probs[possible_mask]
        
        # Calculate outcome sums for each sequence
        active_outcome_sums = np.sum(patterns_arr[possible_mask], axis=(1, 2))
        
        # Normalize probabilities (each sequence equally weighted)
        seq_weights = 1.0 / np.sum(possible_mask)
        active_probs = active_probs * seq_weights
    
    else:
        # BEAM SEARCH MODE (original implementation)
        # Initialize Beam with this ket
        active_kets = np.zeros((1, cutoff_dim), dtype=np.complex128)
        active_kets[0] = initial_ket 
        active_probs = np.array([1.0])
        active_outcome_sums = np.zeros(1, dtype=int)
        active_outcomes = np.zeros((1, 0), dtype=int)
        
        # Precompute slicing info
        meas_modes = [m for m, c in meas_specs]
        meas_cutoffs = [c for m, c in meas_specs]
        perm = [0] + meas_modes

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

            # Vectorized Outcome Tracking
            # 1. Stack current step outcomes: (K, n_meas_modes)
            current_step_outcomes = np.stack(outcomes_unraveled, axis=1)
            # 2. Filter parents and outcomes by mask
            valid_parent_indices = parent_indices[mask]
            valid_step_outcomes = current_step_outcomes[mask]
            # 3. Gather parent history and append
            parent_history = active_outcomes[valid_parent_indices]
            active_outcomes = np.hstack([parent_history, valid_step_outcomes])

    # --- Final Objective (Vectorized) ---
    
    total_prob = np.sum(active_probs)

    # Filter out zero-outcome branches for reward calculation
    mask_nonzero = active_outcome_sums > 0
    
    expected_fidelity = 0.0
    soft_success_prob = 0.0
    ng_loss = 0.0
    photon_dist_similarity = 0.0
    vacuum_excluded_ssd_score = 0.0

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
        # Best target indices for each branch
        best_target_indices = np.argmax(pairwise_fidelities, axis=1)

        # Logarithmic Reward
        min_infidel=1e-2
        infidelities = np.maximum(1.0 - fidelities, min_infidel)
        log_vals = np.log10(infidelities)  /  np.log10(min_infidel)
        capped_fidelities = np.minimum(fidelities, 1-min_infidel)
        expected_fidelity = np.sum((final_probs**prob_power)
                                   * (capped_fidelities**2 *log_vals)**4)
        # expected_fidelity = np.sum(final_probs + capped_fidelities)
        # expected_fidelity = np.log(expected_fidelity)
        # Soft Success Calculation
        steepness = 50.0
        # expit(-x) == 1 / (1 + exp(x))
        diff = success_threshold - fidelities
        x = np.log(np.maximum(1.0 - diff, 1e-12))
        sigmoids = expit(steepness * x)
        soft_success_prob = np.sum(sigmoids * (capped_fidelities + 0.1*final_probs))  
        
        # Non-Gaussianity Penalty
        if ng_weight > 1e-6:
            ng_scores = compute_ng_scores(final_kets, cutoff_dim)

            # Sigmoid penalty: High (1.0) if score < threshold (Gaussian), Low (0.0) if score > threshold
            # S = 1 / (1 + exp(k * (score - threshold)))
            #   = expit( -k * (score - threshold) )
            ng_steepness = 500.0
            # ng_penalty_terms = 1/(1+np.exp(ng_steepness * (ng_scores - ng_threshold)))
            # NOTE: We use expit(-z) to calculate 1/(1+exp(z)) safely
            ng_penalty_terms = 1- expit(ng_steepness * (ng_scores - ng_threshold))
            
            # ng_loss = np.sum(ng_penalty_terms)
            ng_loss = np.sum(abs(ng_threshold - ng_scores))

        # Photon Moment Similarity
        if photon_dist_weight > 1e-6:
            # Compute photon moments for each branch
            branch_photon_moments = np.array([
                _compute_photon_moments(ket, max_photon_dist)
                for ket in final_kets
            ])

            # Get corresponding target moments
            if target_photon_moments is None:
                target_photon_moments = [
                    _compute_photon_moments(ket, max_photon_dist)
                    for ket in target_kets
                ]

            # Compute similarity for each branch with its best matching target
            branch_similarities = np.zeros(len(final_kets))
            for i, branch_moments in enumerate(branch_photon_moments):
                target_idx = best_target_indices[i]
                target_moments = target_photon_moments[target_idx]
                similarity = _compute_moment_similarity(
                    target_moments, branch_moments
                )
                branch_similarities[i] = similarity

            # Weighted average similarity
            photon_dist_similarity = np.sum(branch_similarities)

        vacuum_excluded_weight = 0
        # Vacuum-Excluded SSD (Rotationally Invariant)
        if vacuum_excluded_weight > 1e-12:
            # 1. Mask Vacuum (n=0 set to 0) - preserve original normalization for energy calc
            masked_gen = final_kets.copy()
            masked_gen[:, 0] = 0.0
            
            masked_tgt = np.array(target_kets, dtype=np.complex128)
            masked_tgt[:, 0] = 0.0
            
            # 2. Energies (Squared Norms)
            E_gen = np.sum(np.abs(masked_gen)**2, axis=1) # (N_branches,)
            E_tgt = np.sum(np.abs(masked_tgt)**2, axis=1) # (N_targets,)
            
            # 3. FFT Overlap (Maximize over phase)
            # P = masked_gen * conj(masked_tgt)
            # Broadcasting: (N_branches, 1, D) * (1, N_targets, D)
            prod_ssd = masked_gen[:, None, :] * np.conj(masked_tgt[None, :, :])
            
            # FFT along Fock axis
            fft_vals_ssd = np.fft.fft(prod_ssd, n=256, axis=-1)
            M_max = np.max(np.abs(fft_vals_ssd), axis=-1) # (N_branches, N_targets)
            
            # 4. SSD Calculation
            # SSD = E_gen + E_tgt - 2*M_max
            # Broadcast E terms
            ssd_matrix = E_gen[:, None] + E_tgt[None, :] - 2 * M_max
            ssd_matrix = np.maximum(ssd_matrix, 0.0) # Numerical stability
            
            # 5. Best target per branch
            best_ssd_per_branch = np.min(ssd_matrix, axis=1)
            
            # 6. Weighted Sum
            # vacuum_excluded_ssd_score = np.sum(final_probs * best_ssd_per_branch**4)

            # Modified calculation to prevent zero-probability cheating


            total_prob = np.sum(final_probs)
            if total_prob > 1e-12:
                min_infidel=1e-6
                infidelities = np.maximum(1.0 - best_ssd_per_branch, min_infidel)
                log_vals = np.log10(infidelities)  /  np.log10(min_infidel)
                capped_fidelities = np.minimum(best_ssd_per_branch, 1-min_infidel)
                vacuum_excluded_ssd_score = np.sum((final_probs**prob_power)
                                        * (capped_fidelities**2 *log_vals)**4)
                # Normalize by total probability to get the Conditional Expected SSD
                vacuum_excluded_ssd_score = np.sum(final_probs * best_ssd_per_branch) / total_prob
            else:
                # Penalize if probability is too low
                vacuum_excluded_ssd_score = 10.0


    # Return loss
    # loss = -expected_fidelity - (success_weight * soft_success_prob) \
    #        + (penalty_strength * total_truncation_error) + (ng_weight * ng_loss)

    loss = -1*expected_fidelity + (penalty_strength * total_truncation_error) \
           + (vacuum_excluded_weight * vacuum_excluded_ssd_score)
    
    
    if not return_details:
        return loss

    # --- Build branch details (shared for beam + fixed patterns) ---
    branch_details = []

    if np.any(mask_nonzero):
        # Prepare filtered outcomes
        if measurement_patterns is not None:
            # Flatten patterns: (N_seq, Steps, Modes) -> (N_seq, Steps*Modes)
            flat_patterns = patterns_arr.reshape(patterns_arr.shape[0], -1)
            # Filter by possible sequences (active_kets correspond to these)
            surviving_patterns = flat_patterns[possible_mask]
            # Filter by nonzero sum (final_probs correspond to these)
            final_outcomes_arr = surviving_patterns[mask_nonzero]
        else:
            final_outcomes_arr = active_outcomes[mask_nonzero]

        for i in range(len(final_probs)):
            # Convert to tuple of ints
            outcome = tuple(final_outcomes_arr[i].tolist())
            
            # Compute photon moment similarity if needed
            photon_sim = 0.0
            if photon_dist_weight > 1e-6:
                branch_moments = _compute_photon_moments(
                    final_kets[i], max_photon_dist
                )
                target_idx = best_target_indices[i]
                target_moments = target_photon_moments[target_idx] if target_photon_moments is not None else \
                    _compute_photon_moments(target_kets[target_idx], max_photon_dist)
                photon_sim = _compute_moment_similarity(
                    target_moments, branch_moments
                )

            branch_details.append({
                "outcome": outcome,
                "prob": float(final_probs[i]),
                "fidelity": float(fidelities[i]),
                "target_idx": int(best_target_indices[i]),
                "photon_similarity": float(photon_sim) if photon_dist_weight > 1e-6 else None
            })

    return {
        "loss": loss,
        "expected_fidelity": float(expected_fidelity),
        "photon_similarity": float(photon_dist_similarity) if photon_dist_weight > 1e-6 else 0.0,
        "vacuum_excluded_ssd": float(vacuum_excluded_ssd_score) if vacuum_excluded_weight > 1e-6 else 0.0,
        "branches": branch_details,
        "total_probability": float(np.sum(final_probs)) if np.any(mask_nonzero) else 0.0
    }

class BasinHoppingRunner:
    """
    Optimizes time-domain circuits using Basin-Hopping with a Beam Search strategy.
    
    This runner executes the circuit step-by-step, maintaining a 'beam' of the most 
    probable trajectories (measurement outcomes).
    
    Phases:
    A. Evolution: Serial execution of the circuit step for each active branch.
    B. Virtual Branching: Vectorized calculation of all possible measurement outcome probabilities.
    C. Pruning: Vectorized selection of the top-K probable paths (Beam Search).
    D. Realization: Lazy projection and normalization of the selected states.
    """
    def __init__(self, 
                 circuit: TimeMultiplexedCircuit, 
                 target_gens: list[TargetGenerator], 
                 cutoff_dim: int,
                 beam_width: int = 5,
                 penalty_strength: float = 10.0,
                 success_threshold: float = 0.99,
                 success_weight: float = 5.0,
                 ng_weight: float = 0.0,
                 ng_threshold: float = 0.1,
                 photon_dist_weight: float = 0.0,
                 max_photon_dist: int = 10,
                 photon_dist_metric: str = 'dot_product',
                 measurement_patterns = None,
                 **kwargs):
        
        self.prob_power = 1.0
        self.circuit = circuit

        if not isinstance(target_gens, list):
            target_gens = [target_gens]
        
        self.target_kets = [gen.get_target_ket(cutoff_dim) for gen in target_gens]
        self.cutoff_dim = cutoff_dim
        self.beam_width = beam_width
        self.penalty_strength = penalty_strength
        self.success_threshold = success_threshold
        self.success_weight = success_weight
        self.ng_weight = ng_weight
        self.ng_threshold = ng_threshold
        self.photon_dist_weight = photon_dist_weight
        self.max_photon_dist = max_photon_dist
        self.photon_dist_metric = photon_dist_metric
        self.measurement_patterns = measurement_patterns
        self.eval_count = 0
        self.iteration_count = 0
        
        # Precompute target photon moments if needed
        if photon_dist_weight > 1e-6:
            self.target_photon_moments = [
                _compute_photon_moments(ket, max_photon_dist)
                for ket in self.target_kets
            ]
        else:
            self.target_photon_moments = None
        
    def _loss_function(self, flat_params):
        """
        Calculates loss: -Expected_Fidelity + Penalties
        """
        self.eval_count += 1
        return evaluate_time_domain_circuit(
            flat_params,
            self.circuit,
            self.target_kets,
            self.cutoff_dim,
            self.beam_width,
            self.penalty_strength,
            self.measurement_patterns,
            self.success_threshold,
            self.success_weight,
            self.ng_weight,
            self.ng_threshold,
            prob_power=self.prob_power,
            photon_dist_weight=self.photon_dist_weight,
            max_photon_dist=self.max_photon_dist,
            photon_dist_metric=self.photon_dist_metric,
            target_photon_moments=self.target_photon_moments
        )

    def callback(self, x, f, accept):
        self.iteration_count += 1
        status = "Accept" if accept else "Reject"
        print(f"  [Iteration {self.iteration_count}] [{status}] (Evals: {self.eval_count}) Loss: {f} ")
        self.eval_count = 0

    def run(self, n_iter=20, method="SLSQP", prob_power=1.0, n_generations=None):
        """
        Runs the global optimization.
        """
        if n_generations is not None:
            n_iter = n_generations
            
        self.prob_power = prob_power
        # Construct full parameter bounds
        per_step_bounds = self.circuit.per_step_parameter_bounds
        if self.circuit.time_invariant:
            step_bounds = per_step_bounds
        else:
            step_bounds = per_step_bounds * self.circuit.steps
            
        # 2. Init Bounds (NEW)
        init_bounds = self.circuit.initial_parameter_bounds
        
        # 3. Concatenate
        full_bounds = init_bounds + step_bounds
        
            
        # Initial guess
        x0 = np.array([np.random.uniform(l, h) for l, h in full_bounds])
        
        minimizer_kwargs = {
            "method": method,
            "bounds": full_bounds,
            "tol": 1e-4
        }
        
        print(f"Starting Basin-Hopping Beam Search (Width={self.beam_width}, Steps={self.circuit.steps}, NG_Weight={self.ng_weight})...")
        start_time = time.time()
        
        result = basinhopping(
            self._loss_function,
            x0,
            niter=n_iter,
            minimizer_kwargs=minimizer_kwargs,
            callback=self.callback,
            stepsize=0.5
        )
        
        # --- Final Evaluation (UNIFIED PATH) ---
        final_eval = evaluate_time_domain_circuit(
            result.x,
            circuit=self.circuit,
            target_kets=self.target_kets,
            cutoff_dim=self.cutoff_dim,
            beam_width=self.beam_width,
            penalty_strength=self.penalty_strength,
            measurement_patterns=self.measurement_patterns,
            success_threshold=self.success_threshold,
            success_weight=self.success_weight,
            ng_weight=self.ng_weight,
            ng_threshold=self.ng_threshold,
            prob_power=self.prob_power,
            photon_dist_weight=self.photon_dist_weight,
            max_photon_dist=self.max_photon_dist,
            photon_dist_metric=self.photon_dist_metric,
            target_photon_moments=self.target_photon_moments,
            return_details=True
        )

        duration = time.time() - start_time
        return {
            "x": result.x,
            "loss": final_eval["loss"],
            "expected_fidelity": final_eval.get("expected_fidelity", 0.0),
            "photon_similarity": final_eval.get("photon_similarity", 0.0),
            "branches": final_eval.get("branches", []),
            "total_probability": final_eval.get("total_probability", 0.0),
            "duration": duration,
            "message": result.message
        }



class CMAESOptimizationRunner:
    """
    Optimizes time-domain circuits using CMA-ES (Covariance Matrix Adaptation Evolution Strategy).
    """
    def __init__(self, 
                 circuit: TimeMultiplexedCircuit, 
                 target_gens: list[TargetGenerator], 
                 cutoff_dim: int,
                 beam_width: int = 5,
                 penalty_strength: float = 10.0,
                 success_threshold: float = 0.99,
                 success_weight: float = 5.0,
                 ng_weight: float = 0.0,
                 ng_threshold: float = 0.1,
                 photon_dist_weight: float = 0.0,
                 max_photon_dist: int = 10,
                 photon_dist_metric: str = 'dot_product',
                 num_processes: int = 4,
                 sigma0: float = 0.5,
                 measurement_patterns = None,
                 **kwargs):
        
        self.circuit = circuit
        self.cutoff_dim = cutoff_dim
        self.beam_width = beam_width
        self.penalty_strength = penalty_strength
        self.success_threshold = success_threshold
        self.success_weight = success_weight
        self.ng_weight = ng_weight
        self.ng_threshold = ng_threshold
        self.photon_dist_weight = photon_dist_weight
        self.max_photon_dist = max_photon_dist
        self.photon_dist_metric = photon_dist_metric
        self.num_processes = num_processes
        self.sigma0 = sigma0
        self.measurement_patterns = measurement_patterns

        if not isinstance(target_gens, list):
            target_gens = [target_gens]
        
        # Precompute targets
        self.target_kets = [gen.get_target_ket(cutoff_dim) for gen in target_gens]
        
        # Precompute target photon moments if needed
        if photon_dist_weight > 1e-6:
            self.target_photon_moments = [
                _compute_photon_moments(ket, max_photon_dist)
                for ket in self.target_kets
            ]
        else:
            self.target_photon_moments = None

    def run(self, n_generations=50, population_size=None, prob_power=1):
        """
        Runs the CMA-ES optimization.
        
        Args:
            n_generations (int): Maximum number of generations.
            population_size (int): Size of the population (lambda). If None, CMA defaults are used.
        """
        # 1. Setup Parameters and Bounds
        per_step_bounds = self.circuit.per_step_parameter_bounds
        self.prob_power = prob_power
        
        if self.circuit.time_invariant:
            step_bounds = per_step_bounds
        else:
            step_bounds = per_step_bounds * self.circuit.steps
            
        # 2. Init Bounds
        init_bounds = self.circuit.initial_parameter_bounds
        
        # 3. Concatenate
        full_bounds = init_bounds + step_bounds
        
        # CMA expects bounds in format [[lower_1, lower_2...], [upper_1, upper_2...]]
        lower_bounds = [b[0] for b in full_bounds]
        upper_bounds = [b[1] for b in full_bounds]
        
        # Initial guess (random within bounds)
        x0 = np.array([np.random.uniform(l, h) for l, h in full_bounds])
        
        # 2. Initialize CMA-ES
        opts = {
            'bounds': [lower_bounds, upper_bounds],
            'maxiter': n_generations,
            'verbose': -1,  # Suppress internal printing
        }
        if population_size:
            opts['popsize'] = population_size
            
        es = cma.CMAEvolutionStrategy(x0, self.sigma0, opts)
        
        print(f"Starting CMA-ES (Generations={n_generations}, PopSize={es.popsize}, "
              f"Processes={self.num_processes}, NG_Weight={self.ng_weight}, "
              f"PhotonDistWeight={self.photon_dist_weight}, "
              f"Patterns={'fixed' if self.measurement_patterns is not None else 'beam'})...")
        start_time = time.time()
        
        # Create a partial function with fixed arguments for the worker
        # We bind the complex objects here so the pool only transmits the params
        worker_func = partial(
            evaluate_time_domain_circuit,
            circuit=self.circuit,
            target_kets=self.target_kets,
            cutoff_dim=self.cutoff_dim,
            beam_width=self.beam_width,
            penalty_strength=self.penalty_strength,
            measurement_patterns=self.measurement_patterns,
            success_threshold=self.success_threshold,
            success_weight=self.success_weight,
            ng_weight=self.ng_weight,
            ng_threshold=self.ng_threshold,
            prob_power=self.prob_power,
            photon_dist_weight=self.photon_dist_weight,
            max_photon_dist=self.max_photon_dist,
            photon_dist_metric=self.photon_dist_metric,
            target_photon_moments=self.target_photon_moments
        )
        
        best_loss = float('inf')
        best_x = None

        # 3. Evolution Loop
        with multiprocessing.Pool(processes=self.num_processes) as pool:
            for gen in range(n_generations):
                if es.stop():
                    break
                    
                X = es.ask()
                
                # Parallel Evaluation
                fitness_values = pool.map(worker_func, X)
                
                es.tell(X, fitness_values)
                es.logger.add()  # write to cma files
                
                current_best_loss = min(fitness_values)
                if current_best_loss < best_loss:
                    best_loss = current_best_loss
                    best_x = es.result.xbest
                    
                print(f"  Gen {gen+1}/{n_generations} | Sigma: {es.sigma:.3f} | Min Loss: {current_best_loss} ")

        duration = time.time() - start_time
        final_x = es.result.xbest
        final_loss = best_loss
        
        
        # 4. Final Evaluation (UNIFIED PATH)
        final_eval = evaluate_time_domain_circuit(
            final_x,
            circuit=self.circuit,
            target_kets=self.target_kets,
            cutoff_dim=self.cutoff_dim,
            beam_width=self.beam_width,
            penalty_strength=self.penalty_strength,
            measurement_patterns=self.measurement_patterns,
            success_threshold=self.success_threshold,
            success_weight=self.success_weight,
            ng_weight=self.ng_weight,
            ng_threshold=self.ng_threshold,
            prob_power=self.prob_power,
            photon_dist_weight=self.photon_dist_weight,
            max_photon_dist=self.max_photon_dist,
            photon_dist_metric=self.photon_dist_metric,
            target_photon_moments=self.target_photon_moments,
            return_details=True
        )

        return {
            "x": final_x,
            "loss": final_eval["loss"],
            "expected_fidelity": final_eval["expected_fidelity"],
            "photon_similarity": final_eval.get("photon_similarity", 0.0),
            "branches": final_eval["branches"],
            "total_probability": final_eval["total_probability"],
            "duration": duration,
            "message": "CMA-ES Finished"
        }


class DifferentialEvolutionRunner:
    """
    Optimizes time-domain circuits using Differential Evolution (DE).
    Useful for rugged landscapes where CMA-ES might get stuck in local minima.
    """
    def __init__(self,
                 circuit: TimeMultiplexedCircuit,
                 target_gens: list[TargetGenerator],
                 cutoff_dim: int,
                 beam_width: int = 5,
                 penalty_strength: float = 10.0,
                 success_threshold: float = 0.99,
                 success_weight: float = 5.0,
                 ng_weight: float = 0.0,
                 ng_threshold: float = 0.1,
                 photon_dist_weight: float = 0.0,
                 max_photon_dist: int = 10,
                 photon_dist_metric: str = 'dot_product',
                 num_processes: int = 4,
                 measurement_patterns = None,
                 popsize: int = 15,
                 **kwargs):

        self.circuit = circuit
        self.cutoff_dim = cutoff_dim
        self.beam_width = beam_width
        self.penalty_strength = penalty_strength
        self.success_threshold = success_threshold
        self.success_weight = success_weight
        self.ng_weight = ng_weight
        self.ng_threshold = ng_threshold
        self.photon_dist_weight = photon_dist_weight
        self.max_photon_dist = max_photon_dist
        self.photon_dist_metric = photon_dist_metric
        self.num_processes = num_processes
        self.measurement_patterns = measurement_patterns
        self.popsize = popsize

        if not isinstance(target_gens, list):
            target_gens = [target_gens]

        # Precompute targets
        self.target_kets = [gen.get_target_ket(cutoff_dim) for gen in target_gens]

        # Precompute target photon moments if needed
        if photon_dist_weight > 1e-6:
            self.target_photon_moments = [
                _compute_photon_moments(ket, max_photon_dist)
                for ket in self.target_kets
            ]
        else:
            self.target_photon_moments = None

    def run(self, n_generations=50, prob_power=1):
        """
        Runs the Differential Evolution optimization.

        Args:
            n_generations (int): Maximum number of generations.
            prob_power (float): Power for probability weighting in evaluation.
        """
        # 1. Setup Parameters and Bounds
        per_step_bounds = self.circuit.per_step_parameter_bounds
        self.prob_power = prob_power

        if self.circuit.time_invariant:
            step_bounds = per_step_bounds
        else:
            step_bounds = per_step_bounds * self.circuit.steps

        # 2. Init Bounds
        init_bounds = self.circuit.initial_parameter_bounds

        # 3. Concatenate
        full_bounds = init_bounds + step_bounds

        print(f"Starting Differential Evolution (Generations={n_generations}, PopSize={self.popsize}, "
              f"Processes={self.num_processes}, NG_Weight={self.ng_weight}, "
              f"PhotonDistWeight={self.photon_dist_weight})...")
        start_time = time.time()

        # Create a partial function to freeze arguments for the worker
        # Note: We create args for evaluate_time_domain_circuit
        # Scipy DE requires the objective function to take x as the first argument.
        # evaluate_time_domain_circuit(flat_params, ...) accepts flat_params as first arg.
        # We use partial to bind the rest of keyword arguments.
        objective_wrapper = partial(
            evaluate_time_domain_circuit,
            circuit=self.circuit,
            target_kets=self.target_kets,
            cutoff_dim=self.cutoff_dim,
            beam_width=self.beam_width,
            penalty_strength=self.penalty_strength,
            measurement_patterns=self.measurement_patterns,
            success_threshold=self.success_threshold,
            success_weight=self.success_weight,
            ng_weight=self.ng_weight,
            ng_threshold=self.ng_threshold,
            prob_power=self.prob_power,
            photon_dist_weight=self.photon_dist_weight,
            max_photon_dist=self.max_photon_dist,
            photon_dist_metric=self.photon_dist_metric,
            target_photon_moments=self.target_photon_moments,
            return_details=False
        )

        # Callback to print progress
        def callback(xk, convergence):
            print(f"  DE Step | Convergence: {convergence:.5f}")

        # 3. Run Optimization
        # workers=-1 uses all available processors, or we can pass self.num_processes
        workers_arg = self.num_processes if self.num_processes > 0 else 1
        
        result = differential_evolution(
            objective_wrapper,
            bounds=full_bounds,
            maxiter=n_generations,
            popsize=self.popsize,
            workers=workers_arg,
            callback=callback,
            disp=True,
            polish=True  # Refine result with L-BFGS-B at the end
        )

        duration = time.time() - start_time
        final_x = result.x
        
        # 4. Final Evaluation (Detailed)
        final_eval = evaluate_time_domain_circuit(
            final_x,
            circuit=self.circuit,
            target_kets=self.target_kets,
            cutoff_dim=self.cutoff_dim,
            beam_width=self.beam_width,
            penalty_strength=self.penalty_strength,
            measurement_patterns=self.measurement_patterns,
            success_threshold=self.success_threshold,
            success_weight=self.success_weight,
            ng_weight=self.ng_weight,
            ng_threshold=self.ng_threshold,
            prob_power=self.prob_power,
            photon_dist_weight=self.photon_dist_weight,
            max_photon_dist=self.max_photon_dist,
            photon_dist_metric=self.photon_dist_metric,
            target_photon_moments=self.target_photon_moments,
            return_details=True
        )

        return {
            "x": final_x,
            "loss": final_eval["loss"],
            "expected_fidelity": final_eval["expected_fidelity"],
            "photon_similarity": final_eval.get("photon_similarity", 0.0),
            "branches": final_eval["branches"],
            "total_probability": final_eval["total_probability"],
            "duration": duration,
            "message": result.message
        }


class DualAnnealingRunner:
    """
    Optimizes time-domain circuits using Dual Annealing.
    Combines Generalized Simulated Annealing with local search (L-BFGS-B).
    Useful for rugged landscapes where CMA-ES might get stuck in local minima.
    """
    def __init__(self,
                 circuit: TimeMultiplexedCircuit,
                 target_gens: list[TargetGenerator],
                 cutoff_dim: int,
                 beam_width: int = 5,
                 penalty_strength: float = 10.0,
                 success_threshold: float = 0.99,
                 success_weight: float = 5.0,
                 ng_weight: float = 0.0,
                 ng_threshold: float = 0.1,
                 photon_dist_weight: float = 0.0,
                 max_photon_dist: int = 10,
                 photon_dist_metric: str = 'dot_product',
                 measurement_patterns = None,
                 **kwargs):

        self.circuit = circuit
        self.cutoff_dim = cutoff_dim
        self.beam_width = beam_width
        self.penalty_strength = penalty_strength
        self.success_threshold = success_threshold
        self.success_weight = success_weight
        self.ng_weight = ng_weight
        self.ng_threshold = ng_threshold
        self.photon_dist_weight = photon_dist_weight
        self.max_photon_dist = max_photon_dist
        self.photon_dist_metric = photon_dist_metric
        self.measurement_patterns = measurement_patterns

        if not isinstance(target_gens, list):
            target_gens = [target_gens]

        # Precompute targets
        self.target_kets = [gen.get_target_ket(cutoff_dim) for gen in target_gens]

        # Precompute target photon moments if needed
        if photon_dist_weight > 1e-6:
            self.target_photon_moments = [
                _compute_photon_moments(ket, max_photon_dist)
                for ket in self.target_kets
            ]
        else:
            self.target_photon_moments = None

    def run(self, maxiter=1000, initial_temp=5230.0, restart_temp_ratio=2e-5, 
            visit=2.62, accept=-5.0, prob_power=1, seed=None):
        """
        Runs the Dual Annealing optimization.

        Args:
            maxiter (int): Maximum number of global search iterations.
            initial_temp (float): Initial temperature.
            restart_temp_ratio (float): Restart temperature ratio.
            visit (float): Parameter for visiting distribution.
            accept (float): Parameter for acceptance distribution.
            prob_power (float): Power for probability weighting in evaluation.
            seed (int): Random seed.
        """
        # 1. Setup Parameters and Bounds
        per_step_bounds = self.circuit.per_step_parameter_bounds
        self.prob_power = prob_power

        if self.circuit.time_invariant:
            step_bounds = per_step_bounds
        else:
            step_bounds = per_step_bounds * self.circuit.steps

        # 2. Init Bounds
        init_bounds = self.circuit.initial_parameter_bounds

        # 3. Concatenate
        full_bounds = init_bounds + step_bounds

        print(f"Starting Dual Annealing (MaxIter={maxiter}, InitialTemp={initial_temp}, "
              f"NG_Weight={self.ng_weight}, PhotonDistWeight={self.photon_dist_weight})...")
        start_time = time.time()
        
        # Create a partial function to freeze arguments
        objective_wrapper = partial(
            evaluate_time_domain_circuit,
            circuit=self.circuit,
            target_kets=self.target_kets,
            cutoff_dim=self.cutoff_dim,
            beam_width=self.beam_width,
            penalty_strength=self.penalty_strength,
            measurement_patterns=self.measurement_patterns,
            success_threshold=self.success_threshold,
            success_weight=self.success_weight,
            ng_weight=self.ng_weight,
            ng_threshold=self.ng_threshold,
            prob_power=self.prob_power,
            photon_dist_weight=self.photon_dist_weight,
            max_photon_dist=self.max_photon_dist,
            photon_dist_metric=self.photon_dist_metric,
            target_photon_moments=self.target_photon_moments,
            return_details=False
        )
        
        self.iteration_count = 0

        # Callback to print progress
        def callback(x, f, context):
            # context: 0=Annealing, 1=Local Search, 2=Done
            self.iteration_count += 1
            ctx_map = {0: "Annealing", 1: "Local Search", 2: "Done"}
            status = ctx_map.get(context, str(context))
            print(f"  [Iter {self.iteration_count}] [{status}] Loss: {f}")
            return False

        # 3. Run Optimization
        result = dual_annealing(
            objective_wrapper,
            bounds=full_bounds,
            maxiter=maxiter,
            initial_temp=initial_temp,
            restart_temp_ratio=restart_temp_ratio,
            visit=visit,
            accept=accept,
            seed=seed,
            callback=callback
        )

        duration = time.time() - start_time
        final_x = result.x
        
        # 4. Final Evaluation (Detailed)
        final_eval = evaluate_time_domain_circuit(
            final_x,
            circuit=self.circuit,
            target_kets=self.target_kets,
            cutoff_dim=self.cutoff_dim,
            beam_width=self.beam_width,
            penalty_strength=self.penalty_strength,
            measurement_patterns=self.measurement_patterns,
            success_threshold=self.success_threshold,
            success_weight=self.success_weight,
            ng_weight=self.ng_weight,
            ng_threshold=self.ng_threshold,
            prob_power=self.prob_power,
            photon_dist_weight=self.photon_dist_weight,
            max_photon_dist=self.max_photon_dist,
            photon_dist_metric=self.photon_dist_metric,
            target_photon_moments=self.target_photon_moments,
            return_details=True
        )

        return {
            "x": final_x,
            "loss": final_eval["loss"],
            "expected_fidelity": final_eval["expected_fidelity"],
            "photon_similarity": final_eval.get("photon_similarity", 0.0),
            "branches": final_eval["branches"],
            "total_probability": final_eval["total_probability"],
            "duration": duration,
            "message": result.message
        }
