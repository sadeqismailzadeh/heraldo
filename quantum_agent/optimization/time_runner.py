"""
Optimization runner for time-domain multiplexed circuits using Beam Search.
"""
import time
import numpy as np
from scipy.optimize import basinhopping
import strawberryfields as sf
from strawberryfields.ops import Ket

from quantum_agent.optimization.time_interfaces import TimeMultiplexedCircuit
from quantum_agent.components.targets import TargetGenerator
from quantum_agent.envs.modular_env import fidelity_max_rotation

def evaluate_time_domain_circuit(flat_params, circuit, target_kets, cutoff_dim, beam_width, penalty_strength, success_threshold=0.99, success_weight=5.0):
    """
    Evaluates the circuit using Beam Search and returns the loss.
    """
    # 0. Setup
    mapped_params = circuit.map_parameters(flat_params)
    meas_specs = circuit.get_measurement_specs()
    
    # Initialize Beam: Single vacuum state on Mode 0 (Loop)
    active_kets = np.zeros((1, cutoff_dim), dtype=np.complex128)
    active_kets[0, 0] = 1.0
    active_probs = np.array([1.0])
    active_outcome_sums = np.zeros(1, dtype=int)
    
    # Precompute slicing info
    meas_modes = [m for m, c in meas_specs]
    meas_cutoffs = [c for m, c in meas_specs]
    perm = [0] + meas_modes # Loop mode first, then measured
    
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
    mask_nonzero = active_outcome_sums > 0
    
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
        infidelities = np.maximum(1.0 - fidelities, 1e-2)
        log_vals = -np.log10(infidelities)        
        expected_fidelity = np.sum((final_probs**0.1) * log_vals)

        # Soft Success Calculation
        steepness = 500.0
        sigmoids = 1.0 / (1.0 + np.exp(-steepness * (fidelities - success_threshold)))
        soft_success_prob = np.sum(final_probs * sigmoids)
    
    else:
        expected_fidelity = 0.0
        soft_success_prob = 0.0
        
    # Penalties
    # 1. Total Probability Loss (indicates truncation or dropped branches)
    loss_prob = np.abs(1.0 - total_prob)
    
    return -expected_fidelity - (success_weight * soft_success_prob) + (penalty_strength * loss_prob)


class TimeDomainRunner:
    """
    Optimizes time-domain circuits using a Beam Search strategy.
    
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
                 success_weight: float = 5.0):
        
        self.circuit = circuit

        if not isinstance(target_gens, list):
            target_gens = [target_gens]
        
        self.target_kets = [gen.get_target_ket(cutoff_dim) for gen in target_gens]
        self.cutoff_dim = cutoff_dim
        self.beam_width = beam_width
        self.penalty_strength = penalty_strength
        self.success_threshold = success_threshold
        self.success_weight = success_weight
        self.eval_count = 0
        
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
            self.success_threshold,
            self.success_weight
        )

    def callback(self, x, f, accept):
        if accept:
            print(f"  [Accept] Loss: {f:.5f} (Evals: {self.eval_count})")
        self.eval_count = 0

    def run(self, n_iter=20, method="SLSQP"):
        """
        Runs the global optimization.
        """
        # Construct full parameter bounds
        per_step_bounds = self.circuit.per_step_parameter_bounds
        if self.circuit.time_invariant:
            full_bounds = per_step_bounds
        else:
            full_bounds = per_step_bounds * self.circuit.steps
            
        # Initial guess
        x0 = np.array([np.random.uniform(l, h) for l, h in full_bounds])
        
        minimizer_kwargs = {
            "method": method,
            "bounds": full_bounds,
            "tol": 1e-4
        }
        
        print(f"Starting Time-Domain Beam Search (Width={self.beam_width}, Steps={self.circuit.steps})...")
        start_time = time.time()
        
        result = basinhopping(
            self._loss_function,
            x0,
            niter=n_iter,
            minimizer_kwargs=minimizer_kwargs,
            callback=self.callback,
            stepsize=0.5
        )
        
        # --- Evaluate final details with history ---
        mapped_params = self.circuit.map_parameters(result.x)
        meas_specs = self.circuit.get_measurement_specs()
        meas_modes = [m for m, c in meas_specs]
        meas_cutoffs = [c for m, c in meas_specs]
        perm = [0] + meas_modes

        # Initialize Beam
        active_kets = np.zeros((1, self.cutoff_dim), dtype=np.complex128)
        active_kets[0, 0] = 1.0
        active_probs = np.array([1.0])
        active_outcomes = [()]  # List of tuples

        for step in range(self.circuit.steps):
            step_params = mapped_params[step]

            branch_raw_probs = []
            branch_tensors = []

            # Phase A: Evolution
            for idx in range(len(active_kets)):
                parent_ket = active_kets[idx]
                eng = sf.Engine("fock", backend_options={"cutoff_dim": self.cutoff_dim})

                prog_prep = sf.Program(len(meas_modes) + 1)
                with prog_prep.context as q:
                    Ket(parent_ket) | q[0]
                eng.run(prog_prep)

                result_step = self.circuit.run_step(None, step, step_params, eng)
                full_ket = result_step.state.ket()

                transposed_ket = np.transpose(full_ket, axes=perm)
                slices = [slice(None)] + [slice(0, c) for c in meas_cutoffs]
                sliced_ket = transposed_ket[tuple(slices)]

                branch_tensors.append(sliced_ket)
                probs_tensor = np.sum(np.abs(sliced_ket) ** 2, axis=0)
                branch_raw_probs.append(probs_tensor.flatten())

            # Phase B: Virtual Branching
            P_parent = active_probs
            P_raw = np.stack(branch_raw_probs)
            P_total = P_parent[:, None] * P_raw

            # Phase C: Pruning
            flat_P = P_total.flatten()
            k = min(self.beam_width, flat_P.size)
            top_indices = np.argpartition(flat_P, -k)[-k:]
            top_indices = top_indices[np.argsort(flat_P[top_indices])[::-1]]

            parent_indices, outcome_indices_flat = np.unravel_index(top_indices, P_total.shape)
            outcomes_unraveled = np.unravel_index(outcome_indices_flat, tuple(meas_cutoffs))

            # Phase D: Realization
            tensor_stack = np.stack(branch_tensors)
            indexer = (parent_indices, slice(None)) + outcomes_unraveled
            raw_new_kets = tensor_stack[indexer]
            
            norms = np.linalg.norm(raw_new_kets, axis=1)
            selected_probs = P_total[parent_indices, outcome_indices_flat]

            mask = (norms > 1e-9) & (selected_probs > 1e-12)

            if not np.any(mask):
                break

            active_kets = raw_new_kets[mask] / norms[mask][:, None]
            active_probs = selected_probs[mask]

            # Track history
            new_outcomes = []
            masked_parent_indices = parent_indices[mask]
            
            # Re-slice outcomes unraveled based on mask
            masked_outcomes_tuple = tuple(arr[mask] for arr in outcomes_unraveled)
            
            for i in range(len(masked_parent_indices)):
                p_idx = masked_parent_indices[i]
                step_outcome = tuple(int(arr[i]) for arr in masked_outcomes_tuple)
                new_outcomes.append(active_outcomes[p_idx] + step_outcome)
            active_outcomes = new_outcomes

        branch_details = []
        if len(active_kets) > 0:
            targets_arr = np.array(self.target_kets)
            prod = np.conj(active_kets[:, None, :]) * targets_arr[None, :, :]
            fft_vals = np.fft.fft(prod, n=256, axis=-1)
            pairwise_fidelities = np.max(np.abs(fft_vals) ** 2, axis=-1)

            best_fidelities = np.max(pairwise_fidelities, axis=1)
            best_target_idxs = np.argmax(pairwise_fidelities, axis=1)

            for i in range(len(active_probs)):
                branch_details.append({
                    "outcome": active_outcomes[i],
                    "prob": float(active_probs[i]),
                    "fidelity": float(best_fidelities[i]),
                    "target_idx": int(best_target_idxs[i])
                })

        return {
            "x": result.x,
            "loss": result.fun,
            "duration": time.time() - start_time,
            "message": result.message,
            "branches": branch_details
        }
