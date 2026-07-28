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

from heraldo.optimization.time_interfaces import TimeMultiplexedCircuit
from heraldo.components.targets import TargetGenerator
from heraldo.utils import *


import multiprocessing
from functools import partial
from tqdm import tqdm


def _process_fixed_patterns(circuit: TimeMultiplexedCircuit, initial_ket: np.ndarray, mapped_params, meas_specs,
                            cutoff_dim: int, measurement_patterns):
    """Processes fixed measurement patterns trajectories."""
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
    meas_cutoffs = [min(c, cutoff_dim) for m, c in meas_specs]
    perm = [0] + meas_modes

    if patterns_arr.shape[1] != circuit.steps:
        raise ValueError(
            f"Patterns (steps) mismatch: patterns have {patterns_arr.shape[1]} steps, circuit expects {circuit.steps}"
        )
    if patterns_arr.shape[2] != len(meas_modes):
        raise ValueError(
            f"Patterns (measured modes) mismatch: patterns have {patterns_arr.shape[2]} measured modes, circuit expects {len(meas_modes)}"
        )

    current_kets = np.tile(initial_ket, (n_sequences, 1))
    sequence_probs = np.ones(n_sequences)
    sequence_active = np.ones(n_sequences, dtype=bool)
    total_truncation_error = 0.0

    for step in range(circuit.steps):
        step_params = mapped_params[step]
        step_outcomes = patterns_arr[:, step, :]
        
        if not np.any(sequence_active):
            break
            
        active_indices = np.where(sequence_active)[0]
        n_active = len(active_indices)
        
        batch_kets = current_kets[active_indices]
        batch_outcomes = step_outcomes[active_indices]
        
        next_kets = np.zeros_like(batch_kets, dtype=np.complex128)
        step_probs = np.zeros(n_active)
        step_truncation_errors = np.zeros(n_active)
        
        full_kets = []
        ket_cache = {}
        for i in range(n_active):
            ket = batch_kets[i]
            key = np.round(ket, decimals=8).tobytes()

            if key in ket_cache:
                full_ket, trunc_err = ket_cache[key]
            else:
                eng = sf.Engine("fock", backend_options={"cutoff_dim": cutoff_dim})
                prog_prep = sf.Program(len(meas_modes) + 1)
                with prog_prep.context as q:
                    Ket(ket) | q[0]
                eng.run(prog_prep)

                result = circuit.run_step(None, step, step_params, eng)
                full_ket = result.state.ket()

                flat_ket = full_ket.flatten()
                norm_sq = np.real(np.vdot(flat_ket, flat_ket))
                trunc_err = np.abs(1.0 - norm_sq)

                ket_cache[key] = (full_ket, trunc_err)

            step_truncation_errors[i] = trunc_err
            full_kets.append(full_ket)

        full_kets_arr = np.stack(full_kets, axis=0)

        batch_perm = (0,) + tuple(p + 1 for p in perm)
        transposed = np.transpose(full_kets_arr, axes=batch_perm)

        indexer = (slice(None), slice(None)) + tuple(slice(0, c) for c in meas_cutoffs)
        sliced_batch = transposed[indexer]

        batch_idx = np.arange(n_active)
        adv_index = (batch_idx, ) + (slice(None),) + tuple(batch_outcomes[:, j] for j in range(batch_outcomes.shape[1]))
        projected_batch = sliced_batch[adv_index]

        prob_outcomes = np.sum(np.abs(projected_batch) ** 2, axis=1)
        nonzero_mask = prob_outcomes >= 1e-12

        if np.any(nonzero_mask):
            next_kets[nonzero_mask] = projected_batch[nonzero_mask] / np.sqrt(prob_outcomes[nonzero_mask])[:, None]
            step_probs[nonzero_mask] = prob_outcomes[nonzero_mask]
        
        for idx, active_idx in enumerate(active_indices):
            if step_probs[idx] < 1e-12:
                sequence_active[active_idx] = False
                sequence_probs[active_idx] = 0.0
            else:
                current_kets[active_idx] = next_kets[idx]
                sequence_probs[active_idx] *= step_probs[idx]
                total_truncation_error += step_truncation_errors[idx] * sequence_probs[active_idx]

    possible_mask = sequence_active & (sequence_probs > 1e-12)
    if not np.any(possible_mask):
        return None

    active_kets = current_kets[possible_mask]
    active_probs = sequence_probs[possible_mask]
    active_outcome_sums = np.sum(patterns_arr[possible_mask], axis=(1, 2))

    return active_kets, active_probs, active_outcome_sums, total_truncation_error, possible_mask, patterns_arr


def _process_beam_search(circuit: TimeMultiplexedCircuit, initial_ket: np.ndarray, mapped_params, meas_specs,
                         cutoff_dim: int, beam_width: int):
    """Processes trajectories using Beam Search."""
    active_kets = np.zeros((1, cutoff_dim), dtype=np.complex128)
    active_kets[0] = initial_ket 
    active_probs = np.array([1.0])
    active_outcome_sums = np.zeros(1, dtype=int)
    active_outcomes = np.zeros((1, 0), dtype=int)

    meas_modes = [m for m, c in meas_specs]
    meas_cutoffs = [min(c, cutoff_dim) for m, c in meas_specs]
    perm = [0] + meas_modes

    total_truncation_error = 0.0

    for step in range(circuit.steps):
        step_params = mapped_params[step]

        branch_raw_probs = []
        branch_tensors = []

        for idx in range(len(active_kets)):
            parent_ket = active_kets[idx]

            eng = sf.Engine("fock", backend_options={"cutoff_dim": cutoff_dim})

            prog_prep = sf.Program(len(meas_modes) + 1)
            with prog_prep.context as q:
                Ket(parent_ket) | q[0]
            eng.run(prog_prep)

            result = circuit.run_step(None, step, step_params, eng)
            full_ket = result.state.ket()

            flat_ket = full_ket.flatten()
            norm_sq = np.real(np.vdot(flat_ket, flat_ket))
            total_truncation_error += np.abs(1.0 - norm_sq) * active_probs[idx]

            transposed_ket = np.transpose(full_ket, axes=perm)

            slices = [slice(None)] + [slice(0, c) for c in meas_cutoffs]
            sliced_ket = transposed_ket[tuple(slices)]

            branch_tensors.append(sliced_ket)

            probs_tensor = np.sum(np.abs(sliced_ket)**2, axis=0)
            branch_raw_probs.append(probs_tensor.flatten())

        P_parent = active_probs
        P_raw = np.stack(branch_raw_probs) 
        P_total = P_parent[:, None] * P_raw

        flat_P = P_total.flatten()
        k = min(beam_width, flat_P.size)
        top_indices = np.argpartition(flat_P, -k)[-k:]

        parent_indices, outcome_indices_flat = np.unravel_index(top_indices, P_total.shape)
        outcomes_unraveled = np.unravel_index(outcome_indices_flat, tuple(meas_cutoffs))

        tensor_stack = np.stack(branch_tensors)

        indexer = (parent_indices, slice(None)) + outcomes_unraveled
        raw_new_kets = tensor_stack[indexer]

        norms = np.linalg.norm(raw_new_kets, axis=1)
        selected_probs = P_total[parent_indices, outcome_indices_flat]

        if len(outcomes_unraveled) > 0:
            step_outcome_sums = np.sum(np.stack(outcomes_unraveled), axis=0)
        else:
            step_outcome_sums = np.zeros(len(parent_indices), dtype=int)

        candidate_outcome_sums = active_outcome_sums[parent_indices] + step_outcome_sums

        mask = (norms > 1e-9) & (selected_probs > 1e-12)

        if not np.any(mask):
            return None

        active_kets = raw_new_kets[mask] / norms[mask][:, None]
        active_probs = selected_probs[mask]
        active_outcome_sums = candidate_outcome_sums[mask]

        current_step_outcomes = np.stack(outcomes_unraveled, axis=1)
        valid_parent_indices = parent_indices[mask]
        valid_step_outcomes = current_step_outcomes[mask]
        parent_history = active_outcomes[valid_parent_indices]
        active_outcomes = np.hstack([parent_history, valid_step_outcomes])

    return active_kets, active_probs, active_outcome_sums, total_truncation_error, active_outcomes


def _compute_fidelities_and_loss(active_kets, active_probs, active_outcome_sums, target_kets,
                                 total_truncation_error, penalty_strength):
    """Computes max phase-rotated fidelities and loss objective."""
    mask_nonzero = active_outcome_sums > 0

    expected_fidelity = 0.0
    fidelities = np.array([])
    best_target_indices = np.array([])

    if np.any(mask_nonzero):
        final_kets = active_kets[mask_nonzero]
        final_probs = active_probs[mask_nonzero]

        targets_arr = np.array(target_kets)
        prod = np.conj(final_kets[:, None, :]) * targets_arr[None, :, :]

        fft_vals = np.fft.fft(prod, n=256, axis=-1)

        pairwise_fidelities = np.max(np.abs(fft_vals)**2, axis=-1)

        fidelities = np.max(pairwise_fidelities, axis=1)
        best_target_indices = np.argmax(pairwise_fidelities, axis=1)

        min_infidel = 2e-2
        infidelities = np.maximum(1.0 - fidelities, min_infidel)
        capped_fidelities = np.minimum(fidelities, 1 - min_infidel)

        log_vals = np.log10(infidelities) / np.log10(min_infidel)
        objective2 = np.sum(final_probs * (capped_fidelities**2 * log_vals)**4)
        expected_fidelity = np.log(objective2 + 1e-72) + 1e4 * objective2

    loss = -1 * expected_fidelity + (penalty_strength * total_truncation_error)

    return loss, expected_fidelity, fidelities, best_target_indices, mask_nonzero


def _format_branch_details(mask_nonzero, active_probs, fidelities, best_target_indices,
                           measurement_patterns=None, patterns_arr=None, possible_mask=None,
                           active_outcomes=None):
    """Formats detailed branch metadata dictionary list."""
    branch_details = []

    if np.any(mask_nonzero):
        final_probs = active_probs[mask_nonzero]

        if measurement_patterns is not None:
            flat_patterns = patterns_arr.reshape(patterns_arr.shape[0], -1)
            surviving_patterns = flat_patterns[possible_mask]
            final_outcomes_arr = surviving_patterns[mask_nonzero]
        else:
            final_outcomes_arr = active_outcomes[mask_nonzero]

        for i in range(len(final_probs)):
            outcome = tuple(final_outcomes_arr[i].tolist())

            branch_details.append({
                "outcome": outcome,
                "prob": float(final_probs[i]),
                "fidelity": float(fidelities[i]),
                "target_idx": int(best_target_indices[i]),
            })

    return branch_details


def evaluate_time_domain_circuit(flat_params, circuit: TimeMultiplexedCircuit, target_kets, cutoff_dim, beam_width,
                                  penalty_strength, measurement_patterns=None, return_details: bool = False):
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
        return_details: If True, returns dictionary with detailed evaluation metrics.
    
    Returns:
        Loss value or dict (if return_details=True)
    """
    n_init = circuit.num_initial_parameters
    init_params = flat_params[:n_init]
    step_params = flat_params[n_init:]

    mapped_params = circuit.map_parameters(step_params)
    meas_specs = circuit.get_measurement_specs()
    initial_ket = circuit.get_initial_state_ket(init_params, cutoff_dim)

    use_fixed_patterns = (measurement_patterns is not None)

    if use_fixed_patterns:
        res = _process_fixed_patterns(
            circuit, initial_ket, mapped_params, meas_specs, cutoff_dim, measurement_patterns
        )
        if res is None:
            return 100.0
        active_kets, active_probs, active_outcome_sums, total_truncation_error, possible_mask, patterns_arr = res
        active_outcomes = None
    else:
        res = _process_beam_search(
            circuit, initial_ket, mapped_params, meas_specs, cutoff_dim, beam_width
        )
        if res is None:
            return 100.0
        active_kets, active_probs, active_outcome_sums, total_truncation_error, active_outcomes = res
        possible_mask = None
        patterns_arr = None

    loss, expected_fidelity, fidelities, best_target_indices, mask_nonzero = _compute_fidelities_and_loss(
        active_kets, active_probs, active_outcome_sums, target_kets, total_truncation_error, penalty_strength
    )

    if not return_details:
        return loss

    branch_details = _format_branch_details(
        mask_nonzero, active_probs, fidelities, best_target_indices,
        measurement_patterns=measurement_patterns,
        patterns_arr=patterns_arr,
        possible_mask=possible_mask,
        active_outcomes=active_outcomes
    )

    return {
        "loss": loss,
        "expected_fidelity": float(expected_fidelity),
        "branches": branch_details,
        "total_probability": float(np.sum(active_probs[mask_nonzero])) if np.any(mask_nonzero) else 0.0
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
                 measurement_patterns = None,
                 num_parallel_runs: int = 4,
                 num_processes: int = 4,
                 **kwargs):
        
        self.circuit = circuit

        if not isinstance(target_gens, list):
            target_gens = [target_gens]
        
        self.target_kets = [gen.get_target_ket(cutoff_dim) for gen in target_gens]
        self.cutoff_dim = cutoff_dim
        self.beam_width = beam_width
        self.penalty_strength = penalty_strength
        self.measurement_patterns = measurement_patterns
        self.num_parallel_runs = num_parallel_runs
        self.num_processes = num_processes
        self.eval_count = 0
        self.iteration_count = 0
        
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
        )

    def callback(self, x, f, accept):
        self.iteration_count += 1
        status = "Accept" if accept else "Reject"
        print(f"  [Iteration {self.iteration_count}] [{status}] (Evals: {self.eval_count}) Loss: {f} ")
        self.eval_count = 0

    def _execute_single_run(self, seed, run_idx, total_runs, n_iter, method, full_bounds, **kwargs):
        """Helper to execute a single basin hopping run (for parallelization)."""
        if seed is not None:
            np.random.seed(seed)
            
        self.eval_count = 0
        self.iteration_count = 0
        
        # Initial guess
        x0 = np.array([np.random.uniform(l, h) for l, h in full_bounds])
        
        # Nelder-Mead requires a penalty wrapper for bounds as it doesn't natively support them
        if method == 'Nelder-Mead':
            def bounded_loss(x):
                for val, (low, high) in zip(x, full_bounds):
                    if val < low or val > high:
                        return 1e10
                return self._loss_function(x)
            objective = bounded_loss
            minimizer_kwargs = {"method": method}
        else:
            objective = self._loss_function
            minimizer_kwargs = {
                "method": method,
                "bounds": full_bounds,
            }
        
        def local_callback(x, f, accept):
            self.iteration_count += 1
            status = "Accept" if accept else "Reject"
            prefix = f"[Run {run_idx+1}/{total_runs}] " if total_runs > 1 else ""
            print(f"  {prefix}[Iteration {self.iteration_count}] [{status}] (Evals: {self.eval_count}) Loss: {f} ")
            self.eval_count = 0

        try:
            result = basinhopping(
                objective,
                x0,
                niter=n_iter,
                minimizer_kwargs=minimizer_kwargs,
                callback=local_callback,
                stepsize=0.5
            )
            
            # Final Evaluation
            final_eval = evaluate_time_domain_circuit(
                result.x,
                circuit=self.circuit,
                target_kets=self.target_kets,
                cutoff_dim=self.cutoff_dim,
                beam_width=self.beam_width,
                penalty_strength=self.penalty_strength,
                measurement_patterns=self.measurement_patterns,
                return_details=True
            )
            
            return {
                "x": result.x,
                "loss": final_eval["loss"],
                "expected_fidelity": final_eval.get("expected_fidelity", 0.0),
                "branches": final_eval.get("branches", []),
                "total_probability": final_eval.get("total_probability", 0.0),
                "duration": 0.0, # Calculated in parent
                "message": result.message,
                "seed": seed,
                "success": True
            }
        except Exception as e:
            # Catch exceptions to prevent crashing all runs
            return {"success": False, "error": str(e), "seed": seed}

    def run(self, n_iter=20, method="L-BFGS-B", n_generations=None, 
            num_parallel_runs=None, base_seed=None, **kwargs):
        """
        Runs the global optimization.
        """
        if n_generations is not None:
            n_iter = n_generations
            
        n_parallel = num_parallel_runs if num_parallel_runs is not None else self.num_parallel_runs

        # Construct full parameter bounds
        per_step_bounds = self.circuit.per_step_parameter_bounds
        if self.circuit.time_invariant:
            step_bounds = per_step_bounds
        else:
            step_bounds = per_step_bounds * self.circuit.steps
            
        # Init Bounds
        init_bounds = self.circuit.initial_parameter_bounds
        
        # Concatenate
        full_bounds = init_bounds + step_bounds
        
        start_time = time.time()
        
        if n_parallel <= 1:
            print(f"Starting Basin-Hopping Beam Search (Width={self.beam_width}, Steps={self.circuit.steps})...")
            
            current_seed = base_seed if base_seed is not None else np.random.randint(0, 2**32 - 1)
            
            # Use local execution flow
            res = self._execute_single_run(current_seed, 0, 1, n_iter, method, full_bounds)
            
            if not res.get("success", False):
                raise RuntimeError(f"Basin-Hopping run failed: {res.get('error')}")

            res["duration"] = time.time() - start_time
            return res
            
        else:
            # Parallel Execution
            print(f"Starting Parallel Basin-Hopping ({n_parallel} runs, Width={self.beam_width}, Steps={self.circuit.steps})...")
            
            if base_seed is None:
                base_seed = np.random.randint(0, 100000)
            seeds = [base_seed + i for i in range(n_parallel)]

            n_iter = n_iter // n_parallel
            
            args_list = [(seeds[i], i, n_parallel, n_iter, method, full_bounds) for i in range(n_parallel)]
            
            workers = min(n_parallel, self.num_processes if self.num_processes > 1 else multiprocessing.cpu_count())
            
            with multiprocessing.Pool(processes=workers) as pool:
                results = pool.starmap(self._execute_single_run, args_list)
            
            valid_results = [r for r in results if r.get("success", False)]
            if not valid_results:
                raise RuntimeError("All parallel Basin-Hopping runs failed.")
                
            best_res = min(valid_results, key=lambda x: x["loss"])
            total_duration = time.time() - start_time
            
            final_output = {
                "x": best_res["x"],
                "loss": best_res["loss"],
                "expected_fidelity": best_res.get("expected_fidelity", 0.0),
                "branches": best_res.get("branches", []),
                "total_probability": best_res.get("total_probability", 0.0),
                "duration": total_duration,
                "message": f"Best of {n_parallel} parallel runs",
                "run_results": results,
                "best_run_idx": seeds.index(best_res["seed"])
            }
            return final_output
