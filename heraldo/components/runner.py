"""Evaluation runner for static spatial photonic circuits used in static setup evaluations."""

import multiprocessing
import time
from scipy.optimize import basinhopping

import numpy as np
import strawberryfields as sf

from heraldo._internal import _normalize_outcomes
from heraldo.components.interfaces import ObjectiveFunction, StaticCircuit
from heraldo.components.objectives import (
    BeamSearchLoss,
    FixedPatternCappedLoss,
    beam_search_loss_fn,
    fixed_pattern_capped_loss_fn,
)
from heraldo.components.targets import TargetGenerator
from heraldo.serialization import to_config


def _process_fixed_patterns(circuit: StaticCircuit, full_ket: np.ndarray,
                           cutoff_dim: int, measurement_patterns):
    """Simulates static spatial circuit outcomes under pre-specified measurement patterns.

    Args:
        circuit (StaticCircuit): Static spatial circuit instance.
        full_ket (np.ndarray): Multi-mode joint state vector in Fock basis.
        cutoff_dim (int): Fock space truncation cutoff dimension.
        measurement_patterns: Measurement outcome pattern(s) specified as int, tuple, list, or array.

    Returns:
        tuple or None: Tuple containing ``(kets, probs, outcome_sums, outcomes)``
        if at least one pattern has non-zero probability, or ``None`` if all patterns are impossible.
    """
    # 1. Fetch ancillary measurement specifications and cap detector cutoffs at simulation cutoff_dim
    meas_specs = circuit.get_measurement_specs()
    meas_modes = [m for m, c in meas_specs]
    meas_cutoffs = [min(c, cutoff_dim) for m, c in meas_specs]

    # 2. Normalize input measurement patterns into a standard list of integer tuples
    patterns_list = _normalize_outcomes(measurement_patterns, len(meas_modes))
    if not patterns_list:
        return None

    patterns_arr = np.array(patterns_list, dtype=int)

    # 3. Transpose joint state tensor so Mode 0 (output) is axis 0, followed by measured ancillary modes
    perm = [0] + meas_modes
    transposed = np.transpose(full_ket, axes=perm)

    # 4. Slice each ancilla mode's Fock dimension down to its configured detector cutoff
    indexer = (slice(None),) + tuple(slice(0, c) for c in meas_cutoffs)
    sliced_ket = transposed[indexer]

    # 5. Extract unnormalized Mode 0 state vectors via fancy indexing over requested ancilla detection events
    adv_index = (slice(None),) + tuple(patterns_arr[:, j] for j in range(patterns_arr.shape[1]))
    projected = sliced_ket[adv_index]
    raw_kets = projected.T

    # 6. Calculate probability for each pattern and discard non-viable branches (P < 1e-12)
    probs = np.sum(np.abs(raw_kets)**2, axis=1)
    nonzero_mask = probs >= 1e-12

    if not np.any(nonzero_mask):
        return None

    # 7. Normalize surviving heralded state vectors to unit norm and sum total detected photons per pattern
    kets = raw_kets[nonzero_mask] / np.sqrt(probs[nonzero_mask])[:, None]
    probs = probs[nonzero_mask]
    outcomes = patterns_arr[nonzero_mask]
    outcome_sums = np.sum(outcomes, axis=1)

    return kets, probs, outcome_sums, outcomes


def _process_beam_search(circuit: StaticCircuit, full_ket: np.ndarray,
                         cutoff_dim: int, beam_width: int):
    """Selects top-K outcome trajectories for static spatial circuit mode 0 output using Beam Search.

    Args:
        circuit (StaticCircuit): Static spatial circuit instance.
        full_ket (np.ndarray): Multi-mode joint state vector in Fock basis.
        cutoff_dim (int): Fock space truncation cutoff dimension.
        beam_width (int): Maximum number of top measurement branches retained.

    Returns:
        tuple or None: Tuple containing ``(kets, probs, outcome_sums, outcomes)``
        if non-zero probability outcomes survive, or ``None`` otherwise.
    """
    # 1. Fetch ancillary measurement specifications and cap detector cutoffs at simulation cutoff_dim
    meas_specs = circuit.get_measurement_specs()
    meas_modes = [m for m, c in meas_specs]
    meas_cutoffs = [min(c, cutoff_dim) for m, c in meas_specs]

    # 2. Transpose joint state tensor so Mode 0 (output) is axis 0, followed by measured ancillary modes
    perm = [0] + meas_modes
    transposed = np.transpose(full_ket, axes=perm)

    # 3. Slice each ancilla mode's Fock dimension down to its configured detector cutoff
    indexer = (slice(None),) + tuple(slice(0, c) for c in meas_cutoffs)
    sliced_ket = transposed[indexer]

    # 4. Marginalize over Mode 0 to compute the full multi-dimensional ancilla outcome probability distribution P(n1, n2, ...)
    probs_tensor = np.sum(np.abs(sliced_ket)**2, axis=0)

    # 5. Find indices of top-K probability outcomes in O(N) time using argpartition and sort them descending
    flat_P = probs_tensor.flatten()
    k = min(beam_width, flat_P.size)
    top_indices = np.argpartition(flat_P, -k)[-k:]
    top_indices = top_indices[np.argsort(-flat_P[top_indices])]

    # 6. Convert flat top-K indices back into multi-dimensional mode photon-number index tuples and extract probabilities
    outcomes_unraveled = np.unravel_index(top_indices, probs_tensor.shape)
    selected_probs = probs_tensor[outcomes_unraveled]

    # 7. Extract unnormalized Mode 0 state vectors via fancy indexing using the discovered top-K outcome tuples
    adv_index = (slice(None),) + outcomes_unraveled
    projected = sliced_ket[adv_index]
    raw_kets = projected.T

    # 8. Compute ket norms directly and filter out non-viable branches (near-zero norm or negligible probability < 1e-12)
    norms = np.linalg.norm(raw_kets, axis=1)
    mask = (norms > 1e-9) & (selected_probs > 1e-12)

    if not np.any(mask):
        return None

    # 9. Normalize surviving state vectors to unit norm, stack outcome tuples, and sum detected photons per pattern
    kets = raw_kets[mask] / norms[mask][:, None]
    probs = selected_probs[mask]
    outcomes = np.stack(outcomes_unraveled, axis=1)[mask]
    outcome_sums = np.sum(outcomes, axis=1)

    return kets, probs, outcome_sums, outcomes


def _compute_fidelities_and_loss(kets: np.ndarray, probs: np.ndarray,
                                 outcome_sums: np.ndarray, target_kets: list[np.ndarray],
                                 truncation_error: float, penalty_strength: float,
                                 loss_fn=None, phase_lock: bool = False):
    """Computes target state fidelities and loss metric for projected output states.

    Args:
        kets (np.ndarray): 2D array of state kets for active trajectories.
        probs (np.ndarray): 1D array of trajectory probabilities.
        outcome_sums (np.ndarray): 1D array of total detected photon count per trajectory.
        target_kets (list[np.ndarray]): List of candidate target state vectors in Fock space.
        truncation_error (float): Fock state truncation error.
        penalty_strength (float): Penalty coefficient for truncation errors.
        loss_fn (callable, optional): Objective loss evaluation function.
        phase_lock (bool, optional): If True, enforces a common phase-space rotation across all accepted measurement branches. Defaults to False.

    Returns:
        tuple: A tuple containing ``(loss, objective_score, fidelities, best_target_indices, mask_nonzero)``.
    """
    # 1. Loss function resolution: Default to BeamSearchLoss instance or instantiate class if provided
    if loss_fn is None:
        loss_fn = BeamSearchLoss()
    elif isinstance(loss_fn, type) and issubclass(loss_fn, ObjectiveFunction):
        loss_fn = loss_fn()

    # 2. Vacuum filtering: Filter out vacuum/zero-photon click branches (outcome_sums == 0)
    mask_nonzero = outcome_sums > 0

    objective_score = 0.0
    fidelities = np.array([])
    best_target_indices = np.array([])

    if np.any(mask_nonzero):
        # Extract surviving kets and probabilities for non-vacuum detection branches
        final_kets = kets[mask_nonzero]
        final_probs = probs[mask_nonzero]

        # 3. Batched rotation-invariant fidelity calculation via 256-point FFT:
        # Multiply conj(branch_ket) * target_ket element-wise, broadcasted to shape (num_branches, num_targets, cutoff_dim)
        targets_arr = np.array(target_kets)
        prod = np.conj(final_kets[:, None, :]) * targets_arr[None, :, :]

        # Perform batched FFT along the Fock axis to compute phase-space overlaps across 256 discretized phase angles
        fft_vals = np.fft.fft(prod, n=256, axis=-1)
        all_fidelities = np.abs(fft_vals)**2  # shape: (num_branches, num_targets, 256)

        # 4. Evaluation mode selection: Shared global rotation (phase_lock=True) vs independent per-branch rotation (phase_lock=False)
        if phase_lock:
            # Mode A: Shared global phase-space rotation across all branches
            # Find maximum fidelity over candidate target states for each branch and phase angle -> shape: (num_branches, 256)
            fidelities_all_k = np.max(all_fidelities, axis=1)
            # Identify best-matching target state index for each branch and phase angle -> shape: (num_branches, 256)
            best_targets_all_k = np.argmax(all_fidelities, axis=1)
            # Evaluate objective loss score across branches for every candidate phase angle -> shape: (256,)
            scores = loss_fn(final_probs, fidelities_all_k)
            # Select the global phase angle index that maximizes the total score
            best_k = np.argmax(scores)
            objective_score = float(scores[best_k])
            # Slice per-branch fidelities and best target indices at the optimal locked global phase angle
            fidelities = fidelities_all_k[:, best_k]
            best_target_indices = best_targets_all_k[:, best_k]
        else:
            # Mode B: Per-branch independent phase-space rotation
            # Maximize fidelity over discretized phase angles for each (branch, target) pair -> shape: (num_branches, num_targets)
            pairwise_fidelities = np.max(all_fidelities, axis=-1)
            # Maximize fidelity over candidate target states per branch -> shape: (num_branches,)
            fidelities = np.max(pairwise_fidelities, axis=1)
            # Identify the best target index per branch -> shape: (num_branches,)
            best_target_indices = np.argmax(pairwise_fidelities, axis=1)
            # Compute overall objective score using branch probabilities and maximum per-branch fidelities
            objective_score = loss_fn(final_probs, fidelities)

    # 5. Final loss computation: Negate objective score (since basinhopping minimizes) and add truncation penalty
    loss = -1 * objective_score + (penalty_strength * truncation_error)

    return loss, objective_score, fidelities, best_target_indices, mask_nonzero


def _format_branch_details(mask_nonzero: np.ndarray, probs: np.ndarray,
                           fidelities: np.ndarray, best_target_indices: np.ndarray,
                           outcomes: np.ndarray) -> list[dict]:
    """Formats trajectory and outcome metadata dictionaries.

    Args:
        mask_nonzero (np.ndarray): Boolean mask indicating branches with non-zero photon counts.
        probs (np.ndarray): Array of probabilities for active branches.
        fidelities (np.ndarray): Array of state fidelities for active branches.
        best_target_indices (np.ndarray): Array of best-matching target indices per branch.
        outcomes (np.ndarray): Array of photon detection outcome tuples for measured modes.

    Returns:
        list[dict]: List of metadata dictionaries.
    """
    branch_details = []

    if np.any(mask_nonzero):
        final_probs = probs[mask_nonzero]
        final_outcomes = outcomes[mask_nonzero]

        for i in range(len(final_probs)):
            outcome = tuple(final_outcomes[i].tolist())

            branch_details.append({
                "outcome": outcome,
                "prob": float(final_probs[i]),
                "fidelity": float(fidelities[i]),
                "target_idx": int(best_target_indices[i]),
            })

    return branch_details


def evaluate_circuit(params: np.ndarray,
                     circuit: StaticCircuit,
                     target_kets: list[np.ndarray],
                     cutoff_dim: int,
                     beam_width: int = 20,
                     penalty_strength: float = 0.1,
                     measurement_patterns=None,
                     return_details: bool = False,
                     loss_fn=None,
                     phase_lock: bool = False):
    """Evaluates a static spatial photonic circuit against target state generators.

    Executes a single static circuit run and evaluates Mode 0 output state under
    Beam Search outcome selection or fixed measurement outcome filtering.

    Args:
        params (np.ndarray): 1D array of static circuit parameters.
        circuit (StaticCircuit): Static spatial circuit model to evaluate.
        target_kets (list[np.ndarray]): Target state vectors in Fock basis.
        cutoff_dim (int): Fock space truncation cutoff dimension.
        beam_width (int, optional): Maximum number of outcome patterns retained. Defaults to 20.
        penalty_strength (float, optional): Penalty coefficient applied to Fock truncation errors. Defaults to 0.1.
        measurement_patterns (optional): Pre-specified fixed measurement patterns. Defaults to None.
        return_details (bool, optional): If True, returns evaluation metrics and trajectory metadata. Defaults to False.
        loss_fn (callable, optional): Custom objective loss evaluation function.
        phase_lock (bool, optional): If True, enforces a common phase-space rotation across all branches. Defaults to False.

    Returns:
        float or dict: Objective loss float if `return_details` is False, or dictionary of evaluation results.
    """
    # 1. Simulate the circuit: Run Strawberry Fields Fock engine to obtain the full joint state ket tensor.
    engine = sf.Engine("fock", backend_options={"cutoff_dim": cutoff_dim})
    result = circuit.run_circuit(params, engine)
    full_ket = result.state.ket()

    # 2. Compute truncation error: Measure probability mass leakage beyond the Fock cutoff dimension.
    flat_ket = full_ket.flatten()
    norm_sq = np.real(np.vdot(flat_ket, flat_ket))
    truncation_error = np.abs(1.0 - norm_sq)

    # 3. Delegate state projection: Extract heralded states using fixed patterns or dynamic beam search.
    use_fixed_patterns = (measurement_patterns is not None)

    if use_fixed_patterns:
        res = _process_fixed_patterns(circuit, full_ket, cutoff_dim, measurement_patterns)
    else:
        res = _process_beam_search(circuit, full_ket, cutoff_dim, beam_width)

    # Return fallback loss if no viable measurement outcomes survive projection
    if res is None:
        return 100.0

    kets, probs, outcome_sums, outcomes = res

    # 4. Resolve and apply loss function: Select default loss model and evaluate fidelities and penalties.
    if loss_fn is None:
        loss_fn = FixedPatternCappedLoss() if use_fixed_patterns else BeamSearchLoss()
    elif isinstance(loss_fn, type) and issubclass(loss_fn, ObjectiveFunction):
        loss_fn = loss_fn()

    loss, objective_score, fidelities, best_target_indices, mask_nonzero = _compute_fidelities_and_loss(
        kets, probs, outcome_sums, target_kets, truncation_error, penalty_strength, loss_fn=loss_fn, phase_lock=phase_lock
    )

    # 5. Return result: Fast path returns scalar loss for optimizer; detailed path formats branch metadata dict.
    if not return_details:
        return loss

    branch_details = _format_branch_details(
        mask_nonzero, probs, fidelities, best_target_indices, outcomes
    )

    return {
        "loss": loss,
        "objective_score": float(objective_score),
        "branches": branch_details,
        "total_probability": float(np.sum(probs[mask_nonzero])) if np.any(mask_nonzero) else 0.0
    }


def _single_basinhopping_run(seed: int | None, circuit: StaticCircuit,
                             target_kets: list[np.ndarray], cutoff_dim: int,
                             beam_width: int, penalty_strength: float,
                             measurement_patterns, loss_fn, n_iter: int,
                             method: str, bounds: list[tuple[float, float]],
                             callback=None, run_idx: int = 0, total_runs: int = 1,
                             phase_lock: bool = False) -> dict:
    if seed is not None:
        np.random.seed(seed)

    eval_count = [0]
    iteration_count = [0]

    x0 = np.array([np.random.uniform(low, high) for low, high in bounds])

    def loss_func(x):
        eval_count[0] += 1
        return evaluate_circuit(
            x,
            circuit=circuit,
            target_kets=target_kets,
            cutoff_dim=cutoff_dim,
            beam_width=beam_width,
            penalty_strength=penalty_strength,
            measurement_patterns=measurement_patterns,
            loss_fn=loss_fn,
            phase_lock=phase_lock,
        )

    if method == 'Nelder-Mead':
        def bounded_loss(x):
            for val, (low, high) in zip(x, bounds):
                if val < low or val > high:
                    return 1e10
            return loss_func(x)
        objective = bounded_loss
        minimizer_kwargs = {"method": method}
    else:
        objective = loss_func
        minimizer_kwargs = {"method": method, "bounds": bounds}

    def local_callback(x, f, accept):
        iteration_count[0] += 1
        status = "Accept" if accept else "Reject"
        prefix = f"[Run {run_idx+1}/{total_runs}] " if total_runs > 1 else ""
        print(f"  {prefix}[Iteration {iteration_count[0]}] [{status}] (Evals: {eval_count[0]}) Loss: {f} ")
        eval_count[0] = 0
        if callback is not None:
            callback(x, f, accept)

    try:
        result = basinhopping(
            objective,
            x0,
            niter=n_iter,
            minimizer_kwargs=minimizer_kwargs,
            callback=local_callback,
            stepsize=0.5,
        )

        final_eval = evaluate_circuit(
            result.x,
            circuit=circuit,
            target_kets=target_kets,
            cutoff_dim=cutoff_dim,
            beam_width=beam_width,
            penalty_strength=penalty_strength,
            measurement_patterns=measurement_patterns,
            return_details=True,
            loss_fn=loss_fn,
            phase_lock=phase_lock,
        )

        return {
            "x": result.x,
            "loss": final_eval["loss"],
            "objective_score": final_eval.get("objective_score", 0.0),
            "branches": final_eval.get("branches", []),
            "total_probability": final_eval.get("total_probability", 0.0),
            "message": result.message,
            "seed": seed,
            "success": True,
        }
    except Exception as e:
        return {"success": False, "error": str(e), "seed": seed, "loss": float("inf")}


class BasinHoppingRunner:
    """Optimizes static spatial optical circuits using Basin-Hopping with Beam Search or fixed patterns.

    Args:
        circuit (StaticCircuit): Static spatial circuit model to optimize.
        target_gens (TargetGenerator or list[TargetGenerator]): Target quantum state generators.
        cutoff_dim (int): Fock space cutoff dimension for state vector truncation.
        beam_width (int, optional): Maximum trajectories retained per evaluation. Defaults to 20.
        penalty_strength (float, optional): Multiplier for truncation error penalty. Defaults to 0.1.
        measurement_patterns (optional): Fixed measurement sequences. Defaults to None.
        num_parallel_runs (int, optional): Number of parallel Basin-Hopping optimization runs. Defaults to 4.
        num_processes (int, optional): Number of worker processes for parallel execution. Defaults to 4.
        method (str, optional): Local minimizer algorithm. Defaults to "L-BFGS-B".
        base_seed (int, optional): Base random seed for reproducible runs. Defaults to None.
        loss_fn (callable, optional): Custom objective loss evaluation function.
        callback (callable, optional): Custom callback function `callback(x, f, accept)` invoked at each iteration.
        phase_lock (bool, optional): If True, enforces a common phase-space rotation across all branches. Defaults to False.
    """

    def __init__(self,
                 circuit: StaticCircuit,
                 target_gens: list[TargetGenerator] | TargetGenerator,
                 cutoff_dim: int,
                 beam_width: int = 20,
                 penalty_strength: float = 0.1,
                 measurement_patterns=None,
                 num_parallel_runs: int = 4,
                 num_processes: int = 4,
                 method: str = "L-BFGS-B",
                 base_seed: int | None = None,
                 loss_fn=None,
                 callback=None,
                 phase_lock: bool = False):
        self.circuit = circuit
        if not isinstance(target_gens, list):
            target_gens = [target_gens]
        self.target_gens = target_gens
        self.target_kets = [gen.get_target_ket(cutoff_dim) for gen in target_gens]
        self.cutoff_dim = cutoff_dim
        self.beam_width = beam_width
        self.penalty_strength = penalty_strength
        self.measurement_patterns = measurement_patterns
        self.num_parallel_runs = num_parallel_runs
        self.num_processes = num_processes
        self.method = method
        self.base_seed = base_seed
        self.loss_fn = loss_fn
        self.callback = callback
        self.phase_lock = phase_lock

    def run(self, n_iter: int = 20) -> dict:
        """Executes global static circuit optimization using Basin-Hopping.

        Args:
            n_iter (int, optional): Number of Basin-Hopping iterations per run. Defaults to 20.

        Returns:
            dict: Optimization results containing best parameter vector, loss, fidelities, and branches.
        """
        cb = self.callback
        n_parallel = self.num_parallel_runs
        method = self.method
        bounds = self.circuit.parameter_bounds

        start_time = time.time()

        base_seed = self.base_seed
        if base_seed is None:
            base_seed = np.random.randint(0, 2**31 - 1)
        seeds = [base_seed + i for i in range(n_parallel)]

        meas_patterns_serializable = (
            self.measurement_patterns.tolist()
            if isinstance(self.measurement_patterns, np.ndarray)
            else self.measurement_patterns
        )

        if self.loss_fn is None:
            effective_loss_fn = FixedPatternCappedLoss() if self.measurement_patterns is not None else BeamSearchLoss()
        elif isinstance(self.loss_fn, type) and issubclass(self.loss_fn, ObjectiveFunction):
            effective_loss_fn = self.loss_fn()
        else:
            effective_loss_fn = self.loss_fn

        circuit_cfg = to_config(self.circuit)
        target_cfgs = to_config(self.target_gens)
        loss_cfg = to_config(effective_loss_fn)

        runner_cfg = {
            "cutoff_dim": self.cutoff_dim,
            "beam_width": self.beam_width,
            "penalty_strength": self.penalty_strength,
            "measurement_patterns": meas_patterns_serializable,
            "num_parallel_runs": n_parallel,
            "num_processes": self.num_processes,
            "n_iter": n_iter,
            "method": method,
            "base_seed": base_seed,
            "phase_lock": self.phase_lock,
        }

        if n_parallel <= 1:
            print(f"Starting Basin-Hopping Beam Search (Width={self.beam_width})...")
            res = _single_basinhopping_run(
                seeds[0], self.circuit, self.target_kets, self.cutoff_dim,
                self.beam_width, self.penalty_strength, self.measurement_patterns,
                self.loss_fn, n_iter, method, bounds,
                callback=cb, run_idx=0, total_runs=1,
                phase_lock=self.phase_lock
            )
            if not res.get("success", False):
                raise RuntimeError(f"Basin-Hopping run failed: {res.get('error')}")
            res["duration"] = time.time() - start_time
            res["circuit"] = self.circuit
            res["targets"] = self.target_gens
            res["loss_fn"] = effective_loss_fn
            res["circuit_config"] = circuit_cfg
            res["target_configs"] = target_cfgs
            res["loss_config"] = loss_cfg
            res["runner_config"] = runner_cfg
            return res

        workers = min(n_parallel, self.num_processes if self.num_processes > 1 else multiprocessing.cpu_count())
        print(f"Starting Parallel Basin-Hopping ({n_parallel} runs, Width={self.beam_width})...")
        args_list = [
            (seeds[i], self.circuit, self.target_kets, self.cutoff_dim,
             self.beam_width, self.penalty_strength, self.measurement_patterns,
             self.loss_fn, n_iter, method, bounds,
             cb, i, n_parallel, self.phase_lock)
            for i in range(n_parallel)
        ]

        with multiprocessing.Pool(processes=workers) as pool:
            results = pool.starmap(_single_basinhopping_run, args_list)

        valid_results = [r for r in results if r.get("success", False)]
        if not valid_results:
            raise RuntimeError("All parallel Basin-Hopping runs failed.")

        best_res = min(valid_results, key=lambda x: x["loss"])
        total_duration = time.time() - start_time

        return {
            "x": best_res["x"],
            "loss": best_res["loss"],
            "objective_score": best_res.get("objective_score", 0.0),
            "branches": best_res.get("branches", []),
            "total_probability": best_res.get("total_probability", 0.0),
            "duration": total_duration,
            "message": f"Best of {n_parallel} parallel runs",
            "run_results": results,
            "best_run_idx": seeds.index(best_res["seed"]),
            "circuit": self.circuit,
            "targets": self.target_gens,
            "loss_fn": effective_loss_fn,
            "circuit_config": circuit_cfg,
            "target_configs": target_cfgs,
            "loss_config": loss_cfg,
            "runner_config": runner_cfg,
        }
