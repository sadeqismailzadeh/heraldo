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
from heraldo.factory import to_config


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
    meas_specs = circuit.get_measurement_specs()
    meas_modes = [m for m, c in meas_specs]
    meas_cutoffs = [min(c, cutoff_dim) for m, c in meas_specs]

    patterns_list = _normalize_outcomes(measurement_patterns, len(meas_modes))
    if not patterns_list:
        return None

    patterns_arr = np.array(patterns_list, dtype=int)

    perm = [0] + meas_modes
    transposed = np.transpose(full_ket, axes=perm)

    indexer = (slice(None),) + tuple(slice(0, c) for c in meas_cutoffs)
    sliced_ket = transposed[indexer]

    adv_index = (slice(None),) + tuple(patterns_arr[:, j] for j in range(patterns_arr.shape[1]))
    projected = sliced_ket[adv_index]
    raw_kets = projected.T

    probs = np.sum(np.abs(raw_kets)**2, axis=1)
    nonzero_mask = probs >= 1e-12

    if not np.any(nonzero_mask):
        return None

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
    meas_specs = circuit.get_measurement_specs()
    meas_modes = [m for m, c in meas_specs]
    meas_cutoffs = [min(c, cutoff_dim) for m, c in meas_specs]

    perm = [0] + meas_modes
    transposed = np.transpose(full_ket, axes=perm)

    indexer = (slice(None),) + tuple(slice(0, c) for c in meas_cutoffs)
    sliced_ket = transposed[indexer]

    probs_tensor = np.sum(np.abs(sliced_ket)**2, axis=0)

    flat_P = probs_tensor.flatten()
    k = min(beam_width, flat_P.size)
    top_indices = np.argpartition(flat_P, -k)[-k:]
    top_indices = top_indices[np.argsort(-flat_P[top_indices])]

    outcomes_unraveled = np.unravel_index(top_indices, probs_tensor.shape)
    selected_probs = probs_tensor[outcomes_unraveled]

    adv_index = (slice(None),) + outcomes_unraveled
    projected = sliced_ket[adv_index]
    raw_kets = projected.T

    norms = np.linalg.norm(raw_kets, axis=1)
    mask = (norms > 1e-9) & (selected_probs > 1e-12)

    if not np.any(mask):
        return None

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
        tuple: A tuple containing ``(loss, expected_fidelity, fidelities, best_target_indices, mask_nonzero)``.
    """
    if loss_fn is None:
        loss_fn = BeamSearchLoss()
    elif isinstance(loss_fn, type) and issubclass(loss_fn, ObjectiveFunction):
        loss_fn = loss_fn()

    mask_nonzero = outcome_sums > 0

    expected_fidelity = 0.0
    fidelities = np.array([])
    best_target_indices = np.array([])

    if np.any(mask_nonzero):
        final_kets = kets[mask_nonzero]
        final_probs = probs[mask_nonzero]

        targets_arr = np.array(target_kets)
        prod = np.conj(final_kets[:, None, :]) * targets_arr[None, :, :]

        fft_vals = np.fft.fft(prod, n=256, axis=-1)
        all_fidelities = np.abs(fft_vals)**2

        if phase_lock:
            fidelities_all_k = np.max(all_fidelities, axis=1)
            best_targets_all_k = np.argmax(all_fidelities, axis=1)
            scores = loss_fn(final_probs, fidelities_all_k)
            best_k = np.argmax(scores)
            expected_fidelity = float(scores[best_k])
            fidelities = fidelities_all_k[:, best_k]
            best_target_indices = best_targets_all_k[:, best_k]
        else:
            pairwise_fidelities = np.max(all_fidelities, axis=-1)
            fidelities = np.max(pairwise_fidelities, axis=1)
            best_target_indices = np.argmax(pairwise_fidelities, axis=1)
            expected_fidelity = loss_fn(final_probs, fidelities)

    loss = -1 * expected_fidelity + (penalty_strength * truncation_error)

    return loss, expected_fidelity, fidelities, best_target_indices, mask_nonzero


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
                     penalty_strength: float = 0.001,
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
        beam_width (int, optional): Maximum number of outcome patterns retained. Defaults to 5.
        penalty_strength (float, optional): Penalty coefficient applied to Fock truncation errors. Defaults to 10.0.
        measurement_patterns (optional): Pre-specified fixed measurement patterns. Defaults to None.
        return_details (bool, optional): If True, returns evaluation metrics and trajectory metadata. Defaults to False.
        loss_fn (callable, optional): Custom objective loss evaluation function.

    Returns:
        float or dict: Objective loss float if `return_details` is False, or dictionary of evaluation results.
    """
    engine = sf.Engine("fock", backend_options={"cutoff_dim": cutoff_dim})
    result = circuit.run_circuit(params, engine)
    full_ket = result.state.ket()

    flat_ket = full_ket.flatten()
    norm_sq = np.real(np.vdot(flat_ket, flat_ket))
    truncation_error = np.abs(1.0 - norm_sq)

    use_fixed_patterns = (measurement_patterns is not None)

    if use_fixed_patterns:
        res = _process_fixed_patterns(circuit, full_ket, cutoff_dim, measurement_patterns)
    else:
        res = _process_beam_search(circuit, full_ket, cutoff_dim, beam_width)

    if res is None:
        return 100.0

    kets, probs, outcome_sums, outcomes = res

    if loss_fn is None:
        loss_fn = FixedPatternCappedLoss() if use_fixed_patterns else BeamSearchLoss()
    elif isinstance(loss_fn, type) and issubclass(loss_fn, ObjectiveFunction):
        loss_fn = loss_fn()

    loss, expected_fidelity, fidelities, best_target_indices, mask_nonzero = _compute_fidelities_and_loss(
        kets, probs, outcome_sums, target_kets, truncation_error, penalty_strength, loss_fn=loss_fn, phase_lock=phase_lock
    )

    if not return_details:
        return loss

    branch_details = _format_branch_details(
        mask_nonzero, probs, fidelities, best_target_indices, outcomes
    )

    return {
        "loss": loss,
        "expected_fidelity": float(expected_fidelity),
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
            "expected_fidelity": final_eval.get("expected_fidelity", 0.0),
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
        beam_width (int, optional): Maximum trajectories retained per evaluation. Defaults to 5.
        penalty_strength (float, optional): Multiplier for truncation error penalty. Defaults to 10.0.
        measurement_patterns (optional): Fixed measurement sequences. Defaults to None.
        num_parallel_runs (int, optional): Number of parallel Basin-Hopping optimization runs. Defaults to 4.
        num_processes (int, optional): Number of worker processes for parallel execution. Defaults to 4.
        loss_fn (callable, optional): Custom objective loss evaluation function.
        callback (callable, optional): Custom callback function `callback(x, f, accept)` invoked at each iteration.
    """

    def __init__(self,
                 circuit: StaticCircuit,
                 target_gens: list[TargetGenerator] | TargetGenerator,
                 cutoff_dim: int,
                 beam_width: int = 20,
                 penalty_strength: float = 10.0,
                 measurement_patterns=None,
                 num_parallel_runs: int = 4,
                 num_processes: int = 4,
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
        self.loss_fn = loss_fn
        self.callback = callback
        self.phase_lock = phase_lock

    def run(self, n_iter: int = 20, method: str = "L-BFGS-B",
            num_parallel_runs: int | None = None, base_seed: int | None = None,
            callback=None) -> dict:
        """Executes global static circuit optimization using Basin-Hopping.

        Args:
            n_iter (int, optional): Number of Basin-Hopping iterations per run. Defaults to 20.
            method (str, optional): Local minimizer algorithm. Defaults to "L-BFGS-B".
            num_parallel_runs (int, optional): Override for number of parallel optimization runs.
            base_seed (int, optional): Base random seed for reproducible runs.
            callback (callable, optional): Callback function `callback(x, f, accept)` executed after each basin step.

        Returns:
            dict: Optimization results containing best parameter vector, loss, fidelities, and branches.
        """
        cb = callback if callback is not None else self.callback
        n_parallel = num_parallel_runs if num_parallel_runs is not None else self.num_parallel_runs
        bounds = self.circuit.parameter_bounds

        start_time = time.time()

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
            "expected_fidelity": best_res.get("expected_fidelity", 0.0),
            "branches": best_res.get("branches", []),
            "total_probability": best_res.get("total_probability", 0.0),
            "duration": total_duration,
            "message": f"Best of {n_parallel} parallel runs",
            "run_results": results,
            "best_run_idx": seeds.index(best_res["seed"]),
            "circuit_config": circuit_cfg,
            "target_configs": target_cfgs,
            "loss_config": loss_cfg,
            "runner_config": runner_cfg,
        }
