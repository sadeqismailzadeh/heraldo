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

import quantum_agent
import strawberryfields as sf
from strawberryfields.ops import Ket

# Optimization and Component modules
import quantum_agent.optimization.time_circuits as circuit_module
import quantum_agent.components.targets as target_module
from quantum_agent.optimization.time_circuits import *
from quantum_agent.optimization.time_interfaces import TimeMultiplexedCircuit
from quantum_agent.optimization.time_runner import evaluate_time_domain_circuit
from quantum_agent.components.targets import *
from quantum_agent.utils import *
from quantum_agent.factory import create_from_config


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

def get_all_optimization_results(circuit: TimeMultiplexedCircuit, flat_params: np.ndarray, targets: list, cutoff_dim: int, beam_width: int = 100):
    """
    Runs the circuit evaluation with the provided parameters to generate full branch details.
    
    Args:
        circuit: The circuit instance.
        flat_params: The optimized parameters.
        targets: List of TargetGenerator instances.
        cutoff_dim: Simulation cutoff.
        beam_width: Beam width for search (higher = more branches captured).
        
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
        return_details=True
    )


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

    print("-" * 60)
    print("Dominant Outcome Branches (Sorted by Prob):")
    print(f"{'Outcome':<20} {'Prob':<10} {'Fidelity':<10} {'Best Target':<15}")
    print("-" * 60)
    
    sorted_branches = sorted(branches, key=lambda x: x['prob'], reverse=True)
    
    total_prob = 0.0
    for b in sorted_branches:
        total_prob += b['prob']
        # Filter very small probabilities for display cleanliness if list is huge
        if b['prob'] > 1e-4:
            outcome_str = str(b['outcome'])
            t_idx = b.get('target_idx', 0)
            tgt_name = target_names[t_idx] if t_idx < len(target_names) else f"Target_{t_idx}"
            print(f"{outcome_str:<20} {b['prob']:<10.4f} {b['fidelity']:<10.4f} {tgt_name:<15}")

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
    meas_specs = circuit.get_measurement_specs()
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


def _find_latest_results_dir(base_dir: Path):
    matches = sorted(base_dir.glob("opt_run_*"))
    if not matches:
        return None
    return matches[-1]


def _load_best_from_results(results_dir: Path):
    """Try to load best_result.pkl / best_x.npy / schedule.json from a results directory."""
    best_dir = results_dir / "best"
    best = {}
    if (best_dir / "best_result.pkl").exists():
        with open(best_dir / "best_result.pkl", "rb") as f:
            best['best_res'] = pickle.load(f)
    if (best_dir / "best_x.npy").exists():
        best['x'] = np.load(best_dir / "best_x.npy", allow_pickle=True)
    if (best_dir / "mapped_params.npz").exists():
        npz = np.load(best_dir / "mapped_params.npz")
        best['mapped_params'] = npz['mapped_params']
    if (best_dir / "schedule.json").exists():
        with open(best_dir / "schedule.json", "r") as f:
            best['schedule'] = json.load(f)
    return best


def _load_latest_run(results_dir: Path):
    """
    Load the most recent run_XXXX.pkl from a CMA results directory.
    Returns a dict compatible with downstream evaluation logic.
    """
    runs = sorted(results_dir.glob("run_*.pkl"))
    if not runs:
        return None

    latest = runs[-1]
    with open(latest, "rb") as f:
        data = pickle.load(f)

    # Structure check: {"meta": ..., "res": ...}
    if isinstance(data, dict) and "meta" in data and "res" in data:
        res = data["res"]
        meta = data["meta"]
        # Normalize: Inject recipes from meta into res if they are missing
        if "circuit_config" in meta:
            res["circuit_config"] = meta["circuit_config"]
        if "target_configs" in meta:
            res["target_configs"] = meta["target_configs"]
    else:
        # Fallback for older or flat structures (like best_result.pkl)
        res = data

    out = {
        "best_res": res,
        "x": res.get("x")
    }
    return out


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


def main():
    # Configuration - set these variables directly instead of using command-line arguments
    results_path = None  # Set to specific path if desired
    branch_index = 0  # Index of branch to visualize from best_result['branches']
    measurement = None  # Explicit measurement tuple, e.g., "3,1" or "3,1;2,0" (semicolon separated)
    cutoff = 30  # Cutoff dimension for visualization
    recalc_statistics = True # If True, will print the full branch table and aggregated targets
    
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
    # Prefer latest run_* over best/
    _load_best_from_results
    best = _load_latest_run(results_dir)
    if best is None:
        print("No run_*.pkl found, falling back to best/.")
        best = _load_best_from_results(results_dir)

    if not best:
        print("No usable results found in results directory.")
        return

    # Extract Results
    best_res = best.get('best_res', {})
    
    # Avoid using 'or' with numpy arrays (truth value is ambiguous).
    flat_x = best.get('x')
    if flat_x is None:
        flat_x = best_res.get('x')
    if flat_x is None:
        print("Could not find flat parameter vector (best_x).")
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
            # Example: re-run with potentially higher beam width
            result_to_analyze = get_all_optimization_results(
                circuit, np.asarray(flat_x), targets, cutoff, beam_width=100
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
    # 2. Visualize Specific Outcome
    # -------------------------------------------------------------------------
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

    # Visualize
    plot_ket_wigner(ket, title=f"postselect {measurement_outcomes}", cutoff_dim=cutoff)

if __name__ == "__main__":
    main()
