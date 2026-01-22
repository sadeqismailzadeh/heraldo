# Added imports for file loading, plotting and CLI handling; removed duplicate imports
import argparse
import glob
import json
import pickle
import numpy as np
import os
from pathlib import Path
import matplotlib.pyplot as plt

import strawberryfields as sf
from strawberryfields.ops import Ket

from quantum_agent.optimization.time_circuits import *
from quantum_agent.optimization.time_interfaces import TimeMultiplexedCircuit
from quantum_agent.optimization.time_runner import evaluate_time_domain_circuit
from quantum_agent.components.targets import *
from quantum_agent.utils import *


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
    matches = sorted(base_dir.glob("cma_run_*"))
    if not matches:
        return None
    return matches[-1]


def _load_best_from_results(results_dir: Path):
    """Try to load best_result.pkl / best_x.npy / schedule.json from a results directory."""
    best_dir = results_dir / "best"
    best = {}
    if (best_dir / "best_result.pkl").exists():
        with open(best_dir / "best_result.pkl", "rb") as f:
            try:
                best['best_res'] = pickle.load(f)
            except Exception:
                # maybe it was saved as dict earlier
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


def _reshape_outcome_flat(outcome_flat, circuit: TimeMultiplexedCircuit):
    """
    Convert a flat outcome tuple (o1,o2,o3,...) into per-step tuples:
      result = [ (o_step0_mode0, o_step0_mode1, ...), (o_step1_mode0,...), ... ]
    """
    meas_specs = circuit.get_measurement_specs()
    n_meas_modes = len(meas_specs)
    steps = circuit.steps
    if outcome_flat is None:
        return None
    outcome_list = list(outcome_flat)
    if len(outcome_list) != steps * n_meas_modes:
        # If length equals steps, maybe already per-step single-mode outcomes
        if len(outcome_list) == steps:
            return tuple((int(x),) for x in outcome_list)
        raise ValueError(f"Outcome length {len(outcome_list)} incompatible with circuit (steps={steps}, meas_modes={n_meas_modes})")
    reshaped = []
    for s in range(steps):
        start = s * n_meas_modes
        reshaped.append(tuple(int(x) for x in outcome_list[start:start + n_meas_modes]))
    return tuple(reshaped)


def main():
    # Configuration - set these variables directly instead of using command-line arguments
    results_path = None  # Set to specific path if desired, e.g., "results/cma_run_20240101_120000"
    branch_index = 0  # Index of branch to visualize from best_result['branches']
    measurement = None  # Explicit measurement tuple, e.g., "3,1" or "3,1;2,0" (semicolon separated)
    cutoff = 40  # Cutoff dimension for visualization
    circuit_class = "ThreeModeTimeDomainSqueezeOnly"  # Circuit class to use


    # Optional: define a target to compute fidelity against the post-selected state.
    # If left as None, no fidelity will be computed.
    # Examples:
    #   - Core GKP target:
    #     target = CoreGKPTarget(csv_path=Path(__file__).resolve().parent.parent.parent / "data" / "GKP_core_coefficients.csv", n_max=8, delta_db=10, mu=0)
    #   - Squeezed cat:
    #     target = SqueezedCatTarget(alpha=3, r=1.38, p=0)
    #   - Or create your own TargetGenerator implementation that supports get_target_ket(cutoff)
    target = None
        
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
    best = _load_best_from_results(results_dir)
    if not best:
        print("No best/summary files found under 'best/'. Make sure run_cma_optimization saved results.")
        return

    flat_x = best.get('x') or (best.get('best_res', {}).get('x') if best.get('best_res') else None)
    if flat_x is None:
        print("Could not find flat parameter vector (best_x).")
        return

    # Instantiate the circuit
    squeezing = db_to_r(12)
    if circuit_class == "ThreeModeTimeDomainSqueezeOnly":
        circuit = ThreeModeTimeDomainSqueezeOnly(steps=1,
                                                 time_invariant=False,
                                                 clip_size=squeezing,
                                                 measure_fock_cutoff=30,
                                                 num_single_photon=0,
                                                 train_initial_state=True,
                                                 initial_r=squeezing)
    elif circuit_class == "TwoModeTimeDomainSqueezeOnly":
        circuit = TwoModeTimeDomainSqueezeOnly(steps=1,
                                               time_invariant=False,
                                               clip_size=squeezing,
                                               measure_fock_cutoff=30,
                                               num_single_photon=0,
                                               train_initial_state=True,
                                               initial_r=squeezing)
    else:
        raise ValueError(f"Unknown circuit class: {circuit_class}")

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
        best_res = best.get('best_res')
        if best_res and 'branches' in best_res and len(best_res['branches']) > 0:
            branches = best_res['branches']
            idx = min(branch_index, len(branches) - 1)
            branch = branches[idx]
            outcome_raw = branch.get('outcome')
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
    print("Final ket (truncated):")
    print(ket[:min(len(ket), 20)])

    # If a target is provided, compute fidelity between postselected ket and target ket
    if target is not None:
        try:
            # Request the target ket at the visualization cutoff
            target_ket = target.get_target_ket(cutoff)

            # Make length consistent: pad shorter vector with zeros
            max_len = max(len(target_ket), len(ket))
            t = np.zeros(max_len, dtype=np.complex128)
            s = np.zeros(max_len, dtype=np.complex128)
            t[:len(target_ket)] = target_ket
            s[:len(ket)] = ket

            fid = fidelity_max_rotation(t, s)
            print(f"Fidelity with provided target (cutoff={cutoff}): {fid:.6f}")
        except Exception as e:
            print(f"Failed to compute fidelity with target: {e}")

    # Visualize like demo_target
    plot_ket_wigner(ket, title=f"postselect {measurement_outcomes}", cutoff_dim=cutoff)

if __name__ == "__main__":
    main()
