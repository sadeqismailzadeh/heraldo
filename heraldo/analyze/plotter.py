"""Plotting utilities for quantum states and circuit outcomes."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import matplotlib.pyplot as plt
import strawberryfields as sf

from heraldo._internal import _normalize_outcomes
from heraldo.serialization import reconstruct_objects


def plot_wigner(
    ket: np.ndarray,
    filename: Optional[Union[str, Path]] = None,
    title: str = "State",
    cutoff_dim: Optional[int] = None,
    grid_size: int = 300,
    x_limit: float = 6.0,
    ax_wigner: Optional[plt.Axes] = None,
    ax_fock: Optional[plt.Axes] = None,
    show: bool = True,
    dpi: int = 300,
) -> plt.Figure:
    """Generates a Wigner function and Fock distribution plot for a single-mode state vector.

    Args:
        ket (np.ndarray): Single-mode state vector in Fock basis.
        filename (str or Path, optional): File path to save the generated plot.
        title (str, optional): Title for the Wigner plot. Defaults to "State".
        cutoff_dim (int, optional): Fock space truncation dimension. If None, inferred from len(ket).
        grid_size (int, optional): Resolution of phase space grid. Defaults to 300.
        x_limit (float, optional): Maximum position/momentum bound for phase space. Defaults to 6.0.
        ax_wigner (plt.Axes, optional): Pre-existing matplotlib axis for Wigner plot.
        ax_fock (plt.Axes, optional): Pre-existing matplotlib axis for Fock bar plot.
        show (bool, optional): Whether to display the plot with plt.show(). Defaults to True.
        dpi (int, optional): Resolution DPI for saved image. Defaults to 300.

    Returns:
        plt.Figure: Matplotlib figure object containing the plot.
    """
    ket = np.asarray(ket, dtype=np.complex128).flatten()
    if cutoff_dim is None:
        cutoff_dim = len(ket)

    norm = np.linalg.norm(ket)
    if norm > 1e-12 and abs(norm - 1.0) > 1e-6:
        ket = ket / norm

    prog = sf.Program(1)
    with prog.context as q:
        from strawberryfields.ops import Ket
        Ket(ket) | q[0]

    eng = sf.Engine("fock", backend_options={"cutoff_dim": cutoff_dim})
    result = eng.run(prog)
    state = result.state

    xvec = np.linspace(-x_limit, x_limit, grid_size)
    pvec = np.linspace(-x_limit, x_limit, grid_size)
    W = state.wigner(mode=0, xvec=xvec, pvec=pvec)
    probs = state.all_fock_probs(cutoff=cutoff_dim)

    custom_axes = (ax_wigner is not None) and (ax_fock is not None)

    if custom_axes:
        ax1, ax2 = ax_wigner, ax_fock
        fig = ax1.figure
    else:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    X, P = np.meshgrid(xvec, pvec)
    lim = np.max(np.abs(W)) if np.max(np.abs(W)) > 0 else 1.0
    c = ax1.pcolormesh(X, P, W, cmap='RdBu', shading='auto', vmin=-lim, vmax=lim, rasterized=True)

    cbar = fig.colorbar(c, ax=ax1, fraction=0.046, pad=0.04)
    cbar.set_label('W(x, p)', fontsize=11)
    cbar.ax.tick_params(labelsize=10)

    ax1.set_title(f"Wigner Function: {title}", fontsize=11, pad=10)
    ax1.set_xlabel("x (Position)", fontsize=11)
    ax1.set_ylabel("p (Momentum)", fontsize=11)
    ax1.tick_params(labelsize=10)
    ax1.set_aspect('equal')
    ax1.axhline(0, color='gray', linestyle=':', alpha=0.5, linewidth=1)
    ax1.axvline(0, color='gray', linestyle=':', alpha=0.5, linewidth=1)

    display_cutoff = min(cutoff_dim, 60)
    indices = np.arange(display_cutoff)
    max_p = max(probs[:display_cutoff]) if len(probs) > 0 else 1.0
    ax2.bar(indices, probs[:display_cutoff], color='#2c7bb6', alpha=0.8, edgecolor='black', width=0.7)

    ax2.set_title("Fock State Probabilities", fontsize=11, pad=10)
    ax2.set_xlabel("Fock Number |n>", fontsize=11)
    ax2.set_ylabel("Probability", fontsize=11)
    ax2.tick_params(labelsize=10)

    step = 5 if display_cutoff > 20 else 1
    ax2.set_xticks(np.arange(0, display_cutoff, step))
    ax2.grid(axis='y', linestyle='--', alpha=0.3)
    ax2.set_xlim(-0.5, display_cutoff - 0.5)
    ax2.set_ylim(0, max_p * 1.1 if max_p > 0 else 1.0)

    if not custom_axes:
        fig.tight_layout()
        if filename is not None:
            save_path = Path(filename).resolve()
            save_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, bbox_inches='tight', dpi=dpi)
            print(f"Plot saved to: {save_path}")
        if show:
            plt.show()
        else:
            plt.close(fig)

    return fig


def plot_outcomes(
    results: Dict[str, Any],
    outcomes: Optional[Union[int, Tuple[int, ...], List[Union[int, Tuple[int, ...]]]]] = None,
    cutoff_dim: Optional[int] = None,
    grid_size: int = 300,
    x_limit: float = 6.0,
    save_prefix: Optional[Union[str, Path]] = None,
    show: bool = True,
) -> List[plt.Figure]:
    """Plots the Wigner function and Fock state distribution for specified or all circuit measurement outcomes.

    If beam search optimization was used, the `outcomes` argument MUST be provided to specify
    which outcome patterns to plot. If fixed pattern optimization was used and `outcomes` is None,
    all fixed measurement patterns are plotted by default.

    Args:
        results (dict): Optimization result dictionary returned by `BasinHoppingRunner.run()`
            or loaded via `load_results()`.
        outcomes (int, tuple, or list, optional): Measurement outcome pattern(s) to plot.
            Can be a single outcome (e.g., 4 or (4, 5)) or a list of outcomes.
            Mandatory if beam search was used.
        cutoff_dim (int, optional): Fock space truncation dimension.
            If None, retrieved from runner configuration or defaults to 30.
        grid_size (int, optional): Phase space resolution grid for Wigner function. Defaults to 300.
        x_limit (float, optional): Maximum quadrature extent for x and p axes. Defaults to 6.0.
        save_prefix (str or Path, optional): File path prefix for saving plots.
            If specified, filenames will be formatted as ``{save_prefix}_outcome_{pattern}.png``.
        show (bool, optional): Whether to display plots interactively. Defaults to True.

    Returns:
        list[plt.Figure]: List of generated matplotlib Figure objects.

    Raises:
        ValueError: If beam search optimization was used and `outcomes` is None, or if required
            circuit information is missing.
    """
    if not isinstance(results, dict):
        raise TypeError(f"Expected results to be a dictionary, got {type(results).__name__}")

    circuit = results.get("circuit")
    if circuit is None:
        reconstructed = reconstruct_objects(results)
        circuit = reconstructed.get("circuit")

    if circuit is None:
        raise ValueError("Circuit configuration or object missing from results. Cannot execute circuit.")

    x_params = results.get("x")
    if x_params is None:
        raise ValueError("Optimized parameter vector 'x' missing from results.")

    runner_cfg = results.get("runner_config", {})
    meas_patterns = runner_cfg.get("measurement_patterns") if runner_cfg else None
    is_beam_search = (meas_patterns is None)

    if is_beam_search and outcomes is None:
        raise ValueError(
            "For beam search optimization, specific outcome(s) must be specified via the 'outcomes' parameter "
            "(e.g., outcomes=4 or outcomes=[(4,)] or outcomes=[4, 5])."
        )

    meas_specs = circuit.get_measurement_specs()
    num_meas_modes = len(meas_specs)

    if outcomes is None:
        outcomes_list = _normalize_outcomes(meas_patterns, num_meas_modes)
    else:
        outcomes_list = _normalize_outcomes(outcomes, num_meas_modes)

    if cutoff_dim is None:
        cutoff_dim = runner_cfg.get("cutoff_dim", 30) if runner_cfg else 30

    engine = sf.Engine("fock", backend_options={"cutoff_dim": cutoff_dim})
    eval_res = circuit.run_circuit(np.asarray(x_params), engine)
    full_ket = eval_res.state.ket()

    meas_modes = [m for m, c in meas_specs]
    meas_cutoffs = [min(c, cutoff_dim) for m, c in meas_specs]

    perm = [0] + meas_modes
    transposed = np.transpose(full_ket, axes=perm)
    indexer = (slice(None),) + tuple(slice(0, c) for c in meas_cutoffs)
    sliced_ket = transposed[indexer]

    branches = results.get("branches", [])
    branch_map = {b.get("outcome"): b for b in branches if "outcome" in b}

    figs = []

    for outcome_tuple in outcomes_list:
        for mode_idx, (val, c_dim) in enumerate(zip(outcome_tuple, meas_cutoffs)):
            if val >= c_dim or val < 0:
                raise ValueError(
                    f"Outcome value {val} for measured mode index {mode_idx} exceeds cutoff dimension {c_dim}."
                )

        adv_index = (slice(None),) + outcome_tuple
        mode0_ket_raw = sliced_ket[adv_index]
        prob = np.vdot(mode0_ket_raw, mode0_ket_raw).real

        if prob < 1e-15:
            print(f"Warning: Outcome {outcome_tuple} has near-zero probability ({prob:.2e}).")
            mode0_ket = mode0_ket_raw
        else:
            mode0_ket = mode0_ket_raw / np.sqrt(prob)

        outcome_str = ", ".join(str(v) for v in outcome_tuple)
        if len(outcome_tuple) == 1:
            outcome_disp = f"n={outcome_str}"
        else:
            outcome_disp = f"n=({outcome_str})"

        branch_info = branch_map.get(outcome_tuple)
        if branch_info:
            prob_val = branch_info.get("prob", prob)
            fid_val = branch_info.get("fidelity")
            if fid_val is not None:
                title = f"Outcome {outcome_disp} (P={prob_val:.2%}, F={fid_val:.4f})"
            else:
                title = f"Outcome {outcome_disp} (P={prob_val:.2%})"
        else:
            title = f"Outcome {outcome_disp} (P={prob:.2%})"

        filename = None
        if save_prefix is not None:
            clean_str = "_".join(str(v) for v in outcome_tuple)
            filename = f"{save_prefix}_outcome_{clean_str}.png"

        fig = plot_wigner(
            ket=mode0_ket,
            filename=filename,
            title=title,
            cutoff_dim=cutoff_dim,
            grid_size=grid_size,
            x_limit=x_limit,
            show=show,
        )
        figs.append(fig)

    return figs
