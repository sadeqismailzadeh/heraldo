"""Functions for displaying optimization run results in a clean, user-friendly format."""

from typing import Any, Dict, Optional
import numpy as np


def print_results(
    results: Dict[str, Any],
    show_params: bool = True,
    show_branches: bool = True,
    precision: int = 4,
) -> None:
    """Prints optimization run results in a clean, human-readable format.

    Args:
        results (dict): Result dictionary returned by `BasinHoppingRunner.run()`
            or loaded via `load_results()`.
        show_params (bool, optional): Whether to print the optimized parameter vector ``x``.
            Defaults to True.
        show_branches (bool, optional): Whether to print detailed output branch metadata.
            Defaults to True.
        precision (int, optional): Number of decimal places for floating point values.
            Defaults to 4.
    """
    if not isinstance(results, dict):
        raise TypeError(f"Expected results to be a dictionary, got {type(results).__name__}")

    width = 72
    separator = "=" * width
    sub_separator = "-" * width

    print("\n" + separator)
    print(f"{'HERALDO OPTIMIZATION RESULTS':^{width}}")
    print(separator)

    # General Summary
    message = results.get("message", "Completed")
    loss = results.get("loss")
    exp_fid = results.get("expected_fidelity")
    tot_prob = results.get("total_probability")
    duration = results.get("duration")
    best_run_idx = results.get("best_run_idx")
    parallel_runs = len(results.get("run_results", [])) if "run_results" in results else None

    print(f"  Status / Message      : {message}")
    if parallel_runs:
        print(f"  Parallel Runs         : {parallel_runs} total (Best run index: {best_run_idx})")
    elif "seed" in results and results["seed"] is not None:
        print(f"  Random Seed           : {results['seed']}")

    if duration is not None:
        mins, secs = divmod(duration, 60)
        hours, mins = divmod(mins, 60)
        time_str = f"{duration:.2f} s"
        if hours > 0:
            time_str += f" ({int(hours):02d}:{int(mins):02d}:{secs:05.2f})"
        elif mins > 0:
            time_str += f" ({int(mins):02d}:{secs:05.2f})"
        print(f"  Execution Duration    : {time_str}")

    if loss is not None:
        print(f"  Best Loss Score       : {loss:.{precision}f}")
    if exp_fid is not None:
        print(f"  Expected Fidelity     : {exp_fid:.{precision}f}")
    if tot_prob is not None:
        print(f"  Total Probability     : {tot_prob:.2%} ({tot_prob:.{precision}f})")

    branches = results.get("branches", [])
    print(f"  Active Branches Count : {len(branches)}")

    # Measurement Branches Section
    if show_branches and branches:
        print("\n" + sub_separator)
        print(f"{'MEASUREMENT BRANCHES':^{width}}")
        print(sub_separator)

        header = f"  {'#':<4} {'Outcome':<16} {'Target Idx':<12} {'Probability':<16} {'Fidelity':<12}"
        print(header)
        print("  " + "-" * (width - 4))

        for idx, b in enumerate(branches, 1):
            outcome_str = str(b.get("outcome", "-"))
            target_idx = str(b.get("target_idx", "-"))
            prob = b.get("prob", 0.0)
            fid = b.get("fidelity", 0.0)

            prob_str = f"{prob:.2%} ({prob:.{precision}f})"
            fid_str = f"{fid:.{precision}f}"

            print(f"  {idx:<4} {outcome_str:<16} {target_idx:<12} {prob_str:<16} {fid_str:<12}")

    # Optimized Parameters Section
    x = results.get("x")
    if show_params and x is not None:
        x_arr = np.asarray(x).flatten()
        print("\n" + sub_separator)
        print(f"{'OPTIMIZED PARAMETERS':^{width}}")
        print(sub_separator)
        print(f"  Total Parameters : {len(x_arr)}\n")

        formatted_vals = [f"{val:+.{precision}f}" for val in x_arr]
        cols = 4
        for i in range(0, len(formatted_vals), cols):
            chunk = formatted_vals[i:i + cols]
            indices = [f"[{i+j}]" for j in range(len(chunk))]
            row_str = "   ".join(f"{idx:>5}: {val:>10}" for idx, val in zip(indices, chunk))
            print(f"  {row_str}")

    print("\n" + separator + "\n")
