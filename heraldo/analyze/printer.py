"""Functions for displaying optimization run results in a clean, user-friendly format."""

from typing import Any, Dict
import numpy as np

from heraldo.serialization import create_from_config, reconstruct_objects


def print_results(results: Dict[str, Any]) -> None:
    """Prints optimization run results in a clean, human-readable format.

    Args:
        results (dict): Result dictionary returned by `BasinHoppingRunner.run()`
            or loaded via `load_results()`.
    """
    if not isinstance(results, dict):
        raise TypeError(f"Expected results to be a dictionary, got {type(results).__name__}")

    show_params = True
    show_branches = True
    precision = 4

    width = 72
    separator = "=" * width
    sub_separator = "-" * width

    print("\n" + separator)
    print(f"{'HERALDO OPTIMIZATION RESULTS':^{width}}")
    print(separator)

    # General Summary
    message = results.get("message", "Completed")
    loss = results.get("loss")
    obj_score = results.get("objective_score")
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
    if obj_score is not None:
        print(f"  Objective Score       : {obj_score:.{precision}f}")
    if tot_prob is not None:
        print(f"  Total Probability     : {tot_prob:.2%} ({tot_prob:.{precision}f})")

    branches = results.get("branches", [])
    print(f"  Active Branches Count : {len(branches)}")

    # Circuit Configuration Section
    circuit_cfg = results.get("circuit_config")
    if circuit_cfg and isinstance(circuit_cfg, dict):
        print("\n" + sub_separator)
        print(f"{'CIRCUIT CONFIGURATION':^{width}}")
        print(sub_separator)
        print(f"  Class Name            : {circuit_cfg.get('class_name', 'Unknown')}")
        if "module" in circuit_cfg:
            print(f"  Module                : {circuit_cfg['module']}")
        params = circuit_cfg.get("params", {})
        if params:
            print("  Parameters            :")
            for k, v in params.items():
                val_str = f"{v:.{precision}f}" if isinstance(v, float) else str(v)
                print(f"    • {k:<19}: {val_str}")

    # Target Configuration Section
    target_cfgs = results.get("target_configs")
    if target_cfgs:
        print("\n" + sub_separator)
        print(f"{'TARGET CONFIGURATION(S)':^{width}}")
        print(sub_separator)

        tc_list = target_cfgs if isinstance(target_cfgs, list) else [target_cfgs]
        for idx, tc in enumerate(tc_list, 1):
            if isinstance(tc, dict):
                cls_name = tc.get("class_name", "Unknown")
                print(f"  [Target {idx}] {cls_name}")
                t_params = tc.get("params", {})
                for k, v in t_params.items():
                    val_str = f"{v:.{precision}f}" if isinstance(v, float) else str(v)
                    print(f"    • {k:<19}: {val_str}")

    # Runner & Optimization Settings Section
    runner_cfg = results.get("runner_config")
    if runner_cfg and isinstance(runner_cfg, dict):
        print("\n" + sub_separator)
        print(f"{'RUNNER & OPTIMIZATION SETTINGS':^{width}}")
        print(sub_separator)

        cutoff = runner_cfg.get("cutoff_dim")
        bw = runner_cfg.get("beam_width")
        pen = runner_cfg.get("penalty_strength")
        meas_pat = runner_cfg.get("measurement_patterns")
        loss_cfg = results.get("loss_config") or (runner_cfg.get("loss_fn") if runner_cfg else None)
        method = runner_cfg.get("method")
        n_iter = runner_cfg.get("n_iter")
        n_parallel = runner_cfg.get("num_parallel_runs")
        n_proc = runner_cfg.get("num_processes")
        seed = runner_cfg.get("base_seed")

        if cutoff is not None:
            print(f"  Fock Cutoff Dim       : {cutoff}")
        if bw is not None:
            print(f"  Beam Width            : {bw}")
        if pen is not None:
            print(f"  Penalty Strength      : {pen:.{precision}f}" if isinstance(pen, float) else f"  Penalty Strength      : {pen}")
        if meas_pat is not None:
            print(f"  Measurement Patterns  : {meas_pat}")
        else:
            print(f"  Measurement Strategy  : Dynamic Beam Search")
        if loss_cfg is not None:
            if isinstance(loss_cfg, dict):
                loss_str = loss_cfg.get("class_name", "Unknown")
            elif hasattr(loss_cfg, "__class__"):
                loss_str = loss_cfg.__class__.__name__
            else:
                loss_str = str(loss_cfg)
            print(f"  Loss Function         : {loss_str}")
        if method is not None:
            print(f"  Minimizer Method      : {method}")
        if n_iter is not None:
            print(f"  Basin Iterations      : {n_iter}")
        if n_parallel is not None:
            print(f"  Parallel Runs         : {n_parallel}")
        if n_proc is not None:
            print(f"  Worker Processes      : {n_proc}")
        if seed is not None:
            print(f"  Base Seed             : {seed}")

    # Resolve circuit and target objects for branches and parameters sections
    circuit = results.get("circuit")
    if circuit is None:
        try:
            reconstructed = reconstruct_objects(results)
            circuit = reconstructed.get("circuit")
        except Exception:
            pass

    if circuit is None and "circuit_config" in results and results["circuit_config"]:
        try:
            circuit = create_from_config(results["circuit_config"])
        except Exception:
            pass

    # Resolve target display names for branches table
    target_names = []
    targets = results.get("targets")
    if targets is None and "target_configs" in results and results["target_configs"]:
        tc = results["target_configs"]
        tc_list = tc if isinstance(tc, list) else [tc]
        try:
            targets = [create_from_config(item) for item in tc_list]
        except Exception:
            targets = tc_list

    if targets:
        try:
            from heraldo.analyze.rotations import get_target_display_names
            target_names = get_target_display_names(targets if isinstance(targets, list) else [targets])
        except Exception:
            target_names = []

    # Measurement Branches Section
    if show_branches and branches:
        print("\n" + sub_separator)
        print(f"{'MEASUREMENT BRANCHES':^{width}}")
        print(sub_separator)

        target_col_width = max(24, max((len(name) for name in target_names), default=24))
        header = f"  {'#':<4} {'Outcome':<12} {'Target Name':<{target_col_width}} {'Probability':<18} {'Fidelity':<12}"
        print(header)
        print("  " + "-" * max(width - 4, len(header) - 2))

        for idx, b in enumerate(branches, 1):
            outcome_str = str(b.get("outcome", "-"))
            t_idx_val = b.get("target_idx")

            if isinstance(t_idx_val, (int, np.integer)) and 0 <= int(t_idx_val) < len(target_names):
                target_str = target_names[int(t_idx_val)]
            elif t_idx_val is not None:
                target_str = str(t_idx_val)
            else:
                target_str = "-"

            prob = b.get("prob", 0.0)
            fid = b.get("fidelity", 0.0)

            prob_str = f"{prob:.2%} ({prob:.{precision}f})"
            fid_str = f"{fid:.{precision}f}"

            print(f"  {idx:<4} {outcome_str:<12} {target_str:<{target_col_width}} {prob_str:<18} {fid_str:<12}")

    # Optimized Parameters Section
    x = results.get("x")
    if show_params and x is not None:
        x_arr = np.asarray(x).flatten()
        print("\n" + sub_separator)
        print(f"{'OPTIMIZED PARAMETERS':^{width}}")
        print(sub_separator)
        print(f"  Total Parameters : {len(x_arr)}\n")

        param_names = None
        if circuit is not None:
            if hasattr(circuit, "parameter_names") and circuit.parameter_names:
                param_names = circuit.parameter_names
            elif hasattr(circuit, "_param_names") and circuit._param_names:
                param_names = circuit._param_names

        if param_names and len(param_names) == len(x_arr):
            idx_w = max(2, len(str(len(x_arr) - 1)))
            max_name_len = max(len(str(name)) for name in param_names)
            name_w = max(10, max_name_len)
            formatted_items = []
            for i, (name, val) in enumerate(zip(param_names, x_arr)):
                val_str = f"{val:+.{precision}f}"
                formatted_items.append(f"[{i:>{idx_w}}] {name:<{name_w}} : {val_str:>10}")

            cols = 2
            for i in range(0, len(formatted_items), cols):
                chunk = formatted_items[i:i + cols]
                row_str = "    ".join(chunk)
                print(f"  {row_str}")
        else:
            formatted_vals = [f"{val:+.{precision}f}" for val in x_arr]
            cols = 4
            for i in range(0, len(formatted_vals), cols):
                chunk = formatted_vals[i:i + cols]
                indices = [f"[{i+j}]" for j in range(len(chunk))]
                row_str = "   ".join(f"{idx:>5}: {val:>10}" for idx, val in zip(indices, chunk))
                print(f"  {row_str}")

    print("\n" + separator + "\n")
