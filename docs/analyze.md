# Analyzing and Visualizing Results

The `heraldo.analyze` module provides post-optimization tooling for inspecting, visualizing, and stress-testing a result dictionary returned by `BasinHoppingRunner.run()` or reloaded via `load_results()` (see the [Serialization and Reconstruction Guide](serialization.md)). All functions on this page are importable directly from `heraldo.analyze`.

---

## `print_results`

Prints a formatted, human-readable summary of an optimization result to the console. This is normally the first thing you call after a run finishes or after loading a saved result — it surfaces the overall loss/fidelity/probability metrics, the circuit and target configuration, the runner settings used, a table of individual measurement branches, and the optimized parameter vector.

**Signature:** `print_results(results)`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `results` | `dict` | — (required) | Result dictionary returned by `BasinHoppingRunner.run()` or `load_results()`. |

**Returns:** `None`. Output is printed directly to the console.

**Raises:** `TypeError` if `results` is not a dictionary.

The printed report is organized into sections: status/message and timing, best loss/objective score/total probability, circuit configuration, target configuration(s), runner & optimization settings, a per-outcome measurement branches table (outcome, matched target name, probability, fidelity), and the optimized parameter vector (labeled with parameter names when available). If live `circuit`/`targets` objects aren't present in `results` (e.g. after loading without reconstruction), `print_results` attempts to reconstruct them from `circuit_config`/`target_configs` purely for display, and degrades gracefully if that fails.

```python
from heraldo.analyze import print_results

print_results(loaded_result)
```

---

## Plotting Heralded States

### `plot_outcomes`

Generates a Wigner function and Fock-state-probability plot for one or more measurement outcomes in a result dictionary. Internally, it re-simulates the circuit at the optimized parameters, projects the joint state onto the requested outcome(s), and delegates the actual figure rendering to `plot_wigner`.

**Signature:** `plot_outcomes(results, outcomes=None, cutoff_dim=None, grid_size=300, x_limit=6.0, save_prefix=None, show=True)`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `results` | `dict` | — (required) | Result dictionary from `BasinHoppingRunner.run()` or `load_results()`. |
| `outcomes` | `int`, `tuple`, or `list` | `None` | Measurement outcome pattern(s) to plot. See [Specifying Measurement Patterns](patterns.md) for accepted formats. **Mandatory** if the result came from beam search. |
| `cutoff_dim` | `int` | `None` | Fock space truncation dimension. If `None`, taken from `runner_config`, defaulting to `30`. |
| `grid_size` | `int` | `300` | Phase space resolution for the Wigner function grid. |
| `x_limit` | `float` | `6.0` | Maximum quadrature extent for the x/p axes. |
| `save_prefix` | `str` or `Path` | `None` | If given, each plot is saved as `{save_prefix}_outcome_{pattern}.png`. |
| `show` | `bool` | `True` | Whether to display plots interactively via `plt.show()`. |

**Returns:** `list[plt.Figure]` — one figure per requested outcome.

**Raises:** `TypeError` if `results` is not a dictionary. `ValueError` if beam search optimization was used and `outcomes` is `None`, or if `circuit`/`x` are missing from `results`.

If beam search was used (`runner_config["measurement_patterns"]` is `None`), `outcomes` must be supplied. If fixed-pattern optimization was used and `outcomes` is omitted, all fixed patterns are plotted by default. An outcome with near-zero probability prints a warning and is plotted using the unnormalized state vector.

```python
from heraldo.analyze import plot_outcomes

plot_outcomes(
    loaded_result,
    outcomes=[4, 5],
    save_prefix="example_plot",
    show=True,
)
```

---

## `analyze_rotations`

Computes, for each requested measurement outcome, the phase-space rotation angle that maximizes fidelity with the best-matching target state, using the same rotation-invariant FFT-based fidelity metric used internally by the optimizer. Use this to check whether the heralded states across a set of outcomes share a common phase-space orientation (rotation-invariant) or require per-outcome phase compensation (rotation-variant).

**Signature:** `analyze_rotations(results, outcomes=None, cutoff_dim=None, print_summary=True)`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `results` | `dict` | — (required) | Result dictionary from `BasinHoppingRunner.run()` or `load_results()`. |
| `outcomes` | `int`, `tuple`, or `list` | `None` | Measurement outcome pattern(s) to analyze. See [Specifying Measurement Patterns](patterns.md). **Mandatory** if beam search was used. |
| `cutoff_dim` | `int` | `None` | Fock space truncation dimension. If `None`, taken from `runner_config`, defaulting to `30`. |
| `print_summary` | `bool` | `True` | Whether to print a formatted summary table to the console. |

**Returns:** `list[dict]`, one entry per outcome:

| Key | Description |
|---|---|
| `outcome` | Tuple of detected photon numbers. |
| `prob` | Branch probability (from `results["branches"]` when available, otherwise recomputed). |
| `fidelity` | Maximum fidelity achieved over all candidate targets and phase angles. |
| `target_idx` / `target_name` | Index and display name of the best-matching target. |
| `angle_rad` / `angle_deg` | Optimal rotation angle in `(-π, π]` radians / degrees. |

**Raises:** `TypeError` if `results` is not a dictionary. `ValueError` if beam search was used and `outcomes` is `None`, or if circuit/target information is missing.

The rotation search is discretized over `256` phase angles via FFT (matching the optimizer's internal resolution). When `print_summary=True`, the printed report also includes the overall angular range spanning all requested outcomes, plus a per-target angular range — a range of `0°` indicates a rotation-invariant configuration.

```python
from heraldo.analyze import analyze_rotations

analyze_rotations(loaded_result, outcomes=[4, 5])
```

---

## `analyze_cutoff`

Re-evaluates the circuit at two different Fock-space truncation dimensions and compares the resulting infidelities, to confirm numerical convergence for a chosen `cutoff_dim`.

**Signature:** `analyze_cutoff(results, low_cutoff=30, high_cutoff=50, outcomes=None, print_summary=True)`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `results` | `dict` | — (required) | Result dictionary from `BasinHoppingRunner.run()` or `load_results()`. |
| `low_cutoff` | `int` | `30` | Lower Fock space truncation dimension. |
| `high_cutoff` | `int` | `50` | Higher Fock space truncation dimension. |
| `outcomes` | `int`, `tuple`, or `list` | `None` | Measurement outcome pattern(s) to analyze. See [Specifying Measurement Patterns](patterns.md). If `None`, uses the fixed measurement patterns stored in `results`; **mandatory** if beam search was used. |
| `print_summary` | `bool` | `True` | Whether to print a formatted summary table to the console. |

**Returns:** `dict` with the following keys:

| Key | Description |
|---|---|
| `low_cutoff`, `high_cutoff` | The two cutoff dimensions compared. |
| `truncation_error_low`, `truncation_error_high` | Pre-measurement joint-state truncation error (`1 - ‖ket‖²`) at each cutoff. |
| `outcomes` | Normalized list of outcome tuples analyzed. |
| `analysis` | Per-outcome list of dicts (`outcome`, `target_idx`, `target_name`, `prob_low`/`prob_high`, `fidelity_low`/`fidelity_high`, `infidelity_low`/`infidelity_high`, `abs_error`, `log_discrepancy`). |
| `max_abs_error` | Largest absolute infidelity difference across all outcomes. |
| `max_log_discrepancy` | Largest $\log_{10}(I_{\text{high}}) - \log_{10}(I_{\text{low}})$ across all outcomes (`None` if not computable). |
| `worst_outcome_abs_error`, `worst_outcome_log_discrepancy` | Outcome tuples associated with the worst-case metrics above. |

**Raises:** `TypeError` if `results` is not a dictionary. `ValueError` if circuit/target/parameter information is missing, or if beam search was used and `outcomes` is `None`.

`log_discrepancy` for a given outcome is `None` when either infidelity is at or below `1e-30`, to avoid taking `log10` of a non-positive or vanishing value.

```python
from heraldo.analyze import analyze_cutoff

analyze_cutoff(loaded_result, low_cutoff=30, high_cutoff=45, outcomes=[4, 5])
```

---

## `analyze_loss`

Re-evaluates the optimized circuit under simulated photon loss at one or more channel transmissivities, using full density-matrix simulation, to quantify how success probability and state fidelity degrade under realistic non-ideal optical channels.

**Signature:** `analyze_loss(results, transmissivities=None, outcomes=None, cutoff_dim=None, print_summary=True)`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `results` | `dict` | — (required) | Result dictionary from `BasinHoppingRunner.run()` or `load_results()`. |
| `transmissivities` | `list[float]` | `None` | Channel transmissivities $\eta \in (0, 1]$ to evaluate. Defaults to `[1.0, 0.99, 0.90]` (0%, 1%, and 10% loss). |
| `outcomes` | `int`, `tuple`, or `list` | `None` | Measurement outcome pattern(s) to analyze. See [Specifying Measurement Patterns](patterns.md). If `None`, uses `runner_config["measurement_patterns"]` when set, otherwise falls back to the outcomes recorded in `results["branches"]`. |
| `cutoff_dim` | `int` | `None` | Fock space truncation dimension. If `None`, defaults to `15` for circuits with 3 or more modes and `30` for 2-mode circuits (to manage the memory cost of density-matrix simulation). |
| `print_summary` | `bool` | `True` | Whether to print a formatted summary table to the console. |

**Returns:** `dict` with the following keys:

| Key | Description |
|---|---|
| `transmissivities` | The list of transmissivities evaluated. |
| `outcomes` | Normalized list of outcome tuples analyzed. |
| `analysis` | Per-outcome list of dicts, each with `outcome`, `target_idx`, `target_name`, and `results_by_transmissivity` — a dict keyed by transmissivity value, each holding `prob` and `fidelity`. |

**Raises:** `TypeError` if `results` is not a dictionary. `ValueError` if circuit/target information is missing, or if no outcomes can be resolved (neither `outcomes`, fixed patterns, nor recorded branches are available).

A separate circuit instance is constructed for each transmissivity (from the serialized circuit config, with `loss_transmissivity` overridden), so the original `circuit` object in `results` is left untouched. For each outcome, the best-matching target is determined once — at the first (highest, i.e. least lossy) transmissivity evaluated — and that same target index is reused across all other transmissivities so the comparison stays consistent. If an outcome's projected probability is at or below `1e-12` at a given transmissivity, its fidelity is reported as `0.0`.

```python
from heraldo.analyze import analyze_loss

analyze_loss(loaded_result, outcomes=[4, 5])
```

---

## Summary

| Function | Purpose |
|---|---|
| `print_results` | Console report of overall run quality: loss, fidelity, probability, configuration, and branches. |
| `plot_outcomes` / `plot_wigner` | Visualize heralded states as Wigner functions and Fock-probability distributions. |
| `analyze_rotations` | Determine per-outcome optimal phase-space rotation, and whether a result set is rotation-invariant. |
| `analyze_cutoff` | Sensitivity of fidelity to the chosen Fock-space truncation dimension. |
| `analyze_loss` | Sensitivity of probability and fidelity to simulated photon loss. |
