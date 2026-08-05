# Getting started

This guide walks through a complete optimization workflow with `heraldo`. By the end, you'll have optimized a photonic circuit to herald Schrödinger cat states and inspected the results.

## Quickstart

> **Note**: Optimization runs use Python's `multiprocessing` module to execute parallel basin-hopping searches. Because child processes import the main module on platforms using spawn (such as Windows and macOS), your execution code must be enclosed inside an `if __name__ == "__main__":` block.

```python
import numpy as np
from heraldo.components.circuits import TwoModeStaticSqueezeOnly
from heraldo.components.targets import SqueezedCatTarget
from heraldo.components.runner import BasinHoppingRunner
from heraldo.utils import db_to_r
from heraldo.analyze import print_results


def main():
    # 1. A 2-mode static circuit with 12 dB of squeezing
    circuit = TwoModeStaticSqueezeOnly(clip_size=db_to_r(12.0), measure_fock_cutoff=30)

    # 2. Target states: even (|cat_+>) and odd (|cat_->) squeezed cat states
    targets = [
        SqueezedCatTarget(alpha=np.sqrt(6), r=0.5, p=0),
        SqueezedCatTarget(alpha=np.sqrt(6), r=0.5, p=1),
    ]

    # 3. Optimize the circuit
    runner = BasinHoppingRunner(circuit=circuit, target_gens=targets, cutoff_dim=30)
    result = runner.run(n_iter=5)

    # 4. Inspect the results
    print_results(result)


if __name__ == "__main__":
    main()
```

`print_results` prints a formatted summary: best loss, expected fidelity, total success probability, and the individual measurement branches discovered.

## Step by Step

### 1. Choose a circuit

Circuits live in `heraldo.components.circuits` and describe the physical layout (squeezing, displacement, and beamsplitter network) being optimized:

- `TwoModeStaticGeneral` / `TwoModeStaticSqueezeOnly`
- `ThreeModeStaticGeneral` / `ThreeModeStaticSqueezeOnly`
- `FourModeStaticSqueezeOnly`

Mode 0 is always the unmeasured output mode; the remaining modes are ancillary modes measured with photon-number-resolving detectors.

### 2. Define target states

Targets live in `heraldo.components.targets`:

- `SqueezedCatTarget`, `CatTarget`
- `CubicPhaseTarget`, `CubicResourceTarget`
- `CoreGKPTarget`
- `BinomialCodeTarget`

You can pass a single target or a list — `BasinHoppingRunner` will optimize the circuit to herald whichever combination you provide.

### 3. Run the optimizer

`BasinHoppingRunner` wraps SciPy's basin-hopping algorithm:

```python
runner = BasinHoppingRunner(
    circuit=circuit,
    target_gens=targets,
    cutoff_dim=30,       # Fock space truncation
    beam_width=20,       # top-K outcomes considered during beam search
    penalty_strength=0.1, # penalty for Fock-space truncation error
)
result = runner.run(n_iter=5, method="L-BFGS-B")
```

Leaving `measurement_patterns=None` (the default) runs **beam search**, letting the optimizer discover promising heralding patterns on its own. Passing an explicit list of patterns (e.g. `measurement_patterns=[[4], [5]]`) instead runs **fixed-pattern optimization** against those outcomes.

### 4. Save and reload results

```python
from pathlib import Path
from heraldo.serialization import save_results, load_results

# Define save path relative to the current file
save_path = Path(__file__).parent / "example_results.pkl"

# Save the optimization result dictionary
save_results(result, filepath=save_path)

# Reload results using the saved file path
loaded_result = load_results(save_path, reconstruct=True)
```

`reconstruct=True` rebuilds the `circuit` and `targets` objects from the saved configuration metadata, allowing you to perform post-hoc analysis.

### 5. Analyze rotation, loss sensitivity, and cutoff sensitivity

```python
from heraldo.analyze import analyze_rotations, analyze_loss, analyze_cutoff

# 1. Optimal phase-space rotation per heralding outcome
analyze_rotations(loaded_result, outcomes=[4, 5])

# 2. Fidelity/probability degradation under photon loss
analyze_loss(loaded_result, outcomes=[4, 5])

# 3. Sensitivity to Fock-space truncation dimension
analyze_cutoff(loaded_result, low_cutoff=30, high_cutoff=45, outcomes=[4, 5])
```

- **`analyze_rotations`**: Computes the optimal global phase-space rotation angle $\phi$ that maximizes overlap (fidelity) with each target state for the specified measurement patterns ($n=4$ and $n=5$). It reports the per-outcome target alignment, rotation angles, and overall angular spread to differentiate between rotation-invariant and rotation-variant circuit behavior.
- **`analyze_loss`**: Re-evaluates the circuit under simulated photon loss (defaulting to 0%, 1%, and 10% loss) using density matrix simulation. It calculates how output probabilities and state fidelities degrade across realistic non-ideal optical channels.
- **`analyze_cutoff`**: Simulates the circuit at two different Fock space truncation cutoffs (e.g., $D=30$ vs $D=45$) to quantify truncation errors ($\Delta_{\log} = \log_{10}(I_{\text{high}}) - \log_{10}(I_{\text{low}})$) and confirm numerical convergence.

### 6. Plot heralded states

```python
from heraldo.analyze import plot_outcomes

plot_outcomes(
    loaded_result,
    outcomes=[4, 5],
    save_prefix="example_plot",
    show=True,
)
```

This produces a Wigner function and Fock-probability plot for each requested outcome, optionally saved to `{save_prefix}_outcome_{pattern}.png`.

## Full Example

Below is the complete runnable script combining circuit setup, optimization, result saving/loading, analysis, and visualization into a single pipeline:

```python
from pathlib import Path
import numpy as np

from heraldo.components.circuits import TwoModeStaticSqueezeOnly
from heraldo.components.targets import SqueezedCatTarget
from heraldo.components.runner import BasinHoppingRunner
from heraldo.components.objectives import fixed_pattern_capped_loss_fn
from heraldo.utils import db_to_r
from heraldo.serialization import save_results, load_results, reconstruct_objects
from heraldo.analyze import (
    print_results, plot_outcomes,
    analyze_rotations, analyze_loss, analyze_cutoff
)


def main():
    # 1. Initialize 2-mode static spatial circuit with 12 dB squeezing
    circuit = TwoModeStaticSqueezeOnly(clip_size=db_to_r(12.0), measure_fock_cutoff=30)

    # 2. Define targets: Even (|cat_+>) and Odd (|cat_->) Schrödinger cat states
    targets = [
        SqueezedCatTarget(alpha=np.sqrt(6), r=0.5, p=0),
        SqueezedCatTarget(alpha=np.sqrt(6), r=0.5, p=1),
    ]

    # 3. Optimize for fixed patterns n=4 (even) and n=5 (odd)
    runner = BasinHoppingRunner(
        circuit=circuit,
        target_gens=targets,
        cutoff_dim=30,
        penalty_strength=0
    )

    result = runner.run(n_iter=5)

    # 5. Save results to pickle file in the example script directory
    save_path = Path(__file__).parent / "example_results.pkl"
    saved_file = save_results(result, filepath=save_path)

    # 6. Load results from the saved file with object reconstruction
    print("\n--- Loading Saved Results ---")
    loaded_result = load_results(save_path, reconstruct=True)

    # 7. Display loaded results using print_results
    print_results(loaded_result)

    # 8. Analyze phase rotations for specific outcomes (n=4 and n=5)
    print("--- Rotation Analysis ---")
    analyze_rotations(loaded_result, outcomes=[4, 5])

    # 9. Analyze impact of photon loss (e.g., ideal 100%, 1% loss, 10% loss)
    print("--- Photon Loss Analysis ---")
    analyze_loss(loaded_result, outcomes=[4, 5])

    # 10. Evaluate Fock cutoff truncation fidelity (e.g., low cutoff 30 vs high cutoff 45)
    print("--- Cutoff Truncation Analysis ---")
    analyze_cutoff(loaded_result, low_cutoff=30, high_cutoff=45, outcomes=[4, 5])

    # 11. Plot Wigner functions and Fock probabilities for specific outcomes (n=4 and n=5)
    print("--- Plotting Outcomes ---")
    plot_outcomes(
        loaded_result,
        outcomes=[4, 5],
        save_prefix=Path(__file__).parent / "example_plot",
        show=True
    )


if __name__ == "__main__":
    main()
```

## Next Steps

- See the [User Guide](user_guide.md) for a deeper dive into resource multiplexing vs. single-target harvesting.
- See [Circuits Comparison](circuits_comparison.md) for guidance on choosing between circuit architectures.
- See the [API Reference](api/index.md) for full parameter documentation.
