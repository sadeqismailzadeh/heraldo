# Getting started

This guide walks through a complete optimization workflow with `heraldo`. By the end, you'll have optimized a photonic circuit to herald Schrödinger cat states and inspected the results.

## Quickstart

{{fix: must be inside main, it used multiprocessing}}

```python
import numpy as np
from heraldo.components.circuits import TwoModeStaticSqueezeOnly
from heraldo.components.targets import SqueezedCatTarget
from heraldo.components.runner import BasinHoppingRunner
from heraldo.utils import db_to_r
from heraldo.analyze import print_results

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
from heraldo.analyze import save_results, load_results

saved_file = save_results(result, filepath=Path("example_results.pkl")) {{fix: must define the path}}
loaded_result = load_results(saved_file, reconstruct=True)  {{fix: must use the path itself not the saved_file output}}
```

`reconstruct=True` rebuilds the `circuit` and `targets` objects from the saved configuration, so you can keep analyzing a run without re-running the optimization.

### 5. Analyze rotation, loss sensitivity, and cutoff sensitivity

```python
from heraldo.analyze import analyze_rotations, analyze_loss, analyze_cutoff

# Optimal phase-space rotation per heralding outcome
analyze_rotations(loaded_result, outcomes=[4, 5])

# Fidelity/probability degradation under photon loss (e.g. 1% and 10%)
analyze_loss(loaded_result, outcomes=[4, 5])

# Sensitivity to Fock-space truncation dimension
analyze_cutoff(loaded_result, low_cutoff=30, high_cutoff=45, outcomes=[4, 5])
```

{{fix: explain what is happening}}

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

{{fix:copy pases the example in here instead of refering to it}}

The complete script is available at `scripts/other/example.py` and combines every step above into a single runnable pipeline: circuit setup → optimization → save/load → rotation, loss, and cutoff analysis → plotting.

## Next Steps

- See the [User Guide](user_guide.md) for a deeper dive into resource multiplexing vs. single-target harvesting.
- See [Circuits Comparison](circuits_comparison.md) for guidance on choosing between circuit architectures.
- See the [API Reference](api/index.md) for full parameter documentation.
