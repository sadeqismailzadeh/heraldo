# Optimization Runner

The `BasinHoppingRunner` class is the central optimization engine in `heraldo`. It performs global optimization of static spatial photonic circuits using the **basin‑hopping** algorithm. The runner evaluates candidate parameter vectors, simulates the circuit, extracts heralded states, computes fidelities against target states, and minimizes a customizable loss function.

---


### Constructor Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `circuit` | `StaticCircuit` | — (required) | The static circuit instance to optimize. Its `parameter_names`, `parameter_bounds`, `run_circuit`, and `get_measurement_specs` are used directly by the runner. See [Circuits](circuits.md)|
| `target_gens` | `TargetGenerator` or `list[TargetGenerator]` | — (required) | One or more target state generators. A single instance is automatically wrapped in a list internally. Each generator's `get_target_ket(cutoff_dim)` is called once up front to produce the target kets used throughout optimization. See [Targets](targets.md)|
| `cutoff_dim` | `int` | — (required) | Fock-space truncation dimension used both to generate target kets and to configure the Strawberry Fields `"fock"` backend during simulation. |
| `beam_width` | `int` | `20` | Maximum number of top-probability ancillary outcomes retained per evaluation during **beam search**. Only relevant when `measurement_patterns=None`. |
| `penalty_strength` | `float` | `0.1` | Multiplier applied to the Fock-truncation-error penalty term added to the loss. Higher values penalize parameter regions where probability mass leaks outside the truncated Hilbert space. |
| `measurement_patterns` | see [Specifying Measurement Patterns](patterns.md) | `None` | If `None`, the runner performs beam search. If provided, the runner performs **fixed-pattern optimization** restricted to the given outcome(s). |
| `num_parallel_runs` | `int` | `4` | Number of independent basin-hopping runs (each with its own random seed and initial guess) launched in parallel. The run with the lowest final loss is returned. |
| `num_processes` | `int` | `4` | Number of worker processes used by `multiprocessing.Pool` when more than one parallel run is requested. Ignored when the effective number of parallel runs is `1`. |
| `method` | `str` | `"L-BFGS-B"` | Local minimizer algorithm name passed to basin-hopping. |
| `base_seed` | `int` or `None` | `None` | Base random seed for reproducible multi-run optimization runs. |
| `loss_fn` | `ObjectiveFunction`, subclass, or `None` | `None` | loss function (objective function). If `None`, the runner automatically selects `FixedPatternCappedLoss()` when `measurement_patterns` is set, or `BeamSearchLoss()` otherwise. See [Optimization Objectives](objectives.md). |
| `callback` | `callable` or `None` | `None` | Optional function `callback(x, f, accept)` invoked after every basin-hopping step (in addition to the runner's built-in progress logging). |
| `phase_lock` | `bool` | `False` | If `True`, enforces a single global phase-space rotation shared across all accepted measurement branches when computing fidelities, rather than optimizing the rotation independently per branch. |

---

## Functionality

Given a circuit, a set of target states, and a Fock-space cutoff dimension, `BasinHoppingRunner`:

1. Draws a random initial parameter vector `x0` within the circuit's `parameter_bounds`.
2. Repeatedly simulates the circuit (via Strawberry Fields) at candidate parameter vectors, using SciPy's basin-hopping algorithm to propose new candidates and a local minimizer (`L-BFGS-B` by default) to refine each candidate.
3. For every simulated state, either:
   - runs **beam search**, dynamically keeping the `beam_width` highest-probability ancillary measurement outcomes (when `measurement_patterns=None`), or
   - runs **fixed-pattern optimization**, evaluating only the outcomes you specify via `measurement_patterns`.
4. Scores each candidate using an `ObjectiveFunction` (`BeamSearchLoss` or `FixedPatternCappedLoss` by default, depending on the mode above — see [Optimization Objectives](objectives.md)).
5. Optionally repeats this whole process across several **independent parallel runs** (different random seeds, run in separate worker processes) and keeps the best one.
6. Returns the best-found parameters together with the resulting fidelities, probabilities, and per-outcome branch details.

---

## Instantiating a Runner

```python
from heraldo.components.runner import BasinHoppingRunner
from heraldo.components.circuits import TwoModeStaticSqueezeOnly
from heraldo.components.targets import SqueezedCatTarget
from heraldo.utils import db_to_r

circuit = TwoModeStaticSqueezeOnly(clip_size=db_to_r(12.0), measure_fock_cutoff=30)
target = SqueezedCatTarget(alpha=3.0, r=0.5, p=0)

runner = BasinHoppingRunner(
    circuit=circuit,
    target_gens=target,
    cutoff_dim=30,
)
```


## Running the Optimization: `run()`

### `run()` Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `n_iter` | `int` | `20` | Number of basin-hopping iterations performed **per parallel run**. Each iteration proposes a new candidate (via a random perturbation) and refines it with the local minimizer. |

### Execution Modes

- **`num_parallel_runs <= 1`**: the optimization runs synchronously in the current process (no `multiprocessing.Pool` involved).
- **`num_parallel_runs > 1`** (the default, since the constructor default is `4`): the runner launches a `multiprocessing.Pool` with `min(num_parallel_runs, num_processes)` workers, runs all seeds in parallel via `pool.starmap`, and returns the result with the lowest `loss` among the successful runs.

---

## Set thread limits

To prevent thread oversubscription during parallel optimization runs, ensure thread-limiting environment variables (`OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS`, `VECLIB_MAXIMUM_THREADS`, `NUMEXPR_NUM_THREADS`) are set to `1` at the very top of your entry-point script before importing `numpy`, `scipy`, or `heraldo`. For a detailed explanation of thread limits and backend behavior, see [Internals: Thread Limits & Backend Patches](internals.md).

## Required: call `run()` from `if __name__ == "__main__":`


because `BasinHoppingRunner.run()` uses Python's `multiprocessing` module to execute parallel basin-hopping searches, and platforms that use the `spawn` start method (Windows and macOS) re-import your main module in every child process, **your call to `runner.run(...)` must be guarded by `if __name__ == "__main__":`**. Calling `runner.run(...)` at top-level module scope will cause each worker process to re-execute your script recursively upon import, raising a `RuntimeError`.



```python
# Correct
import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

from heraldo.components.runner import BasinHoppingRunner
from heraldo.components.circuits import TwoModeStaticSqueezeOnly
from heraldo.components.targets import SqueezedCatTarget
from heraldo.utils import db_to_r

def main():
    circuit = TwoModeStaticSqueezeOnly(clip_size=db_to_r(12.0), measure_fock_cutoff=30)
    target = SqueezedCatTarget(alpha=3.0, r=0.5, p=0)

    runner = BasinHoppingRunner(
        circuit=circuit,
        target_gens=target,
        cutoff_dim=30,
    )
    result = runner.run(n_iter=10)

if __name__ == "__main__":
    main()
```

```python
# Incorrect — will misbehave / error on spawn platforms
from heraldo.components.runner import BasinHoppingRunner
from heraldo.components.circuits import TwoModeStaticSqueezeOnly
from heraldo.components.targets import SqueezedCatTarget
from heraldo.utils import db_to_r

circuit = TwoModeStaticSqueezeOnly(clip_size=db_to_r(12.0), measure_fock_cutoff=30)
target = SqueezedCatTarget(alpha=3.0, r=0.5, p=0)

runner = BasinHoppingRunner(
    circuit=circuit,
    target_gens=target,
    cutoff_dim=30,
)

result = runner.run(n_iter=10)  # module-level call
```

This applies even if you pass `num_parallel_runs=1` in some calls, since the constructor default (`4`) means parallel execution is the common case; keeping `run()` inside a guarded function is a safe habit regardless of the parallelism level used.

---

## Return Value

`run()` returns a `dict` with the following keys:

| Key | Type | Description |
|---|---|---|
| `x` | `np.ndarray` | Optimized parameter vector for `circuit`. |
| `loss` | `float` | Final loss value (`-objective_score + penalty_strength * truncation_error`) of the best run. |
| `objective_score` | `float` | Raw score returned by the loss function (before negation/penalty). |
| `branches` | `list[dict]` | Per-outcome details for the best run, each entry containing `outcome` (tuple of detected photon numbers), `prob`, `fidelity`, and `target_idx` (index into `target_gens` of the best-matching target). |
| `total_probability` | `float` | Sum of probabilities across all branches with non-zero photon counts. |
| `duration` | `float` | Wall-clock time in seconds for the full `run()` call. |
| `message` | `str` | Human-readable status message (e.g. `"Best of 4 parallel runs"` or the SciPy termination message for a single run). |
| `circuit`, `targets`, `loss_fn` | `StaticCircuit` / `list` / `ObjectiveFunction` | Live object instances of the circuit, target generator(s), and loss function used during optimization. |
| `circuit_config`, `target_configs`, `loss_config`, `runner_config` | `dict` / `list[dict]` | Serializable configuration metadata (via `heraldo.serialization.to_config`) enabling later reconstruction — see [Serialization and Reconstruction Guide](serialization.md). |
| `run_results` *(only if `num_parallel_runs > 1`)* | `list[dict]` | Raw result dict from every parallel run (successful or not), useful for inspecting run-to-run variance. |
| `best_run_idx` *(only if `num_parallel_runs > 1`)* | `int` | Index (into the internal seed list) of the run that produced `x`/`loss`. |

If all evaluated outcomes for a given parameter vector have zero probability, `evaluate_circuit` (used internally) returns a large fallback loss of `100.0` rather than raising an error, keeping the optimizer well-defined across the whole search space.

---

## Full Example

```python
import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

from heraldo.components.circuits import TwoModeStaticSqueezeOnly
from heraldo.components.targets import SqueezedCatTarget
from heraldo.components.runner import BasinHoppingRunner
from heraldo.utils import db_to_r
from heraldo.analyze import print_results
import numpy as np


def main():
    circuit = TwoModeStaticSqueezeOnly(clip_size=db_to_r(12.0), measure_fock_cutoff=30)

    targets = [
        SqueezedCatTarget(alpha=np.sqrt(6), r=0.5, p=0),
        SqueezedCatTarget(alpha=np.sqrt(6), r=0.5, p=1),
    ]

    runner = BasinHoppingRunner(
        circuit=circuit,
        target_gens=targets,
        cutoff_dim=30,
        beam_width=20,
        penalty_strength=0.1,
        num_parallel_runs=4,
        num_processes=4,
        method="L-BFGS-B",
        base_seed=42,
    )

    result = runner.run(n_iter=10)
    print_results(result)


if __name__ == "__main__":
    main()
```

## Summary

- `BasinHoppingRunner` optimizes a `StaticCircuit`'s parameters against one or more `TargetGenerator` targets, using either beam search or fixed-pattern evaluation.
- Configure simulation and search behavior at construction time (`cutoff_dim`, `beam_width`, `measurement_patterns`, `loss_fn`, `phase_lock`, parallelism settings).
- Execute the optimization run itself via `run()`.
- Always call `run()` from inside `if __name__ == "__main__":`, since it relies on `multiprocessing` whenever more than one parallel run is used.
- The returned dictionary contains the optimized parameters, loss/fidelity/probability summary, per-branch details, and serializable configuration metadata for later reconstruction via [`load_results`](serialization.md).
