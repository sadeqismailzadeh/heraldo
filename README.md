# heraldo

**heraldo** is a Python framework for multi-outcome optimization of static continuous-variable (CV) photonic circuits. It targets Gaussian Boson Sampling (GBS)-like devices — squeezing, displacement, and beam-splitter networks followed by photon-number-resolving (PNR) detection — and optimizes them to herald non-Gaussian quantum states such as Gottesman–Kitaev–Preskill (GKP) core states, Schrödinger cat states, binomial codes, and cubic phase states.

Conventionally, these circuits are optimized to herald a single specific measurement outcome, discarding the potential utility of every other pattern the same physical setup could produce. heraldo implements the multi-outcome strategy described in the paper below: a *beam search* phase autonomously discovers promising heralding patterns without any a priori physical intuition, followed by *fixed-pattern* refinement that either

- **multiplexes** a diverse set of target resource states across different measurement outcomes from a single fixed hardware layout, or
- **harvests** several degenerate outcomes to maximize the production rate of one specific target state.

Circuits are simulated with [Strawberry Fields](https://strawberryfields.ai/) in the Fock basis. State fidelities are evaluated in a rotation-invariant way via a batched FFT over phase-space rotations, since any such rotation of a heralded state can be corrected downstream with a simple optical delay. Several performance patches (a JIT-compiled $O(D^3)$ beam splitter, disabled gate-matrix caching, purity-preserving multimode state preparation) are applied automatically on import to keep large-cutoff, long-running optimizations fast and memory-bounded.

## Features

- **Static spatial GBS-like circuits**: two-, three-, and four-mode CV circuits combining squeezing, displacement, and fixed beam-splitter networks, with PNR detection on all but one (heralded) mode.
- **Multi-outcome optimization**: `BasinHoppingRunner` supports both *beam search* (to discover high-probability heralding patterns) and *fixed-pattern* optimization, enabling **resource multiplexing** (a diverse set of targets across different outcomes) or **single-target probability harvesting** (aggregating degenerate outcomes to boost one target's production rate).
- **Rotation-invariant fidelity**: fidelities are maximized over phase-space rotation via a batched FFT, so a heralded state isn't penalized for a rotation a downstream phase shift can correct.
- **Target state generators**: pre-built generators for GKP core states, Schrödinger cat states, binomial codes, and cubic phase states, plus a simple interface for custom targets.
- **Serialization & reconstruction**: optimization results can be saved to disk and later reloaded with full object reconstruction (circuit, targets, loss function) for further analysis.
- **Post-optimization analysis**: utilities to summarize results, plot Wigner functions and Fock distributions, check rotation invariance, and stress-test circuits against photon loss and Fock-cutoff truncation.
- **JIT-compiled performance patches**: an $O(D^3)$ Numba-compiled beam-splitter implementation, disabled Strawberry Fields gate caching, and purity-preserving multimode state preparation, all applied automatically on `import heraldo`.

## How the Optimization Works

Instead of tuning a circuit for one heralding outcome, `heraldo` optimizes over a *set* of outcomes $S = \{\mathbf{n}_k\}$ and target states $\{\ket{\phi_i}\}$ at once, using a two-phase strategy:

1. **Beam search** (exploratory): when the useful heralding patterns aren't known in advance, the optimizer dynamically tracks the top-`beam_width` highest-probability ancilla detection events at each step and scores them with a non-linear objective that suppresses low-fidelity patterns and sharpens the gradient around promising ones.
2. **Fixed-pattern refinement**: once a set of patterns has been identified (via beam search, or from physical intuition), the optimizer switches to a direct objective that maximizes probability and fidelity summed over exactly that fixed set — either across several *different* target states (**resource multiplexing**), or across several outcomes that all herald the *same* target (**single-target probability harvesting**).

Within fixed-pattern optimization, two regimes trade off probability against fidelity: a **capped** loss (`FixedPatternCappedLoss`) that saturates fidelity at a baseline so the optimizer shifts focus to maximizing success probability once that baseline is met, and a **free** loss (`FixedPatternFreeLoss`) that leaves fidelity uncapped. See [`docs/objectives.md`](docs/objectives.md) for the full loss functions and [`docs/runner.md`](docs/runner.md) for how they plug into `BasinHoppingRunner`.

## Installation

`heraldo` requires **Python 3.13** and is managed with [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/sadeqismailzadeh/GKP_state_code
cd GKP_state_code
uv sync
```

On Windows, you can instead double-click `install.bat`, which installs `uv` (if missing) and runs `uv sync` for you.

Verify the installation:

```bash
uv run python -c "import heraldo; print('heraldo installed successfully')"
```

See [`docs/installation.md`](docs/installation.md) for full details, including manual `pip`/virtualenv usage.

## Quickstart

```python
import os
# Set thread limits *before* importing numpy/scipy/heraldo to avoid BLAS
# thread oversubscription during parallel basin-hopping runs.
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

import numpy as np
from heraldo.components.circuits import TwoModeStaticSqueezeOnly
from heraldo.components.targets import SqueezedCatTarget
from heraldo.components.runner import BasinHoppingRunner
from heraldo.utils import db_to_r
from heraldo.analyze import print_results


def main():
    # 12 dB source squeezing, two-mode circuit (Mode 0 = heralded output, Mode 1 = ancilla)
    circuit = TwoModeStaticSqueezeOnly(clip_size=db_to_r(12.0), measure_fock_cutoff=30)

    # Herald even and odd squeezed cat states
    targets = [
        SqueezedCatTarget(alpha=np.sqrt(6), r=0.5, p=0),
        SqueezedCatTarget(alpha=np.sqrt(6), r=0.5, p=1),
    ]

    runner = BasinHoppingRunner(circuit=circuit, target_gens=targets, cutoff_dim=30)
    result = runner.run(n_iter=10)

    print_results(result)


if __name__ == "__main__":
    main()
```

`runner.run(...)` must be called from inside `if __name__ == "__main__":`, since `BasinHoppingRunner` uses `multiprocessing` for its parallel basin-hopping runs. See [`docs/quickstart.md`](docs/quickstart.md) for a full walkthrough including saving/loading results, rotation and loss-sensitivity analysis, and Wigner-function plotting.

## Documentation

Full documentation — circuit models, target state generators, optimization objectives, the runner, serialization, post-optimization analysis, and internals — lives in [`docs/`](docs/) and can be built locally:

```bash
cd docs
pip install -r requirements.txt
sphinx-build -b html . _build/html
```

(or just double-click `docs/build_docs.bat` on Windows).

## Citing heraldo

heraldo implements the multi-outcome optimization strategy introduced in:

> S. Ismailzadeh and B. Abedi Ravan, *"Multi-Outcome Circuit Optimization for Enhanced Non-Gaussian State Generation."*

If heraldo is useful in your research, please cite the paper above alongside the software itself:

```bibtex
@article{ismailzadeh_multioutcome,
  title   = {Multi-Outcome Circuit Optimization for Enhanced Non-Gaussian State Generation},
  author  = {Ismailzadeh, Sadeq and Abedi Ravan, B.},
}

@software{ismailzadeh_code,
  title   = {heraldo},
  author  = {Ismailzadeh, Sadeq},
  url     = {https://github.com/sadeqismailzadeh/GKP_state_code},
}
```

## License

MIT © 2025 Sadeq Ismailzadeh. See the license header in [`heraldo/__init__.py`](heraldo/__init__.py).
