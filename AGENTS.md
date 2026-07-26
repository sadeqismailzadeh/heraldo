# AGENTS.md

## Project Overview

`quantum_agent` is a Python library for optimizing time-domain multiplexed quantum circuits using Strawberry Fields (SF) as the simulation backend. The core task: find circuit parameters that produce target non-Gaussian quantum states (GKP, binomial codes, cat states, cubic phase states) with high fidelity via basin-hopping optimization with beam search.

## Environment Setup

- **Virtualenv is required**: Dependencies live in `~/vnev/`, not in the default Python. See `VERY IMPORTANT.txt`.
- Install the package: `pip install -e .` (uses `setup.py` with `find_packages`).
- Core dependency: `strawberryfields`. Also uses `qutip`, `scipy`, `numpy`, `pandas`, `numba`, `scikit-learn`.

## Running

- **Optimization**: `python scripts/optimize/run_time_optimization.py` — this is the main entrypoint. Configurable targets, circuits, and measurement patterns are defined in `main()`.
- **Tests**: `python -m unittest tests/` or `python tests/test_prepare_multimode_patch.py`. Single test file covering the multimode preparation patch.
- **Benchmarks**: `cd benchmarks && python benchmark_gates.py` — NOTE: this script imports `monitored_loss_measure_fock_patch` which is missing from the repo. May fail at runtime.
- No formal lint, typecheck, or formatter is configured. No CI workflows exist.

## Architecture

```
quantum_agent/
  components/targets.py    — Target state generators (GKP, cat, binomial, cubic, trisqueeze, quadsqueeze)
  optimization/
    time_interfaces.py     — TimeMultiplexedCircuit ABC (steps, parameter bounds, map_parameters)
    time_circuits.py       — Concrete circuits: TwoModeTimeDomainGadget, ThreeModeTimeDomainGadget, etc.
    time_runner.py         — BasinHoppingRunner: beam search + basin-hopping optimization loop
  patches/
    prepare_multimode_patch.py  — Patches SF's fock backend to preserve purity for unentangled partial state prep
    beamsplitter_patch.py       — JIT-compiled beamsplitter tensor (numba)
    sf_operations_no_cache.py   — Disables SF fock backend caching
  factory.py               — Dynamic class instantiation from config dicts
  utils.py                 — Fidelity functions, Wigner helpers, scipy simps compat patch
scripts/
  optimize/run_time_optimization.py  — Main experiment runner
  analysis/                — Post-hoc analysis scripts
  plotting/                — Paper figure generation
data/
  GKP_core_coefficients.csv — GKP target state coefficients
results/                   — Timestamped optimization run directories (gitignored)
```

## Key Conventions

- **Config-driven instantiation**: Circuits and targets are created via `create_from_config(dict, module)` using `{'class_name': '...', 'params': {...}}` dicts. This is the standard pattern throughout `scripts/optimize/`.
- **Scipy compat patch**: `scipy.integrate.simps` was renamed to `simpson` in newer scipy. The codebase monkey-patches this back — repeated in `__init__.py`, `utils.py`, and several patches. Do not remove these.
- **Measurement patterns**: Fixed patterns passed as `np.ndarray` with shape `(n_sequences, steps, n_meas_modes)`. Beam search mode uses `None`.
- **Results naming**: `opt_{CircuitTag}_{TargetTag}_{ISO_timestamp}/` in `results/`. Tags: Gadget, Sq2, Sq3, Sq4 for circuits; GKP, Cat, SqCat, Bin, Cub for targets.
- **Fidelity metric**: Uses `fidelity_max_rotation` (FFT-based global phase optimization) throughout the optimizer.

## Gotchas

- `__init__.py` has `from sympy import true` at line 25 — this is an unused leftover import, not a functional dependency.
- The `time_runner.py` evaluation is vectorized across branches but serial across SF engine runs. Multiprocessing is used at the outer basin-hopping level, not inside `evaluate_time_domain_circuit`.
- The `Clip size` parameter for circuit parameter bounds defaults to the squeezing level `r` (in dB-converted units).
