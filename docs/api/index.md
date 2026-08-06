# API Reference

Welcome to the **heraldo** API reference documentation. This section provides detailed auto-generated module references, class definitions, and function signatures for all public components of the framework.

```{toctree}
:maxdepth: 2

components
analyze
serialization
utils
patches
```

## Module Overview

### [Core Components](components.md)
Contains abstract base interfaces, circuit models, target state generators, objective loss functions, and the optimization runner:
- **`heraldo.components.interfaces`**: Abstract base classes (`ObjectiveFunction`, `StaticCircuit`, `TargetGenerator`).
- **`heraldo.components.circuits`**: Concrete multi-mode static circuit implementations (`TwoModeStaticGeneral`, `TwoModeStaticSqueezeOnly`, `ThreeModeStaticGeneral`, `ThreeModeStaticSqueezeOnly`, `FourModeStaticSqueezeOnly`).
- **`heraldo.components.targets`**: Prebuilt non-Gaussian target state generators (`CatTarget`, `SqueezedCatTarget`, `CubicPhaseTarget`, `CubicResourceTarget`, `CoreGKPTarget`, `BinomialCodeTarget`).
- **`heraldo.components.objectives`**: Optimization objective and loss classes (`BeamSearchLoss`, `FixedPatternCappedLoss`, `FixedPatternFreeLoss`).
- **`heraldo.components.runner`**: The `BasinHoppingRunner` class and circuit evaluation function (`evaluate_circuit`).

### [Analysis & Visualization](analyze.md)
Post-optimization tools for inspecting results, plotting Wigner functions, assessing phase rotation invariance, and testing against photon loss and Fock space truncation:
- **`heraldo.analyze`**: Top-level API surfacing `print_results`, `plot_outcomes`, `plot_wigner`, `analyze_rotations`, `analyze_loss`, and `analyze_cutoff`.

### [Serialization & Reconstruction](serialization.md)
Functions for saving run results to disk, converting objects to/from configuration dictionaries, and reloading results with full object reconstruction:
- **`heraldo.serialization`**: `save_results`, `load_results`, `reconstruct_objects`, `create_from_config`, `to_config`.

### [Utilities](utils.md)
Helper functions for fidelity metrics, squeezing parameters, and path utilities:
- **`heraldo.utils`**: `fidelity_pure_state`, `fidelity_max_rotation`, `db_to_r`, `windows_to_wsl_path`.

### [Performance Patches](patches.md)
Backend optimizations applied automatically on import to optimize performance and memory usage during Strawberry Fields simulations:
- **`heraldo.patches`**: JIT-compiled $O(D^3)$ beam-splitter patch, gate-caching disabler, and purity-preserving state preparation patch.
