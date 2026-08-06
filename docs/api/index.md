# API Reference Overview

The `heraldo` package is structured into core components, patch modules.

- **`heraldo.components.interfaces`**: Abstract base classes defining circuit and target state interfaces.
- **`heraldo.components.circuits`**: Concrete implementations of static continuous-variable photonic circuits.
- **`heraldo.components.targets`**: Quantum target state generators (GKP, Cat, Binomial, Cubic Phase).
- **`heraldo.components.objectives`**: Optimization objective and loss functions.
- **`heraldo.components.runner`**: Static circuit optimization runner based on Basin-Hopping.
- **`heraldo.utils`**: Fidelity metrics, WSL path tools, and conversion utilities.
