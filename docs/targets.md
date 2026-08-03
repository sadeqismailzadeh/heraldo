# Target State Generators

Target generators produce quantum state vectors (kets) in the Fock basis. They are used as the `target_gens` argument to `BasinHoppingRunner` and determine which non-Gaussian states the optimizer tries to herald.


## Prebuilt Target Generators

| Class | Description | Key Parameters |
|-------|-------------|----------------|
| `CatTarget` | Unsqueezed Schrödinger cat state | `alpha` (amplitude), `p` (parity) |
| `SqueezedCatTarget` | Squeezed Schrödinger cat state | `alpha`, `r` (squeezing), `p` |
| `CubicPhaseTarget` | Displaced cubic phase state | `gamma`, `r`, `alpha` |
| `CubicResourceTarget` | Finite superposition for cubic resource | `a` (scaling) |
| `CoreGKPTarget` | Approximate GKP core state (stellar representation) | `csv_path`, `n_max`, `delta_db`, `mu`, `apply_squeezing` |
| `BinomialCodeTarget` | Binomial code logical codeword | `N` (order), `S` (spacing), `mu` (logical) |

All target classes are defined in `heraldo.components.targets` and inherit from the abstract base class `TargetGenerator`.

### `CatTarget`

Unsqueezed Schrödinger cat state:

```math
\ket{\text{Cat}_p(\alpha)} = \mathcal{N}_p \left( \ket{\alpha} + (-1)^p \ket{-\alpha} \right)
```

- `alpha` (float): coherent amplitude, default 3.0
- `p` (int): parity (0 = even, 1 = odd), default 0

```python
from heraldo.components.targets import CatTarget
target = CatTarget(alpha=2.5, p=1)   # odd cat state
```

### `SqueezedCatTarget`

Squeezed cat state:

```math
\ket{\text{SqCat}_p(\alpha, r)} = \hat{S}(r) \mathcal{N}_p \left( \ket{\alpha} + (-1)^p \ket{-\alpha} \right)
```

- `alpha` (float): coherent amplitude, default 3.0
- `r` (float): squeezing parameter, default 1.38
- `p` (int): parity, default 0

```python
target = SqueezedCatTarget(alpha=np.sqrt(6), r=0.5, p=0)
```

### `CubicPhaseTarget`

Displaced cubic phase state:

```math
\ket{\gamma, r, \alpha} = \hat{D}(\alpha) \exp(i \gamma \hat{Q}^3) \hat{S}(r) \ket{0}
```

- `gamma` (float): cubic nonlinearity, default -0.2
- `r` (float): squeezing, default -0.7
- `alpha` (float): displacement amplitude (imaginary), default 1.25

```python
target = CubicPhaseTarget(gamma=-0.2, r=-0.7, alpha=1.25)
```

### `CubicResourceTarget`

Finite superposition used as a cubic phase resource:

```math
\ket{\psi} \propto \ket{0} + i a \sqrt{1.5} \ket{1} + i a \ket{3}
```

- `a` (float): scaling coefficient, default 0.61

```python
target = CubicResourceTarget(a=0.61)
```

### `CoreGKPTarget`

Approximate Gottesman–Kitaev–Preskill (GKP) core state using the stellar representation. Coefficients are read from a CSV file (see section on CSV path below).

- `csv_path` (str): path to the Tzitrin *et al.* (2020) coefficient file (required)
- `n_max` (int): maximum stellar rank / cutoff (e.g., 2, 4, 6, 8, 10, 12), default 4
- `delta_db` (float): envelope parameter Δ in dB, default 10.0
- `mu` (int): logical state (0 or 1), default 0
- `apply_squeezing` (bool): if `True`, applies the squeezing transformation to the core state; default `False`

```python
target = CoreGKPTarget(csv_path="data/GKP_core_coefficients.csv", n_max=4, mu=0, delta_db=10.0)
```

#### CSV file format

The CSV contains columns:
- `n_max`
- `Delta (dB)`
- `r0 (dB)`, `r1 (dB)` (squeezing for μ=0,1)
- `c0_0`, `c0_2`, `c0_4`, ... (coefficients for μ=0, even Fock states)
- `c1_0`, `c1_2`, ... (coefficients for μ=1)

This file is provided in the repository at `data/GKP_core_coefficients.csv`, sourced from the official Xanadu repository for approximate GKP state preparation ([XanaduAI/approximate-GKP-prep](https://github.com/XanaduAI/approximate-GKP-prep)).

### `BinomialCodeTarget`

Binomial code logical codewords:

```math
\ket{W_\mu} = \frac{1}{\sqrt{2^N}} \sum_{p \equiv \mu \pmod{2}} \sqrt{\binom{N+1}{p}} \ket{p(S+1)}
```

- `N` (int): code order, default 1
- `S` (int): spacing, default 1
- `mu` (int): logical state (0 or 1), default 0

```python
target = BinomialCodeTarget(N=2, S=2, mu=0)
```

## Instantiating Targets

Simply create an instance of any target class with the desired parameters. You can combine multiple targets in a list when passing to the runner.

```python
targets = [
    SqueezedCatTarget(alpha=np.sqrt(6), r=0.5, p=0),
    SqueezedCatTarget(alpha=np.sqrt(6), r=0.5, p=1),
]
```

## Using Targets in Optimization

Pass the target instance(s) to `BasinHoppingRunner` via the `target_gens` argument. The runner will compute the fidelity between each heralded state and every target for each measurement outcome, and use the best match during optimization.

```python
runner = BasinHoppingRunner(
    circuit=circuit,
    target_gens=targets,   # list or single target
    cutoff_dim=30,
)
```

## Creating Custom Target Generators

To define your own target state, inherit from `TargetGenerator` and implement the `get_target_ket` method.

### Example: Custom Fock-state superposition

```python
import numpy as np
from heraldo.components.interfaces import TargetGenerator

class MyCustomTarget(TargetGenerator):
    def __init__(self, coeffs=None):
        self.coeffs = coeffs if coeffs is not None else [1.0, 0.5, 0.2]

    def get_target_ket(self, cutoff_dim: int) -> np.ndarray:
        ket = np.zeros(cutoff_dim, dtype=np.complex128)
        for n, c in enumerate(self.coeffs):
            if n >= cutoff_dim:
                break
            ket[n] = c
        ket /= np.linalg.norm(ket)
        return ket
```

{{apply: create a new file named serilaization reconstructin guide or something like that. then put this guide there. then refere to that file in here}}
## Saving and Loading Custom Targets

For details on serializing and reconstructing custom targets, see the dedicated **[Serialization and Reconstruction Guide](serialization.md)**.

## Reference

- Base class: `heraldo.components.interfaces.TargetGenerator`
- Factory: `heraldo.factory.create_from_config` and `to_config`
- All prebuilt targets are in `heraldo.components.targets`
