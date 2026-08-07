# Optimization Objectives (Loss Functions)

**Objectives** (loss functions) define the metric that the optimizer minimizes during circuit parameter search. They quantify how well a candidate set of parameters produces heralded states that match your target states, combining state fidelity and success probability into a single scalar.

Objectives are implemented as subclasses of `ObjectiveFunction` (defined in `heraldo.components.interfaces`). They receive arrays of probabilities and fidelities for the measurement branches that survive the beam‑search or fixed‑pattern filter, and return a score that the optimizer attempts to maximize (internally, the runner minimizes `-score` plus penalties).

---

## Prebuilt Objective Classes

All prebuilt objective **classes** are located in `heraldo.components.objectives`:

| Class                     | Description                                                                                     |
|---------------------------|-------------------------------------------------------------------------------------------------|
| `BeamSearchLoss`          | Non‑linear loss for the exploratory **beam search** phase (filters low‑fidelity patterns).      |
| `FixedPatternCappedLoss`  | Capped‑fidelity loss for **fixed‑pattern** optimization (saturates at a fidelity cap).          |
| `FixedPatternFreeLoss`    | Un‑capped (free) fidelity loss for **fixed‑pattern** optimization.                              |

### Pre‑instantiated Instances

For convenience, default instances of the above classes are provided:

| Instance                     | Description                                                                                     |
|------------------------------|-------------------------------------------------------------------------------------------------|
| `beam_search_loss_fn`        | Pre‑instantiated `BeamSearchLoss` with default parameters.                                      |
| `fixed_pattern_capped_loss_fn`| Pre‑instantiated `FixedPatternCappedLoss` with default parameters.                              |
| `fixed_pattern_free_loss_fn`  | Pre‑instantiated `FixedPatternFreeLoss` with default parameters.                                |
| `default_loss_fn`            | Alias for `beam_search_loss_fn` (used when no loss is explicitly provided).                     |

### `BeamSearchLoss`

Designed for the **beam search** phase, where the set of evaluated measurement patterns is **dynamic**. The loss is a non‑linear function that amplifies gradients for high‑fidelity patterns and suppresses low‑fidelity ones:

$$
L_{\text{beam}} = -\left[ \log\left( \mathcal{S} + \delta \right) + \lambda \mathcal{S} \right],
$$

where

$$
\mathcal{S} = \sum_k p_k \left( \tilde{F}_k^2 \Lambda_k \right)^4,
$$

with `Λ_k = log(1 - F̃_k) / log(ε)` and `F̃_k = min(F_k, 1-ε)`.  
The parameters `ε` (epsilon), `δ` (delta), and `λ` (lam) control the sensitivity.

**Constructor:**
```python
BeamSearchLoss(epsilon=2e-2, delta=1e-72, lam=1e4)
```

### `FixedPatternCappedLoss`

Used during **fixed‑pattern** optimization when you want to cap the fidelity contribution to a baseline value, allowing the optimizer to focus on increasing the overall success probability once the fidelity exceeds the cap:

$$
L_{\text{fixed}} = -\sum_k \left( \alpha p_k + \min(F_k, F_{\text{cap}}) \right).
$$

**Constructor:**
```python
FixedPatternCappedLoss(f_cap=0.95, alpha=None)
```
- `f_cap`: saturation value for fidelity (default `0.95`).
- `alpha`: weight on probability. If `None`, set to `len(probs)` (number of branches) automatically.

### `FixedPatternFreeLoss`

Un‑capped fixed‑pattern loss. It encourages both fidelity and probability without saturation:

$$
L_{\text{fixed}} = -\sum_k \left( \alpha p_k + F_k \right).
$$

**Constructor:**
```python
FixedPatternFreeLoss(alpha=None)
```
- `alpha`: weight on probability. If `None`, set to `0.1 * len(probs)`.

---


## Using Objectives in Optimization

You can pass an objective instance (or a pre‑instantiated function) to `BasinHoppingRunner` via the `loss_fn` argument. If omitted, the runner automatically selects the appropriate default:

- If you provide `measurement_patterns` (fixed‑pattern mode) → `FixedPatternCappedLoss()` is used.
- Otherwise (beam‑search mode) → `BeamSearchLoss()` is used.

**Example – Beam search with default loss:**
```python
from heraldo.components.runner import BasinHoppingRunner
from heraldo.components.circuits import TwoModeStaticSqueezeOnly
from heraldo.components.targets import CatTarget

def main():
    circuit = TwoModeStaticSqueezeOnly()
    targets = [CatTarget(alpha=3.0, p=0)]
    runner = BasinHoppingRunner(circuit, targets, cutoff_dim=30)
    # No loss_fn passed → uses beam_search_loss_fn automatically
    result = runner.run(n_iter=10)

if __name__ == "__main__":
    main()
```

**Example – Fixed‑pattern with capped loss:**
```python
from heraldo.components.objectives import fixed_pattern_capped_loss_fn

def main():
    runner = BasinHoppingRunner(
        circuit,
        targets,
        cutoff_dim=30,
        measurement_patterns=[[4], [5]],   # fixed patterns
        loss_fn=fixed_pattern_capped_loss_fn
    )
    result = runner.run(n_iter=10)

if __name__ == "__main__":
    main()
```

**Example – Customizing loss parameters:**
```python
from heraldo.components.objectives import FixedPatternFreeLoss

def main():
    custom_loss = FixedPatternFreeLoss(alpha=2.0) 
    runner = BasinHoppingRunner(..., loss_fn=custom_loss)
    result = runner.run(n_iter=10)

if __name__ == "__main__":
    main()
```

---



## Creating Custom Objectives

To define your own loss function, inherit from `ObjectiveFunction` and implement the `_compute` method. The method receives:

- `probs` : `np.ndarray` of shape `(num_branches, )` – probabilities of each surviving measurement branch.
- `fidelities` : `np.ndarray` – state fidelities (1D or 2D as described below).

**Return type:**  
- If `fidelities` is 1D, return a scalar (`float`).  
- If `fidelities` is 2D, return a 1D array of length `num_phases` (one score per phase angle). The runner will later pick the phase that yields the best (maximum) score.  

The runner **negates** the returned score internally because it minimizes the loss; your `_compute` method should produce a positive score that increases with quality.


**Example:**
```python
class CombinedLoss(ObjectiveFunction):
    def __init__(self, alpha=0.5):
        self.alpha = alpha

    def _compute(self, probs, fidelities):
        return np.sum(probs * fidelities, axis=0) + self.alpha * np.sum(probs, axis=0)
```
---


## Fidelity Array Shapes: Phase-Lock vs. Phase-Free

To be able to create custom objectives evectively, it's important to understand the shape of the arrays passed to the objective's `_compute` method. The shape of the `fidelities` array depends on the `phase_lock` parameter of `BasinHoppingRunner` (or `evaluate_circuit`).

- **Phase‑free (`phase_lock=False`, default):**  
  For each measurement branch, we find the optimal phase‑space rotation independently. The fidelity is then a single number per branch (the maximum over all rotations).  
  `fidelities.shape == (num_branches, )` → 1D array.

- **Phase‑lock (`phase_lock=True`):**  
  A *global* phase rotation is enforced across all branches. The optimizer considers a discrete set of phase angles (e.g., 256 points via FFT). For each branch, we have an array of fidelities evaluated at each phase angle.  
  `fidelities.shape == (num_branches, num_phases)` → 2D array.


To simplify arithmetic with `probs` (which is always 1D of length `num_branches`), the runner **automatically reshapes** `probs` to a column vector (shape `(num_branches, 1)`) when `fidelities` is 2D, so that operations like `probs * fidelities` and `probs + fidelities` broadcast correctly. In your `_compute` method, you can safely write expressions like:

```python
score = np.sum(probs * fidelities, axis=0)
```

or, using addition:

```python
score = np.sum(probs + fidelities, axis=0)
```

This works regardless of whether `fidelities` is 1D or 2D, because:
- if `fidelities` is 1D, `probs` remains 1D and the sum yields a scalar.
- if `fidelities` is 2D, `probs` is automatically expanded to shape `(num_branches, 1)` and the sum over `axis=0` yields a 1D array of length `num_phases`.


if `fidelities` is 2D (phase‑locked mode), the runner reshapes `probs` to a column vector of shape `(num_branches, 1)`. This enables NumPy broadcasting: operations like `probs * fidelities` or `probs + fidelities` are performed element‑wise, where the single probability value for each branch is applied to all phase angles of that branch. The result is an array of shape `(num_branches, num_phases)`. Summing over `axis=0` then collapses the branch dimension, yielding a 1D array of length `num_phases` — one score per phase angle. For the 1D case, `probs` remains a 1D array, and summing over `axis=0` (the only axis) produces a scalar score

Your objective must return a **scalar** when `fidelities` is 1D, and a **1D array** (length = number of phases) when `fidelities` is 2D.

---

## Serialization and Reconstruction

All objective classes are serializable via the `heraldo.serialization` module. When you save an optimization result (using `save_results`), the objective configuration is stored as a dictionary. On loading with `load_results(..., reconstruct=True)`, the objective instance is re‑created from that config.

To ensure your custom objective can be reconstructed:

1. **Store all constructor parameters as attributes** with the same name.
2. **Avoid non‑picklable or non‑serializable members** (like file handles, large arrays). Keep parameters simple (ints, floats, strings).
3. **Define your class in a module** that can be imported when loading (not in the main script).

For full details, see the **[Serialization and Reconstruction Guide](serialization.md)**.
---

## Summary

- Objectives are callable classes that turn probabilities and fidelities into an optimisation score.
- Use pre‑built objectives for standard workflows, or subclass `ObjectiveFunction` for custom needs.
- The shape of `fidelities` changes with `phase_lock`; ensure your `_compute` handles both cases by using `axis=0` and returning a scalar or 1D array accordingly.
- Objectives are serializable; follow the naming and import rules for seamless reconstruction.

For further details on the internal `_compute` mechanics, refer to the source code in `heraldo/components/objectives.py`.
