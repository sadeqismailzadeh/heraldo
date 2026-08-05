# Serialization and Reconstruction Guide

`heraldo` provides utilities to save optimization results to disk and later reload them with full object reconstruction. This is essential for post‑processing, analysis, and sharing results.

## Saving Results

Use `heraldo.serialization.save_results` to save the dictionary returned by the runner:

```python
from heraldo.serialization import save_results

save_results(result, filepath="my_results.pkl")
```

Note that `save_results` automatically pops live object instances (such as `circuit`, `targets`, and `loss_fn`) from the result dictionary before saving to ensure clean serialization. The saved file retains serializable configuration metadata (`circuit_config`, `target_configs`, `loss_config`), which is used to reconstruct the original objects when reloaded.


## Loading Results with Reconstruction

Use `heraldo.serialization.load_results` with `reconstruct=True` to rebuild the original objects from the stored metadata:

```python
from heraldo.serialization import load_results

loaded = load_results("my_results.pkl", reconstruct=True)
```

When `reconstruct=True`, the loader uses `heraldo.serialization.create_from_config` to instantiate:

- The `StaticCircuit`
- Each `TargetGenerator`
- The `ObjectiveFunction` (loss function)

This allows you to run further analysis (e.g., `analyze_loss`, `plot_outcomes`) without having to manually recreate the configuration.

## Requirements for Custom Objects

To ensure your custom classes can be reconstructed, follow these guidelines:

1. **Attribute names must match `__init__` parameters**  
   The factory collects attribute values that correspond to constructor arguments. For example:

   ```python
   def __init__(self, alpha=1.0):
       self.alpha = alpha   # same name as parameter
   ```

2. **Define the class in its own module**  
   Do not define the class inside your main script. Place it in a separate `.py` file (e.g., `my_targets.py`). This ensures the class can be imported when the pickle is loaded.

3. **Make the module importable**  
   The module containing your class must be on the Python path when loading. The easiest way to ensure this is to place both the module and your execution script in the same directory. If you are working in a project, ensure the directory is in `sys.path` or use a package structure.

### Example

Suppose you have a custom target in `my_targets.py`:

```python
# my_targets.py
import numpy as np
from heraldo.components.interfaces import TargetGenerator

class MyCustomTarget(TargetGenerator):
    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def get_target_ket(self, cutoff_dim: int) -> np.ndarray:
        ket = np.zeros(cutoff_dim, dtype=np.complex128)
        ket[0] = 1.0
        ket[1] = self.alpha
        ket /= np.linalg.norm(ket)
        return ket
```

Then in your optimization script:

```python
from my_targets import MyCustomTarget
target = MyCustomTarget(alpha=2.0)
runner = BasinHoppingRunner(..., target_gens=[target])
result = runner.run()
save_results(result, "results.pkl")
```

Later, when loading:

```python
loaded = load_results("results.pkl", reconstruct=True)
# loaded now contains a MyCustomTarget instance with alpha=2.0
```

## Supported Object Types

The `heraldo.serialization` module can reconstruct:
- Pre‑built classes from `heraldo.components.circuits`, `heraldo.components.targets`, and `heraldo.components.objectives`
- Custom classes that inherits from `StaticCircuit`, `TargetGenerator`, or `ObjectiveFunction` as long as they follow the rules above


For more details, see the `heraldo.serialization` module documentation.
