# Quickstart & Usage

## 1. Defining a Target State

Target states implement the `TargetGenerator` interface:

```python
from heraldo.components.targets import SqueezedCatTarget, CoreGKPTarget

# Create a squeezed cat target state
cat_target = SqueezedCatTarget(alpha=2.45, r=0.5, p=0)
ket = cat_target.get_target_ket(cutoff_dim=30)
```

## 2. Setting Up Static Spatial Circuits

Use `StaticCircuit` implementations to define optical setups:

```python
from heraldo.components.circuits import TwoModeStaticGeneral
from heraldo.components.runner import BasinHoppingRunner

circuit = TwoModeStaticGeneral(clip_size=2.0)
runner = BasinHoppingRunner(
    circuit=circuit,
    target_gens=[cat_target],
    cutoff_dim=30,
    beam_width=5
)

# Run optimization
result = runner.run(n_iter=20)
print("Optimal parameters:", result["x"])
print("Achieved Loss:", result["loss"])
```

## 3. Time-Domain Multiplexed Circuits

For time-domain loop configurations, use `TimeMultiplexedCircuit`:

```python
from heraldo.experimental.time_circuits import TwoModeTimeDomainGeneral
from heraldo.experimental.time_runner import BasinHoppingRunner as TimeRunner

circuit = TwoModeTimeDomainGeneral(steps=3)
runner = TimeRunner(
    circuit=circuit,
    target_gens=[cat_target],
    cutoff_dim=30,
    beam_width=5
)

result = runner.run(n_iter=10)
```