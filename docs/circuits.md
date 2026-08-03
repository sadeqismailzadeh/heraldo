# Circuit Models

This guide covers how to instantiate, use, and extend static continuous‑variable (CV) photonic circuits in `heraldo`. Circuits define the optical parameters – squeezing, displacement, beam‑splitter networks, and loss channels – that are optimized to herald non‑Gaussian states.

These circuits are Gaussian Boson Sampling (GBS)‑like devices: Gaussian operations (squeezing, displacement, and passive linear optics) are applied to input modes, and non‑Gaussianity is induced by photon‑number‑resolving measurements on ancillary modes.

All static circuits inherit from the abstract base class `StaticCircuit` (defined in `heraldo.components.interfaces`). They operate on a fixed number of spatial modes, where **Mode 0** is the unmeasured output (heralded) mode and modes `1..N-1` are measured with photon‑number‑resolving detectors (PNRDs).

---

## Prebuilt Circuits

The following classes are ready to use and located in `heraldo.components.circuits`. They provide standard architectures with varying numbers of modes and gate sets.

| Circuit class | Modes | Gates | Parameters |
|---------------|-------|-------|------------|
| `TwoModeStaticGeneral` | 2 | Squeeze, Displace, BS | 10 |
| `TwoModeStaticSqueezeOnly` | 2 | Squeeze, BS | 6 |
| `ThreeModeStaticGeneral` | 3 | Squeeze, Displace, 3×BS | 18 |
| `ThreeModeStaticSqueezeOnly` | 3 | Squeeze, 3×BS | 12 |
| `FourModeStaticSqueezeOnly` | 4 | Squeeze, 6×BS | 20 |

**Key parameters** (common to all):

- `clip_size` (float): maximum absolute value for squeezing and displacement magnitudes (default `2.0`). This bounds the optimization search space.
- `measure_fock_cutoff` (int): Fock‑space cutoff dimension for PNR detectors on ancillary modes (default `5`). This controls the resolution of the photon‑number measurements.
- `num_single_photon` (int): number of ancillary modes initialised in `|1⟩` instead of vacuum (default `0`). Useful for injecting non‑Gaussian seed states.
- `loss_transmissivity` (float): transmissivity η ∈ (0,1] for loss channels applied to all modes (default `1.0`, lossless). Simulates photon loss.

**Architecture details**:

- `*General` circuits include both squeezing and displacement on every mode.
- `*SqueezeOnly` circuits omit displacement (only squeezing is applied).
- Beam‑splitter networks are fixed:
  - Two‑mode: a single `BS(θ, φ)` between modes 0 and 1.
  - Three‑mode: sequence `BS(0,1) → BS(1,2) → BS(0,1)`.
  - Four‑mode: ladder network `BS(0,1) → BS(2,3) → BS(1,2) → BS(0,1) → BS(2,3) → BS(1,2)`.

---

### Three-Mode General Circuit Architecture

```text
Mode 0: |0⟩   ───[ S0 ]──[ D0 ]───(BS1)────────────(BS3)───────── |ψ⟩
                                    │                │
Mode 1: |0/1⟩ ───[ S1 ]──[ D1 ]───(BS1)───(BS2)────(BS3)───[ PNRD ]
                                            │
Mode 2: |0/1⟩ ───[ S2 ]──[ D2 ]───────────(BS2)────────────[ PNRD ]
```

## Instantiating a Circuit

Simply create an instance with the desired parameters. For example, a two‑mode squeeze‑only circuit with 12 dB of squeezing (converted to the squeezing parameter `r`) and a detector cutoff of 30:

```python
from heraldo.components.circuits import TwoModeStaticSqueezeOnly
from heraldo.utils import db_to_r

circuit = TwoModeStaticSqueezeOnly(
    clip_size=db_to_r(12.0),   # r ≈ 1.38
    measure_fock_cutoff=30,
    num_single_photon=0,
    loss_transmissivity=1.0
)
```

To use a three‑mode general circuit with a single‑photon seeded ancilla:

```python
from heraldo.components.circuits import ThreeModeStaticGeneral

circuit = ThreeModeStaticGeneral(
    clip_size=2.0,
    measure_fock_cutoff=20,
    num_single_photon=1,          # Mode 1 starts in |1⟩
    loss_transmissivity=0.98      # 2% loss on all modes
)
```

---

## Using a Circuit in Optimization


Pass the circuit instance to a  `BasinHoppingRunner`. The runner will:

- Use the circuit’s `parameter_names` and `parameter_bounds` to set up the optimization variables.
- Call `run_circuit(params, engine)` to evaluate the state for a given parameter vector.
- Use `get_measurement_specs()` to know which modes are measured and with what cutoff.


```python
from heraldo.components.runner import BasinHoppingRunner
from heraldo.components.targets import SqueezedCatTarget

def main():
    circuit = TwoModeStaticSqueezeOnly(clip_size=1.5, measure_fock_cutoff=30)
    target = SqueezedCatTarget(alpha=3.0, r=0.5, p=0)

    runner = BasinHoppingRunner(
        circuit=circuit,
        target_gens=[target],
        cutoff_dim=30,
        beam_width=20
    )

    result = runner.run(n_iter=10)

if __name__ == "__main__":
    main()
```

---

## Extending: Creating a Custom Circuit

To implement your own static circuit, inherit from `StaticCircuit` (imported from `heraldo.components.interfaces`) and implement the abstract methods:

1. `parameter_names` → list of strings naming each optimizable parameter.
2. `parameter_bounds` → list of `(min, max)` tuples for each parameter.
3. `run_circuit(self, params, engine)` → build and run the Strawberry Fields program.
4. `get_measurement_specs(self)` → list of `(mode_index, fock_cutoff)` for ancillary modes.

Optionally, you can override `__init__` to accept custom parameters (like `clip_size`, `num_single_photon`, etc.) and store them as attributes.

### Example: A two‑mode circuit with only a beam splitter (no squeezing)

```python
import numpy as np
import strawberryfields as sf
from strawberryfields.ops import BSgate
from heraldo.components.interfaces import StaticCircuit

class TwoModeBSOnly(StaticCircuit):
    num_modes = 2

    def __init__(self, measure_fock_cutoff=5, loss_transmissivity=1.0):
        self.measure_fock_cutoff = measure_fock_cutoff
        self.loss_transmissivity = loss_transmissivity
        self._param_names = ['bs_theta', 'bs_phi']
        self._bounds = [(-8*np.pi, 8*np.pi), (-8*np.pi, 8*np.pi)]

    @property
    def parameter_names(self):
        return self._param_names

    @property
    def parameter_bounds(self):
        return self._bounds

    def run_circuit(self, params, engine):
        bs_th, bs_ph = params
        prog = sf.Program(2)
        with prog.context as q:
            # No initial gates – both modes start in vacuum
            BSgate(bs_th, bs_ph) | (q[0], q[1])
            # Optionally add loss
            if self.loss_transmissivity < 1.0:
                from strawberryfields.ops import LossChannel
                LossChannel(self.loss_transmissivity) | q[0]
                LossChannel(self.loss_transmissivity) | q[1]
        return engine.run(prog)

    def get_measurement_specs(self):
        # Measure Mode 1 only
        return [(1, self.measure_fock_cutoff)]
```

### Important: Serialization and Reconstruction

For your custom circuit to be saved and later reconstructed via `load_results(..., reconstruct=True)`, you must follow the serialization rules described in the **[Serialization and Reconstruction Guide](serialization.md)**. In short:

- All `__init__` parameters that matter for reconstruction must be stored as attributes with the same name.
- Your class must be importable from its module (i.e., defined in a separate `.py` file, not in the main script).
- The module must be on the Python path when loading.

The `heraldo.factory` will automatically capture the parameter values and use them to recreate your circuit when `reconstruct=True`.

---

## Parameter Access and Inspection

You can always inspect the current parameter names and bounds:

```python
print(circuit.parameter_names)
# ['bs_theta', 'bs_phi']
print(circuit.parameter_bounds)
# [(-25.1327, 25.1327), (-25.1327, 25.1327)]
```

The measurement specifications can also be retrieved:

```python
print(circuit.get_measurement_specs())
# [(1, 5)]   for TwoModeBSOnly example
```

---

## Summary

- **Prebuilt circuits** cover the most common architectures – choose based on number of modes and whether you need displacement.
- **Instantiate** with the desired `clip_size`, `measure_fock_cutoff`, `num_single_photon`, and `loss_transmissivity`.
- **Use** them by passing to a runner; the runner handles optimization and evaluation.
- **Extend** by subclassing `StaticCircuit` and implementing the required methods.
- **Ensure serializability** for reliable saving/loading; see the [serialization guide](serialization.md).

For a deeper comparison of circuit architectures and their performance, see [Circuits Comparison](circuits_comparison.md).
