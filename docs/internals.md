# Internals: Thread Limits & Backend Patches

This page documents low-level behavior that happens automatically the moment `heraldo` (or
`heraldo.components`) is imported — thread-limiting environment variables, a SciPy compatibility
shim, and three Strawberry Fields backend monkey-patches — plus one related setup step that
remains the **user's** responsibility. None of the functions described here need to be called
directly; they exist so that circuit simulation and optimization behave correctly and efficiently
out of the box.

---

## 1. Thread-Limiting Environment Variables

At the very top of `heraldo/__init__.py` and duplicated in `heraldo/components/__init__.py`, before any other imports:

```python
import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
```

### Why this exists

Left unconfigured, NumPy/SciPy's underlying linear-algebra libraries (OpenBLAS, MKL, Accelerate)
will spawn one BLAS thread per available CPU core for *every* matrix operation. This is fine in
isolation, but `BasinHoppingRunner` already parallelizes across `num_parallel_runs` independent
worker processes via Python's `multiprocessing` (see [Optimization Runner](runner.md)). If each of
those worker processes *also* multithreads its own BLAS calls, you get massive oversubscription —
`P` processes each spawning `N` threads, all competing for the same `N` physical cores — which
slows the optimization down rather than speeding it up. Pinning every thread count to `1` ensures
each worker process uses exactly one core for its BLAS operations, so parallelism comes purely
from the process-level `multiprocessing.Pool`, not from nested, contended threading underneath it.

### Critical caveat: ordering matters

These environment variables only take effect if they are set **before** NumPy, SciPy, OpenBLAS,
or MKL are first imported/initialized in the process. Setting them afterwards has no effect,
because those libraries read the environment once at load time to configure their internal
thread pools — there is no supported way to change it later in the same process.

Because `heraldo/__init__.py` sets these variables before importing anything heavy, simply
`import heraldo` first in your script is safe. However, **if your own script imports `numpy`,
`scipy`, or any other heavy numerical library *before* `import heraldo`**, the assignments inside
`heraldo/__init__.py` will already be too late for that import — the library will have already
read the (unset or default) environment and configured its thread pool accordingly.

**Recommendation:** set these five `os.environ[...]` lines yourself at the very top of your own
entry-point script, before any other imports (including `import heraldo`). This is especially
important given that `multiprocessing`'s `spawn` start method re-imports your entire module in
every worker process on Windows and macOS — the same ordering hazard applies independently in
each worker (see the note on `if __name__ == "__main__":` in [Optimization Runner](runner.md)).

```python
# Recommended: top of your own script, before any other imports
import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

import heraldo
# ... rest of your script
```

---

## 2. SciPy `simps` / `simpson` Compatibility Patch

This patch appears — defensively repeated, guarded by `hasattr` so it is idempotent — in three
places: `heraldo/__init__.py`, `heraldo/components/targets.py`, and
`heraldo/patches/sf_operations_no_cache.py`:

```python
import scipy.integrate
if not hasattr(scipy.integrate, 'simps'):
    print("Monkey patching scipy.integrate: 'simps' not found. Pointing to 'simpson'.")
    scipy.integrate.simps = scipy.integrate.simpson
else:
    print("'simps' already exists in scipy.integrate. No patch needed.")
```

### Why this exists

Newer SciPy releases removed the deprecated `scipy.integrate.simps` alias in favor of
`scipy.integrate.simpson`. However, `strawberryfields` still calls `scipy.integrate.simps` internally, and has not been updated for
newer SciPy versions. This patch aliases `simps` back to `simpson` so that `strawberryfields`
keeps working unmodified on modern SciPy installations, without needing to pin an old SciPy
version.

---

## 3. Strawberry Fields Backend Patches

`heraldo/components/__init__.py` applies three monkey-patches automatically as soon as
`heraldo.components` (or anything importing it) is loaded:

```python
from heraldo.patches.sf_operations_no_cache import disable_fock_caching
disable_fock_caching()

from heraldo.patches.beamsplitter_patch import patch_beamsplitter
patch_beamsplitter()

from heraldo.patches.prepare_multimode_patch import patch_prepare_multimode
patch_prepare_multimode()
```

The user never calls these directly — they take effect the moment `heraldo` is imported.

### `disable_fock_caching()` — `heraldo/patches/sf_operations_no_cache.py`

Replaces Strawberry Fields' `@functools.lru_cache()`-decorated gate-matrix builders
(`beamsplitter`, `squeezing`, `displacement`, `two_mode_squeeze`, `mzgate`, `phase`, `kerr`, and
`cross_kerr` in `strawberryfields.backends.fockbackend.ops`) with uncached equivalents.

**Motivation:** this is purely a memory-footprint reduction. During a long basin-hopping
optimization run, gate parameters are essentially always different on every single evaluation, so
the LRU cache doesn't save meaningful recomputation — it just accumulates unboundedly and grows
RAM usage over the course of a run, which becomes serious at large `cutoff_dim` values, since each
cached matrix scales with the Fock-space cutoff. Disabling the cache trades a small amount of
recomputation for flat, bounded memory usage.

### `patch_beamsplitter()` — `heraldo/patches/beamsplitter_patch.py`

Replaces `Circuit.beamsplitter` in `strawberryfields.backends.fockbackend.circuit` with a
Numba-JIT-compiled implementation that generates the beamsplitter recurrence tensor and applies it
via diagonal-manifold traversal (exploiting total photon-number conservation across the
interaction), achieving $O(D^3)$ memory and algorithm complexity instead of the naive $O(D^4)$ approach.

**Motivation:**  this is a **memory reduction** and **speed increase** optimization which makes it
significantly faster than the default implementation.

### `patch_prepare_multimode()` — `heraldo/patches/prepare_multimode_patch.py`

Replaces `Circuit.prepare_multimode` with a version that detects when a newly-prepared subsystem
is actually separable/unentangled from the rest of the circuit's state, and in that case keeps the
overall state representation **pure** (a state vector) instead of falling back to Strawberry
Fields' default behavior of converting the entire circuit to a mixed-state density matrix
representation.

**Motivation:** Enables injecting single-photon states $|1\rangle$ into ancillary modes (`Fock(1) | q[i]` when `num_single_photon >= 1`) while keeping the state pure. Keeping the state pure (a vector) rather than mixed (a density matrix)
is a large memory/compute win, since a mixed-state representation over $N$ modes at cutoff $D$
scales as $D^{2N}$ instead of $D^N$.

---

## Summary

| Patch / Setting | File | Applied | Purpose |
|---|---|---|---|
| Thread-limiting env vars | `heraldo/__init__.py` (and duplicated in `heraldo/components/__init__.py`, example scripts) | Automatic on `import heraldo`, **but only if `heraldo` is imported before other heavy numerical libraries** — otherwise requires manual action in your own script | Prevents BLAS thread oversubscription when combined with `multiprocessing`-based parallel optimization runs. |
| `simps`/`simpson` compatibility shim | `heraldo/__init__.py`, `heraldo/components/targets.py`, `heraldo/patches/sf_operations_no_cache.py` | Automatic on `import heraldo` | Keeps `strawberryfields` working on modern SciPy versions that removed `scipy.integrate.simps`. |
| `disable_fock_caching()` | `heraldo/patches/sf_operations_no_cache.py` | Automatic on `import heraldo.components` | Prevents unbounded RAM growth from Strawberry Fields' `lru_cache`-based gate-matrix caching during long optimization runs. |
| `patch_beamsplitter()` | `heraldo/patches/beamsplitter_patch.py` | Automatic on `import heraldo.components` | Memory-efficient, JIT-compiled $O(D^3)$ beamsplitter application via diagonal-manifold traversal. |
| `patch_prepare_multimode()` | `heraldo/patches/prepare_multimode_patch.py` | Automatic on `import heraldo.components` | Keeps separable states pure (vector) instead of mixed (density matrix) when injecting non-vacuum $|1\rangle$ inputs. |