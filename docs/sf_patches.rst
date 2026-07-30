========================================================================
Strawberry Fields Performance & Backend Patches
========================================================================

.. contents:: Table of Contents
   :local:
   :depth: 2

Overview & Motivation
=====================

Simulating continuous-variable (CV) quantum circuits in the Fock basis requires truncating the infinite-dimensional Hilbert space to a finite cutoff dimension :math:`D`. Standard backend implementations in `Strawberry Fields <https://strawberryfields.ai/>`_ can encounter severe memory bottlenecks and computational scaling issues during global parameter optimizations involving thousands of circuit evaluations at high cutoff dimensions (e.g., :math:`D = 30`).

To enable high-speed global optimization and maintain a minimal RAM footprint on standard consumer laptop hardware, ``heraldo`` automatically applies specialized performance and backend patches upon importing ``heraldo.components``:

1. **Single-Threaded BLAS/NumPy Configuration** (``OMP_NUM_THREADS=1``, etc.)
2. **Fock Gate Cache Removal** (``disable_fock_caching``)
3. **JIT-Compiled :math:`O(D^3)` Beam Splitter Kernel** (``patch_beamsplitter``)
4. **Pure-State Preservation in Multi-Mode Preparation** (``patch_prepare_multimode``)

Together, these patches reduce memory growth from unbounded :math:`O(\text{evals} \times D^4)` to constant, bounded storage, prevent thread contention during parallel Basin-Hopping runs, and accelerate tensor contractions by orders of magnitude.

------------------------------------------------------------------------

1. Disabling Fock Gate Caching
==============================

* **Source File**: ``heraldo/patches/sf_operations_no_cache.py``
* **Function**: ``disable_fock_caching()``
* **Invocation**: ``heraldo/components/__init__.py`` (Line 10)

Problem
~~~~~~~

By default, Strawberry Fields uses internal Least-Recently-Used (LRU) caching for all Fock-basis gate matrices (squeezing, displacement, beam splitters, phase shifters, Kerr gates). For optimization routines evaluating tens of thousands of continuous parameter values, this gate cache grows unbounded in RAM, causing severe memory leaks and eventual out-of-memory (OOM) crashes.

Solution
~~~~~~~~

``disable_fock_caching()`` overrides the backend gate functions in ``strawberryfields.backends.fockbackend.ops`` with uncached implementations powered by ``thewalrus.fock_gradients`` and direct NumPy array operations.

Benefits
~~~~~~~~

* **Bounded Memory**: Memory usage remains flat and constant regardless of the number of optimization iterations.
* **No Memory Leaks**: Prevents RAM buildup across long global Basin-Hopping sweeps.

------------------------------------------------------------------------

2. JIT-Compiled :math:`O(D^3)` Beam Splitter Kernel
===================================================

* **Source File**: ``heraldo/patches/beamsplitter_patch.py``
* **Function**: ``patch_beamsplitter()``
* **Invocation**: ``heraldo/components/__init__.py`` (Line 11)

Problem
~~~~~~~

The beam splitter operation :math:`\hat{BS}(\theta, \phi)` is the primary multi-mode gate in CV photonic circuits. Standard implementations compute or contract full 4D tensors of shape :math:`(D, D, D, D)`, resulting in an :math:`O(D^4)` time and memory complexity per beam splitter gate. For :math:`D = 30`, :math:`D^4 = 810,000` elements per tensor, making multi-step optimization intractable.

Solution
~~~~~~~~

``patch_beamsplitter()`` replaces Strawberry Fields' default ``Circuit.beamsplitter`` method with a custom Numba JIT-compiled kernel:

* **Photon-Number Conservation**: Exploits total photon conservation :math:`S = n_1 + n_2 = m_1 + m_2` to restrict contraction strictly along diagonal manifolds.
* **JIT Recurrence & Arithmetic Optimization**: Computes recurrence matrix entries on-the-fly using Numba ``@jit(nopython=True, fastmath=True)`` routines with tight, branchless loops.

Benefits
~~~~~~~~

* **Complexity Reduction**: Reduces storage and calculation scaling from :math:`O(D^4)` down to :math:`O(D^3)`.
* **Massive Speedup**: Contraction loop times drop by over an order of magnitude.

------------------------------------------------------------------------

3. Pure-State Preservation in Multi-Mode Preparation
====================================================

* **Source File**: ``heraldo/patches/prepare_multimode_patch.py``
* **Function**: ``patch_prepare_multimode()``
* **Invocation**: ``heraldo/components/__init__.py`` (Line 13)

Problem
~~~~~~~

In time-domain multiplexed (TDM) or recirculating loop circuits, ancillary modes are reset and re-prepared at each time step. Standard Strawberry Fields behavior treats partial-mode re-preparation as a general channel operation, automatically converting pure state vectors (:math:`\psi \in \mathbb{C}^D`) into density matrices (:math:`\rho \in \mathbb{C}^{D \times D}`). Density matrix representations square the memory footprint and turn :math:`O(D^3)` operations into :math:`O(D^6)` processes.

Solution
~~~~~~~~

``patch_prepare_multimode()`` introduces an **unentangled state detection fast-path**:

1. After PNR measurement or partial mode reset, it checks if the retained system subsystem remains separable (unentangled).
2. If the subsystem is pure, it computes the pure state projection directly and updates the state vector ket.
3. It bypasses the fallback to density matrix simulation, keeping the simulation strictly within the pure-state regime.

Benefits
~~~~~~~~

* **State Purity Maintenance**: Keeps recirculating loop mode states pure across arbitrary numbers of time steps :math:`T`.
* **Low Memory Scale**: Avoids the quadratic memory penalty of density matrix evolution.

------------------------------------------------------------------------

4. Single-Threaded BLAS/NumPy Threading for Parallel Basin-Hopping
===================================================================

* **Source File**: ``heraldo/components/__init__.py``
* **Mechanism**: Sets environment variables (``OMP_NUM_THREADS=1``, ``OPENBLAS_NUM_THREADS=1``, ``MKL_NUM_THREADS=1``, ``VECLIB_MAXIMUM_THREADS=1``, ``NUMEXPR_NUM_THREADS=1``) before backend imports.

Problem
~~~~~~~

By default, NumPy linear algebra backends (OpenBLAS, MKL, OMP) attempt to use all available CPU threads for matrix computations. When ``BasinHoppingRunner`` executes parallel optimization runs using Python's multiprocessing workers, each process spawning its own multi-threaded BLAS pool leads to severe CPU thread oversubscription, context-switching thrashing, and degraded optimization throughput.

Solution
~~~~~~~~

Upon importing ``heraldo.components``, thread limits for NumPy and underlying BLAS backends are automatically set to ``1``. Each Basin-Hopping worker process executes matrix operations in a single thread, allowing parallel Basin-Hopping optimization runs to scale cleanly across available CPU cores without thread contention.

------------------------------------------------------------------------

Performance Impact Summary
==========================

The table below summarizes the computational impact of these patches:

.. list-table::
   :widths: 30 35 35
   :header-rows: 1

   * - Metric / Operation
     - Stock Strawberry Fields
     - With ``heraldo`` Patches
   * - **NumPy Threading in Parallel Runs**
     - Multi-threaded BLAS thread contention
     - Single-threaded per process (optimized for multiprocessing)
   * - **RAM Usage over 10,000 Evals**
     - Unbounded growth (OOM crash)
     - Flat & constant (~ hundreds of MB)
   * - **Beam Splitter Complexity**
     - :math:`O(D^4)`
     - :math:`O(D^3)` (Numba JIT)
   * - **Loop State Representation**
     - Density matrix :math:`\rho \in \mathbb{C}^{D \times D}`
     - Pure ket vector :math:`|\psi\rangle \in \mathbb{C}^D`
   * - **Practical Cutoff Dimension**
     - :math:`D \approx 10 - 15`
     - :math:`D = 30+` on laptop CPU
==========================

The table below summarizes the computational impact of these patches:

.. list-table::
   :widths: 30 35 35
   :header-rows: 1

   * - Metric / Operation
     - Stock Strawberry Fields
     - With ``heraldo`` Patches
   * - **RAM Usage over 10,000 Evals**
     - Unbounded growth (OOM crash)
     - Flat & constant (~ hundreds of MB)
   * - **Beam Splitter Complexity**
     - :math:`O(D^4)`
     - :math:`O(D^3)` (Numba JIT)
   * - **Loop State Representation**
     - Density matrix :math:`\rho \in \mathbb{C}^{D \times D}`
     - Pure ket vector :math:`|\psi\rangle \in \mathbb{C}^D`
   * - **Practical Cutoff Dimension**
     - :math:`D \approx 10 - 15`
     - :math:`D = 30+` on laptop CPU

Thanks to these optimizations, high-fidelity circuit simulations with Fock space cutoffs up to :math:`D = 30` run with fast execution times and a small memory footprint directly on consumer laptop processors.
