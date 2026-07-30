========================================
Heraldo User Guide
========================================

.. contents:: Table of Contents
   :local:
   :depth: 2

This guide is a practical, "read this first" introduction to the ``heraldo``
package. It is aimed at new students and collaborators who need to
understand what the package does, how its pieces fit together, and how to
run their own optimizations — without having to reverse-engineer the source
first. For exhaustive parameter-by-parameter documentation, see the
:doc:`API Reference <modules>`, which is generated directly from the
docstrings.

--------------------------------------------------------------------------

1. What is heraldo?
====================

``heraldo`` is an optimization framework for **heralded non-Gaussian state
generation** in continuous-variable (CV) photonic circuits. It numerically
optimizes the classical control parameters of a Gaussian Boson
Sampling (GBS)-like circuit — squeezing, displacement, and beam-splitter
angles — so that particular photon-number-resolving (PNR) detection
outcomes herald a desired non-Gaussian quantum state (e.g. a GKP state, a
Schrödinger cat state, a binomial code, or a cubic phase state) in an
unmeasured output mode.

It is the reference implementation for the paper *"Multi-Outcome Circuit
Optimization for Enhanced Non-Gaussian State Generation"* (``paper.tex`` in
this repository), and it implements the two-phase optimization strategy
described there:

1. **Beam Search** — an exploratory phase that autonomously discovers
   which measurement outcomes are worth targeting, without any prior
   assumption about which patterns are "good".
2. **Fixed-Pattern Optimization** — a refinement phase that polishes the
   circuit parameters for a chosen, fixed set of heralding patterns,
   either to multiplex several different target states, or to harvest
   several outcomes that all herald the *same* target state.

Under the hood, simulation is done in the Fock basis using
`Strawberry Fields <https://strawberryfields.ai/>`_, with several
performance and correctness patches applied automatically on import (see
:ref:`patches-section`).

.. note::
   The package can simulate **static spatial circuits** (all modes act at
   once, :math:`T=1`) *and* **time-domain multiplexed (TDM) circuits**
   (a single loop mode reused over :math:`T>1` steps). If you're not sure
   which one you need, read :doc:`circuits_comparison` first — most
   day-to-day usage (and everything the paper evaluates) uses ``steps=1``.

--------------------------------------------------------------------------

2. Installation & Requirements
===============================

``heraldo`` depends on:

* ``numpy``, ``scipy``, ``pandas``
* ``strawberryfields`` (Fock-basis CV simulation backend)
* ``thewalrus`` (fast Gaussian-boson-sampling primitives used by the patches)
* ``numba`` (JIT compilation for the custom beam-splitter kernel)
* ``tqdm`` (progress bars)

There is nothing to configure — simply ``import heraldo`` (or, more
commonly, import the submodules you need, e.g.
``import heraldo.components.circuits``). Import side effects apply a small
number of compatibility/performance patches automatically; see
:ref:`patches-section`.

.. note::
   Some parts of ``heraldo`` (notably :class:`~heraldo.components.targets.CoreGKPTarget`)
   read tabulated coefficients from a CSV file
   (``data/GKP_core_coefficients.csv``, courtesy of Tzitrin *et al.*). Make
   sure the ``csv_path`` you pass points at that file.

--------------------------------------------------------------------------

3. Package Layout
====================

.. code-block:: text

    heraldo/
    ├── __init__.py                     # scipy compatibility patch (simps -> simpson)
    ├── factory.py                      # create_from_config(): build objects from dict configs
    ├── utils.py                        # fidelity metrics, dB<->r conversion, path helpers
    ├── components/
    │   ├── __init__.py                 # applies the performance/compat patches on import
    │   ├── interfaces.py               # TargetGenerator & TimeMultiplexedCircuit ABCs
    │   ├── circuits.py                 # concrete circuit architectures (2/3/4-mode)
    │   ├── targets.py                  # concrete target-state generators (GKP, cat, ...)
    │   └── runner.py                   # BasinHoppingRunner, loss functions, evaluation core
    └── patches/
        ├── sf_operations_no_cache.py   # disables SF's gate-tensor caching (bounds memory use)
        ├── beamsplitter_patch.py       # custom O(D^3) JIT beam-splitter (big speedup)
        └── prepare_multimode_patch.py  # keeps pure states pure when re-preparing ancillae

    scripts/
    ├── optimize/    # end-to-end optimization drivers (single runs and batch sweeps)
    ├── analysis/    # standalone loss/robustness analyses
    ├── plotting/    # publication-quality Wigner-function figures
    └── state_visualization/  # quick-look Wigner + Fock-histogram plots for a target

Everything you need for day-to-day work lives in ``heraldo.components``.
The ``scripts/`` directory is not part of the installable package — it's a
collection of runnable examples/drivers, and the best place to copy a
starting point from.

--------------------------------------------------------------------------

4. Core Concepts
====================

4.1 The circuit anatomy: loop mode + ancillae
------------------------------------------------

Every circuit in ``heraldo`` is described relative to a single output mode
and one or more measured ancillary modes:

* **Mode 0 — the loop / memory mode.** This is the mode that ends up
  carrying the heralded output state :math:`|\psi\rangle`. It is *never*
  measured. When ``steps > 1`` it also persists across time steps (hence
  "loop"); when ``steps == 1`` it is just the paper's single unmeasured
  signal mode.
* **Modes 1 … N-1 — ancilla modes.** Freshly prepared each step (vacuum,
  optionally squeezed/displaced/single-photon), mixed with Mode 0 through
  beam splitters, and then measured with photon-number-resolving (PNR)
  detectors. The detected pattern :math:`\mathbf{n} = (n_1, \dots,
  n_{N-1})` is what "heralds" the state left behind on Mode 0.

This maps directly onto the two circuit diagrams in
:doc:`circuits_comparison`: setting ``steps=1`` gives you the paper's
static spatial circuit; ``steps>1`` gives you the time-domain multiplexed
loop architecture.

4.2 ``time_invariant``
------------------------

Every circuit constructor takes a ``time_invariant`` flag:

* ``True`` — the *same* control parameters (squeezing, displacement,
  beam-splitter angles) are reused at every step ``t``.
* ``False`` (default) — each step gets its own independent set of
  parameters, giving the optimizer more freedom at the cost of a larger
  search space.

This only matters when ``steps > 1``; for ``steps=1`` it has no effect.

4.3 Beam Search vs. Fixed-Pattern optimization
--------------------------------------------------

``heraldo`` supports two evaluation modes, selected by whether you pass a
``measurement_patterns`` argument to :class:`~heraldo.components.runner.BasinHoppingRunner`:

+---------------------------+-----------------------------------------------+----------------------------------------------+
| ``measurement_patterns``  | Behaviour                                      | Typical loss function                         |
+===========================+=================================================+================================================+
| ``None``                  | **Beam Search.** All possible outcome           | :func:`~heraldo.components.runner.beam_search_loss_fn` |
|                            | combinations are explored, keeping only the top |                                                |
|                            | ``beam_width`` most probable branches at each   |                                                |
|                            | step. Use this when you don't know in advance   |                                                |
|                            | which detection patterns will be useful.        |                                                |
+---------------------------+-----------------------------------------------+----------------------------------------------+
| a list/array of patterns  | **Fixed-Pattern optimization.** Only the paths  | :func:`~heraldo.components.runner.fixed_pattern_capped_loss_fn` |
|                            | that produce exactly the specified detection    | or                                             |
|                            | outcome(s) are simulated. Use this once you know| :func:`~heraldo.components.runner.fixed_pattern_free_loss_fn` |
|                            | (from a Beam Search, or from physical intuition)|                                                |
|                            | which patterns to refine.                       |                                                |
+---------------------------+-----------------------------------------------+----------------------------------------------+

In practice, the recommended workflow mirrors the paper: run a **Beam
Search** first to discover promising patterns (see
``scripts/optimize/run_beam_search_sweeps.py``), then lock those patterns
in and run **Fixed-Pattern** refinement (see
``scripts/optimize/run_table_sweeps.py``).

4.4 Pattern format
----------------------

A measurement pattern describes, for every time step, the photon numbers
expected on every ancilla mode:

.. code-block:: python

    # A single-step (steps=1), 1-ancilla circuit heralding on n=4:
    patterns = [[(4,)]]

    # Two different single-step patterns to optimize simultaneously
    # (e.g. resource multiplexing of an even and an odd cat state):
    patterns = [[(4,)], [(5,)]]

    # A 3-mode (2-ancilla) circuit, single step, pattern (1, 3):
    patterns = [[(1, 3)]]

    # Harvesting several equivalent 2-ancilla outcomes for one target:
    patterns = [[(1, 3)], [(3, 1)], [(2, 2)]]

Each top-level list entry is one *pattern sequence* (one candidate outcome
to optimize for); each pattern sequence is itself a list with one tuple
per time step; each tuple has one integer per ancilla mode. For
``steps=1`` circuits (the common case), that's just a list of
one-tuple-per-outcome, wrapped once more.

4.5 Fidelity is rotation-invariant
---------------------------------------

Fidelities reported by ``heraldo`` are **maximized over phase-space
rotation** (:math:`\mathcal{F} = \max_\phi |\langle \phi_{\text{target}} |
e^{i\hat n \phi} | \psi \rangle|^2`), computed efficiently via an FFT over
256 discretized angles. This is done because a global phase-space rotation
of the heralded state can always be corrected afterwards with a simple
optical delay. It also means two branches with identical fidelity can
still differ in their *actual* orientation — see
:func:`~scripts.optimize.eval_time_optimized.evaluate_and_report_rotations`
in ``scripts/optimize/eval_time_optimized.py`` for how to check whether a
discovered pattern cluster is "rotation-invariant" (correctable by one
shared phase shifter) or "rotation-variant" (needs per-pattern
feed-forward correction) — this corresponds to Sec. III.C of the paper.

--------------------------------------------------------------------------

5. Quick Start
==================

The smallest complete example: optimize a 2-mode circuit (1 loop mode + 1
ancilla) to herald even/odd squeezed cat states on ancilla outcomes
``n=4`` and ``n=5`` respectively.

.. code-block:: python

    import numpy as np
    from heraldo.components.circuits import TwoModeTimeDomainSqueezeOnly
    from heraldo.components.targets import SqueezedCatTarget
    from heraldo.components.runner import BasinHoppingRunner, fixed_pattern_capped_loss_fn
    from heraldo.utils import db_to_r

    # 1. Convert 12 dB of source squeezing into the squeezing parameter r
    squeezing = db_to_r(12)

    # 2. Build the circuit: 1 loop mode + 1 ancilla, single spatial stage (T=1)
    circuit = TwoModeTimeDomainSqueezeOnly(
        steps=1,
        time_invariant=False,
        clip_size=squeezing,
        measure_fock_cutoff=30,
    )

    # 3. Define the target(s) to herald
    targets = [
        SqueezedCatTarget(alpha=np.sqrt(6), r=0.5, p=0),  # even cat
        SqueezedCatTarget(alpha=np.sqrt(6), r=0.5, p=1),  # odd cat
    ]

    # 4. Fix the heralding patterns to optimize for (see Sec. 4.4 above)
    patterns = [[(4,)], [(5,)]]

    # 5. Build and run the optimizer
    runner = BasinHoppingRunner(
        circuit=circuit,
        target_gens=targets,
        cutoff_dim=30,
        beam_width=200,
        penalty_strength=1.0,
        measurement_patterns=patterns,
        loss_fn=fixed_pattern_capped_loss_fn,
    )

    result = runner.run(n_iter=20, method="L-BFGS-B")

    print(f"Loss:               {result['loss']:.5f}")
    print(f"Expected fidelity:  {result['expected_fidelity']:.5f}")
    for branch in result["branches"]:
        print(branch)

Running an unconstrained **Beam Search** instead only requires dropping
``measurement_patterns`` (and switching the loss function):

.. code-block:: python

    from heraldo.components.runner import beam_search_loss_fn

    runner = BasinHoppingRunner(
        circuit=circuit,
        target_gens=targets,
        cutoff_dim=30,
        beam_width=200,          # keep the 200 most-probable branches per step
        penalty_strength=1.0,
        measurement_patterns=None,   # <-- triggers Beam Search discovery mode
        loss_fn=beam_search_loss_fn,
    )
    result = runner.run(n_iter=200)

For fully worked, copy-pasteable versions of both flows (including saving
results, generating reports, and reproducing the paper's tables), see
``scripts/optimize/run_time_optimization.py`` (single run),
``scripts/optimize/run_beam_search_sweeps.py`` (batch discovery), and
``scripts/optimize/run_table_sweeps.py`` (batch refinement).

--------------------------------------------------------------------------

6. Understanding the Output
================================

:meth:`BasinHoppingRunner.run() <heraldo.components.runner.BasinHoppingRunner.run>`
returns a dictionary with the following keys:

``x``
    The optimized flat parameter vector (initial-state parameters followed
    by per-step gate parameters — see Sec. 7.3 below for how to unpack it).
``loss``
    The final objective value (lower is better; the sign convention makes
    this the *negative* of the reported score, plus any truncation
    penalty).
``expected_fidelity``
    The scalar objective score computed by the chosen loss function (its
    exact meaning depends on which loss function was used — see Sec. 8).
``branches``
    A list of dictionaries, one per surviving measurement outcome:

    .. code-block:: python

        {
            "outcome":    (4,),      # detected photon numbers, per step
            "prob":       0.058,     # probability of this heralding event
            "fidelity":   0.994,     # rotation-maximized fidelity to the best target
            "target_idx": 0,         # which entry of target_gens this best matches
        }

``total_probability``
    Sum of ``prob`` over all surviving branches.
``duration``
    Wall-clock time of the run, in seconds.
``message``
    Human-readable status (e.g. which parallel run was best, if
    ``num_parallel_runs > 1``).

If a pattern is physically unreachable (e.g. it demands more photons than
the circuit can plausibly produce given its bounds), that branch is simply
dropped; if *all* branches are dropped, the loss function short-circuits
to ``100.0`` — a large-but-finite value the optimizer will steer away
from.

--------------------------------------------------------------------------

7. Working with Circuits
=============================

7.1 Built-in circuit classes
--------------------------------

All circuits live in :mod:`heraldo.components.circuits` and subclass
:class:`~heraldo.components.circuits.BaseTimeDomainGeneral`, itself a
:class:`~heraldo.components.interfaces.TimeMultiplexedCircuit`.

+---------------------------------------+-------+----------+---------------------------------------------------------------+
| Class                                 | Modes | Ancillae | Per-step gate sequence                                         |
+========================================+=======+==========+=================================================================+
| ``TwoModeTimeDomainGeneral``          | 2     | 1        | :math:`S,D` on ancilla, then 1 beam splitter (loop–ancilla)     |
+---------------------------------------+-------+----------+---------------------------------------------------------------+
| ``TwoModeTimeDomainSqueezeOnly``      | 2     | 1        | :math:`S` only on ancilla, then 1 beam splitter                 |
+---------------------------------------+-------+----------+---------------------------------------------------------------+
| ``ThreeModeTimeDomainGeneral``        | 3     | 2        | :math:`S,D` on both ancillae, 3 beam splitters (0-1, 1-2, 0-1)  |
+---------------------------------------+-------+----------+---------------------------------------------------------------+
| ``ThreeModeTimeDomainSqueezeOnly``    | 3     | 2        | :math:`S` only, 3 beam splitters (0-1, 1-2, 0-1)                |
+---------------------------------------+-------+----------+---------------------------------------------------------------+
| ``FourModeTimeDomainSqueezeOnly``     | 4     | 3        | :math:`S` only, 6-beam-splitter nearest-neighbour ladder        |
+---------------------------------------+-------+----------+---------------------------------------------------------------+

All of them share the same constructor keywords (see the docstrings for
full detail):

``steps`` (required)
    Number of time-domain recirculation steps :math:`T`. Use ``1`` for a
    static spatial circuit matching the paper.
``time_invariant`` (default ``False``)
    See Sec. 4.2.
``clip_size`` (default ``2.0``)
    Bound on squeezing/displacement magnitudes (dimensionless :math:`r`).
    A convenient value is ``heraldo.utils.db_to_r(source_dB)``.
``measure_fock_cutoff`` (default ``5``)
    PNR detector cutoff for ancilla modes. This is automatically capped by
    ``cutoff_dim`` wherever it's used, so it's safe to set it equal to
    ``cutoff_dim``.
``num_single_photon`` (default ``0``)
    How many ancilla modes to initialize in :math:`|1\rangle` instead of
    vacuum (useful for photon-subtraction-style protocols).
``initial_fock_one`` (default ``False``)
    If ``True``, initializes the loop mode (Mode 0) in :math:`|1\rangle`
    before squeezing, instead of vacuum.
``loss_transmissivity`` (default ``1.0``)
    Channel transmissivity :math:`\eta \in (0, 1]` applied to every mode
    at every step, to simulate photon loss (see Sec. 10.2).

7.2 Choosing a circuit
--------------------------

Pick the number of modes based on how many detection events you're
willing to condition on (more ancillae = richer heralding statistics, but
exponentially larger Fock-space simulation cost). Pick "General" vs.
"SqueezeOnly" based on whether you want displacement gates available on
the ancillae — cubic phase state generation, for instance, needs
displacement, so it uses ``ThreeModeTimeDomainGeneral``; GKP/cat/binomial
targets in the paper only need squeezing, so they use the
``SqueezeOnly`` variants.

7.3 Unpacking optimized parameters
---------------------------------------

The flat vector returned by the optimizer, ``result["x"]``, is
``[initial-state params..., per-step params (flattened over steps)...]``.
To turn it into something human-readable:

.. code-block:: python

    n_init = circuit.num_initial_parameters          # usually 2: (r, phi)
    init_params = result["x"][:n_init]
    step_params = result["x"][n_init:]

    # shape: (steps, n_params_per_step)
    mapped = circuit.map_parameters(step_params)
    names = circuit.per_step_parameter_names

.. warning::
   ``per_step_parameter_names`` is prefixed with two legacy labels,
   ``'init_sq_r'`` and ``'init_sq_phi'``, that do **not** correspond to
   columns of ``map_parameters()``'s output — the true initial-state
   squeezing parameters are governed separately, by
   ``circuit.num_initial_parameters`` /
   ``circuit.initial_parameter_bounds`` (see above). When zipping names
   against mapped-parameter columns, skip the first two entries, e.g.
   ``names[2:]``.

--------------------------------------------------------------------------

8. Working with Target States
==================================

All targets live in :mod:`heraldo.components.targets` and implement a
single method, ``get_target_ket(cutoff_dim) -> np.ndarray``, returning the
target's state vector in the Fock basis.

+-----------------------------+---------------------------------------------+------------------------------------------------+
| Class                       | Parameters                                   | Produces                                        |
+==============================+===============================================+==================================================+
| ``SqueezedCatTarget``       | ``alpha``, ``r``, ``p`` (0=even, 1=odd)      | Squeezed Schrödinger cat state                  |
+-----------------------------+---------------------------------------------+------------------------------------------------+
| ``CatTarget``               | ``alpha``, ``p``                              | Unsqueezed Schrödinger cat state                |
+-----------------------------+---------------------------------------------+------------------------------------------------+
| ``CubicPhaseTarget``        | ``gamma``, ``r``, ``alpha``                   | Displaced approximate cubic phase state         |
+-----------------------------+---------------------------------------------+------------------------------------------------+
| ``CubicResourceTarget``     | ``a``                                          | 3-term Fock superposition cubic-gate resource   |
+-----------------------------+---------------------------------------------+------------------------------------------------+
| ``CoreGKPTarget``           | ``csv_path``, ``n_max``, ``delta_db``, ``mu``, | GKP "core state" (stellar representation) from  |
|                              | ``apply_squeezing``                            | tabulated Tzitrin *et al.* coefficients         |
+-----------------------------+---------------------------------------------+------------------------------------------------+
| ``BinomialCodeTarget``      | ``N``, ``S``, ``mu``                          | Binomial quantum-error-correcting code word     |
+-----------------------------+---------------------------------------------+------------------------------------------------+

``CoreGKPTarget`` is the one target that needs an external data file — it
looks up the row matching ``(n_max, delta_db)`` in the CSV and, if
``apply_squeezing=True``, applies the tabulated squeezing to turn the
finite-Fock "core state" into the actual approximate GKP codeword.

Add your own target by subclassing
:class:`~heraldo.components.targets.TargetGenerator` (see Sec. 11).

--------------------------------------------------------------------------

9. Loss Functions
======================

:mod:`heraldo.components.runner` exposes three ready-made loss functions,
all with the signature ``loss_fn(probs, fidelities) -> float``:

:func:`~heraldo.components.runner.beam_search_loss_fn`
    Non-linear, gradient-sharpening objective used during pattern
    *discovery*. Suppresses low-fidelity branches with a capped,
    power-4 weighting, then combines a ``log`` term (to keep gradients
    alive when the score is tiny) with a large linear term (to reward
    genuinely high scores). Tunable via ``epsilon`` (minimum-infidelity
    threshold, default ``0.02``), ``delta`` (log regularizer), and
    ``lam`` (linear-term weight).
:func:`~heraldo.components.runner.fixed_pattern_capped_loss_fn`
    :math:`\sum_k (\alpha\, p_k + \min(\mathcal{F}_k, F_{\text{cap}}))` —
    used for refinement once you've capped the target fidelity (default
    ``f_cap=0.95``) so the optimizer stops "over-polishing" one branch and
    instead spends its effort raising the *probability* of all targeted
    branches. ``alpha`` defaults to the number of targeted patterns.
:func:`~heraldo.components.runner.fixed_pattern_free_loss_fn`
    :math:`\sum_k (\alpha\, p_k + \mathcal{F}_k)` — the uncapped version:
    the optimizer is free to push fidelity as high as it can. ``alpha``
    defaults to ``0.1 × (number of targeted patterns)``, i.e. it weighs
    fidelity more heavily than the capped variant.

You can pass a custom ``loss_fn`` to :class:`~heraldo.components.runner.BasinHoppingRunner`
as long as it matches this signature — e.g. wrap one of the built-ins with
``functools.partial`` to change ``f_cap``/``alpha``/``epsilon``.

--------------------------------------------------------------------------

10. Evaluating & Visualizing Results
=========================================

Once you have an optimized parameter vector, ``scripts/optimize/eval_time_optimized.py``
provides a toolbox of post-processing utilities (these operate on the
``opt_*``/``job_*`` result directories written by the ``run_*`` scripts):

10.1 Deterministic path evaluation
--------------------------------------

:func:`run_deterministic_path` re-simulates the optimized circuit for one
*specific* measurement outcome, returning the exact heralded state ket and
its probability — this is what powers the Wigner-function plots and
density-matrix exports.

10.2 Loss / robustness analysis
------------------------------------

:func:`evaluate_time_domain_circuit_dm` (density-matrix version of the
core evaluator) lets you re-run an optimized circuit with
``loss_transmissivity < 1`` to see how fidelity and success probability
degrade under realistic photon loss — this is exactly what produces
Table IV of the paper (:func:`evaluate_loss_influence` automates it across
every optimized configuration and emits a LaTeX table). For a quick
single-target scan, see ``scripts/analysis/loss_impact.py``.

10.3 Truncation-error sanity check
----------------------------------------

:func:`evaluate_cutoff_fidelity` compares infidelities computed at two
different Fock cutoffs (e.g. 30 vs. 50) to make sure your chosen
``cutoff_dim`` isn't introducing artificial error — see Sec. II of the
paper for why this matters.

10.4 Rotation-invariance report
------------------------------------

:func:`evaluate_and_report_rotations` computes, for every pattern in a
targeted cluster, the optimal correcting phase-rotation angle, and reports
the overall angular spread. A spread of :math:`0^\circ` means a single
static phase shifter can correct every branch in the cluster; a nonzero
spread means you'd need per-pattern feed-forward correction (see Sec. 4.5
above, and Sec. III.C of the paper).

10.5 Figures
----------------

``scripts/plotting/paper_plot_comparison.py`` and
``paper_plot_comparison_5_patterns.py`` render publication-quality,
multi-panel 3D Wigner-function comparisons (target vs. several heralded
branches), matching the style of the paper's figures.
``scripts/state_visualization/demo_target.py`` is the quickest way to
sanity-check what a *target* state alone looks like (Wigner function +
Fock-number histogram) before you even start optimizing.

--------------------------------------------------------------------------

11. Extending heraldo
==========================

11.1 Adding a new target state
------------------------------------

Subclass :class:`~heraldo.components.targets.TargetGenerator` and
implement one method:

.. code-block:: python

    from heraldo.components.targets import TargetGenerator
    import numpy as np

    class MyTarget(TargetGenerator):
        def __init__(self, some_param=1.0):
            self.some_param = some_param

        def get_target_ket(self, cutoff_dim: int) -> np.ndarray:
            ket = np.zeros(cutoff_dim, dtype=np.complex128)
            # ... fill in your state's Fock coefficients ...
            ket /= np.linalg.norm(ket)
            return ket

That's it — it can now be passed to ``BasinHoppingRunner(target_gens=[...])``
just like any built-in target.

11.2 Adding a new circuit architecture
--------------------------------------------

The simplest route is subclassing
:class:`~heraldo.components.circuits.BaseTimeDomainGeneral`, which already
handles initial-state preparation, measurement specs, and parameter
bookkeeping — you only need to define the per-step gate sequence:

.. code-block:: python

    import numpy as np
    import strawberryfields as sf
    from strawberryfields.ops import Sgate, BSgate
    from heraldo.components.circuits import BaseTimeDomainGeneral

    class MyTwoModeCircuit(BaseTimeDomainGeneral):
        num_modes = 2

        def __init__(self, steps, **kwargs):
            super().__init__(steps=steps, **kwargs)
            self._param_names = ['sq_r', 'bs_theta', 'bs_phi']
            self._param_names = ['init_sq_r', 'init_sq_phi'] + self._param_names
            self._bounds = [
                (-self.clip_size, self.clip_size),  # sq_r
                (-8*np.pi, 8*np.pi),                 # bs_theta
                (-8*np.pi, 8*np.pi),                 # bs_phi
            ]

        def run_step(self, state, step_idx, step_params, engine):
            sq_r, bs_theta, bs_phi = step_params
            prog = sf.Program(2)
            with prog.context as q:
                Sgate(sq_r) | q[1]
                BSgate(bs_theta, bs_phi) | (q[0], q[1])
            return engine.run(prog)

For architectures that don't fit the "loop + squeeze/displace ancillae"
mold at all, subclass
:class:`~heraldo.components.interfaces.TimeMultiplexedCircuit` directly
and implement its four abstract members: ``per_step_parameter_names``,
``per_step_parameter_bounds``, ``run_step``, and
``get_measurement_specs``.

11.3 Custom loss functions
--------------------------------

Any callable with signature ``(probs: np.ndarray, fidelities: np.ndarray) -> float``
can be passed as ``loss_fn=`` to ``BasinHoppingRunner``. Remember that
``BasinHoppingRunner`` *minimizes* this value, so if higher is better in
your metric, negate it internally (as the built-ins do).

--------------------------------------------------------------------------

.. _patches-section:

12. Why the monkey-patches?
================================

Importing ``heraldo.components`` (which happens automatically the first
time you import anything from it) applies three patches to Strawberry
Fields' Fock backend:

``sf_operations_no_cache.disable_fock_caching()``
    Strawberry Fields normally caches every gate tensor it builds (keyed
    by its parameters). For an optimizer that evaluates thousands of
    distinct parameter vectors, that cache grows without bound and can
    exhaust memory. This patch swaps in uncached gate constructors so
    memory use stays flat, at a small recomputation cost per step.
``beamsplitter_patch.patch_beamsplitter()``
    Replaces Strawberry Fields' default beam-splitter application with a
    custom, Numba-JIT-compiled :math:`O(D^3)` diagonal-traversal
    implementation, which is substantially faster for the
    repeated-many-times workload of basin-hopping optimization.
``prepare_multimode_patch.patch_prepare_multimode()``
    Adds an "unentangled detection" fast path: when re-preparing a subset
    of modes (e.g. resetting ancillae for the next time step) would leave
    the *remaining* modes in a separable pure state, this patch updates
    the state vector directly instead of falling back to a full
    density-matrix representation — keeping simulation in the cheaper
    pure-state regime whenever physically valid.

You will also see a small ``scipy.integrate.simps -> scipy.integrate.simpson``
compatibility patch scattered across several files (``heraldo/__init__.py``,
``heraldo/utils.py``, ``heraldo/components/targets.py``, and the patch
modules). This exists because newer ``scipy`` releases removed the
deprecated ``simps`` alias that some of ``heraldo``'s dependencies still
call; the patch simply restores it.

None of this requires any action on your part — it's applied once, the
first time you import from ``heraldo.components``.

--------------------------------------------------------------------------

13. Static vs. Time-Domain Circuits
========================================

If you're choosing between ``steps=1`` and ``steps>1``, or trying to
understand how the loop-based physical realization maps onto the
simulated circuit, read :doc:`circuits_comparison` — it walks through the
Motes *et al.* (2014) loop-unraveling result that justifies simulating a
recirculating fiber loop as an equivalent multi-stage spatial beam-splitter
network, and gives a side-by-side comparison table plus runnable code for
both regimes.

**tl;dr:** everything the paper evaluates uses ``steps=1`` (static
spatial circuits); ``steps>1`` is a forward-looking extension of the same
framework to physically loop-based, time-multiplexed hardware.

--------------------------------------------------------------------------

14. Common Pitfalls / FAQ
==============================

**My run always returns loss = 100.0 — what went wrong?**
    Every candidate branch had (numerically) zero probability — the
    requested pattern is unreachable given the circuit's current
    parameter bounds, or a bug in the pattern shape. Double-check that
    your ``measurement_patterns`` shape matches
    ``(n_sequences, circuit.steps, n_ancilla_modes)`` and that the photon
    counts you're asking for are within ``measure_fock_cutoff``.

**Why does my ``schedule.json`` / parameter table have more names than
columns?**
    See the warning in Sec. 7.3 — ``per_step_parameter_names`` includes
    two leading legacy labels (``init_sq_r``, ``init_sq_phi``) that don't
    correspond to a column of ``map_parameters()``'s output.

**I passed ``train_initial_state`` / ``initial_r`` to a circuit and
nothing happened.**
    These keyword arguments appear in some of the example scripts but are
    not consumed by any current circuit class — they're silently absorbed
    by ``**kwargs`` and ignored. The actual initial-state parameters are
    controlled via ``circuit.num_initial_parameters`` /
    ``circuit.initial_parameter_bounds`` and optimized automatically as
    the first ``n_init`` entries of ``result["x"]``.

**What cutoff dimension should I use?**
    The paper uses ``cutoff_dim=30`` for the main results and cross-checks
    against ``cutoff_dim=50`` (see Sec. 10.3 /
    :func:`evaluate_cutoff_fidelity`) to confirm truncation error stays
    below roughly one order of magnitude in infidelity. For loss-inclusive
    (density-matrix) simulations of 3-mode circuits, memory pressure
    typically forces a smaller cutoff (e.g. ``15``) — see
    ``evaluate_loss_influence`` for the pattern used in this repo.

**How many parallel basin-hopping runs should I use?**
    ``BasinHoppingRunner(num_parallel_runs=..., num_processes=...)``
    splits the requested ``n_iter`` basin-hopping iterations *evenly*
    across ``num_parallel_runs`` independent, differently-seeded runs
    (executed with ``multiprocessing.Pool``), and keeps whichever run
    found the lowest loss. This trades iteration depth per run for
    broader exploration of the parameter space — tune to your
    core count and how multi-modal you expect the loss landscape to be.

--------------------------------------------------------------------------

15. Where to Go Next
=========================

* :doc:`circuits_comparison` — static spatial vs. time-domain multiplexed
  architectures, with the Motes *et al.* loop-unraveling background.
* :doc:`modules` — full, docstring-generated API reference for every
  class and function mentioned above.
* ``scripts/optimize/run_time_optimization.py`` — the best single file to
  copy as a starting point for a new optimization.
* ``scripts/optimize/run_beam_search_sweeps.py`` /
  ``run_table_sweeps.py`` — how the paper's Tables I & II were generated,
  end-to-end (batch discovery, then batch refinement, with automatic
  Markdown report generation).
* ``paper.tex`` — the full theoretical background, target-state
  definitions, and numerical results this package was built to produce.
