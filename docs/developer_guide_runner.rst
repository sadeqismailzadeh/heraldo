========================================================================
Developer Guide: The Circuit Evaluation Engine (``runner.py``)
========================================================================

.. contents:: Table of Contents
   :local:
   :depth: 2

Audience & Scope
=================

This guide is for contributors who need to modify, debug, or extend the
core simulation/evaluation machinery in
:mod:`heraldo.components.runner`. It does **not** cover
:class:`~heraldo.components.runner.BasinHoppingRunner` (the outer
optimization loop) or the public loss functions
(:func:`~heraldo.components.runner.beam_search_loss_fn`,
:func:`~heraldo.components.runner.fixed_pattern_capped_loss_fn`,
:func:`~heraldo.components.runner.fixed_pattern_free_loss_fn`) in detail
— those are documented from a *usage* perspective in
:doc:`user_guide`. Instead, this document walks through the five
functions that do the actual step-by-step quantum state propagation and
scoring:

* :func:`evaluate_time_domain_circuit` — the single public entry point
* :func:`_process_fixed_patterns` — deterministic replay of chosen outcomes
* :func:`_process_beam_search` — pruned exploration of the full outcome tree
* :func:`_compute_fidelities_and_loss` — turns surviving branches into a scalar loss
* :func:`_format_branch_details` — turns surviving branches into human-readable metadata

--------------------------------------------------------------------------

1. Call Graph & Mental Model
=============================

Every circuit evaluation — whether triggered by a basin-hopping objective
call or a final "give me the details" call — goes through one function:

.. code-block:: text

   evaluate_time_domain_circuit(flat_params, circuit, target_kets, ...)
           │
           ├── circuit.map_parameters(...)          # unpack flat_params -> (steps, n_params)
           ├── circuit.get_initial_state_ket(...)    # prepare Mode 0 (loop) at t=0
           │
           ├── measurement_patterns is not None?
           │     │
           │     ├── YES ──> _process_fixed_patterns(...)
           │     └── NO  ──> _process_beam_search(...)
           │
           ├── _compute_fidelities_and_loss(...)     # -> loss, expected_fidelity, fidelities, ...
           │
           └── return_details?
                 ├── False -> return loss (float)                 # fast path for the optimizer
                 └── True  -> _format_branch_details(...) 
                              -> return dict (loss, branches, ...) # slow path for reporting
                              
Both ``_process_fixed_patterns`` and ``_process_beam_search`` implement
the *same* physical loop — apply :meth:`circuit.run_step` for each of the
``T`` time steps, then project the ancilla modes onto some measurement
outcome and renormalize Mode 0 — but they differ in **which** outcomes
they keep:

* ``_process_fixed_patterns`` keeps only outcomes you already told it to
  care about (:doc:`Phase 2 <user_guide>` — refinement).
* ``_process_beam_search`` keeps whichever outcomes happen to be most
  probable, discovered fresh at every step (:doc:`Phase 1 <user_guide>` —
  discovery).

Understanding one is most of the way to understanding the other; the
sections below cover ``_process_fixed_patterns`` first since its control
flow is simpler (no pruning), then highlight what changes in
``_process_beam_search``.

--------------------------------------------------------------------------

2. ``evaluate_time_domain_circuit``: The Entry Point
=======================================================

.. code-block:: python

   def evaluate_time_domain_circuit(flat_params, circuit, target_kets, cutoff_dim,
                                     beam_width, penalty_strength,
                                     measurement_patterns=None, return_details=False,
                                     loss_fn=None):

This is the only function in the module that the rest of the codebase
should call directly — :class:`BasinHoppingRunner` calls it once per
objective evaluation (``return_details=False``) and once more at the end
of each run to build the reported ``branches`` list
(``return_details=True``).

Responsibilities, in order:

1. **Split the flat parameter vector.** ``flat_params`` is
   ``[init_params..., step_params...]``. The first
   ``circuit.num_initial_parameters`` entries configure Mode 0's initial
   state; everything after that is per-step gate parameters.
2. **Reshape step parameters.** :meth:`circuit.map_parameters` turns the
   remaining flat array into a ``(steps, n_params_per_step)`` matrix,
   either by reshaping (time-variant) or tiling a single row
   (time-invariant) — see :ref:`the interfaces docs
   <heraldo-components-interfaces>` for that logic.
3. **Prepare the initial ket.** :meth:`circuit.get_initial_state_ket`
   builds Mode 0's t=0 state (usually vacuum, optionally squeezed).
4. **Dispatch on evaluation mode.** If ``measurement_patterns`` is
   supplied, delegate to :func:`_process_fixed_patterns`; otherwise
   delegate to :func:`_process_beam_search`. Both return ``None`` when
   *every* candidate branch has (numerically) zero probability — in that
   case this function short-circuits and returns the fixed penalty
   ``100.0``. This is the mechanism behind the FAQ entry "my run always
   returns loss = 100.0" in :doc:`user_guide` — it means every requested
   pattern (fixed mode) or every explored branch (beam search mode) was
   unreachable given the current parameters.
5. **Score the survivors.** :func:`_compute_fidelities_and_loss` turns
   the surviving ``(kets, probs, outcome_sums)`` into a scalar loss plus
   per-branch fidelities.
6. **Return.** If ``return_details`` is ``False``, only the bare
   ``loss`` float is returned — this is the hot path called thousands of
   times per basin-hopping run, so it deliberately skips building the
   branch dictionaries. If ``True``, :func:`_format_branch_details`
   assembles the reported ``branches`` list and the function returns the
   full result dict (``loss``, ``expected_fidelity``, ``branches``,
   ``total_probability``) — this is what ends up in
   ``BasinHoppingRunner.run()``'s return value.

.. note::
   The choice of default ``loss_fn`` also happens here: if the caller
   didn't pass one explicitly, fixed-pattern mode defaults to
   :func:`fixed_pattern_capped_loss_fn` and beam-search mode defaults to
   :func:`beam_search_loss_fn`. Passing your own ``loss_fn`` overrides
   this in both modes.

--------------------------------------------------------------------------

3. ``_process_fixed_patterns``: Deterministic Outcome Replay
===============================================================

.. code-block:: python

   def _process_fixed_patterns(circuit, initial_ket, mapped_params, meas_specs,
                                cutoff_dim, measurement_patterns):

This function answers one question: *if the PNR detectors are forced to
report exactly these outcomes, step by step, what state survives on Mode
0, and with what probability?* It's the workhorse behind Phase 2
(:func:`fixed_pattern_capped_loss_fn` / :func:`fixed_pattern_free_loss_fn`).

3.1 Input normalization
------------------------

``measurement_patterns`` arrives as a nested list/array of shape
``(n_sequences, steps, n_meas_modes)`` (or ``(steps, n_meas_modes)`` for
a single sequence, which gets promoted with ``patterns_arr[None, ...]``).
Each "sequence" is one candidate outcome you want the optimizer to chase
— see Sec. 4.4 of :doc:`user_guide` for the pattern-format convention.
Shape mismatches against ``circuit.steps`` or the number of measured
modes raise ``ValueError`` immediately, since a silent shape mismatch
here would otherwise fail in a much more confusing way deep inside the
advanced-indexing logic below.

3.2 Per-step propagation
--------------------------

The function tracks, in parallel, one state-vector "lane" per surviving
sequence:

* ``current_kets`` — shape ``(n_sequences, cutoff_dim)``, Mode 0's state
  for each sequence.
* ``sequence_probs`` — cumulative probability of matching the requested
  pattern so far, per sequence.
* ``sequence_active`` — boolean mask; a sequence is deactivated the
  moment its probability underflows ``1e-12`` (the requested outcome is
  physically unreachable at this point in the parameter space).

For each step ``t = 0 .. steps-1``:

1. Only **active** sequences are propagated (``active_indices``).
2. For each active sequence, its current ket is re-injected into a fresh
   :class:`strawberryfields.Program` via ``Ket(ket) | q[0]``, then
   :meth:`circuit.run_step` applies that step's gates (squeezing,
   displacement, beam splitters — whatever the concrete circuit class
   defines) to get the full multi-mode ``full_ket``.
3. **Deduplication cache** (``ket_cache``): if two sequences happen to
   share the exact same Mode-0 state going into this step (rounded to 8
   decimals), the simulation is only run once and the result is reused.
   This matters because fixed-pattern sequences frequently share a
   common prefix — e.g. requesting outcomes ``(1,3)`` and ``(3,1)`` for
   the same circuit both start from the same Mode 0 state at ``t=0``.
4. The multi-mode ket is transposed so Mode 0 comes first, followed by
   the measured ancilla modes (``perm = [0] + meas_modes``), then sliced
   down to each detector's cutoff dimension.
5. **Projection onto the requested outcome**: advanced indexing pulls
   out exactly the amplitude slice matching ``step_outcomes`` for that
   sequence — this is the "measurement". The retained probability is
   ``sum(|projected|**2)`` over Mode 0's remaining Fock dimension.
6. Sequences whose projected probability falls below ``1e-12`` are
   marked inactive and dropped from further propagation (their
   ``sequence_probs`` is zeroed).
7. Surviving sequences have their ket renormalized
   (``/ sqrt(prob)``), ``sequence_probs`` multiplied by this step's
   conditional probability, and a running ``total_truncation_error`` is
   accumulated — this tracks how much probability mass "leaked" out of
   the truncated Fock space due to the finite ``cutoff_dim`` (norm drift
   away from 1.0), weighted by how likely that branch is. It feeds
   directly into the ``penalty_strength`` term of the loss (see
   :ref:`section 5 <compute-fidelities-loss>`).

3.3 Return value
------------------

If every sequence died before the end, the function returns ``None``
(this is what triggers the ``loss = 100.0`` fallback upstream). Otherwise
it returns a 6-tuple:

.. code-block:: python

   active_kets, active_probs, active_outcome_sums, total_truncation_error, possible_mask, patterns_arr

``active_outcome_sums`` is the **total photon count** summed across all
detectors and all steps for each surviving sequence — used downstream to
filter out trivial all-zero-photon branches (see
:ref:`compute-fidelities-loss`). ``possible_mask`` (which original
sequences survived) and ``patterns_arr`` (the original dense pattern
array) are threaded through so :func:`_format_branch_details` can later
recover which literal outcome tuple each surviving branch corresponds
to.

--------------------------------------------------------------------------

4. ``_process_beam_search``: Pruned Outcome-Tree Exploration
=================================================================

.. code-block:: python

   def _process_beam_search(circuit, initial_ket, mapped_params, meas_specs,
                             cutoff_dim, beam_width):

Where :func:`_process_fixed_patterns` only ever simulates the outcomes
you specify, this function has no prior idea which detector outcomes are
worth keeping — it has to *discover* them, one step at a time, without
letting the number of tracked branches explode combinatorially. This is
the implementation of Phase 1 (pattern discovery) described in
:doc:`introduction` and :doc:`quickstart_tutorial`.

4.1 The branching problem
----------------------------

At every step, each currently-tracked branch can, in principle, produce
*any* combination of photon counts across all measured ancilla modes up
to their cutoff — i.e. up to ``prod(meas_cutoffs)`` distinct children per
parent. Left unchecked, the number of tracked branches would grow as
``prod(meas_cutoffs) ** steps``, which is intractable for more than a
couple of steps. **Beam search** bounds this by keeping only the
``beam_width`` globally-most-probable branches after each step,
discarding the rest.

4.2 Per-step propagation
--------------------------

For each step:

1. For every currently active parent branch, run the step's gates
   exactly as in the fixed-pattern case (fresh ``Ket(parent_ket) | q[0]``
   into a new engine, then :meth:`circuit.run_step`), producing a full
   multi-mode ``full_ket``.
2. Transpose + slice down to the measured modes' cutoffs, same as
   before, giving a tensor ``sliced_ket`` of shape
   ``(cutoff_dim, *meas_cutoffs)`` — Mode 0's amplitude for *every
   possible* combination of ancilla photon numbers, not just one.
3. Sum ``|sliced_ket|**2`` over the Mode-0 axis to get
   ``probs_tensor``: the **raw** (parent-conditional) probability of
   each possible outcome combination for this branch alone.
4. Multiply by the parent's cumulative probability
   (``P_total = P_parent[:, None] * P_raw``) to get the *joint*
   probability of "reach this parent, then observe this outcome" for
   every (parent, outcome) pair simultaneously — a 2D array of shape
   ``(n_active_parents, prod(meas_cutoffs))``.
5. **Prune**: flatten ``P_total`` and use
   ``np.argpartition(..., -k)`` (``k = min(beam_width, size)``) to find
   the indices of the top-``k`` joint probabilities in :math:`O(n)`
   time, avoiding a full sort. ``np.unravel_index`` recovers which
   parent and which outcome-combination each survivor corresponds to.
6. The surviving (parent, outcome) pairs are used to gather the
   corresponding un-normalized ket slices out of the stacked branch
   tensors via advanced indexing, then normalized by their norm to
   become the new Mode-0 states for the next step.
7. A ``mask`` filters out numerically negligible survivors (near-zero
   norm or probability) — mirroring the "sequence death" logic in the
   fixed-pattern function, except here it can prune the beam down to
   fewer than ``beam_width`` branches, or return ``None`` entirely if
   nothing survives.
8. The running **outcome history** (``active_outcomes``) is extended by
   concatenating each surviving branch's parent history with its new
   step's outcome tuple — this is what lets
   :func:`_format_branch_details` later report, e.g., ``(4, 2)`` as the
   full two-step detection record for a branch, even though no single
   step "knew" about the other.

4.3 Return value
------------------

Returns ``None`` if the beam empties out at any step, otherwise:

.. code-block:: python

   active_kets, active_probs, active_outcome_sums, total_truncation_error, active_outcomes

Structurally identical in spirit to the fixed-pattern return value,
except ``active_outcomes`` (the full per-branch outcome history array)
replaces the ``(possible_mask, patterns_arr)`` pair — beam search has no
predefined pattern array to point back into, so it has to carry its own
discovered history forward explicitly.

--------------------------------------------------------------------------

.. _compute-fidelities-loss:

5. ``_compute_fidelities_and_loss``: Scoring the Survivors
===============================================================

.. code-block:: python

   def _compute_fidelities_and_loss(active_kets, active_probs, active_outcome_sums,
                                     target_kets, total_truncation_error,
                                     penalty_strength, loss_fn=None):

Both processing functions above hand back a pile of surviving
``(ket, probability)`` branches; this function is where "how good are
these states, and how good is this whole configuration overall" gets
decided.

5.1 Filtering trivial branches
---------------------------------

.. code-block:: python

   mask_nonzero = active_outcome_sums > 0

Branches where **every** detector registered zero photons across every
step are excluded from fidelity scoring entirely. A total photon count
of zero generally means no actual heralding event occurred — the ancilla
modes passed through undetected/empty, which is not a useful
non-Gaussian resource and would otherwise dominate the probability-
weighted objective with an uninteresting, typically high-probability,
low-value branch.

5.2 Rotation-maximized fidelity via FFT
------------------------------------------

For the remaining branches:

.. code-block:: python

   prod = np.conj(final_kets[:, None, :]) * targets_arr[None, :, :]
   fft_vals = np.fft.fft(prod, n=256, axis=-1)
   pairwise_fidelities = np.max(np.abs(fft_vals)**2, axis=-1)

This computes, for **every** (branch, target) pair simultaneously, the
fidelity maximized over an arbitrary global phase-space rotation
:math:`\hat R(\phi) = e^{i\hat n \phi}`. The reasoning (also explained in
Sec. 4.5 of :doc:`user_guide`) is that a phase-space rotation acts on
Fock coefficients as :math:`c_n \to c_n e^{in\phi}`, so

.. math::
   \max_\phi \left| \sum_n \psi_n^{*} t_n e^{-in\phi} \right|^2

is exactly the magnitude-squared of the discrete Fourier transform of
the per-Fock-order overlap ``prod``, maximized over frequency bins. Using
``n=256`` zero-pads the FFT to interpolate a finer angular grid than the
raw cutoff dimension would give for free. This is why a rotated version
of a target state (correctable afterward with a simple optical delay)
doesn't get penalized during optimization — see Sec. III.C of the paper
for the physical justification, and the FAQ in :doc:`user_guide` on
"rotation-variant" vs. "rotation-invariant" pattern clusters.

``fidelities`` then takes the max over the target axis (best-matching
target state per branch) and ``best_target_indices`` records *which*
target won, for later reporting — this is what makes resource
multiplexing (Sec. "Resource Multiplexing" in :doc:`introduction`) work:
a single optimization run can herald different targets on different
outcomes, and this argmax is what assigns each branch to its target.

5.3 Combining into a scalar loss
-----------------------------------

.. code-block:: python

   expected_fidelity = loss_fn(final_probs, fidelities)
   loss = -1 * expected_fidelity + (penalty_strength * total_truncation_error)

``loss_fn`` is one of the probability/fidelity aggregators described in
:doc:`user_guide` Sec. 9 (:func:`beam_search_loss_fn`,
:func:`fixed_pattern_capped_loss_fn`, or a user-supplied callable with
the same ``(probs, fidelities) -> float`` signature). Since
``scipy.optimize.basinhopping`` **minimizes**, and higher
``expected_fidelity`` is always better, the sign is flipped here. The
truncation-error penalty is added (not subtracted) so that
parameter regions producing large Fock-truncation leakage are
disfavored regardless of how good the reported fidelity looks — a high
score built on a badly-truncated simulation is not trustworthy.

The function returns a 5-tuple —
``(loss, expected_fidelity, fidelities, best_target_indices,
mask_nonzero)`` — deliberately including the *unmasked-length* arrays
plus the mask itself, rather than pre-filtering, so that callers (in
particular :func:`_format_branch_details`) can re-apply the same mask
against other same-length arrays (probabilities, outcome tuples) without
re-deriving it.

--------------------------------------------------------------------------

6. ``_format_branch_details``: Human-Readable Branch Metadata
==================================================================

.. code-block:: python

   def _format_branch_details(mask_nonzero, active_probs, fidelities,
                               best_target_indices, measurement_patterns=None,
                               patterns_arr=None, possible_mask=None,
                               active_outcomes=None):

This function exists purely to translate the numeric arrays produced
above into the list-of-dicts shape documented in :doc:`user_guide` Sec.
6 (``result["branches"]``), and it is **only** called when
``return_details=True`` — it's intentionally kept out of the hot
optimization loop.

Its only nontrivial logic is reconstructing each surviving branch's
**outcome tuple**, which differs depending on which processing function
produced the data:

* **Fixed-pattern mode** (``measurement_patterns is not None``): the
  outcome is already known in advance — it's recovered by flattening
  ``patterns_arr`` to ``(n_sequences, steps * n_meas_modes)``, indexing
  with ``possible_mask`` (which original sequences survived
  propagation), then with ``mask_nonzero`` (which of those had nonzero
  total photon count).
* **Beam search mode** (``measurement_patterns is None``): the outcome
  was accumulated step-by-step during the search itself, so it's read
  directly out of ``active_outcomes`` (already flat, shape
  ``(n_survivors, steps * n_meas_modes)``), filtered by ``mask_nonzero``
  only (there's no ``possible_mask`` in this mode — beam search only
  ever tracks branches it decided to keep).

Each retained branch becomes:

.. code-block:: python

   {
       "outcome": (4,),        # flattened per-step, per-mode photon counts
       "prob": 0.058,          # this branch's absolute (not conditional) probability
       "fidelity": 0.994,      # best rotation-maximized fidelity, across all targets
       "target_idx": 0,        # which entry of target_gens/target_kets fidelity matched
   }

--------------------------------------------------------------------------

7. Data Shape Quick Reference
================================

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - Variable
     - Shape / Meaning
   * - ``mapped_params``
     - ``(steps, n_params_per_step)`` — from ``circuit.map_parameters()``
   * - ``initial_ket``
     - ``(cutoff_dim,)`` — Mode 0's state at :math:`t=0`
   * - ``meas_specs``
     - ``list[(mode_index, cutoff)]`` — one entry per ancilla mode
   * - ``patterns_arr`` (fixed mode)
     - ``(n_sequences, steps, n_meas_modes)``
   * - ``active_kets``
     - ``(n_surviving_branches, cutoff_dim)``
   * - ``active_probs``
     - ``(n_surviving_branches,)`` — absolute probability per branch
   * - ``active_outcome_sums``
     - ``(n_surviving_branches,)`` — total photons detected, all steps/modes
   * - ``active_outcomes`` (beam search)
     - ``(n_surviving_branches, steps * n_meas_modes)`` — flat outcome history
   * - ``pairwise_fidelities``
     - ``(n_nonzero_branches, n_targets)`` — before target-argmax
   * - ``fidelities``
     - ``(n_nonzero_branches,)`` — after target-argmax

--------------------------------------------------------------------------

8. Performance Notes
========================

* **Why the ``ket_cache`` in fixed-pattern mode but not beam search?**
  Fixed patterns are known in advance and frequently share prefixes
  (e.g. harvesting several degenerate outcomes for one target — see
  :doc:`user_guide` Sec. 4.4), so identical intermediate states are
  common and worth deduplicating. Beam search branches are the *result*
  of pruning by probability, so two branches sharing an identical state
  at the same step is comparatively rare and not worth the bookkeeping
  overhead.
* **Why serial loops with advanced indexing instead of full
  vectorization across steps?** The circuit itself (via
  :meth:`circuit.run_step`) must be applied sequentially — each step's
  gates depend on the previous step's output state — so steps cannot be
  parallelized. Within a step, however, all active/candidate branches
  *are* vectorized together via NumPy advanced indexing and batched
  tensor operations, which is what keeps both functions reasonably fast
  despite Python-level looping over steps.
* **Where does :math:`O(D^3)` vs :math:`O(D^4)` matter here?** The
  per-step call into ``engine.run(prog)`` inside both functions is where
  the beam-splitter patch documented in :doc:`sf_patches` actually gets
  exercised — every ``run_step`` call in this file indirectly triggers
  the JIT-compiled diagonal-traversal beam splitter kernel, not just the
  gate application in isolation.