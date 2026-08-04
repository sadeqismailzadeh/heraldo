# Developer Guide: Inside `evaluate_circuit`

This guide explains, function by function, what actually happens on every single call to `evaluate_circuit` — the routine that `BasinHoppingRunner` (and the SciPy `basinhopping` minimizer it wraps) calls thousands of times per optimization run. It is aimed at contributors who need to modify the projection logic, the objective wiring, or the branch-reporting format, and who want to understand the tensor shapes flowing through the pipeline rather than just the public API.

The pipeline has one orchestrator and four private workers, called in this order:

```
evaluate_circuit
 ├─ circuit.run_circuit(...)                 # Strawberry Fields simulation
 ├─ _process_fixed_patterns   (fixed-pattern mode)   ─┐
 │  or                                                 ├─ produces (kets, probs, outcome_sums, outcomes)
 ├─ _process_beam_search      (beam-search mode)     ─┘
 ├─ _compute_fidelities_and_loss              # scoring
 └─ _format_branch_details                    # only if return_details=True
```

---

## 1. `evaluate_circuit` — orchestrator

```python
heraldo/components/runner.py:226  def evaluate_circuit(...)
```

This function turns a raw parameter vector `params` into either a scalar loss (used inside the optimizer's inner loop) or a rich results dictionary (used once at the end to report the winning configuration). It does five things, strictly in order:

**1. Simulate the circuit.** It creates a fresh Strawberry Fields Fock-backend engine sized to `cutoff_dim`, and calls `circuit.run_circuit(params, engine)`. Every `StaticCircuit` subclass builds its own `sf.Program` inside `run_circuit`, so this is the only place the runner touches Strawberry Fields directly — everything downstream operates purely on the resulting Fock-basis ket tensor, `full_ket = result.state.ket()`. This tensor has one axis per circuit mode, each of length `cutoff_dim` (mode 0 is the unmeasured output mode; modes `1..N-1` are the ancillas).

**2. Compute the truncation error.** `full_ket` is flattened and its squared norm is compared to 1:

```python
truncation_error = np.abs(1.0 - norm_sq)
```

Because the true, untruncated joint state should be exactly normalized, any leakage into Fock levels above `cutoff_dim` shows up as a norm deficit. This scalar is later folded into the loss as a penalty term, discouraging the optimizer from drifting into parameter regions where the finite-cutoff simulation stops being trustworthy.

**3. Delegate the projection.** Depending on whether `measurement_patterns` was supplied, `evaluate_circuit` picks exactly one of the two projection helpers:

```python
if use_fixed_patterns:
    res = _process_fixed_patterns(circuit, full_ket, cutoff_dim, measurement_patterns)
else:
    res = _process_beam_search(circuit, full_ket, cutoff_dim, beam_width)
```

Both helpers return the same four-tuple shape, `(kets, probs, outcome_sums, outcomes)`, or `None` if nothing usable survived. `evaluate_circuit` treats `None` uniformly: it short-circuits and returns the fixed fallback loss `100.0`, a large-but-finite sentinel that keeps the optimizer well-defined (rather than raising) whenever a candidate parameter vector produces no viable heralding outcomes.

**4. Resolve and apply the loss function.** If the caller didn't pass a `loss_fn` instance, `evaluate_circuit` chooses a sensible default based on the mode (`FixedPatternCappedLoss` for fixed patterns, `BeamSearchLoss` for beam search), matching the two-phase optimization strategy described in the paper. If a *class* (rather than an instance) was passed, it's instantiated with defaults. The resolved `loss_fn` and the four projection outputs are handed to `_compute_fidelities_and_loss`, which does the actual fidelity/scoring work and returns `(loss, objective_score, fidelities, best_target_indices, mask_nonzero)`.

**5. Return the result.** If `return_details=False` (the default, used by the optimizer's inner loop), only the scalar `loss` is returned — this keeps each basin-hopping evaluation cheap. If `return_details=True` (used once, on the final optimized parameters), `evaluate_circuit` additionally calls `_format_branch_details` to turn the raw arrays into a human-readable list of per-outcome dictionaries, and packages everything into the result dict (`loss`, `objective_score`, `branches`, `total_probability`) that `BasinHoppingRunner.run()` ultimately returns to the user.

---

## 2. `_process_fixed_patterns` — projection for user-specified outcomes

```python
heraldo/components/runner.py:266  def _process_fixed_patterns(...)
```

Used when `measurement_patterns` is not `None`. Its job is to project `full_ket` onto exactly the ancilla photon-number patterns the caller asked for, and return the resulting (unmeasured-mode) heralded states and their probabilities.

- It reads `circuit.get_measurement_specs()` to get the ancilla mode indices and their per-mode PNR detector cutoffs, then caps each cutoff at the simulation's `cutoff_dim`.
- `measurement_patterns` (which can be an int, tuple, list, list-of-lists, or ndarray — see [Specifying Measurement Patterns](patterns.md)) is normalized into a  `(num_patterns, num_ancilla_modes)` integer array via `_normalize_outcomes` which then converted to a NumPy array. An empty result here means "nothing to evaluate," so the function returns `None` immediately.
- `full_ket` is transposed so that mode 0 (output) is axis 0, followed by the ancilla modes in the order given by `get_measurement_specs()`, then sliced down to each mode's detector cutoff. Mode 0's axis is left untouched (`slice(None)`) since it is not measured.
- The requested patterns are used as a **NumPy advanced (fancy) index** into the sliced tensor — one integer index array per ancilla axis — which simultaneously extracts, for every requested pattern, the sub-vector of mode-0 amplitudes conditioned on exactly that ancilla detection event. The result is transposed to shape `(num_patterns, mode0_dim)`, i.e. one (unnormalized) candidate heralded ket per requested pattern.
- The probability of each pattern is `sum_n |amplitude[n]|^2` over the mode-0 axis. Patterns whose probability falls below `1e-12` are dropped (this also guards the subsequent normalization against division by zero). If every requested pattern turns out negligible, the function returns `None` — signalling to `evaluate_circuit` that this parameter vector doesn't produce any of the requested outcomes.
- Surviving kets are normalized to unit norm (turning them into proper single-mode quantum states), and the corresponding `probs`, `outcomes` (the surviving pattern tuples), and `outcome_sums` (total ancilla photon count per pattern) are returned alongside them.


---

## 3. `_process_beam_search` — projection for the exploratory phase

```python
heraldo/components/runner.py:268  def _process_beam_search(...)
```

Used when `measurement_patterns is None` (the default). Instead of projecting onto a caller-supplied list, this function computes the *entire* ancilla outcome probability distribution and dynamically keeps only the top-`beam_width` most likely outcomes — this is the algorithmic core of the paper's "beam search" pattern-discovery phase.

- Setup mirrors `_process_fixed_patterns`: fetch ancilla modes/cutoffs, transpose `full_ket` so mode 0 is axis 0, and slice each ancilla axis to its detector cutoff.
- `probs_tensor = np.sum(np.abs(sliced_ket)**2, axis=0)` marginalizes out the unmeasured mode-0 axis, producing the *full* multi-dimensional probability distribution `P(n_1, n_2, ...)` over every possible ancilla detection pattern — this is the object the beam search explores.
- Rather than sorting the entire (potentially huge) flattened distribution, `np.argpartition` is used to find the indices of the top `k = min(beam_width, total_outcomes)` entries in `O(n)` time, and only those `k` indices are then sorted by descending probability. `np.unravel_index` converts the flat top-k indices back into per-mode photon-number tuples (`outcomes_unraveled`).
- The same fancy-indexing trick as in `_process_fixed_patterns` is applied — but this time using the discovered top-k index arrays instead of caller-supplied patterns — to extract the corresponding `(k, mode0_dim)` block of unnormalized heralded kets.
- Each candidate ket's own norm is computed directly (`np.linalg.norm`, independent of `selected_probs`, for numerical robustness) and used both to filter out near-zero branches (`norms > 1e-9` and `selected_probs > 1e-12`) and to normalize the surviving kets.
- Returns the same `(kets, probs, outcome_sums, outcomes)` shape as the fixed-pattern version, with `outcomes` stacked from the per-axis index arrays.

Because this rebuilds the *global* outcome distribution every call, it lets the optimizer autonomously discover useful heralding patterns without any physical intuition being hard-coded — the only human-set knob is `beam_width` (how many of the highest-probability outcomes are kept in play at each optimization step).

---

## 4. `_compute_fidelities_and_loss` — scoring phase

```python
heraldo/components/runner.py:122  def _compute_fidelities_and_loss(...)
```

This is where the projected heralded states from either projection helper are turned into a single scalar loss. It receives `kets`, `probs`, and `outcome_sums` from whichever projection function ran, plus the list of `target_kets`.

- `loss_fn` is resolved the same defensive way as in `evaluate_circuit` (default to `BeamSearchLoss`, or instantiate a passed-in class) — this makes the function safely callable on its own, outside `evaluate_circuit`.
- **Vacuum filtering:** `mask_nonzero = outcome_sums > 0` excludes any branch where *all* ancilla modes detected zero photons. An all-zero click pattern is not treated as a useful heralding event, so it never contributes to the fidelity/probability score (though it's still present in the raw arrays passed to `_format_branch_details` later, filtered out identically there).
- If no branch survives the mask, the function short-circuits with `objective_score = 0.0` and empty `fidelities`/`best_target_indices` arrays — the returned `loss` then reduces to just the truncation penalty.
- **Rotation-invariant fidelity via FFT.** For every surviving branch and every target state, the function computes `conj(branch_ket) * target_ket` element-wise (broadcasting to shape `(num_branches, num_targets, cutoff_dim)`), then takes a 256-point FFT along the Fock-basis axis. This batched FFT is the vectorized equivalent of the single-pair `fidelity_max_rotation` helper in `heraldo/utils.py`: because a phase-space rotation by angle φ corresponds to multiplying each Fock amplitude `n` by `e^{-inφ}`, the FFT simultaneously evaluates the overlap at 256 discretized rotation angles for every (branch, target) pair in one shot. `all_fidelities = |FFT|^2` has shape `(num_branches, num_targets, 256)`.
- **Two evaluation modes, controlled by `phase_lock`:**
  - **`phase_lock=False` (default, per-branch rotation):** each branch independently picks whichever target and whichever phase angle maximizes its own fidelity. `pairwise_fidelities = max over phase axis` collapses to `(num_branches, num_targets)`, then `fidelities = max over target axis` and `best_target_indices = argmax over target axis` collapse to `(num_branches,)`. The resolved `loss_fn` is then called once with 1D `fidelities`, returning a single scalar `objective_score`.
  - **`phase_lock=True` (shared global rotation):** all branches are forced to share one common phase-space rotation. For every candidate phase angle, the best fidelity across targets is taken per branch (`fidelities_all_k`, shape `(num_branches, 256)`), and `loss_fn` is evaluated once *per phase angle* (it naturally supports this because `ObjectiveFunction.__call__` broadcasts a 1D `probs` against a 2D `fidelities` array — see `docs/objectives.md`). The phase angle `best_k` that maximizes the resulting 256-length score vector is chosen, and the per-branch `fidelities`/`best_target_indices` are sliced at that one column.
- **Final loss.** Regardless of mode, the objective function returns a *quality score* that should increase with better fidelity/probability. Since SciPy's `basinhopping` minimizes, the sign is flipped and the truncation penalty from step 2 of `evaluate_circuit` is added:

```python
loss = -1 * objective_score + (penalty_strength * truncation_error)
```

- Returns `(loss, objective_score, fidelities, best_target_indices, mask_nonzero)`. Note that `mask_nonzero` is returned alongside the (already-filtered) `fidelities`/`best_target_indices` so that downstream code can re-align them with the original, unfiltered `probs`/`outcomes` arrays — which is exactly what `_format_branch_details` does next.

---

## 5. `_format_branch_details` — presentation phase

```python
heraldo/components/runner.py:192  def _format_branch_details(...)
```

This function only runs when `evaluate_circuit(..., return_details=True)` — i.e. once, on the final winning parameter vector, not on every optimizer step. Its sole job is to turn the parallel arrays produced upstream into a list of self-contained, JSON/pickle-friendly dictionaries.

- It takes `mask_nonzero` (from `_compute_fidelities_and_loss`) and re-applies it to the *original* `probs` and `outcomes` arrays (the ones returned by whichever projection helper ran) — this re-alignment step is necessary because `fidelities` and `best_target_indices` were already computed only for the mask-surviving branches, while `probs`/`outcomes` as received here are still in their pre-filter, full-length form.
- For each surviving branch `i`, it assembles:
  - `"outcome"`: the ancilla photon-number pattern as a plain Python tuple (e.g. `(2, 4)`),
  - `"prob"`: that outcome's success probability,
  - `"fidelity"`: the fidelity achieved against the best-matching target (under whichever rotation policy `phase_lock` selected),
  - `"target_idx"`: which entry of `target_gens`/`target_kets` that fidelity was computed against.
- Returns the list of these dictionaries — this is exactly the `"branches"` field surfaced in the dict returned by `evaluate_circuit` and, ultimately, by `BasinHoppingRunner.run()`, and is what `print_results`, `plot_outcomes`, `analyze_rotations`, `analyze_loss`, and `analyze_cutoff` in `heraldo.analyze` all consume.

---

## Summary

| Function | Called when | Purpose |
|---|---|---|
| `evaluate_circuit` | Every optimizer step | Orchestrates simulation → projection → scoring → (optional) reporting |
| `_process_fixed_patterns` | `measurement_patterns` given | Projects `full_ket` onto caller-specified ancilla outcomes |
| `_process_beam_search` | `measurement_patterns=None` | Computes the full outcome distribution and keeps the top-`beam_width` outcomes |
| `_compute_fidelities_and_loss` | Every call with a valid projection | Computes rotation-invariant fidelities (FFT trick) and reduces them to a scalar loss |
| `_format_branch_details` | Only when `return_details=True` | Converts raw arrays into a reportable list of per-outcome dictionaries |

Understanding this pipeline is the prerequisite for two common extension points: adding a new projection strategy (mirror `_process_fixed_patterns`/`_process_beam_search`'s four-tuple contract), or adding a new `ObjectiveFunction` (must accept both the 1D and 2D `fidelities` shapes described in `docs/objectives.md`, since `_compute_fidelities_and_loss` calls it in both `phase_lock` modes).