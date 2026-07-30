========================================================================
Research Student Onboarding & Quickstart Tutorial
========================================================================

Welcome to the ``heraldo`` research group! This hands-on tutorial is designed to onboard new research students and collaborators into the simulation, optimization, and analysis of non-Gaussian quantum optical circuits.

By following this tutorial, you will learn how to:

1. **Verify your environment** and understand how ``heraldo`` automatically patches Strawberry Fields for high performance.
2. **Define target non-Gaussian quantum states** (GKP core states, Schrödinger cat states, binomial codes, cubic phase states).
3. **Construct continuous-variable photonic circuits** (static spatial setups and time-domain multiplexed architectures).
4. **Execute Phase 1 optimization (Pattern Discovery)** using **Beam Search** to identify high-probability heralding patterns without prior assumptions.
5. **Execute Phase 2 optimization (Refinement)** using **Fixed-Pattern Optimization** for *resource multiplexing* or *single-target probability harvesting*.
6. **Analyze output quantum states**, check for rotation invariance in phase space, and evaluate performance under realistic photon loss.
7. **Run a complete, copy-pasteable end-to-end Python script** out of the box.

.. contents:: Table of Contents
   :local:
   :depth: 2

---

1. Prerequisites & Environment Check
------------------------------------

Before running optimizations, ensure that all required dependencies are installed:

.. code-block:: bash

   pip install numpy scipy pandas strawberryfields thewalrus numba tqdm

When you import ``heraldo.components``, the package automatically applies key performance patches:

* **Single-Threaded BLAS/NumPy Configuration**: Sets NumPy thread limits (``OMP_NUM_THREADS=1``, etc.) to 1 to prevent CPU thread oversubscription during parallel Basin-Hopping optimization runs.
* **Fock Caching Disable**: Prevents memory buildup during thousands of optimizer iterations.
* **JIT Beam Splitter Kernel**: Speeds up circuit tensor contraction to :math:`O(D^3)` complexity.
* **Unentangled State Optimization**: Preserves pure-state simulation when ancillae are re-prepared.

.. note::
   **Main Function Requirement**:
   Because ``BasinHoppingRunner`` uses multiprocessing to execute parallel optimization runs across CPU cores, all Python code that invokes circuit runner optimizations must be structured inside a ``main()`` function protected by an ``if __name__ == '__main__':`` block.

Test your setup by running the following Python snippet:

.. code-block:: python

   import heraldo.components.circuits
   import heraldo.components.targets
   import heraldo.components.runner
   from heraldo.utils import db_to_r

   print("Heraldo initialized successfully!")
   print(f"12 dB Squeezing in r-parameter: {db_to_r(12):.4f}")

---

2. Understanding the Physics & Target States
--------------------------------------------

Continuous-variable (CV) quantum computing requires non-Gaussian quantum resources to achieve fault tolerance and quantum advantage. ``heraldo`` provides pre-built target generators for the four major non-Gaussian state families used in CV quantum optics:

1. **Gottesman-Kitaev-Preskill (GKP) Core States** (:class:`~heraldo.components.targets.CoreGKPTarget`): Stellar representations of GKP logical qubits.
2. **Schrödinger Cat States** (:class:`~heraldo.components.targets.SqueezedCatTarget` / :class:`~heraldo.components.targets.CatTarget`): Superpositions of coherent states :math:`|\alpha\rangle \pm |-\alpha\rangle`.
3. **Binomial Quantum Error-Correcting Codes** (:class:`~heraldo.components.targets.BinomialCodeTarget`): Bosonic codewords constructed with binomial weighting.
4. **Cubic Phase States** (:class:`~heraldo.components.targets.CubicPhaseTarget`): States exhibiting non-Gaussian cubic phase nonlinearity :math:`\exp(i \gamma \hat{q}^3)`.

Code Example: Generating & Inspecting a Target State
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Here is how to instantiate and inspect a squeezed Schrödinger cat target state in the Fock basis:

.. code-block:: python

   import numpy as np
   from heraldo.components.targets import SqueezedCatTarget

   # Define an even squeezed cat state target (|alpha| = sqrt(6), r = 0.5, parity = 0)
   target = SqueezedCatTarget(alpha=np.sqrt(6), r=0.5, p=0)

   # Extract the state ket vector truncated at cutoff dimension D = 30
   cutoff_dim = 30
   target_ket = target.get_target_ket(cutoff_dim)

   print(f"Target Ket Shape: {target_ket.shape}")
   print(f"State Norm: {np.linalg.norm(target_ket):.6f}")
   print("Top Fock Coefficients:")
   for n, val in enumerate(target_ket[:10]):
       if np.abs(val) > 1e-4:
           print(f"  |{n:2d}>: {val:.6f}")

---

3. Setting Up a Photonic Quantum Circuit
----------------------------------------

In ``heraldo``, photonic circuits consist of:

* **Mode 0 (Loop / Memory Mode)**: The persistent optical mode carrying the unmeasured heralded output state :math:`|\psi\rangle`.
* **Modes 1 to N-1 (Ancilla Modes)**: Prepared with single-mode squeezing :math:`S(r)` and displacement :math:`D(\alpha)`, mixed with Mode 0 via beam splitters, and measured with Photon-Number-Resolving (PNR) detectors.

Setting ``steps = 1`` isolates a **single static spatial circuit** (reproducing the paper model).

Code Example: Initializing a 2-Mode Circuit
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from heraldo.components.circuits import TwoModeTimeDomainSqueezeOnly
   from heraldo.utils import db_to_r

   # Convert 12 dB experimental source squeezing to squeezing magnitude r
   squeezing_r = db_to_r(12.0)

   # Initialize a 2-mode circuit (1 signal mode + 1 ancilla mode, single spatial step T=1)
   circuit = TwoModeTimeDomainSqueezeOnly(
       steps=1,                   # T = 1 matches the research paper's spatial architecture
       time_invariant=False,      # Step-independent control parameters
       clip_size=squeezing_r,     # Bound on squeezing magnitude
       measure_fock_cutoff=30,    # PNR detector cutoff dimension
   )

   print(f"Number of modes: {circuit.num_modes}")
   print(f"Initial parameters required: {circuit.num_initial_parameters}")
   print(f"Per-step parameters: {circuit.per_step_parameter_names[2:]}")

---

4. Phase 1: Unconstrained Pattern Discovery (Beam Search)
---------------------------------------------------------

When starting optimization for a new target state, you usually do not know beforehand which PNR detection outcomes :math:`(n_1, \dots, n_{N-1})` will herald high-fidelity states.

In **Phase 1**, we set ``measurement_patterns = None`` and use the **Beam Search loss function** (:func:`~heraldo.components.runner.beam_search_loss_fn`). The optimizer dynamically explores the combinatorial tree of PNR detection outcomes, retaining the top ``beam_width`` most probable trajectories at each step.

Code Example: Running Beam Search Pattern Discovery
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import numpy as np
   from heraldo.components.circuits import TwoModeTimeDomainSqueezeOnly
   from heraldo.components.targets imp   # 1. Setup circuit & targets
   squeezing_r = db_to_r(12.0)
   circuit = TwoModeTimeDomainSqueezeOnly(steps=1, clip_size=squeezing_r, measure_fock_cutoff=30)

   targets = [
       SqueezedCatTarget(alpha=np.sqrt(6), r=0.5, p=0),  # even cat
       SqueezedCatTarget(alpha=np.sqrt(6)   def main():
       # 1. Setup circuit & targets
       squeezing_r = db_to_r(12.0)
       circuit = TwoModeTimeDomainSqueezeOnly(steps=1, clip_size=squeezing_r, measure_fock_cutoff=30)

       targets = [
           SqueezedCatTarget(alpha=np.sqrt(6), r=0.5, p=0),  # even cat
           SqueezedCatTarget(alpha=np.sqrt(6), r=0.5, p=1),  # odd cat
       ]

       # 2. Configure Basin-Hopping runner in Beam Search discovery mode
       runner = BasinHoppingRunner(
           circuit=circuit,
           target_gens=targets,
           cutoff_dim=30,
           beam_width=200,             # Keep top 200 trajectories
           penalty_strength=1.0,
           measurement_patterns=None,  # Triggers Beam Search discovery mode!
           loss_fn=beam_search_loss_fn,
           num_processes=4,
       )

       # 3. Run optimization
       result = runner.run(n_generations=20, method="L-BFGS-B")

       print(f"Discovery Loss: {result['loss']:.4f}")
       print("\nDiscovered Heralding Outcomes:")
       for branch in result["branches"]:
           print(f"  Outcome: {branch['outcome']} | Prob: {branch['prob']:.4f} | Fidelity: {branch['fidelity']:.4f} | Target Index: {branch['target_idx']}")

   if __name__ == "__main__":
       main()

---

5. Phase 2: Fixed-Pattern Optimization & Refinement
---------------------------------------------------

Once Phase 1 discovers promising heralding patterns (e.g. outcome ``(4,)`` for even cat and ``(5,)`` for odd cat), you transition to **Phase 2 (Fixed-Pattern Optimization)** to fine-tune the parameters.

Depending on your research goal, you can choose between two strategies:

* **Resource Multiplexing**: Optimize a single physical circuit to simultaneously generate *different* target states on different heralding outcomes (e.g., target even cat on ``(4,)`` and odd cat on ``(5,)``).
* **Single-Target Probability Harvesting**: Aggregate *multiple degenerate* heralding outcomes that all prepare the *same* target state (e.g., harvest outcomes ``(1,3)``, ``(3,1)``, and ``(2,2)`` for GKP logical zero).

Code Example: Fixed-Pattern Optimization for Resource Multiplexing
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import numpy as np
   from heraldo.components.circuits import TwoModeTimeDomainSqueezeOnly
   from heraldo.components.targets import SqueezedCatTarget
   from heraldo.components.runner import BasinHoppingRunner, fixed_pattern_capped_loss_fn
   from heraldo.utils import db_to_r

   def main():
       squeezing_r = db_to_r(12.0)
       circuit = TwoModeTimeDomainSqueezeOnly(steps=1, clip_size=squeezing_r, measure_fock_cutoff=30)
       targets = [
           SqueezedCatTarget(alpha=np.sqrt(6), r=0.5, p=0),
           SqueezedCatTarget(alpha=np.sqrt(6), r=0.5, p=1),
       ]

       # Define fixed heralding patterns discovered in Phase 1
       patterns = [[(4,)], [(5,)]]

       runner_fixed = BasinHoppingRunner(
           circuit=circuit,
           target_gens=targets,
           cutoff_dim=30,
           beam_width=200,
           penalty_strength=1.0,
           measurement_patterns=patterns,       # Fixed outcome patterns
           loss_fn=fixed_pattern_capped_loss_fn, # Capped fidelity objective
           num_processes=4,
       )

       res_fixed = runner_fixed.run(n_iter=20, method="L-BFGS-B")

       print(f"Refined Objective Score: {res_fixed['expected_fidelity']:.4f}")
       print(f"Total Aggregated Success Probability: {res_fixed['total_probability']:.2%}")

   if __name__ == "__main__":
       main()
on Invariance
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Heralded states across different measurement outcomes may differ by a global phase-space rotation :math:`\hat{R}(\phi) = e^{i \hat{n} \phi}`. Because a phase-space rotation can be corrected post-generation using a simple optical delay, ``heraldo`` uses an FFT-accelerated rotation-invariant fidelity metric:

.. code-block:: python

   from heraldo.utils import fidelity_max_rotation, fidelity_pure_state

   # Compare a heralded state with a target state
   target_ket = targets[0].get_target_ket(cutoff_dim=30)
   heralded_ket = target_ket  # Replace with actual heralded ket vector

   # Standard fidelity F = |<phi|psi>|^2
   standard_fid = fidelity_pure_state(target_ket, heralded_ket)

   # Rotation-maximized fidelity F_max = max_phi |<phi| R(phi) |psi>|^2
   max_rot_fid = fidelity_max_rotation(target_ket, heralded_ket, n_fft=256)

   print(f"Standard Fidelity: {standard_fid:.6f}")
   print(f"Max Rotation-Invariant Fidelity: {max_rot_fid:.6f}")

6.2 Simulating Photon Loss
~~~~~~~~~~~~~~~~~~~~~~~~~~

To model realistic experimental loss, configure ``loss_transmissivity`` on the circuit (where :math:`\eta = 1 - \text{loss\_rate}`):

.. code-block:: python

   # Simulate 1% channel loss (transmissivity eta = 0.99)
   lossy_circuit = TwoModeTimeDomainSqueezeOnly(
       steps=1,
       clip_size=squeezing_r,
       measure_fock_cutoff=30,
       loss_transmissivity=0.99,
   )

---

7. End-to-End Onboarding Script
--------------------------------

Here is a complete, copy-pasteable script that executes both Phase 1 (Pattern Discovery) and Phase 2 (Fixed-Pattern Refinement) for a 3-mode circuit preparing GKP core states. You can run this directly in Python:

.. code-block:: python

   """
   Complete Onboarding Example: GKP Core State Generation in a 3-Mode Circuit
   -------------------------------------------------------------------------
   Demonstrates:
     1. Setting up a 3-mode circuit (1 signal mode + 2 ancillae).
     2. Loading a GKP logical 0 core state target from CSV data.
     3. Running Phase 1 Beam Search to discover viable heralding patterns.
     4. Running Phase 2 Fixed-Pattern Optimization to harvest multiple outcomes.
   """

   import numpy as np
   from pathlib import Path

   import heraldo.components.circuits as circuits
   import heraldo.components.targets as targets
   from heraldo.components.runner import (
       BasinHoppingRunner,
       beam_search_loss_fn,
       fixed_pattern_capped_loss_fn,
   )
   from heraldo.utils import db_to_r, fidelity_max_rotation


   def main():
       print("=" * 70)
       print("HERALDO RESEARCH ONBOARDING: GKP CORE STATE OPTIMIZATION")
       print("=" * 70)

       # --- 1. Parameters & Configuration ---
       cutoff_dim = 30
       squeezing_db = 12.0
       squeezing_r = db_to_r(squeezing_db)

       # Locate CSV file for GKP core state coefficients
       csv_path = Path("data/GKP_core_coefficients.csv").resolve()

       # --- 2. Instantiate Target & Circuit ---
       print("\n[Step 1] Initializing GKP Core Target (|0_A4>, n_max=4, Delta=10dB)...")
       target = targets.CoreGKPTarget(
           csv_path=str(csv_path),
           n_max=4,
           delta_db=10.0,
           mu=0,
           apply_squeezing=False,
       )

       target_ket = target.get_target_ket(cutoff_dim)
       print(f"Target state norm: {np.linalg.norm(target_ket):.6f}")

       print("\n[Step 2] Building 3-Mode Photonic Circuit...")
       circuit = circuits.ThreeModeTimeDomainSqueezeOnly(
           steps=1,
           time_invariant=False,
           clip_size=squeezing_r,
           measure_fock_cutoff=cutoff_dim,
       )

       # --- 3. Phase 1: Beam Search Pattern Discovery ---
       print("\n[Step 3] Running Phase 1: Unconstrained Beam Search Pattern Discovery...")
       runner_phase1 = BasinHoppingRunner(
           circuit=circuit,
           target_gens=[target],
           cutoff_dim=cutoff_dim,
           beam_width=100,
           penalty_strength=1.0,
           measurement_patterns=None,  # Triggers Beam Search discovery
           loss_fn=beam_search_loss_fn,
           num_processes=2,
       )

       res_phase1 = runner_phase1.run(n_iter=10, method="L-BFGS-B")
       print(f"Phase 1 Objective Loss: {res_phase1['loss']:.4f}")
       print("Discovered outcomes in Phase 1:")
       for b in res_phase1["branches"][:5]:
           print(f"  Outcome: {b['outcome']} | Prob: {b['prob']:.4f} | Fidelity: {b['fidelity']:.4f}")

       # --- 4. Phase 2: Fixed-Pattern Single-Target Harvesting ---
       print("\n[Step 4] Running Phase 2: Single-Target Probability Harvesting...")
       # Target symmetric degenerate outcomes: (1,3) and (3,1)
       harvest_patterns = [[(1, 3)], [(3, 1)]]

       runner_phase2 = BasinHoppingRunner(
           circuit=circuit,
           target_gens=[target],
           cutoff_dim=cutoff_dim,
           beam_width=100,
           penalty_strength=1.0,
           measurement_patterns=harvest_patterns,
           loss_fn=fixed_pattern_capped_loss_fn,
           num_processes=2,
       )

       res_phase2 = runner_phase2.run(n_iter=10, method="L-BFGS-B")
       print("\n" + "=" * 70)
       print("FINAL HARVESTING OPTIMIZATION RESULTS")
       print("=" * 70)
       print(f"Total Aggregated Target Success Probability: {res_phase2['total_probability']:.2%}")
       for b in res_phase2["branches"]:
           print(f"  Outcome Pattern: {b['outcome']} -> Prob: {b['prob']:.4f}, Fidelity: {b['fidelity']:.4f}")

       print("\nTutorial complete! You are ready to run your own optimizations.")


   if __name__ == "__main__":
       main()

---

8. Summary & Next Steps
-----------------------

* Read :doc:`circuits_comparison` for an architectural comparison of static spatial and time-domain multiplexed circuits.
* Explore the :doc:`user_guide` for details on circuit architectures, loss functions, and patch mechanisms.
* Browse the :doc:`modules` section for complete API specifications of all classes and methods in ``heraldo``.
