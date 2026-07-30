========================================================================
Introduction & Core Concepts
========================================================================

.. contents:: Table of Contents
   :local:
   :depth: 2

Executive Summary
=================

**Heraldo** is an advanced Python modeling, simulation, and optimization framework for continuous-variable (CV) photonic quantum computing circuits. It specializes in the **conditional generation of non-Gaussian quantum states** using Gaussian operations (squeezing, displacement, beam splitters) combined with Photon-Number-Resolving (PNR) detectors.

``heraldo`` is the official implementation of the research paper:

   *Multi-Outcome Circuit Optimization for Enhanced Non-Gaussian State Generation* (S. Ismailzadeh & B. Abedi Ravan, 2026)

While conventional photonic circuit design optimizes for a single, fixed measurement outcome and discards all other detection events as useless waste, ``heraldo`` introduces a **multi-outcome optimization framework**. By jointly optimizing classical control parameters across multiple photon-number detection patterns, ``heraldo`` turns waste measurement events into usable quantum resources, dramatically enhancing the total state generation rate on near-term optical hardware.

------------------------------------------------------------------------

The Problem: Waste Outcomes in Photonic State Generation
=========================================================

Continuous-variable photonic quantum computing encodes information in the infinite-dimensional Hilbert space of harmonic modes. While Gaussian states (squeezed and coherent states) and Gaussian operations (beam splitters, phase shifters) are easy to prepare and manipulate, **non-Gaussian resources** are strictly required to achieve quantum computational universality and fault-tolerant quantum error correction.

In optical systems, non-Gaussianity is typically induced probabilistically through **heralded state engineering**:

1. Squeezed vacuum modes are prepared and interfered in a linear optical network.
2. PNR detectors measure :math:`N-1` ancillary modes.
3. Measuring a specific photon number pattern :math:`\mathbf{n} = (n_1, n_2, \dots, n_{N-1})` heralds a non-Gaussian target state :math:`|\psi\rangle` in the remaining unmeasured mode.

.. image:: _static/circuit_comparison.svg
   :align: center
   :alt: Photonic Circuit Architecture Comparison
   :width: 100%

*Figure 1: Comparison between (a) static spatial photonic circuits and (b) time-domain multiplexed loop architectures supported by heraldo.*

The Central Bottleneck
~~~~~~~~~~~~~~~~~~~~~~

Because quantum measurement is inherently probabilistic, the success probability of heralding a specific target state under a single outcome pattern :math:`\mathbf{n}` is often prohibitively low (frequently :math:`< 1\%`). In traditional setups, whenever the PNR detectors observe *any other* outcome pattern :math:`\mathbf{n}' \neq \mathbf{n}`, the experiment fails, and the state in the output mode is discarded.

------------------------------------------------------------------------

The Solution: Multi-Outcome Circuit Optimization
=================================================

``heraldo`` shifts the design paradigm by asking: **What if a single physical circuit could be tuned to herald useful quantum states across MULTIPLE detection outcomes?**

Instead of optimizing circuit parameters :math:`\boldsymbol{\theta}` for one target and one fixed detection event, ``heraldo`` optimizes the circuit across a portfolio of accepted outcome patterns :math:`S = \{\mathbf{n}_k\}`.

``heraldo`` implements two key operational mechanisms:

1. Resource Multiplexing
------------------------
A single physical circuit is optimized to generate a **diverse set of distinct non-Gaussian states** across different heralding outcomes.

* *Example*: A 2-mode circuit is tuned so that outcome :math:`n=4` heralds an **even Schrödinger cat state** :math:`|\text{cat}_+\rangle`, while outcome :math:`n=5` heralds an **odd Schrödinger cat state** :math:`|\text{cat}_-\rangle`.
* *Benefit*: A single fixed hardware layout acts as a multi-resource quantum state generator, doubling or tripling the total useful resource yield (:math:`P_{\text{agg}}`).

2. Single-Target Probability Harvesting
---------------------------------------
A single circuit is optimized to generate **one specific target state** by accepting **multiple degenerate heralding outcomes**.

* *Example*: A 3-mode circuit preparing a GKP logical zero state :math:`|0_{A4}\rangle` is configured to accept outcomes :math:`(1,3)`, :math:`(3,1)`, and :math:`(2,2)` simultaneously.
* *Benefit*: Aggregating degenerate detection outcomes substantially boosts the production rate of a single target state (e.g. increasing the generation probability from :math:`2.3\%` to :math:`6.6\%`).

3. Rotation-Invariant Optimization
-----------------------------------
Target states heralded across different measurement outcomes may differ by a phase-space rotation angle :math:`\hat{R}(\phi) = e^{i \hat{n} \phi}`. Because a phase-space rotation in optics is physically equivalent to a simple optical delay line or LO phase shift, ``heraldo`` employs an **FFT-accelerated rotation-invariant fidelity metric**:

.. math::

   \mathcal{F}_{i,k} = \max_{\phi} \left| \langle \phi_i | e^{i \hat{n} \phi} | \psi_k \rangle \right|^2

This removes phase constraints during global parameter search, accelerating convergence without sacrificing physical state quality.

------------------------------------------------------------------------

Supported Non-Gaussian Target State Families
============================================

``heraldo`` includes built-in state generators for four major families of non-Gaussian targets required for fault-tolerant continuous-variable quantum error correction and universal gate execution:

.. list-table::
   :widths: 25 35 40
   :header-rows: 1

   * - Target Family
     - Mathematical Representation
     - Primary Application
   * - **Gottesman-Kitaev-Preskill (GKP) Core States**
     - :math:`\ket{\psi_A} \approx \hat{D}(\beta)\hat{S}(\xi) \sum_{n=0}^{n_{\text{max}}} c_n \ket{n}`
     - Grid-code logical qubits for fault-tolerant CV quantum computing.
   * - **Schrödinger Cat States**
     - :math:`\hat{S}(r) \mathcal{N}_{\pm} (\ket{\alpha} \pm \ket{-\alpha})`
     - Parity-encoded logical qubits resilient to photon loss.
   * - **Binomial Quantum Codes**
     - :math:`\frac{1}{\sqrt{2^N}} \sum_{p} \sqrt{\binom{N+1}{p}} |p(S+1)\rangle`
     - Exact error-correcting codes protecting against photon loss/gain.
   * - **Cubic Phase States**
     - :math:`\hat{D}(\alpha) e^{i \gamma \hat{Q}^3} \hat{S}(r) \ket{0}`
     - Universal non-Gaussian resource for non-linear gate synthesis.

------------------------------------------------------------------------

Photonic Architecture Frameworks
================================

``heraldo`` seamlessly supports two complementary physical circuit paradigms:

1. Static Spatial Architectures (:math:`T = 1`)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
* Modeled in the research paper.
* All :math:`N` modes exist as parallel physical waveguides/fibers on a chip.
* Photons enter simultaneously and interfere via a static network of spatial beam splitters and phase shifters.
* Set ``steps = 1`` in ``heraldo`` to simulate this architecture.

2. Time-Domain Multiplexed (TDM) Loop Architectures (:math:`T \ge 1`)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
* Pulse trains travel down a single optical line into a recirculating delay loop.
* **Mode 0 (Loop Memory)** carries the evolving state sequentially across time steps :math:`t = 1, 2, \dots, T`.
* **Modes :math:`1 \dots N-1` (Ancillae)** represent fresh temporal pulses injected at each step.
* Set ``steps > 1`` in ``heraldo`` to simulate TDM recirculating delay loops.

To learn more about the equivalence between temporal loops and spatial networks, see :doc:`circuits_comparison`.

------------------------------------------------------------------------

Two-Phase Optimization Pipeline
================================

To navigate the complex non-convex optimization landscape of continuous-variable circuits, ``heraldo`` implements a two-phase optimization pipeline:

.. code-block:: text

   +-----------------------------------------------------------------------+
   |                       PHASE 1: BEAM SEARCH                            |
   |           Unconstrained Pattern Discovery (measurement_patterns=None) |
   |  * Dynamically tracks top-B probable trajectories at each step        |
   |  * Uses non-linear score function L_beam to locate outcome clusters   |
   +-----------------------------------------------------------------------+
                                      |
                                      v
   +-----------------------------------------------------------------------+
   |                   PHASE 2: FIXED-PATTERN REFINEMENT                   |
   |        Polishing Target Cluster (measurement_patterns=[...])           |
   |  * Capped Regime (f_cap=0.95): Maximizes probability yield             |
   |  * Free Regime: Pushes state fidelity to hardware limits              |
   +-----------------------------------------------------------------------+

------------------------------------------------------------------------


Simulating high-dimensional continuous-variable quantum circuits in the Fock basis is computationally intensive. When ``heraldo`` is imported, it automatically applies three optimized patches to the Strawberry Fields engine:

1. **Fock Caching Disable**: Replaces global gate caching with uncached evaluation, preventing memory leaks during long optimization sweeps.
2. **JIT-Compiled Beam Splitter**: Implements an :math:`O(D^3)` diagonal-traversal kernel compiled with Numba, speeding up tensor contraction by orders of magnitude.
3. **Unentangled State Preservation**: Detects separable pure states when resetting ancilla modes, avoiding unnecessary conversion to density matrices.
=======
High-Performance Backend & Patches
==================================

Importing ``heraldo.components`` automatically applies three patches to
the Strawberry Fields Fock backend: disabled gate-tensor caching (bounds
memory across long optimization sweeps), a JIT-compiled :math:`O(D^3)`
beam-splitter kernel, and an unentangled-state fast path that avoids
dropping to density-matrix simulation when it isn't necessary.

See :ref:`patches-section` in the :doc:`user_guide` for the full
explanation of each patch, including why the ``scipy.integrate.simps``
compatibility shim shows up alongside them.
==================================

Simulating high-dimensional continuous-variable quantum circuits in the Fock basis is computationally intensive. When ``heraldo`` is imported, it automatically applies three optimized patches to the Strawberry Fields engine:

1. **Fock Caching Disable**: Replaces global gate caching with uncached evaluation, preventing memory leaks during long optimization sweeps.
2. **JIT-Compiled Beam Splitter**: Implements an :math:`O(D^3)` diagonal-traversal kernel compiled with Numba, speeding up tensor contraction by orders of magnitude.
3. **Unentangled State Preservation**: Detects separable pure states when resetting ancilla modes, avoiding unnecessary conversion to density matrices.

------------------------------------------------------------------------

Documentation Roadmap
=====================

* :doc:`quickstart_tutorial`: A step-by-step hands-on guide for onboarding new research students.
* :doc:`user_guide`: In-depth user manual detailing package components, parameter structures, and workflows.
* :doc:`circuits_comparison`: Technical guide comparing static spatial circuits and time-domain multiplexed loop architectures.
* :doc:`modules`: Auto-generated API reference for all modules, classes, and utility functions.
