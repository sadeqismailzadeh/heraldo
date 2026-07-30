====================================
Heraldo Documentation
====================================

.. image:: _static/circuit_comparison.svg
   :align: center
   :alt: Heraldo Photonic Circuit Architecture
   :width: 100%

|

**Heraldo** is an optical quantum state engineering and optimization framework for continuous-variable (CV) photonic quantum circuits. It combines custom JIT-compiled Fock simulation backends with multi-outcome optimization algorithms to turn "waste" photon-number measurement outcomes into useful non-Gaussian quantum states.

``heraldo`` is the official open-source software implementation accompanying the paper:

   *Multi-Outcome Circuit Optimization for Enhanced Non-Gaussian State Generation* (S. Ismailzadeh & B. Abedi Ravan, 2026)

.. note::
   **New to Heraldo?** Start with the :doc:`introduction` for core concepts and the :doc:`quickstart_tutorial` for a step-by-step hands-on guide to running your first optimization!

Key Features
============

* **Multi-Outcome Optimization Framework**:
  * **Resource Multiplexing**: Tune a single physical circuit to herald a portfolio of distinct target states across different PNR measurement patterns.
  * **Single-Target Probability Harvesting**: Aggregate multiple degenerate measurement outcomes to drastically boost the generation rate of a single target state.
* **Rotation-Invariant Optimization**: Uses an FFT-accelerated metric to evaluate state fidelity across phase-space rotations :math:`\hat{R}(\phi)`, avoiding unnecessary orientation constraints during global parameter search.
* **Supported Non-Gaussian Resource Families**: Built-in support for Gottesman-Kitaev-Preskill (GKP) core states, Schrödinger cat states, binomial quantum codes, and cubic phase states.
* **Flexible Architecture Engine**:
  * **Static Spatial Circuits** (:math:`T = 1`): Parallel waveguide/fiber channels evaluated simultaneously.
  * **Time-Domain Multiplexed (TDM) Circuits** (:math:`T \ge 1`): Recirculating optical delay loops carrying evolving states across temporal steps.
* **High-Performance JIT Backend**: Custom JIT-compiled Numba kernels for :math:`O(D^3)` beam splitters, disabled memory-leak gate caching, and unentangled pure-state detection optimizations for Strawberry Fields.

Quick Example
=============

.. code-block:: python

   import numpy as np
   from heraldo.components.circuits import TwoModeTimeDomainSqueezeOnly
   from heraldo.components.targets import SqueezedCatTarget
   from heraldo.components.runner import BasinHoppingRunner, fixed_pattern_capped_loss_fn
   from heraldo.utils import db_to_r

   # 1. Initialize 2-mode spatial circuit with 12 dB squeezing
   circuit = TwoModeTimeDomainSqueezeOnly(steps=1, clip_size=db_to_r(12.0), measure_fock_cutoff=30)

   # 2. Define targets: Even (|cat_+>) and Odd (|cat_->) Schrödinger cat states
   targets = [
       SqueezedCatTarget(alpha=np.sqrt(6), r=0.5, p=0),
       SqueezedCatTarget(alpha=np.sqrt(6), r=0.5, p=1),
   ]

   # 3. Optimize for fixed patterns n=4 (even) and n=5 (odd)
   runner = BasinHoppingRunner(
       circuit=circuit,
       target_gens=targets,
       cutoff_dim=30,
       measurement_patterns=[[(4,)], [(5,)]],
       loss_fn=fixed_pattern_capped_loss_fn
   )

   result = runner.run(n_iter=20)
   print(f"Total Success Probability: {result['total_probability']:.2%}")


Documentation Contents
======================

.. toctree::
   :maxdepth: 2
   :caption: Getting Started & Onboarding

   introduction
   quickstart_tutorial
   user_guide

.. toctree::
   :maxdepth: 2
   :caption: Architecture & Theory

   circuits_comparison

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   modules


Indices and Tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
