========================================================================
Circuit Architecture Comparison: Static (Paper) vs. Time-Multiplexed (Package)
========================================================================

Overview
--------

This document clarifies the architectural connection and differences between the photonic circuit model presented in the theoretical paper (``paper.tex``) and the software implementation provided in the ``heraldo`` Python package.

In summary:
- **Paper Model**: Uses a **static spatial multi-mode circuit** (Gaussian Boson Sampling-like layout) where all mode preparations, unitary operations, and photon-number-resolving detections (PNRDs) occur in a single spatial stage across $N$ physical modes.
- **Package Implementation**: Uses a **Time-Domain Multiplexed (TDM) circuit** model where a single loop mode (Mode 0) interacts sequentially over $T$ time steps with ancillary modes (Modes $1, \dots, N-1$).
- **Equivalence**: The static spatial circuit described in the paper is a **special case** of the package's time-multiplexed architecture corresponding to setting ``steps = 1``.

------------------------------------------------------------------------

Architecture Diagram
--------------------

The following diagram illustrates the architectural comparison between the static spatial setup in the paper and the time-domain multiplexed layout in ``heraldo``:

.. image:: _static/circuit_comparison.svg
   :align: center
   :alt: Comparison between Static Spatial and Time-Domain Multiplexed Circuits
   :width: 100%

------------------------------------------------------------------------

Time-Domain Unravelling to Spatial Networks
-------------------------------------------

A time-domain multiplexed fiber loop fed by a pulse train can be mathematically 'unraveled' into an equivalent spatial network of beam splitters:

.. image:: _static/tdm_loop_unravelling.svg
   :align: center
   :alt: Time-Domain Loop Unravelling into Equivalent Spatial Beam Splitter Network
   :width: 100%

------------------------------------------------------------------------

Static Spatial Circuits (Paper Architecture)
--------------------------------------------

The circuit described in Section II.A of the paper represents a static spatial setup:

1. **Input State**: An $N$-mode vacuum state $|0\rangle^{\otimes N}$.
2. **Gaussian State Preparation**: Single-mode squeezing $S_i(r_i, \phi_i)$ and displacement $D_i(\alpha_i, \theta_i)$ operators are applied independently to each spatial mode $i \in \{1, \dots, N\}$.
3. **Linear Optical Interferometer**: Modes pass through a passive, multi-mode linear unitary transformation $U$ constructed from a spatial network of beam splitters ($BS$) and phase shifters.
4. **Heralded Measurement**: Photon-Number-Resolving Detectors (PNRDs) are applied to $N-1$ ancillary spatial modes. Detecting photon outcome $\mathbf{n} = (n_1, n_2, \dots, n_{N-1})$ heralds a non-Gaussian target state $|\psi\rangle$ in the single unmeasured output mode.

Key characteristics:
- All operations occur simultaneously across physical spatial channels.
- Hardcoded spatial mode count $N$ (e.g., $N=2$ or $N=3$).
- Single-shot execution without temporal loopback.

------------------------------------------------------------------------

Time-Domain Multiplexed Circuits (Package Architecture)
-------------------------------------------------------

The ``heraldo`` package generalizes circuit simulation through the ``TimeMultiplexedCircuit`` abstract interface (located in ``heraldo.components.interfaces`` and ``heraldo.components.circuits``).

Mode Roles and Step Transitions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In ``heraldo``, modes are categorized as follows:

- **Mode 0 (Loop Mode)**: The persistent storage/signal mode that holds the evolving quantum state across all time steps.
- **Modes $1, \dots, N-1$ (Ancilla Modes)**: Temporal pulse modes that interact with Mode 0 during step $t$, are measured at the end of step $t$, and are discarded.

.. note::
   **Does Mode 0 get swapped with Ancilla Modes at each step?**
   No. Mode 0 is **never swapped or discarded**. Mode 0 acts as the continuous quantum memory line across all $T$ steps. At each step $t$:
   
   1. Mode 0 is loaded with the state carried over from Step $t-1$ (``parent_ket``).
   2. Ancilla modes (Modes $1, \dots, N-1$) are freshly prepared.
   3. Beam splitters mix Mode 0 with the ancilla modes.
   4. Ancilla modes are measured via PNRDs and discarded.
   5. The post-measurement conditional state in Mode 0 is normalized and becomes the input for Step $t+1$.

   Any physical swapping or routing in optical delay hardware is mathematically absorbed into the parameterization of the beam splitter gates $\text{BS}(\theta, \phi)$ acting between Mode 0 and the ancillae.

Step Control & Parameters
~~~~~~~~~~~~~~~~~~~~~~~~~

- **Sequential Steps ($T \ge 1$)**: Specified by the ``steps`` parameter.
- **Time-Invariant** (``time_invariant=True``): The same gate parameters are applied at every step $t$.
- **Time-Variant** (``time_invariant=False``): Independent parameters $\boldsymbol{\theta}(t)$ are tuned for each step $t$.

------------------------------------------------------------------------

The Paper as a Special Case (``steps = 1``)
-------------------------------------------

When setting ``steps = 1`` in ``heraldo``, the time-domain multiplexed loop architecture reduces exactly to the static spatial circuit described in the paper:

- A 1-step, 2-mode TDM circuit (``TwoModeTimeDomainGeneral(steps=1)``) simulates a 2-mode spatial GBS setup (1 signal mode + 1 ancillary mode).
- A 1-step, 3-mode TDM circuit (``ThreeModeTimeDomainGeneral(steps=1)``) simulates a 3-mode spatial GBS setup (1 signal mode + 2 ancillary modes).

Mathematical Mapping
~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :widths: 30 35 35
   :header-rows: 1

   * - Feature
     - Paper (Static Spatial)
     - Package (TDM, ``steps = 1``)
   * - Signal / Target Mode
     - Unmeasured mode $N$
     - Loop Mode 0
   * - Ancillary Modes
     - Spatial modes $1, \dots, N-1$
     - Ancilla Modes $1, \dots, N-1$
   * - Initial Squeezing
     - $S_i$ applied to all $N$ modes
     - Initial squeezing on Mode 0 + step squeezing on Ancillas
   * - Interferometer $U$
     - Multi-mode unitary $U$
     - Beam splitter sequence (e.g., $BS_{0,1} \rightarrow BS_{1,2} \rightarrow BS_{0,1}$)
   * - Measurements
     - PNRDs on $N-1$ spatial modes
     - PNRDs on Modes $1, \dots, N-1$ after Step 1

------------------------------------------------------------------------

Comparison Summary Table
------------------------

.. list-table::
   :widths: 25 35 40
   :header-rows: 1

   * - Property
     - Paper Circuit Model
     - Package Circuit Framework
   * - **Physical Domain**
     - Spatial channels
     - Time-domain pulse trains / delay loops
   * - **Temporal Steps**
     - Fixed at 1
     - Arbitrary $T \ge 1$ (configurable)
   * - **Mode Persistence**
     - $N$ parallel spatial modes
     - Mode 0 persists; Modes $1 \dots N-1$ reset each step
   * - **Scalability**
     - Requires physical optical components per mode
     - Reuses same loop hardware across $T$ steps
   * - **Parameterization**
     - Single static parameter set $\boldsymbol{\theta}$
     - Static or step-dependent $\boldsymbol{\theta}(t)$
   * - **Python Class**
     - N/A (Analytical/Numerical equations)
     - ``TwoModeTimeDomainGeneral``, ``ThreeModeTimeDomainGeneral``, etc.

------------------------------------------------------------------------

Code Example: Reproducing Paper Results in Package
--------------------------------------------------

To run optimizations corresponding to the static circuits in the paper, instantiate the target circuit with ``steps=1``:

.. code-block:: python

   from heraldo.components.circuits import TwoModeTimeDomainGeneral, ThreeModeTimeDomainGeneral
   from heraldo.components.runner import BasinHoppingRunner
   from heraldo.components.targets import CoreGKPTarget

   # 1. Recreate the 2-mode static paper circuit using TDM with steps=1
   circuit_2mode = TwoModeTimeDomainGeneral(
       steps=1,                # steps=1 reproduces the paper's static model
       time_invariant=True,
       clip_size=2.0,
       measure_fock_cutoff=10
   )

   # 2. Recreate the 3-mode static paper circuit using TDM with steps=1
   circuit_3mode = ThreeModeTimeDomainGeneral(
       steps=1,                # steps=1 reproduces the paper's static model
       time_invariant=True,
       clip_size=2.0,
       measure_fock_cutoff=10
   )

   # 3. Define target state (e.g., GKP core logical 1)
   target = CoreGKPTarget(n_max=4, delta_db=10.0, mu=1)

   # 4. Initialize runner
   runner = BasinHoppingRunner(
       circuit=circuit_2mode,
       target_gens=[target],
       cutoff_dim=30,
       beam_width=100
   )

   # Running optimization with steps=1 evaluates the static spatial layout
   # Increasing steps > 1 enables full time-domain multiplexing multi-step evolution.
