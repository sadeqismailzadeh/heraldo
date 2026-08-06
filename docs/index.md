# heraldo Documentation

**heraldo** is a Python framework for multi-outcome optimization of continuous-variable (CV) photonic circuits. It targets Gaussian Boson Sampling (GBS)-like devices — squeezing, displacement, and beam-splitter networks followed by photon-number-resolving (PNR) detection — and optimizes them to herald non-Gaussian quantum states such as Gottesman–Kitaev–Preskill (GKP) core states, Schrödinger cat states, binomial codes, and cubic phase states.

Rather than tuning a circuit for a single heralding outcome, heraldo implements the multi-outcome strategy described in the paper: a *beam search* phase autonomously discovers promising measurement patterns, followed by *fixed-pattern* refinement that either multiplexes a diverse set of target resources across different outcomes, or harvests several degenerate outcomes to boost the production rate of one target state. Circuits are simulated with [Strawberry Fields](https://strawberryfields.ai/) in the Fock basis, with several performance patches (JIT-compiled beam splitters, disabled gate caching, purity-preserving state preparation) applied automatically to keep large-cutoff optimization runs fast and memory-bounded.

```{toctree}
:maxdepth: 2
:caption: Getting Started

installation
quickstart
```

```{toctree}
:maxdepth: 2
:caption: Guides & Architecture

circuits
targets
objectives
patterns
runner
serialization
analyze
internals
developer_guide
```

```{toctree}
:maxdepth: 2
:caption: API Reference

api/index
api/components
api/analyze
api/serialization
api/utils
api/patches
```

## Features

- **Static spatial GBS-like circuits**: Two-, three-, and four-mode continuous-variable circuits combining squeezing, displacement, and fixed beam-splitter networks, with photon-number-resolving detection on all but one (heralded) mode. See [Circuit Models](circuits.md).

- **Multi-outcome optimization**: `BasinHoppingRunner` supports both *beam search* (to autonomously discover high-probability heralding patterns) and *fixed-pattern* optimization — enabling either **resource multiplexing** (a diverse set of targets across different outcomes) or **single-target probability harvesting** (aggregating degenerate outcomes for one target). Users can run beam search to discover patterns and subsequently refine them with fixed-pattern optimization. See [Optimization Runner](runner.md) and [Optimization Objectives](objectives.md).
- **Rotation-invariant fidelity**: State fidelities are evaluated over all phase-space rotations via a batched FFT, so a heralded state isn't penalized for a rotation that a downstream optical delay or software phase-tracking can correct. See [Analyzing and Visualizing Results](analyze.md).
- **Target state generators**: Pre-built generators for GKP core states, Schrödinger cat states, binomial codes, and cubic phase states, plus a simple interface for defining custom targets. See [Target State Generators](targets.md).
- **Serialization & reconstruction**: Optimization results can be saved to disk and later reloaded with full object reconstruction for further analysis. See [Serialization and Reconstruction Guide](serialization.md).
- **Post-optimization analysis**: Utilities to summarize results, plot Wigner functions and Fock distributions, check rotation invariance, and stress-test circuits against photon loss and Fock-cutoff truncation. See [Analyzing and Visualizing Results](analyze.md).
- **JIT-compiled performance patches**: An $O(D^3)$ Numba-compiled beam-splitter implementation, disabled Strawberry Fields gate caching, and purity-preserving multimode state preparation, all applied automatically on import. See [Internals](internals.md).

## Citing heraldo

heraldo implements the multi-outcome optimization strategy introduced in:

> S. Ismailzadeh and B. Abedi Ravan, *"Multi-Outcome Circuit Optimization for Enhanced Non-Gaussian State Generation."*

If heraldo is useful in your research, please cite the paper above alongside the software itself.
