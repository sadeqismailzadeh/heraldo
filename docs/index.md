# heraldo Documentation

**heraldo** is a Python framework for  multi-outcome quantum state optimization.

```{toctree}
:maxdepth: 2
:caption: Getting Started

installation
quickstart
```

```{toctree}
:maxdepth: 2
:caption: Guides & Architecture

circuits_comparison
sf_patches
targets
circuits
objectives
runner
patterns
serialization


```

```{toctree}
:maxdepth: 2
:caption: API Reference

api/index
api/components
api/experimental
```

## Features

- **Static Spatial Circuits**: Simulate continuous-variable spatial optical networks with photon-number-resolving (PNR) detectors.
- **Multi-Outcome Optimization**: Optimization algorithms supporting both beam search pattern discovery and fixed-pattern optimization.
- **Target State Generators**: Pre-built generators for Gottesman-Kitaev-Preskill (GKP) core states, Schrödinger cat states, binomial codes, and cubic phase states.
- **JIT-Compiled Performance Patches**: Optimized tensor representations and fast diagonal traversals for beam splitter interactions.

