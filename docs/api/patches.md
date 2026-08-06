# Performance Patches API

The `heraldo.patches` package contains monkey-patches applied to Strawberry Fields backend operations to optimize execution speed, enforce $O(D^3)$ beam-splitter complexity, disable unneeded LRU gate caching, and maintain pure state representations during multi-mode state preparation.

## Beam Splitter $O(D^3)$ Patch

```{eval-rst}
.. automodule:: heraldo.patches.beamsplitter_patch
   :members:
   :undoc-members:
   :show-inheritance:
```

## Pure State Multimode Preparation Patch

```{eval-rst}
.. automodule:: heraldo.patches.prepare_multimode_patch
   :members:
   :undoc-members:
   :show-inheritance:
```

## Gate Caching Disabler Patch

```{eval-rst}
.. automodule:: heraldo.patches.sf_operations_no_cache
   :members:
   :undoc-members:
   :show-inheritance:
```