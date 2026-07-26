"""
Optimization module for deterministic quantum circuit parameter search.
"""

# disable caching to save memory for large cutoff dims
from quantum_agent.patches.sf_operations_no_cache import disable_fock_caching
disable_fock_caching()

from quantum_agent.patches.beamsplitter_patch import patch_beamsplitter
patch_beamsplitter()

from quantum_agent.patches.prepare_multimode_patch import patch_prepare_multimode
patch_prepare_multimode()