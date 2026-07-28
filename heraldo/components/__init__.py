"""
Components module for quantum targets, circuit models, and runner components.
"""

# disable caching to save memory for large cutoff dims
from heraldo.patches.sf_operations_no_cache import disable_fock_caching
disable_fock_caching()

from heraldo.patches.beamsplitter_patch import patch_beamsplitter
patch_beamsplitter()

from heraldo.patches.prepare_multimode_patch import patch_prepare_multimode
patch_prepare_multimode()