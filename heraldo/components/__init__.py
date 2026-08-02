"""
Components module for quantum targets, circuit models, and runner components.
"""

import os

# --- Set thread limits for NumPy/OpenBLAS/MKL before importing heavy backend libraries ---
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'


# disable caching to save memory for large cutoff dims
from heraldo.patches.sf_operations_no_cache import disable_fock_caching
disable_fock_caching()

from heraldo.patches.beamsplitter_patch import patch_beamsplitter
patch_beamsplitter()

from heraldo.patches.prepare_multimode_patch import patch_prepare_multimode
patch_prepare_multimode()