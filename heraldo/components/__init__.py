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

# 1. Import the module we need to patch
import scipy.integrate

# 2. Check if the patch is needed to avoid errors
if not hasattr(scipy.integrate, 'simps'):
    print("Monkey patching scipy.integrate: 'simps' not found. Pointing to 'simpson'.")
    # 3. Create the 'simps' attribute and point it to the existing 'simpson' function.
    scipy.integrate.simps = scipy.integrate.simpson
else:
    print("'simps' already exists in scipy.integrate. No patch needed.")

# disable caching to save memory for large cutoff dims
from heraldo.patches.sf_operations_no_cache import disable_fock_caching
disable_fock_caching()

from heraldo.patches.beamsplitter_patch import patch_beamsplitter
patch_beamsplitter()

from heraldo.patches.prepare_multimode_patch import patch_prepare_multimode
patch_prepare_multimode()