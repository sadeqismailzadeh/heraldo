"""
Optimization module for deterministic quantum circuit parameter search.
"""

# disable caching to save memory for large cutoff dims
from quantum_agent.patches.sf_operations_no_cache import disable_fock_caching
disable_fock_caching()

# optimized loss channel
from quantum_agent.patches.monitored_loss_measure_fock_patch import MonitoredLossMeasureFock, patch_monitored_loss_measure_fock, decode_measurement_result
patch_monitored_loss_measure_fock()

from quantum_agent.patches.beamsplitter_patch import patch_beamsplitter
patch_beamsplitter()

from quantum_agent.patches.prepare_multimode_patch import patch_prepare_multimode
patch_prepare_multimode()