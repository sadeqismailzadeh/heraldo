from .sf_operations_no_cache import disable_fock_caching
from .beamsplitter_patch import patch_beamsplitter
from .monitored_loss_measure_fock_patch import patch_fock_backend

def apply_performance_patches():
    print("Applying Strawberry Fields Performance Patches...")
    disable_fock_caching()
    patch_beamsplitter()
    patch_fock_backend()