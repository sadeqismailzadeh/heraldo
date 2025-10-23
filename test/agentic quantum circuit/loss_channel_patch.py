# 1. Import the module we need to patch
import scipy.integrate

# 2. Check if the patch is needed to avoid errors
if not hasattr(scipy.integrate, 'simps'):
    print("Monkey patching scipy.integrate: 'simps' not found. Pointing to 'simpson'.")
    # 3. Create the 'simps' attribute and point it to the existing 'simpson' function.
    scipy.integrate.simps = scipy.integrate.simpson
else:
    print("'simps' already exists in scipy.integrate. No patch needed.")

import strawberryfields as sf
from strawberryfields.ops import *
import numpy as np
from itertools import product
import strawberryfields.backends.fockbackend.ops as ops
import types

def _apply_loss_channel_fast(self, T, mode):
    """
    A specialized, high-performance replacement for applying the loss channel.
    This function leverages the sparse structure of the loss channel's Kraus operators
    to avoid slow, dense matrix multiplications.
    
    The key insight: Each Kraus operator E_k is non-zero only on the k-th superdiagonal.
    This means E_k @ ρ @ E_k† can be computed by extracting blocks and scaling them,
    rather than performing full matrix multiplications.
    """
    trunc = self._trunc
    n_modes = self._num_modes

    # Convert pure state to mixed state (crucial for consistency)
    if self._pure:
        # Use the correct ops.mix function with proper arguments
        self._state = ops.mix(self._state, self._num_modes)
        self._pure = False

    # # Handle the edge case of total loss (T=0)
    # if T == 0:
    #     # State collapses to vacuum, but trace must be preserved
    #     new_state = np.zeros_like(self._state)
    #     vacuum_idx = tuple([0] * (2 * n_modes))
    #     # The trace of the original state becomes the vacuum state probability
    #     new_state[vacuum_idx] = np.trace(
    #         self._state.reshape(trunc**n_modes, trunc**n_modes)
    #     )
    #     self._state = new_state
    #     return

    # Generate the Kraus operators for the loss channel
    kraus_ops = ops.lossChannel(T, trunc)

    # Identify which modes are NOT being operated on
    other_modes = [i for i in range(n_modes) if i != mode]
    
    # Create a transpose list that moves the target mode's indices to the end
    # Original order: [i0, j0, i1, j1, i2, j2, ...]
    # New order: [i_others, j_others, ..., i_mode, j_mode]
    # Identify the axes for spectator modes and the target mode
    # based on an (i0, j0, i1, j1, ...) interleaved layout.
    other_mode_axes = [ax for m in other_modes for ax in (2 * m, 2 * m + 1)]
    target_mode_axes = [2 * mode, 2 * mode + 1]

    # The new permutation list moves all spectator axes to the front
    # and the target mode's axes to the end.
    transpose_list = tuple(other_mode_axes + target_mode_axes)


    # Create the inverse permutation to restore original ordering
    untranspose_list = [0] * len(transpose_list)
    for i, p in enumerate(transpose_list):
        untranspose_list[p] = i
        
    # Transpose the state so target mode is at the end
    state_view = self._state.transpose(transpose_list)
    
    # Allocate the new state (not a view, to avoid aliasing issues)
    new_state = np.zeros_like(state_view)

    # Pre-extract the diagonals from each Kraus operator
    # E_k has non-zero elements only on its k-th superdiagonal
    kraus_diags = [np.diagonal(E_k, offset=k) for k, E_k in enumerate(kraus_ops)]

    # Iterate over all configurations of the spectator modes
    # For each configuration, we apply the channel to the target mode's sub-matrix
    other_modes_iterator = product(*([range(trunc)] * (2 * (n_modes - 1))))

    for other_indices in other_modes_iterator:
        # Extract the 2D density matrix for the target mode
        sub_rho = state_view[other_indices]
        
        # This will accumulate the result of applying all Kraus operators
        new_sub_rho = np.zeros((trunc, trunc), dtype=ops.def_type)

        # Apply each Kraus operator using the block-shift optimization
        for k, diag_k in enumerate(kraus_diags):
            if k >= trunc or len(diag_k) == 0:
                continue
            
            # The mathematical operation E_k @ sub_rho @ E_k† can be computed efficiently:
            # Since E_k[i,j] is non-zero only when j = i+k, the result is:
            # (E_k @ ρ @ E_k†)[i,j] = E_k[i,i+k] * ρ[i+k,j+k] * conj(E_k[j,j+k])
            
            # Extract the source block from the original density matrix
            source_block = sub_rho[k:, k:]
            
            # Calculate the scaling factors (outer product of diagonal elements)
            scaling_matrix = np.outer(diag_k, diag_k.conj())
            
            # Compute the scaled block and add it to the result
            # Note: We need to handle the size mismatch between scaling_matrix and source_block
            block_size = min(len(diag_k), trunc - k)
            new_sub_rho[:block_size, :block_size] += (
                scaling_matrix[:block_size, :block_size] * 
                source_block[:block_size, :block_size]
            )
            
        # Store the transformed sub-matrix in the new state
        new_state[other_indices] = new_sub_rho

    # Restore the original mode ordering and update the state
    self._state = new_state.transpose(untranspose_list)


# Apply the monkey patch
print("=" * 60)
print("Applying Loss Channel Performance Optimization Patch")
print("=" * 60)

def patch_loss_channel():
    from strawberryfields.backends.fockbackend.circuit import Circuit
    Circuit._apply_loss_channel_fast = _apply_loss_channel_fast

    # Preserve the original loss method for reference/testing
    if not hasattr(Circuit, 'loss_original'):
        Circuit.loss_original = Circuit.loss

    Circuit.loss = _apply_loss_channel_fast
    print("loss_channel patch applied successfully")
    print("=" * 60)


import strawberryfields as sf
from strawberryfields.ops import Fock
import numpy as np
# Test function
def test_loss_channel(T_values=[0.0, 0.3, 0.5, 0.9, 1.0], truncs=[3, 5, 10]):
    """Compare original vs optimized loss channel"""
    for trunc in truncs:
        for T in T_values:
            # Create two identical states
                # Configuration
            cutoff = trunc
            n_modes = 2
            
            prog = sf.Program(n_modes)
            with prog.context as q:
                # State preparation to ensure gates have a non-trivial effect
                Dgate(0.5, np.pi/4) | q[0] # Corresponds to ops.displacement
                Sgate(0.6, 0) | q[1]          # Corresponds to ops.squeezing
                LossChannel(T) | q[0]        # Apply loss channel to mode 0
                BSgate(np.pi/4, np.pi/6) | (q[0], q[1]) # Corresponds to ops.beamsplitter

            
            # ============================================================================
            # STEP 1: Make state with default operators (caching is ON by default)
            # ============================================================================
            print("--- 1. Calculating state with default (cached) operators ---")
            eng_original = sf.Engine("fock", backend_options={"cutoff_dim": cutoff})
            result_original = eng_original.run(prog)
            state1= result_original.state
            print("Original state calculated successfully.")

            # ============================================================================
            # STEP 2: patch the loss channel
            # ============================================================================
            print("\n--- 2. Disabling Fock backend caching ---")
            patch_loss_channel()

            # ============================================================================
            # STEP 3: Re-run the program and get the new state
            # ============================================================================
            print("\n--- 3. Recalculating state with cache disabled ---")
            # It's good practice to create a new engine after modifying the backend
            eng_no_cache = sf.Engine("fock", backend_options={"cutoff_dim": cutoff})
            result_no_cache = eng_no_cache.run(prog)
            state2 = result_no_cache.state
            print("State with cache disabled calculated successfully.")
            
            # Compare the results
            diff = np.max(np.abs(state1.dm()- state2.dm()))
            print(f"T={T}: Max difference = {diff:.2e}")
            
            assert diff < 1e-10, f"States differ! T={T}, diff={diff}"
    
    print("\n✓ All tests passed!")

def main():
    print("\nRunning loss channel tests...")
    test_loss_channel()

if __name__ == "__main__":
    main()