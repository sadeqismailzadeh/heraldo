# Scipy compatibility patch
import scipy.integrate
if not hasattr(scipy.integrate, 'simps'):
    scipy.integrate.simps = scipy.integrate.simpson

import strawberryfields as sf
from strawberryfields.ops import *
import numpy as np
import numba
from numba import jit

# -----------------------------------------------------------------------------
# OPTIMIZED JIT-COMPILED CORE FUNCTIONS
# -----------------------------------------------------------------------------

@jit(nopython=True, fastmath=True, cache=True)
def _generate_bs_tensor_optimized(theta, phi, cutoff):
    """
    Generates the BS tensor with O(D^3) complexity using optimized bounds.
    Removes internal branching by calculating exact loop ranges.
    """
    dtype = np.complex128
    
    # Precompute 1/sqrt to turn divisions into multiplications
    # We need indices up to cutoff. 
    # sqrt array size is cutoff+1 to handle edge cases safely if needed, 
    # though strict logic usually avoids it.
    sqrt_arr = np.sqrt(np.arange(cutoff, dtype=dtype))
    with numba.objmode(inv_sqrt='float64[:]'):
        # Safety for 0 division, though 0 index usually handled by logic
        tmp = np.arange(cutoff, dtype=np.float64)
        tmp[0] = 1.0 
        inv_sqrt = 1.0 / np.sqrt(tmp)
        inv_sqrt[0] = 0.0 # Just to be clean
        
    ct = np.cos(theta)
    st = np.sin(theta) * np.exp(1j * phi)
    
    # Recurrence coefficients
    R02 = ct
    R12 = st
    R03 = -np.conj(st)
    R13 = ct

    Z = np.zeros((cutoff, cutoff, cutoff), dtype=dtype)
    Z[0, 0, 0] = 1.0

    # We iterate m, n. 
    # Inside, we handle the two recurrence cases:
    # Case A: q > 0 (The "Rank 4" equivalent, general case)
    # Case B: q = 0 (The "Rank 3" equivalent, boundary case)
    
    for m in range(cutoff):
        for n in range(cutoff):
            # Base case already set
            if m == 0 and n == 0:
                continue

            # --- Case A: General Recurrence (q > 0) ---
            # Condition: 0 < m + n - p < cutoff
            # implies: p < m + n  AND  p > m + n - cutoff
            p_start = max(0, m + n - cutoff + 1)
            p_end = min(cutoff - 1, m + n - 1)
            
            if p_start <= p_end:
                # Pre-calculate factors to avoid re-fetching
                f_m = 0.0
                f_n = 0.0
                if m > 0: f_m = R03 * sqrt_arr[m]
                if n > 0: f_n = R13 * sqrt_arr[n]
                
                # Tight loop, no branches
                for p in range(p_start, p_end + 1):
                    q = m + n - p
                    inv_sqrt_q = inv_sqrt[q]
                    
                    val = 0.0 + 0.0j
                    if m > 0:
                        val += f_m * inv_sqrt_q * Z[m - 1, n, p]
                    if n > 0:
                        val += f_n * inv_sqrt_q * Z[m, n - 1, p]
                    Z[m, n, p] = val

            # --- Case B: Boundary Recurrence (q = 0 => p = m + n) ---
            p = m + n
            if p < cutoff:
                # Recurrence uses R02, R12 and shifts index p-1
                # Terms: R02 * sqrt(m)/sqrt(p) * Z[m-1, n, p-1]
                inv_sqrt_p = inv_sqrt[p]
                val = 0.0 + 0.0j
                
                if m > 0:
                    val += R02 * sqrt_arr[m] * inv_sqrt_p * Z[m - 1, n, p - 1]
                if n > 0:
                    val += R12 * sqrt_arr[n] * inv_sqrt_p * Z[m, n - 1, p - 1]
                Z[m, n, p] = val
                
    return Z

@jit(nopython=True, fastmath=True, cache=True)
def _apply_bs_diagonal(state_flat, bs_tensor, cutoff, dim_rest):
    """
    Applies the tensor using Diagonal Traversal (Photon Number Conserved Manifolds).
    
    Complexity: O(D^3) arithmetic, but with drastically better memory strides 
    and no branch mispredictions compared to standard loops.
    """
    new_state = np.zeros_like(state_flat)
    
    # Serial loop over spectator modes (as requested, no parallel)
    for idx in range(dim_rest):
        
        # Iterate over the total photon number S = out1 + out2 = in1 + in2
        # Max photons is 2 * (cutoff - 1)
        for S in range(2 * cutoff - 1):
            
            # Determine valid range for index 'i' (out1) for this diagonal S
            # Constraints: 
            # 1. 0 <= i < cutoff
            # 2. 0 <= j < cutoff  =>  0 <= S - i < cutoff  =>  i > S - cutoff
            i_min = max(0, S - cutoff + 1)
            i_max = min(S, cutoff - 1)
            
            # Range for 'k' (in1) is identical to range for 'i' because bounds are same
            k_min = i_min
            k_max = i_max
            
            # Loop over output modes along the diagonal
            for i in range(i_min, i_max + 1):
                j = S - i
                
                sum_val = 0.0 + 0.0j
                
                # Loop over input modes along the diagonal
                # This loop has constant stride access on bs_tensor and state_flat
                for k in range(k_min, k_max + 1):
                    # l = S - k (implicit)
                    # Z is [out1, out2, in1] -> [i, j, k]
                    # state is [idx, k, S-k]
                    sum_val += bs_tensor[i, j, k] * state_flat[idx, k, S - k]
                
                new_state[idx, i, j] = sum_val
                
    return new_state


# -----------------------------------------------------------------------------
# THE PATCH METHOD
# -----------------------------------------------------------------------------

def _beamsplitter_patched_optimized(self, theta, phi, mode1, mode2):
    trunc = self._trunc
    
    # 1. Generate Tensor (Optimized)
    bs_tensor = _generate_bs_tensor_optimized(theta, phi, trunc)

    if self._pure:
        # --- PURE STATE ---
        all_modes = np.arange(self._num_modes)
        switch_list = all_modes.copy()
        
        switch_list[[0, mode1]] = switch_list[[mode1, 0]]
        switch_list[[1, mode2]] = switch_list[[mode2, 1]]
        inverse_switch = np.argsort(switch_list)
        
        state_view = self._state.transpose(switch_list)
        
        orig_shape = state_view.shape
        dim_rest = 1
        for d in orig_shape[2:]:
            dim_rest *= d
            
        state_reshaped = state_view.reshape(trunc, trunc, dim_rest)
        # Transpose to (rest, m1, m2) to match JIT signature
        state_for_jit = np.ascontiguousarray(state_reshaped.transpose(2, 0, 1))
        
        # 2. Apply Tensor (Optimized Diagonal)
        new_state_jit = _apply_bs_diagonal(state_for_jit, bs_tensor, trunc, dim_rest)
        
        new_state_reshaped = new_state_jit.transpose(1, 2, 0).reshape(orig_shape)
        self._state = new_state_reshaped.transpose(inverse_switch)
        
    else:
        # --- MIXED STATE ---
        # (Handling logic remains same, just calling optimized JIT core)
        t1 = 2 * mode1
        t2 = 2 * mode2
        
        # Bra Indices
        switch_list_1 = np.arange(2 * self._num_modes)
        switch_list_1[[0, t1]] = switch_list_1[[t1, 0]]
        switch_list_1[[1, t2]] = switch_list_1[[t2, 1]]
        inverse_switch_1 = np.argsort(switch_list_1)
        
        state_view = self._state.transpose(switch_list_1)
        orig_shape = state_view.shape
        dim_rest = 1
        for d in orig_shape[2:]:
            dim_rest *= d
            
        state_reshaped = state_view.reshape(trunc, trunc, dim_rest)
        state_for_jit = np.ascontiguousarray(state_reshaped.transpose(2, 0, 1))
        
        # Apply
        new_state_jit = _apply_bs_diagonal(state_for_jit, bs_tensor, trunc, dim_rest)
        
        state_view = new_state_jit.transpose(1, 2, 0).reshape(orig_shape).transpose(inverse_switch_1)
        self._state = state_view

        # Ket Indices
        k1 = t1 + 1
        k2 = t2 + 1
        
        switch_list_2 = np.arange(2 * self._num_modes)
        switch_list_2[[0, k1]] = switch_list_2[[k1, 0]]
        switch_list_2[[1, k2]] = switch_list_2[[k2, 1]]
        inverse_switch_2 = np.argsort(switch_list_2)
        
        state_view = self._state.transpose(switch_list_2)
        orig_shape = state_view.shape
        
        state_reshaped = state_view.reshape(trunc, trunc, dim_rest)
        state_for_jit = np.ascontiguousarray(state_reshaped.transpose(2, 0, 1))
        
        # Use Conjugate Tensor
        bs_tensor_conj = np.ascontiguousarray(bs_tensor.conj())
        
        # Apply
        new_state_jit = _apply_bs_diagonal(state_for_jit, bs_tensor_conj, trunc, dim_rest)
        
        state_view = new_state_jit.transpose(1, 2, 0).reshape(orig_shape).transpose(inverse_switch_2)
        self._state = state_view


def patch_beamsplitter():
    from strawberryfields.backends.fockbackend.circuit import Circuit
    if not hasattr(Circuit, 'beamsplitter_original'):
        Circuit.beamsplitter_original = Circuit.beamsplitter
    Circuit.beamsplitter = _beamsplitter_patched_optimized
    print("BeamSplitter patch applied (Optimized O(D^3) Diagonal Traversal).")

def revert_beamsplitter_patch():
    from strawberryfields.backends.fockbackend.circuit import Circuit
    if hasattr(Circuit, 'beamsplitter_original'):
        Circuit.beamsplitter = Circuit.beamsplitter_original
        print("BeamSplitter patch reverted.")
