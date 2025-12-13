# Scipy compatibility patch
import scipy.integrate
if not hasattr(scipy.integrate, 'simps'):
    scipy.integrate.simps = scipy.integrate.simpson

import strawberryfields as sf
from strawberryfields.ops import *
import numpy as np
import numba
from numba import jit
import time

import scipy.integrate
if not hasattr(scipy.integrate, 'simps'):
    scipy.integrate.simps = scipy.integrate.simpson

import strawberryfields as sf
from strawberryfields.ops import *
import numpy as np
import numba
from numba import jit
import time

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

def revert_patch():
    from strawberryfields.backends.fockbackend.circuit import Circuit
    if hasattr(Circuit, 'beamsplitter_original'):
        Circuit.beamsplitter = Circuit.beamsplitter_original
        print("BeamSplitter patch reverted.")
        
# -----------------------------------------------------------------------------
# VERIFICATION AND BENCHMARKING
# -----------------------------------------------------------------------------


def verify_correctness(theta_phi_pairs=[(np.pi/4, np.pi/6), (np.pi/3, 0), (np.pi, np.pi/7), (np.pi/2, np.pi/6), (np.pi/5, np.pi/9),
                                        (np.pi/9, np.pi/12), (2*np.pi/15, np.pi/7), (2*np.pi/3, np.pi/16) , (7*np.pi/5, 11*np.pi/7)],
                       truncs=[10, 15], n_modes_list=[2, 3]):
    """
    Verify that the patched beamsplitter produces identical results to the original.
    """
    print("--- Running correctness verification ---")

    # Initialize the patching mechanism to ensure original is saved
    patch_beamsplitter()
    revert_patch()

    for n_modes in n_modes_list:
        for trunc in truncs:
            for theta, phi in theta_phi_pairs:
                print(f"\nVerifying: trunc={trunc}, theta={theta:.3f}, phi={phi:.3f}, n_modes={n_modes}")

                # Test with pure states
                prog = sf.Program(n_modes)
                with prog.context as q:
                    for i in range(n_modes):
                        Dgate(0.3 * (i + 1), np.pi / (i + 3)) | q[i]
                    if n_modes >= 2:
                        BSgate(theta, phi) | (q[0], q[1])
                    if n_modes >= 3:
                        BSgate(theta/2, phi/2) | (q[1], q[2])

                # Original (no patch)
                revert_patch()  # Ensure original is active
                eng_original = sf.Engine("fock", backend_options={"cutoff_dim": trunc})
                state1 = eng_original.run(prog).state

                # Patched version
                patch_beamsplitter()  # Apply patch
                eng_patched = sf.Engine("fock", backend_options={"cutoff_dim": trunc})
                state2 = eng_patched.run(prog).state
                assert state1.is_pure
                assert state2.is_pure
                # Compare results
                if state1.is_pure:
                    diff = np.max(np.abs(state1.ket() - state2.ket()))
                else:
                    diff = np.max(np.abs(state1.dm() - state2.dm()))

                print(f"Max difference = {diff:.2e}")
                assert diff < 1e-10, f"States differ! theta={theta}, phi={phi}, trunc={trunc}, diff={diff}"

    print("\nCorrectness verification passed!\n")
    revert_patch()


def run_dynamic_benchmark(truncs=[10, 20, 30, 40, 50, 60], 
                          n_modes_list=[2], 
                          num_random_runs=10):
    """
    Benchmarks performance where every run uses a DIFFERENT random theta/phi.
    This simulates a variational loop where parameters change constantly.
    """
    print("--- Running Dynamic Parameter Benchmark ---")
    results = []

    # 1. Run ORIGINAL (Unpatched)
    print("\nBenchmarking ORIGINAL beamsplitter...")
    revert_patch()

    for n_modes in n_modes_list:
        for trunc in truncs:
            
            # Reset seed so Original and Optimized get the exact same random numbers
            np.random.seed(42) 
            
            eng = sf.Engine("fock", backend_options={"cutoff_dim": trunc})
            timings = []

            for _ in range(num_random_runs):
                # Generate random parameters for this specific run
                theta = np.random.uniform(0, 2 * np.pi)
                phi = np.random.uniform(0, 2 * np.pi)

                prog = sf.Program(n_modes)
                with prog.context as q:
                    # Setup state
                    for i in range(n_modes):
                        Dgate(0.3 * (i + 1), 0) | q[i]
                    
                    # Apply BS with random angles
                    BSgate(theta, phi) | (q[0], q[1])
                    
                    if n_modes > 2:
                        BSgate(theta / 2, phi / 2) | (q[1], q[2])

                start = time.time()
                eng.run(prog)
                end = time.time()
                timings.append(end - start)

            avg_time = np.mean(timings)
            print(f"Original: Modes={n_modes}, Cutoff={trunc} -> {avg_time:.6f}s (avg of {num_random_runs} random angles)")
            
            results.append({
                'n_modes': n_modes,
                'trunc': trunc,
                'original': avg_time
            })

    # 2. Run OPTIMIZED (Patched)
    print("\nBenchmarking OPTIMIZED beamsplitter...")
    patch_beamsplitter()

    # JIT Warmup (Crucial so compilation time isn't counted)
    print("Running JIT warmup...")
    prog_warm = sf.Program(2)
    with prog_warm.context as q:
        BSgate(0.1, 0.1) | (q[0], q[1])
    sf.Engine("fock", backend_options={"cutoff_dim": 5}).run(prog_warm)
    print("Warmup complete.")

    for i, res in enumerate(results):
        n_modes = res['n_modes']
        trunc = res['trunc']
        
        # Reset seed to match the sequence used in Original
        np.random.seed(42)
        
        eng = sf.Engine("fock", backend_options={"cutoff_dim": trunc})
        timings = []

        for _ in range(num_random_runs):
            # Generate SAME random parameters as above
            theta = np.random.uniform(0, 2 * np.pi)
            phi = np.random.uniform(0, 2 * np.pi)

            prog = sf.Program(n_modes)
            with prog.context as q:
                for j in range(n_modes):
                    Dgate(0.3 * (j + 1), 0) | q[j]
                
                BSgate(theta, phi) | (q[0], q[1])
                
                if n_modes > 2:
                    BSgate(theta / 2, phi / 2) | (q[1], q[2])

            start = time.time()
            eng.run(prog)
            end = time.time()
            timings.append(end - start)

        avg_time = np.mean(timings)
        results[i]['optimized'] = avg_time
        print(f"Optimized: Modes={n_modes}, Cutoff={trunc} -> {avg_time:.6f}s")

    # 3. Summary
    print("\n" + "="*80)
    print(f"{'Modes':<6} {'Cutoff':<8} {'Original (s)':<15} {'Optimized (s)':<15} {'Speedup':<10}")
    print("-" * 80)
    
    for res in results:
        t_orig = res['original']
        t_opt = res['optimized']
        speedup = t_orig / t_opt if t_opt > 0 else 0
        
        print(f"{res['n_modes']:<6} {res['trunc']:<8} {t_orig:<15.6f} {t_opt:<15.6f} {speedup:.2f}x")
    print("="*80)
    
    revert_patch()

def main():
    """Run verification and benchmarks."""
    verify_correctness()
    run_dynamic_benchmark()


if __name__ == "__main__":
    main()
