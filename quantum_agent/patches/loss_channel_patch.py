# 1. Import the module we need to patch
import scipy.integrate
import time
import numba

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

# Numba-jitted helper functions for fast binomial coefficients
@numba.jit(nopython=True)
def factorial(n):
    res = 1.0
    for i in range(1, n + 1):
        res *= i
    return res

@numba.jit(nopython=True)
def binom_numba(n, k):
    if k < 0 or k > n:
        return 0
    if k == 0 or k == n:
        return 1
    if k > n // 2:
        k = n - k
    
    res = 1.0
    for i in range(k):
        res = res * (n - i) / (i + 1)
    return res

@numba.jit(nopython=True)
def _core_loss_channel(state_view_flat, trunc, kraus_diags_mat):
    """
    JIT-compiled core function to apply the loss channel.
    This function iterates over all spectator mode configurations in parallel.
    """
    other_modes_dim = state_view_flat.shape[0]
    new_state_flat = np.zeros_like(state_view_flat)

    for i in numba.prange(other_modes_dim):
        sub_rho = state_view_flat[i].copy().reshape(trunc, trunc)
        new_sub_rho = np.zeros((trunc, trunc), dtype=np.complex128)

        for k in range(trunc):
            diag_k = kraus_diags_mat[k]
            block_size = trunc - k
            
            if block_size <= 0:
                continue

            source_block = sub_rho[k:, k:]
            
            for row in range(block_size):
                for col in range(block_size):
                    scaling_factor = diag_k[row] * np.conj(diag_k[col])
                    new_sub_rho[row, col] += scaling_factor * source_block[row, col]

        new_state_flat[i] = new_sub_rho.flatten()
    
    return new_state_flat

def _apply_loss_channel_fast(self, T, mode):
    """
    A specialized, high-performance replacement for applying the loss channel.
    This function leverages the sparse structure of the loss channel's Kraus operators
    and uses a Numba JIT-compiled core to achieve high performance.
    
    The Kraus operators are not pre-computed. Instead, their elements are
    calculated on-the-fly inside the JIT-compiled function.
    """
    trunc = self._trunc
    n_modes = self._num_modes

    if self._pure:
        self._state = ops.mix(self._state, self._num_modes)
        self._pure = False

    # Handle the edge case of T=1 (no loss) separately for speed
    if T == 1.0:
        return

    # Pre-calculate all Kraus diagonals
    kraus_diags = []
    for k in range(trunc):
        diag_k_len = trunc - k
        diag_k = np.zeros(diag_k_len, dtype=np.complex128)
        term1 = (1 - T) ** (k / 2.0)
        for j in range(diag_k_len):
            diag_k[j] = np.sqrt(binom_numba(j + k, k)) * (T ** (j / 2.0)) * term1
        kraus_diags.append(diag_k)
    
    # Numba works best with uniform arrays. We pad the diagonals to the same length.
    kraus_diags_mat = np.zeros((trunc, trunc), dtype=np.complex128)
    for i, diag in enumerate(kraus_diags):
        kraus_diags_mat[i, :len(diag)] = diag

    other_modes = [i for i in range(n_modes) if i != mode]
    other_mode_axes = [ax for m in other_modes for ax in (2 * m, 2 * m + 1)]
    target_mode_axes = [2 * mode, 2 * mode + 1]
    transpose_list = tuple(other_mode_axes + target_mode_axes)

    untranspose_list = [0] * len(transpose_list)
    for i, p in enumerate(transpose_list):
        untranspose_list[p] = i
        
    state_view = self._state.transpose(transpose_list)
    
    other_modes_dim = trunc**(2 * (n_modes - 1))
    state_view_flat = state_view.reshape(other_modes_dim, trunc*trunc)

    # Call the JIT-compiled core function
    new_state_flat = _core_loss_channel(state_view_flat, trunc, kraus_diags_mat)
    
    new_state = new_state_flat.reshape(state_view.shape)
    self._state = new_state.transpose(untranspose_list)


def patch_loss_channel():
    from strawberryfields.backends.fockbackend.circuit import Circuit
    
    # Preserve the original loss method for reference/testing
    if not hasattr(Circuit, 'loss_original'):
        Circuit.loss_original = Circuit.loss

    Circuit.loss = _apply_loss_channel_fast
    print("loss_channel patch applied successfully")


def revert_loss_channel_patch():
    """Reverts the monkey patch on Circuit.loss."""
    from strawberryfields.backends.fockbackend.circuit import Circuit
    
    if hasattr(Circuit, 'loss_original'):
        Circuit.loss = Circuit.loss_original
        print("loss_channel patch reverted.")
    else:
        print("Could not revert: original loss method not found.")


def verify_correctness(T_values=[0.8], truncs=[7, 10], n_modes_list=[2]):
    """
    Verifies that the patched loss channel produces the same results as the original
    for a given set of parameters.
    """
    print("--- Running correctness verification ---")
    for n_modes in n_modes_list:
        for trunc in truncs:
            for T in T_values:
                print(f"\nVerifying with trunc={trunc}, T={T}, n_modes={n_modes}")
                cutoff = trunc
                
                prog = sf.Program(n_modes)
                with prog.context as q:
                    for i in range(n_modes):
                        Dgate(0.2 * (i + 1), np.pi / (i + 2)) | q[i]
                        Sgate(0.2 * (i + 1), 0) | q[i]
                        LossChannel(T) | q[0]
                    if n_modes > 1:
                        BSgate(np.pi / 4, np.pi / 6) | (q[0], q[1])
                    if n_modes > 2:
                        BSgate(np.pi / 3, np.pi / 5) | (q[1], q[2])

                revert_loss_channel_patch()
                eng_original = sf.Engine("fock", backend_options={"cutoff_dim": cutoff})
                state1 = eng_original.run(prog).state

                patch_loss_channel()
                eng_optimized = sf.Engine("fock", backend_options={"cutoff_dim": cutoff})
                state2 = eng_optimized.run(prog).state
                
                diff = np.max(np.abs(state1.dm() - state2.dm()))
                print(f"Max difference between density matrices = {diff:.2e}")
                assert diff < 1e-10, f"States differ significantly! T={T}, trunc={trunc}, max diff={diff}"
    
    print("\n✓ Correctness verification passed!\n")
    revert_loss_channel_patch()


def run_benchmark(T_values=[0.3, 0.8], truncs=[5, 8 , 10], n_modes_list=[2], repeats=10):
    """
    Benchmarks the performance of the original vs. patched loss channel.
    """
    print("--- Running performance benchmark ---")
    results = []

    print("\nBenchmarking ORIGINAL loss channel...")
    revert_loss_channel_patch()
    for n_modes in n_modes_list:
        for trunc in truncs:
            for T in T_values:
                prog = sf.Program(n_modes)
                with prog.context as q:
                    for i in range(n_modes):
                        Dgate(0.2 * (i + 1), np.pi / (i + 2)) | q[i]
                        Sgate(0.2 * (i + 1), 0) | q[i]
                        LossChannel(T) | q[0]
                    if n_modes > 1:
                        BSgate(np.pi / 4, np.pi / 6) | (q[0], q[1])
                    if n_modes > 2:
                        BSgate(np.pi / 3, np.pi / 5) | (q[1], q[2])

                eng = sf.Engine("fock", backend_options={"cutoff_dim": trunc})
                timings = []
                for _ in range(repeats):
                    start = time.time()
                    eng.run(prog)
                    end = time.time()
                    timings.append(end - start)
                
                time_original = np.mean(timings)
                print(f"Original: n_modes={n_modes}, trunc={trunc}, T={T} -> {time_original:.6f}s")
                results.append({'n_modes': n_modes, 'trunc': trunc, 'T': T, 'original': time_original})

    patch_loss_channel()
    print("\nJIT warmup run...")
    eng_warmup = sf.Engine("fock", backend_options={"cutoff_dim": 5})
    prog_warmup = sf.Program(2)
    with prog_warmup.context as q:
        LossChannel(0.5) | q[0]
    eng_warmup.run(prog_warmup)
    print("Warmup complete.")

    print("\nBenchmarking OPTIMIZED loss channel...")
    for i, params in enumerate(results):
        n_modes, trunc, T = params['n_modes'], params['trunc'], params['T']
        prog = sf.Program(n_modes)
        with prog.context as q:
            for j in range(n_modes):
                Dgate(0.2 * (j + 1), np.pi / (j + 2)) | q[j]
                Sgate(0.2 * (j + 1), 0) | q[j]
            LossChannel(T) | q[0]
            if n_modes > 1:
                BSgate(np.pi / 4, np.pi / 6) | (q[0], q[1])
            if n_modes > 2:
                BSgate(np.pi / 3, np.pi / 5) | (q[1], q[2])

        eng = sf.Engine("fock", backend_options={"cutoff_dim": trunc})
        timings = []
        for _ in range(repeats):
            start = time.time()
            eng.run(prog)
            end = time.time()
            timings.append(end - start)
        
        results[i]['optimized'] = np.mean(timings)
        print(f"Optimized: n_modes={n_modes}, trunc={trunc}, T={T} -> {results[i]['optimized']:.6f}s")

    print("\n--- Benchmark Summary ---")
    print(f"{'Modes':<6} {'Trunc':<6} {'T':<4} {'Original (s)':<15} {'Optimized (s)':<15} {'Speedup':<10}")
    print("-" * 60)
    for res in results:
        speedup = res['original'] / res['optimized'] if res.get('optimized', 0) > 0 else float('inf')
        speedup_str = f"{speedup:.2f}x" if speedup != float('inf') else "inf"
        print(f"{res['n_modes']:<6} {res['trunc']:<6} {res['T']:<4.2f} {res['original']:<15.6f} {res.get('optimized', 0):<15.6f} {speedup_str:<10}")

    revert_loss_channel_patch()


def main():
    verify_correctness()
    run_benchmark()


if __name__ == "__main__":
    main()
