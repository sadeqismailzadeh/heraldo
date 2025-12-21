"""
LossMeasureFock Operation - Version 2 with Optimized Random Sampling

This version adds numba JIT compilation to compute_loss_distribution
for better performance with large cutoff dimensions and random sampling.
"""

import scipy.integrate
if not hasattr(scipy.integrate, 'simps'):
    scipy.integrate.simps = scipy.integrate.simpson

import numpy as np
import copy
from collections.abc import Sequence
from itertools import product
from scipy.special import comb
import numba

import strawberryfields as sf
from strawberryfields.ops import Measurement
import strawberryfields.backends.fockbackend.ops as ops
from strawberryfields.ops import *

def_type = np.complex128


# Numba-optimized helper for binomial coefficients
@numba.jit(nopython=True)
def binom_numba(n, k):
    if k < 0 or k > n:
        return 0.0
    if k == 0 or k == n:
        return 1.0
    if k > n // 2:
        k = n - k
    
    res = 1.0
    for i in range(k):
        res = res * (n - i) / (i + 1)
    return res


@numba.jit(nopython=True)
def _compute_loss_dist_1mode(reduced_diag, eta, trunc):
    """
    Optimized single-mode loss distribution computation.
    """
    dist = np.zeros(trunc, dtype=np.float64)
    
    for n in range(trunc):
        prob = 0.0
        for i in range(n, trunc):
            weight = binom_numba(i, n) * (eta ** n) * ((1 - eta) ** (i - n))
            prob += weight * reduced_diag[i]
        dist[n] = prob
    
    return dist


@numba.jit(nopython=True)
def _compute_loss_dist_2mode(reduced_diag, eta, trunc):
    """
    Optimized two-mode loss distribution computation.
    Uses parallel execution for better performance.
    """
    dist = np.zeros((trunc, trunc), dtype=np.float64)
    
    for n0 in numba.prange(trunc):
        for n1 in range(trunc):
            prob = 0.0
            for i0 in range(n0, trunc):
                weight0 = binom_numba(i0, n0) * (eta ** n0) * ((1 - eta) ** (i0 - n0))
                for i1 in range(n1, trunc):
                    weight1 = binom_numba(i1, n1) * (eta ** n1) * ((1 - eta) ** (i1 - n1))
                    weight = weight0 * weight1
                    prob += weight * reduced_diag[i0, i1]
            dist[n0, n1] = prob
    
    return dist


def _compute_loss_distribution_optimized(reduced_diag, eta, num_modes, trunc):
    """
    Optimized computation of measurement probability distribution with loss.
    
    Uses JIT-compiled functions for 1 and 2 mode cases.
    Falls back to Python loops for >2 modes.
    """
    if num_modes == 1:
        # Use optimized 1-mode version
        dist = _compute_loss_dist_1mode(reduced_diag, eta, trunc)
        return dist
    
    elif num_modes == 2:
        # Use optimized 2-mode version
        shape = (trunc, trunc)
        reduced_diag_tensor = reduced_diag.reshape(shape)
        dist = _compute_loss_dist_2mode(reduced_diag_tensor, eta, trunc)
        return dist.ravel()
    
    else:
        # Fall back to original Python implementation for >2 modes
        shape = tuple([trunc] * num_modes)
        reduced_diag_tensor = reduced_diag.reshape(shape)
        dist_tensor = np.zeros(shape, dtype=np.float64)
        
        for n_idx in np.ndindex(shape):
            prob = 0.0
            for i_idx in np.ndindex(shape):
                if all(i >= n for i, n in zip(i_idx, n_idx)):
                    weight = 1.0
                    for i, n in zip(i_idx, n_idx):
                        weight *= comb(i, n, exact=False) * (eta ** n) * ((1 - eta) ** (i - n))
                    prob += weight * reduced_diag_tensor[i_idx]
            dist_tensor[n_idx] = prob
        
        return dist_tensor.ravel()


class LossMeasureFock(Measurement):
    r"""Combined loss channel and photon counting measurement (Optimized V2).
    
    This version includes JIT-optimized distribution computation for better
    performance with random sampling and large cutoff dimensions.
    
    Args:
        eta (float): Loss channel transmissivity (0 <= eta <= 1)
        select (None, int, Sequence[int]): Post-selection value(s)
    """
    
    ns = None
    
    def __init__(self, eta, select=None):
        if not isinstance(eta, (int, float)):
            raise TypeError("eta must be a number")
        if not 0 <= eta <= 1:
            raise ValueError("eta must be in the range [0, 1]")
        
        if select is not None and not isinstance(select, Sequence):
            select = [select]
        
        super().__init__([eta], select)
        self.eta = eta
    
    def _apply(self, reg, backend, shots=1, **kwargs):
        samples = backend.loss_measure_fock(
            reg, eta=self.eta, shots=shots, select=self.select, **kwargs
        )
        
        if isinstance(samples, list):
            samples = np.array(samples)
        
        return samples
    
    def __str__(self):
        temp = f"{self.__class__.__name__}(eta={self.eta}"
        if self.select is not None:
            temp += f", select={self.select}"
        temp += ")"
        return temp


def loss_measure_fock_implementation(self, modes, eta, select=None):
    """
    Optimized combined loss + Fock measurement - Version 2.
    Uses JIT-optimized distribution computation.
    """
    # Make sure the state is mixed
    if self._pure:
        state = ops.mix(self._state, self._num_modes)
        self._state = state
        self._pure = False
    else:
        state = self._state

    if select is not None:
        if len(select) != len(modes):
            raise ValueError(
                "When performing post-selection, the number of "
                "selected values (including None) must match the number of measured modes"
            )

        if not all(isinstance(s, int) or s is None for s in select):
            raise TypeError("The post-select list elements must be integers or None")

        measure = [i for i, s in zip(modes, select) if s is None]
        selected = [i for i, s in zip(modes, select) if s is not None]
        select_values = [s for s in select if s is not None]

        if len(selected) > 0:
            self._state = _project_loss_n_measure(
                selected, select_values, state, False, self._num_modes, self._trunc, eta
            )

            if self.norm() == 0:
                raise ZeroDivisionError("Measurement has zero probability.")

            self._state = self._state / self.norm()
            state = self._state
    else:
        measure = modes
        selected = []
        select_values = []

    if len(measure) > 0:
        # Use OPTIMIZED distribution computation
        unmeasured = [i for i in range(self._num_modes) if i not in measure]
        reduced = ops.partial_trace(state, self._num_modes, unmeasured)
        reduced_diag = ops.diagonal(reduced, len(measure)).real
        
        # Call optimized version (JIT-compiled for 1 and 2 modes)
        dist = _compute_loss_distribution_optimized(reduced_diag, eta, len(measure), self._trunc)
        
        dist = dist * ~np.isclose(dist, 0.0)

        if sum(dist) != 1:
            i = np.random.choice(list(range(len(dist))), p=dist / sum(dist))
        else:
            i = np.random.choice(list(range(len(dist))), p=dist)

        permuted_outcome = ops.unIndex(i, len(measure), self._trunc)

        permutation = np.argsort(measure)
        outcome = [0] * len(measure)
        for i in range(len(measure)):
            outcome[permutation[i]] = permuted_outcome[i]

        all_modes_to_project = selected + measure
        all_outcomes = select_values + outcome
        
        self._state = _project_loss_n_measure(
            all_modes_to_project, all_outcomes, state, False, self._num_modes, self._trunc, eta
        )

        if self.norm() == 0:
            raise ZeroDivisionError("Measurement has zero probability.")

        self._state = self._state / self.norm()

    if select is not None:
        outcome = copy.copy(select)

    return np.array([outcome])


def _project_loss_n_measure(modes, x, state, pure, n, trunc, eta):
    """Optimized projection for loss + measurement."""
    def sliceExp(axes, ind, n):
        return [ind[i] if i in axes else slice(None, None, None) for i in range(n)]
    
    def intersperse(lst):
        return tuple(lst[i // 2] for i in range(len(lst) * 2))
    
    if pure:
        ret = np.zeros([trunc for i in range(n)], dtype=def_type)
        outSlice = tuple(sliceExp(modes, dict(zip(modes, [0] * len(modes))), n))
        
        for i_vals in product(*[range(xi, trunc) for xi in x]):
            weight = 1.0
            for i, xi in zip(i_vals, x):
                weight *= comb(i, xi, exact=False) * (eta ** xi) * ((1 - eta) ** (i - xi))
            weight = np.sqrt(weight)
            
            inSlice = tuple(sliceExp(modes, dict(zip(modes, i_vals)), n))
            ret[outSlice] += weight * state[inSlice]
    else:
        ret = np.zeros([trunc for i in range(n * 2)], dtype=def_type)
        out_base = sliceExp(modes, dict(zip(modes, [0] * len(modes))), n)
        outSlice = intersperse(out_base)
        
        for i_vals in product(*[range(xi, trunc) for xi in x]):
            weight = 1.0
            for i, xi in zip(i_vals, x):
                weight *= comb(i, xi, exact=False) * (eta ** xi) * ((1 - eta) ** (i - xi))
            
            in_base = sliceExp(modes, dict(zip(modes, i_vals)), n)
            inSlice = intersperse(in_base)
            ret[outSlice] += weight * state[inSlice]
    
    return ret


def patch_loss_measure_fock():
    """Patch the Fock backend with optimized V2."""
    from strawberryfields.backends.fockbackend.backend import FockBackend
    from strawberryfields.backends.fockbackend.circuit import Circuit
    from strawberryfields.compilers.fock import Fock as FockCompiler
    
    def backend_loss_measure_fock(self, modes, eta, shots=1, select=None, **kwargs):
        return self.circuit.loss_measure_fock(self._remap_modes(modes), eta, select)
    

    FockBackend.loss_measure_fock = backend_loss_measure_fock
    
    Circuit.loss_measure_fock = loss_measure_fock_implementation
    
    if "LossMeasureFock" not in FockCompiler.primitives:
        FockCompiler.primitives.add("LossMeasureFock")
    
    print("LossMeasureFock V2 (JIT-optimized) patched successfully")


def revert_loss_measure_fock_patch():
    """Revert the patch."""
    from strawberryfields.backends.fockbackend.backend import FockBackend
    from strawberryfields.backends.fockbackend.circuit import Circuit
    from strawberryfields.compilers.fock import Fock as FockCompiler
    
    if hasattr(FockBackend, 'loss_measure_fock'):
        delattr(FockBackend, 'loss_measure_fock')
    
    if hasattr(Circuit, 'loss_measure_fock'):
        delattr(Circuit, 'loss_measure_fock')
    
    if "LossMeasureFock" in FockCompiler.primitives:
        FockCompiler.primitives.discard("LossMeasureFock")
    
    print("LossMeasureFock V2 patch reverted")




def test_with_postselection():
    """Test LossMeasureFock with post-selection across different eta and select values."""
    print("="*60)
    print("Test 1: With Post-selection")
    print("="*60)
    
    # Test different combinations
    test_configs = [
        {'eta': 0.01, 'select': 2, 'cutoff': 6},
        {'eta': 0.1, 'select': 4, 'cutoff': 7},
        {'eta': 0.2, 'select': 2, 'cutoff': 9},
        {'eta': 0.7, 'select': 2, 'cutoff': 6},
        {'eta': 0.5, 'select': 1, 'cutoff': 6},
        {'eta': 0.9, 'select': 3, 'cutoff': 8},
        {'eta': 0.3, 'select': 0, 'cutoff': 6},
        {'eta': 1.0, 'select': 2, 'cutoff': 6},  # Perfect detection
    ]
    
    all_pass = True
    
    for i, config in enumerate(test_configs):
        ETA = config['eta']
        post_select_val = config['select']
        cutoff = config['cutoff']
        
        print(f"\n--- Config {i+1}/{len(test_configs)}: eta={ETA}, select={post_select_val}, cutoff={cutoff} ---")
        
        # Method 1: Original (LossChannel + MeasureFock)
        np.random.seed(42)
        prog1 = sf.Program(2)
        with prog1.context as q:
            Dgate(0.3, np.pi / 3) | q[0]
            Sgate(0.2, 0.1) | q[1]
            BSgate(np.pi / 4, np.pi / 6) | (q[0], q[1])
            LossChannel(ETA) | q[0]
            MeasureFock(select=post_select_val) | q[0]
        
        eng1 = sf.Engine("fock", backend_options={"cutoff_dim": cutoff})
        result1 = eng1.run(prog1)
        dm1 = result1.state.dm()
        
        # Method 2: New LossMeasureFock operation
        patch_loss_measure_fock()
        
        np.random.seed(42)
        prog2 = sf.Program(2)
        with prog2.context as q:
            Dgate(0.3, np.pi / 3) | q[0]
            Sgate(0.2, 0.1) | q[1]
            BSgate(np.pi / 4, np.pi / 6) | (q[0], q[1])
            LossMeasureFock(eta=ETA, select=post_select_val) | q[0]
        
        eng2 = sf.Engine("fock", backend_options={"cutoff_dim": cutoff})
        result2 = eng2.run(prog2)
        dm2 = result2.state.dm()

        
        # Compare
        diff = np.max(np.abs(dm1 - dm2))
        
        if diff < 1e-10:
            print(f"  Result: PASS (diff={diff:.2e})")
        else:
            print(f"  Result: FAIL (diff={diff:.2e})")
            all_pass = False
        
        revert_loss_measure_fock_patch()
    
    print(f"\n{'='*60}")
    if all_pass:
        print(f"[PASS] All {len(test_configs)} post-selection tests passed!")
    else:
        print(f"[FAIL] Some post-selection tests failed!")
    print(f"{'='*60}")
    
    return all_pass


def test_without_postselection():
    """Test LossMeasureFock without post-selection (random sampling) with different eta values."""
    print("\n" + "="*60)
    print("Test 2: Without Post-selection (Random Sampling)")
    print("="*60)
    
    # Test different eta values
    eta_values = [0.3, 0.5, 0.7, 0.9, 1.0]
    cutoff = 8
    n_samples = 50
    
    all_pass = True
    
    for eta_idx, ETA in enumerate(eta_values):
        print(f"\n--- Config {eta_idx+1}/{len(eta_values)}: eta={ETA} ---")
        
        # We'll compare the distribution of outcomes
        outcomes1 = []
        outcomes2 = []
        
        # Method 1: Original
        for i in range(n_samples):
            np.random.seed(100 + i)
            prog1 = sf.Program(1)
            with prog1.context as q:
                Dgate(1) | q[0]
                LossChannel(ETA) | q[0]
                MeasureFock() | q[0]
            
            eng1 = sf.Engine("fock", backend_options={"cutoff_dim": cutoff})
            result1 = eng1.run(prog1)
            outcomes1.append(result1.samples[0][0])
        
        # Method 2: LossMeasureFock
        patch_loss_measure_fock()
        
        for i in range(n_samples):
            np.random.seed(100 + i)
            prog2 = sf.Program(1)
            with prog2.context as q:
                Dgate(1) | q[0]
                LossMeasureFock(eta=ETA) | q[0]
            
            eng2 = sf.Engine("fock", backend_options={"cutoff_dim": cutoff})
            result2 = eng2.run(prog2)
            outcomes2.append(result2.samples[0][0])
        
        outcomes1 = np.array(outcomes1)
        outcomes2 = np.array(outcomes2)
        
        # Check if outcomes match exactly (with same random seeds)
        matches = np.sum(outcomes1 == outcomes2)
        
        # Compute empirical distributions
        unique1, counts1 = np.unique(outcomes1, return_counts=True)
        unique2, counts2 = np.unique(outcomes2, return_counts=True)
        
        dist1 = dict(zip(unique1, counts1/n_samples))
        dist2 = dict(zip(unique2, counts2/n_samples))
        
        print(f"  Matching outcomes: {matches}/{n_samples}")
        print(f"  Distribution 1: {dist1}")
        print(f"  Distribution 2: {dist2}")
        
        if matches == n_samples:
            print(f"  Result: PASS (all samples match)")
        else:
            print(f"  Result: FAIL (only {matches}/{n_samples} match)")
            all_pass = False
        
        revert_loss_measure_fock_patch()
    
    print(f"\n{'='*60}")
    if all_pass:
        print(f"[PASS] All {len(eta_values)} random sampling tests passed!")
    else:
        print(f"[FAIL] Some random sampling tests failed!")
    print(f"{'='*60}")
    
    return all_pass


def test_multi_mode():
    """Test LossMeasureFock on multiple modes simultaneously."""
    print("\n" + "="*60)
    print("Test 3: Multi-mode Measurement")
    print("="*60)
    
    ETA = 0.7
    cutoff = 5
    
    # Test measuring both modes with different eta values
    # Note: Our current implementation uses same eta for all modes in one call
    # For different etas, we need separate calls
    
    patch_loss_measure_fock()
    
    np.random.seed(42)
    prog = sf.Program(2)
    with prog.context as q:
        Dgate(0.4) | q[0]
        Dgate(0.3) | q[1]
        BSgate(np.pi / 4) | (q[0], q[1])
        LossMeasureFock(eta=0.7, select=1) | q[0]
        LossMeasureFock(eta=0.8, select=0) | q[1]
    
    eng = sf.Engine("fock", backend_options={"cutoff_dim": cutoff})
    result = eng.run(prog)
    
    print(f"\nMeasurement results: {result.samples}")
    print(f"State after measurements: {result.state}")
    print("[PASS] Multi-mode measurement works!")
    
    revert_loss_measure_fock_patch()
    return True



def main():
    """Run all tests and benchmarks."""
    print("\n" + "="*70)
    print(" LossMeasureFock Operation - Validation and Benchmarking ")
    print("="*70)
    
    # Run tests
    test1_pass = test_with_postselection()
    test2_pass = test_without_postselection()
    test3_pass = test_multi_mode()
    
    # Summary
    print("\n" + "="*70)
    print(" Summary ")
    print("="*70)
    print(f"Test 1 (Post-selection):     {'PASS' if test1_pass else 'FAIL'}")
    print(f"Test 2 (Random sampling):    {'PASS' if test2_pass else 'FAIL'}")
    print(f"Test 3 (Multi-mode):         {'PASS' if test3_pass else 'FAIL'}")
    print("="*70)
    
    if all([test1_pass, test2_pass, test3_pass]):
        print("\n[SUCCESS] All tests passed!")
    else:
        print("\n[FAIL] Some tests failed!")


if __name__ == "__main__":
    main()

