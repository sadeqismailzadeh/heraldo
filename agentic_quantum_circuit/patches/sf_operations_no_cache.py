"""
COMPLETE SOLUTION: Your circuit with NO CACHING
Copy this entire script or just copy the disable_fock_caching() function to your code
"""

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

# ============================================================================
# STEP 1: DISABLE CACHING FUNCTION (Copy this to your code!)
# ============================================================================

def disable_fock_caching():
    """
    Completely disables caching in Fock backend.
    Call this ONCE at the very beginning of your script.
    
    After calling this:
    - ops.beamsplitter.cache_clear() will NOT exist (no cache!)
    - Memory usage stays constant
    - Slightly slower (recomputes gates each time)
    """
    from strawberryfields.backends.fockbackend import ops
    from thewalrus.fock_gradients import (
        beamsplitter as beamsplitter_tw,
        squeezing as squeezing_tw,
        displacement as displacement_tw,
        two_mode_squeezing as two_mode_squeezing_tw,
        mzgate as mzgate_tw,
    )
    
    # Create uncached versions of all gate functions
    def beamsplitter_uncached(theta, phi, trunc):
        BS_tw = beamsplitter_tw(theta, phi, cutoff=trunc)
        return BS_tw.transpose((0, 2, 1, 3))
    
    def squeezing_uncached(r, theta, trunc):
        return squeezing_tw(r, theta, cutoff=trunc)
    
    def displacement_uncached(r, phi, trunc):
        return displacement_tw(r, phi, cutoff=trunc)
    
    def two_mode_squeeze_uncached(r, theta, trunc):
        ret = two_mode_squeezing_tw(r, theta, cutoff=trunc)
        return np.transpose(ret, [0, 2, 1, 3])
    
    def mzgate_uncached(phi_in, phi_ex, cutoff):
        ret = mzgate_tw(phi_in, phi_ex, cutoff)
        return ret.transpose((0, 2, 1, 3))
    
    def phase_uncached(theta, trunc):
        return np.array(np.diag([np.exp(1j * n * theta) for n in range(trunc)]), 
                       dtype=np.complex128)
    
    def kerr_uncached(kappa, trunc):
        n = np.arange(trunc)
        return np.diag(np.exp(1j * kappa * n**2))
    
    def cross_kerr_uncached(kappa, trunc):
        n1 = np.arange(trunc)[None, :]
        n2 = np.arange(trunc)[:, None]
        n1n2 = np.ravel(n1 * n2)
        return np.diag(np.exp(1j * kappa * n1n2)).reshape([trunc] * 4).swapaxes(1, 2)
    
    # Replace the cached functions with uncached versions
    ops.beamsplitter = beamsplitter_uncached
    ops.squeezing = squeezing_uncached
    ops.displacement = displacement_uncached
    ops.two_mode_squeeze = two_mode_squeeze_uncached
    ops.mzgate = mzgate_uncached
    ops.phase = phase_uncached
    ops.kerr = kerr_uncached
    ops.cross_kerr = cross_kerr_uncached
    
    print("Caching DISABLED for all Fock backend operations!")
    print("   Gates will be recomputed every time.")
    print("   Memory usage will stay constant - no buildup!")



def main():
    """
    This function demonstrates that disabling caching does not affect the
    final state of the quantum simulation. It does this by:
    1. Calculating a state with default (cached) operators.
    2. Disabling the cache.
    3. Recalculating the same state with the new (uncached) operators.
    4. Asserting that the two states are numerically identical.
    """
    # Configuration
    cutoff = 10  # Use a smaller cutoff for a quick check
    n_modes = 2

    
    # Define the quantum program to be used in both scenarios
    prog = sf.Program(n_modes)
    # Use fixed parameters for a deterministic and repeatable comparison
    theta_1 = np.pi / 4
    theta_2 = np.pi / 3

    with prog.context as q:
        # State preparation to ensure gates have a non-trivial effect
        Dgate(0.5, np.pi/4) | q[0] # Corresponds to ops.displacement
        Sgate(0.6, 0) | q[1]          # Corresponds to ops.squeezing

        # Apply all other disabled operators
        Rgate(np.pi/3) | q[0]             # Corresponds to ops.phase
        # Two-mode gates
        BSgate(np.pi/4, np.pi/6) | (q[0], q[1]) # Corresponds to ops.beamsplitter
        S2gate(0.7, np.pi/2) | (q[0], q[1])     # Corresponds to ops.two_mode_squeeze
        MZgate(np.pi/3, np.pi/5) | (q[0], q[1]) # Corresponds to ops.mzgate
        CKgate(0.3) | (q[0], q[1])          # Corresponds to ops.cross_kerr
    
    # ============================================================================
    # STEP 1: Make state with default operators (caching is ON by default)
    # ============================================================================
    print("--- 1. Calculating state with default (cached) operators ---")
    eng_original = sf.Engine("fock", backend_options={"cutoff_dim": cutoff})
    result_original = eng_original.run(prog)
    state_original = result_original.state
    print("Original state calculated successfully.")

    # ============================================================================
    # STEP 2: Disable the cache
    # ============================================================================
    print("\n--- 2. Disabling Fock backend caching ---")
    disable_fock_caching()

    # ============================================================================
    # STEP 3: Re-run the program and get the new state
    # ============================================================================
    print("\n--- 3. Recalculating state with cache disabled ---")
    # It's good practice to create a new engine after modifying the backend
    eng_no_cache = sf.Engine("fock", backend_options={"cutoff_dim": cutoff})
    result_no_cache = eng_no_cache.run(prog)
    state_no_cache = result_no_cache.state
    print("State with cache disabled calculated successfully.")

    # ============================================================================
    # STEP 4: Assert that the states are identical
    # ============================================================================
    print("\n--- 4. Asserting that the two states are identical ---")
    
    # Use np.allclose for robust floating-point comparison of the state vectors
    are_states_equal = np.allclose(state_original.dm(), state_no_cache.dm())
    
    assert are_states_equal, "State with cache disabled is DIFFERENT from the original state!"

    print("\n" + "=" * 70)
    print("✅ SUCCESS: The state calculated with caching disabled is identical to the original.")
    print("This confirms the `disable_fock_caching` function works as expected.")
    print("=" * 70)


if __name__ == "__main__":
    main()
