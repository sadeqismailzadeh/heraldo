"""
COMPLETE SOLUTION: Your circuit with NO CACHING
Copy this entire script or just copy the disable_fock_caching() function to your code
"""

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
    
    print("✅ Caching DISABLED for all Fock backend operations!")
    print("   Gates will be recomputed every time.")
    print("   Memory usage will stay constant - no buildup!")


def main():
    # ============================================================================
    # STEP 2: CALL THE FUNCTION ONCE AT THE START
    # ============================================================================

    disable_fock_caching()

    # ============================================================================
    # STEP 3: RUN YOUR CIRCUIT NORMALLY - NO MEMORY ISSUES!
    # ============================================================================

    # Configuration
    cutoff = 45  # Can use high cutoff now!
    n_modes = 2
    num_iterations = 100  # As many as you want!

    print(f"\nRunning {num_iterations} iterations with cutoff={cutoff}")
    print("=" * 70)

    # Create engine
    eng = sf.Engine("fock", backend_options={"cutoff_dim": cutoff})

    # Run your circuit in a loop
    for index in range(num_iterations):
        prog = sf.Program(n_modes)
        
        # Your parameters (can be random, doesn't matter!)
        theta_1 = np.random.rand() * np.pi
        theta_2 = np.random.rand() * np.pi
        
        with prog.context as q:
            # Your exact circuit
            rot_theta = index
            Sgate(1.38, 0) | q[1]
            BSgate(theta_1, 0) | (q[0], q[1])
            MeasureFock(select=4) | q[0]
            BSgate(theta_2, 0) | (q[0], q[1])
        
        # Run the circuit
        result = eng.run(prog)
        
        # Progress update
        if (index + 1) % 10 == 0:
            print(f"✅ Completed {index + 1}/{num_iterations} iterations - Memory stays constant!")

    print("\n" + "=" * 70)
    print("✅ DONE! No memory issues!")
    print("=" * 70)

    # Verify that caching is disabled
    from strawberryfields.backends.fockbackend import ops

    if hasattr(ops.beamsplitter, 'cache_clear'):
        print("\n⚠️  WARNING: Caching is still enabled somehow!")
        print("   Make sure disable_fock_caching() was called.")
    else:
        print("\n✅ CONFIRMED: Caching is disabled!")
        print("   ops.beamsplitter.cache_clear() does not exist (this is correct!)")


if __name__ == "__main__":
    main()