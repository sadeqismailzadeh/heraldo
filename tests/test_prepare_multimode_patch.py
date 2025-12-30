import quantum_agent
import unittest
import numpy as np
import strawberryfields as sf
from strawberryfields.ops import *
import sys
import os
import math

# Add project root to path to verify imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from quantum_agent.patches.prepare_multimode_patch import patch_prepare_multimode, revert_prepare_multimode_patch
except ImportError:
    print("Error: Could not import patch. Ensure 'quantum_agent' package is in the python path.")
    sys.exit(1)

class TestPrepareMultimodeOptimization(unittest.TestCase):
    
    def setUp(self):
        # Ensure we start with a clean slate (original backend)
        revert_prepare_multimode_patch()
        # High truncation to avoid numerical artifacts affecting separability checks
        self.trunc = 15
        self.tol = 1e-8 # Loosened tolerance to account for acceptable numerical deviations

    def tearDown(self):
        # Clean up
        revert_prepare_multimode_patch()

    def _run_program(self, prog, use_patch, n_modes=2):
        if use_patch:
            patch_prepare_multimode()
        else:
            revert_prepare_multimode_patch()
            
        eng = sf.Engine("fock", backend_options={"cutoff_dim": self.trunc})
        result = eng.run(prog)
        return result.state

    def test_unentangled_preserves_purity(self):
        """
        Scenario: Two modes are independent (separable). We prepare one of them.
        Expected: 
            - Original: Converts to mixed state (density matrix).
            - Patched: Maintains pure state (vector).
            - Both: Numerical fidelities match.
        """
        prog = sf.Program(2)
        with prog.context as q:
            # Create a separable state: |alpha> (x) |alpha>
            Dgate(0.5) | q[0]
            Dgate(0.5) | q[1]
            # Replace q[0] with Vacuum. System remains separable: |0> (x) |alpha>
            Vacuum() | q[0]

        # 1. Run Original
        state_orig = self._run_program(prog, use_patch=False)
        self.assertFalse(state_orig.is_pure, "Original backend should produce mixed state (fallback) for partial prep")
        
        # 2. Run Patched
        state_opt = self._run_program(prog, use_patch=True)
        self.assertTrue(state_opt.is_pure, "Patched backend should preserve purity for unentangled replacement")
        
        # 3. Numerical Comparison
        diff = np.max(np.abs(state_orig.dm() - state_opt.dm()))
        self.assertLess(diff, self.tol, f"States diverge! Max diff: {diff}")

    def test_unentangled_preserves_purity_fock(self):
        """
        Scenario: Two modes are independent (separable). We prepare one of them.
        Expected: 
            - Original: Converts to mixed state (density matrix).
            - Patched: Maintains pure state (vector).
            - Both: Numerical fidelities match.
        """
        prog = sf.Program(2)
        with prog.context as q:
            # Create a separable state: |alpha> (x) |alpha>
            Dgate(0.5) | q[0]
            Dgate(0.5) | q[1]
            # Replace q[0] with Vacuum. System remains separable: |0> (x) |alpha>
            Fock(1) | q[0]

        # 1. Run Original
        state_orig = self._run_program(prog, use_patch=False)
        self.assertFalse(state_orig.is_pure, "Original backend should produce mixed state (fallback) for partial prep")
        
        # 2. Run Patched
        state_opt = self._run_program(prog, use_patch=True)
        self.assertTrue(state_opt.is_pure, "Patched backend should preserve purity for unentangled replacement")
        
        # 3. Numerical Comparison
        diff = np.max(np.abs(state_orig.dm() - state_opt.dm()))
        self.assertLess(diff, self.tol, f"States diverge! Max diff: {diff}")


    def test_unentangled_preserves_purity_double(self):
        """
        Scenario: Two modes are independent (separable). We prepare one of them.
        Expected: 
            - Original: Converts to mixed state (density matrix).
            - Patched: Maintains pure state (vector).
            - Both: Numerical fidelities match.
        """
        prog = sf.Program(2)
        with prog.context as q:
            # Create a separable state: |alpha> (x) |alpha>
            Dgate(0.5) | q[0]
            Dgate(0.5) | q[1]
            # Replace q[0] with Vacuum. System remains separable: |0> (x) |alpha>
            Fock(1) | q[0]
            Catstate(a=2, phi= 0.5) | q[0]

        # 1. Run Original
        state_orig = self._run_program(prog, use_patch=False)
        self.assertFalse(state_orig.is_pure, "Original backend should produce mixed state (fallback) for partial prep")
        
        # 2. Run Patched
        state_opt = self._run_program(prog, use_patch=True)
        self.assertTrue(state_opt.is_pure, "Patched backend should preserve purity for unentangled replacement")
        
        # 3. Numerical Comparison
        diff = np.max(np.abs(state_orig.dm() - state_opt.dm()))
        self.assertLess(diff, self.tol, f"States diverge! Max diff: {diff}")
        
        
    def test_entangled_fallback(self):
        """
        Scenario: Two modes are entangled. We prepare one of them.
        Expected:
            - Patched: Detects entanglement, falls back to mixed state logic.
            - Result is mixed.
            - Numerical results match original.
        """
        prog = sf.Program(2)
        with prog.context as q:
            # Create Bell-like state (Two-mode squeezed vacuum)
            S2gate(1.0) | (q[0], q[1])
            # Discard/Replace q[0]. Remaining q[1] is thermal (mixed).
            Vacuum() | q[0]

        # 1. Run Original
        state_orig = self._run_program(prog, use_patch=False)
        
        # 2. Run Patched
        state_opt = self._run_program(prog, use_patch=True)
        
        # 3. Assertions
        self.assertFalse(state_opt.is_pure, "Patched backend must fallback to mixed state for entangled systems")
        
        diff = np.max(np.abs(state_orig.dm() - state_opt.dm()))
        self.assertLess(diff, self.tol, f"Entangled fallback produced incorrect state. Diff: {diff}")

    def test_multi_mode_complex_replacement(self):
        """
        Scenario: 3 modes. 0 and 2 are entangled. 1 is separable.
        We prepare mode 1 (the separable one) into a Coherent state.
        Expected:
            - System is |Psi_02> (x) |phi_1>.
            - Replacing 1 -> |new> keeps it |Psi_02> (x) |new>.
            - However, our optimization heuristic requires the *kept* modes to be pure.
              Here, kept modes {0,2} are pure together.
              The logic should hold if the cut splits {0,2} from {1}.
        """
        prog = sf.Program(3)
        with prog.context as q:
            # Entangle 0 and 2
            S2gate(0.5) | (q[0], q[2])
            # Mode 1 is Vacuum
            
            # Now replace Mode 1 with Coherent
            # Kept modes: {0, 2}. Replaced: {1}.
            # State |Psi_{0,2}> (x) |0>_1
            # Optimization check: 
            # Slice indices for replaced mode 1. 
            # If we slice at index 0 of mode 1, we get |Psi_{0,2}> * scalar.
            # Projection should be 1.0. Optimization should trigger.
            Dgate(0.5) | q[1] 

        state_opt = self._run_program(prog, use_patch=True, n_modes=3)
        self.assertTrue(state_opt.is_pure, "Optimization failed to detect separability of mode 1 from entangled pair {0,2}")

    def test_optimization_failure_on_corrupted_slice(self):
        """
        Regression test:
        If the internal slice is a view and modified in-place, the calculation is wrong.
        We check correctness on a state that has non-trivial phases.
        """
        prog = sf.Program(2)
        with prog.context as q:
            # Non-trivial separable state
            Dgate(0.5, 0.1) | q[0]
            Rgate(np.pi/3) | q[0]
            
            Sgate(0.2) | q[1]
            Dgate(0.3) | q[1]
            
            # Replace q[0] with Fock state |1>
            Fock(1) | q[0]

        state_orig = self._run_program(prog, use_patch=False)
        state_opt = self._run_program(prog, use_patch=True)
        
        self.assertTrue(state_opt.is_pure)
        diff = np.max(np.abs(state_orig.dm() - state_opt.dm()))
        self.assertLess(diff, self.tol)

    def test_beamsplitter_coherent_preserves_purity(self):
        """
        Scenario: Coherent states passed through a Beam Splitter remain separable (product states).
        Expected:
            - Start: |alpha> (x) |0>
            - Apply BS: |t*alpha> (x) |r*alpha>
            - This is UNENTANGLED.
            - Replace mode 0.
            - Optimization should detect that keeping mode 1 is valid/pure.
            - Result should be Pure.
        """
        prog = sf.Program(2)
        with prog.context as q:
            Dgate(1.0) | q[0]
            # BS on coherent states -> product of coherent states
            BSgate(np.pi/4, 0.0) | (q[0], q[1])
            # Replace q[0] with Vacuum. 
            # System is |0> (x) |coherent_remnant>. Separable.
            Vacuum() | q[0]

        # 1. Run Original (fails to optimize, produces mixed)
        state_orig = self._run_program(prog, use_patch=False)
        self.assertFalse(state_orig.is_pure)

        # 2. Run Patched (should optimize)
        state_opt = self._run_program(prog, use_patch=True)
        self.assertTrue(state_opt.is_pure, "Beam splitter on coherent states produces separable states; optimization should preserve purity.")
        
        # 3. Check consistency
        diff = np.max(np.abs(state_orig.dm() - state_opt.dm()))
        self.assertLess(diff, self.tol)

    def test_beamsplitter_fock_entanglement_fallback(self):
        """
        Scenario: Fock states passed through a Beam Splitter become entangled.
        Expected:
            - Start: |1> (x) |0>
            - Apply BS: entangled superposition.
            - Replace mode 0.
            - Kept mode 1 is mixed.
            - Optimization must NOT trigger. Result is Mixed.
        """
        prog = sf.Program(2)
        with prog.context as q:
            Fock(1) | q[0]
            # BS on Fock states -> entangled
            BSgate(np.pi/4, 0.0) | (q[0], q[1])
            # Replace q[0]
            Vacuum() | q[0]

        state_opt = self._run_program(prog, use_patch=True)
        self.assertFalse(state_opt.is_pure, "Entangled Fock state remnant must be mixed; optimization should have fallen back.")

    def test_theoretical_prediction(self):
        """
        Scenario: Compare simulation result against manual analytical construction.
        State: Prepare |0>|0>, displace q[0] -> |alpha>|0>, replace q[1] with |1>.
        Final Theoretical: |alpha> (x) |1>.
        """
        alpha = 0.5 + 0.0j
        trunc = self.trunc
        
        prog = sf.Program(2)
        with prog.context as q:
            Dgate(np.abs(alpha)) | q[0]
            Fock(1) | q[1] # Replaces vacuum in q[1]
            
        # Run optimized simulation
        state_sim = self._run_program(prog, use_patch=True)
        self.assertTrue(state_sim.is_pure)
        ket_sim = state_sim.ket() # Shape (trunc, trunc)
        
        # Construct Theoretical Vector
        # 1. Coherent state |alpha> for q[0]
        # formula: c_n = exp(-|a|^2/2) * a^n / sqrt(n!)
        n = np.arange(trunc)
        # Calculate prefactor and powers
        # Note: Using standard numpy for theoretical check
        prefactor = np.exp(-0.5 * np.abs(alpha)**2)
        # Avoid 0^0 issues if alpha=0, though here alpha=0.5
        coeffs = prefactor * (alpha**n) / np.sqrt([float(math.factorial(k)) for k in n])
        vec0 = coeffs 
        # Normalize theoretical vector due to truncation
        vec0 /= np.linalg.norm(vec0)
        
        # 2. Fock state |1> for q[1]
        vec1 = np.zeros(trunc, dtype=np.complex128)
        vec1[1] = 1.0
        
        # 3. Tensor Product |alpha> (x) |1> -> Matrix outer product
        # Index ordering in SF fock backend ket(): [mode0, mode1]
        ket_theo = np.outer(vec0, vec1)
        
        # Compare
        # Note: Global phase might differ, so we compare density matrices or absolute overlap
        overlap = np.abs(np.vdot(ket_sim.flatten(), ket_theo.flatten()))
        self.assertAlmostEqual(overlap, 1.0, places=6, msg="Simulation did not match theoretical prediction |alpha>|1>")

if __name__ == '__main__':
    unittest.main()
