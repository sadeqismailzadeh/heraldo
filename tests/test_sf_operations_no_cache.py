import os
import sys
import unittest
import numpy as np
import quantum_agent
import strawberryfields as sf
from strawberryfields.ops import (
    BSgate,
    CKgate,
    Dgate,
    MZgate,
    Rgate,
    S2gate,
    Sgate,
)

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from quantum_agent.patches.sf_operations_no_cache import disable_fock_caching


class TestSfOperationsNoCache(unittest.TestCase):

    def test_disable_fock_caching(self):
        cutoff = 10
        n_modes = 2

        prog = sf.Program(n_modes)

        with prog.context as q:
            Dgate(0.5, np.pi / 4) | q[0]
            Sgate(0.6, 0) | q[1]
            Rgate(np.pi / 3) | q[0]
            BSgate(np.pi / 4, np.pi / 6) | (q[0], q[1])
            S2gate(0.7, np.pi / 2) | (q[0], q[1])
            MZgate(np.pi / 3, np.pi / 5) | (q[0], q[1])
            CKgate(0.3) | (q[0], q[1])

        # 1. Calculate state with default (cached) operators
        eng_original = sf.Engine("fock", backend_options={"cutoff_dim": cutoff})
        result_original = eng_original.run(prog)
        state_original = result_original.state

        # 2. Disable backend caching
        disable_fock_caching()

        # 3. Recalculate state with cache disabled
        eng_no_cache = sf.Engine("fock", backend_options={"cutoff_dim": cutoff})
        result_no_cache = eng_no_cache.run(prog)
        state_no_cache = result_no_cache.state

        # 4. Assert that the two states are identical
        are_states_equal = np.allclose(state_original.dm(), state_no_cache.dm())
        self.assertTrue(
            are_states_equal,
            "State with cache disabled is DIFFERENT from the original state!",
        )


if __name__ == "__main__":
    unittest.main()
