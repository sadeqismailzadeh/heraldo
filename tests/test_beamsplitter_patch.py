import unittest
import numpy as np
import quantum_agent
import strawberryfields as sf
from strawberryfields.ops import Dgate, BSgate
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from quantum_agent.patches.beamsplitter_patch import patch_beamsplitter, revert_beamsplitter_patch


class TestBeamsplitterPatch(unittest.TestCase):

    def setUp(self):
        revert_beamsplitter_patch()

    def tearDown(self):
        revert_beamsplitter_patch()

    def test_verify_correctness(self):
        theta_phi_pairs = [
            (np.pi / 4, np.pi / 6),
            (np.pi / 3, 0),
            (np.pi, np.pi / 7),
            (np.pi / 2, np.pi / 6),
            (np.pi / 5, np.pi / 9),
            (np.pi / 9, np.pi / 12),
            (2 * np.pi / 15, np.pi / 7),
            (2 * np.pi / 3, np.pi / 16),
            (7 * np.pi / 5, 11 * np.pi / 7),
            (np.pi / 2, 0),
        ]
        truncs = [10, 15]
        n_modes_list = [2, 3]

        for n_modes in n_modes_list:
            for trunc in truncs:
                for theta, phi in theta_phi_pairs:
                    with self.subTest(n_modes=n_modes, trunc=trunc, theta=theta, phi=phi):
                        prog = sf.Program(n_modes)
                        with prog.context as q:
                            for i in range(n_modes):
                                Dgate(0.3 * (i + 1), np.pi / (i + 3)) | q[i]
                            if n_modes >= 2:
                                BSgate(theta, phi) | (q[0], q[1])
                            if n_modes >= 3:
                                BSgate(theta / 2, phi / 2) | (q[1], q[2])

                        revert_beamsplitter_patch()
                        eng_original = sf.Engine("fock", backend_options={"cutoff_dim": trunc})
                        state1 = eng_original.run(prog).state

                        patch_beamsplitter()
                        eng_patched = sf.Engine("fock", backend_options={"cutoff_dim": trunc})
                        state2 = eng_patched.run(prog).state

                        self.assertTrue(state1.is_pure)
                        self.assertTrue(state2.is_pure)

                        if state1.is_pure:
                            diff = np.max(np.abs(state1.ket() - state2.ket()))
                        else:
                            diff = np.max(np.abs(state1.dm() - state2.dm()))

                        self.assertLess(
                            diff,
                            1e-10,
                            f"States differ! theta={theta}, phi={phi}, trunc={trunc}, diff={diff}",
                        )


if __name__ == "__main__":
    unittest.main()
