"""
Test suite asserting mathematical and numerical equivalence between:
1. evaluate_time_domain_circuit (heraldo.components.runner) with steps = 1
2. evaluate_circuit (heraldo.components.static_runner)

This test verifies that time-domain multiplexed circuits with steps = 1 produce
identical state vector evolution, outcome probabilities, target fidelities,
and objective loss metrics as their static spatial circuit counterparts across
2-mode and 3-mode configurations in both Beam Search and Fixed Pattern modes.
"""

import os
import sys
import warnings
import pytest
import numpy as np

# Set thread limits before backend imports
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

warnings.filterwarnings("ignore", category=RuntimeWarning, module="scipy.optimize")

from heraldo.experimental.time_circuits import (
    TwoModeTimeDomainSqueezeOnly,
    ThreeModeTimeDomainSqueezeOnly,
)
from heraldo.components.circuits import (
    TwoModeStaticSqueezeOnly,
    ThreeModeStaticSqueezeOnly,
)
from heraldo.components.targets import (
    SqueezedCatTarget,
    CatTarget,
    BinomialCodeTarget,
    CubicPhaseTarget,
)
from heraldo.components.objectives import (
    beam_search_loss_fn,
    fixed_pattern_capped_loss_fn,
    fixed_pattern_free_loss_fn,
)
from heraldo.experimental.time_runner import (
    evaluate_time_domain_circuit,
)
from heraldo.components.runner import evaluate_circuit


def map_2mode_tdm_to_static(tdm_params: np.ndarray) -> np.ndarray:
    """Maps 6 flat parameters from TwoModeTimeDomainSqueezeOnly (steps=1)
    to TwoModeStaticSqueezeOnly parameter ordering.
    
    TDM flat params (6): [init_sq_r, init_sq_phi, sq_r, sq_phi, bs_theta, bs_phi]
    Static flat params (6): [sq0_r, sq0_phi, sq1_r, sq1_phi, bs_theta, bs_phi]
    """
    return tdm_params.copy()


def map_3mode_tdm_to_static(tdm_params: np.ndarray) -> np.ndarray:
    """Maps 12 flat parameters from ThreeModeTimeDomainSqueezeOnly (steps=1)
    to ThreeModeStaticSqueezeOnly parameter ordering.

    TDM flat params (12):
        [init_sq_r, init_sq_phi, sq1_r, sq2_r, sq1_phi, sq2_phi, bs_theta1, bs_theta2, bs_theta3, bs_phi1, bs_phi2, bs_phi3]

    Static flat params (12):
        [sq0_r, sq1_r, sq2_r, sq0_phi, sq1_phi, sq2_phi, bs_theta1, bs_theta2, bs_theta3, bs_phi1, bs_phi2, bs_phi3]
    """
    init_sq_r, init_sq_phi = tdm_params[0], tdm_params[1]
    sq1_r, sq2_r = tdm_params[2], tdm_params[3]
    sq1_phi, sq2_phi = tdm_params[4], tdm_params[5]
    bs_theta = tdm_params[6:9]
    bs_phi = tdm_params[9:12]

    return np.array([
        init_sq_r, sq1_r, sq2_r,
        init_sq_phi, sq1_phi, sq2_phi,
        bs_theta[0], bs_theta[1], bs_theta[2],
        bs_phi[0], bs_phi[1], bs_phi[2]
    ])


def assert_evaluation_details_match(res_tdm: dict, res_static: dict, rtol: float = 1e-6, atol: float = 1e-6):
    """Asserts that two detailed evaluation dictionary outputs match within numerical tolerances."""
    # 1. Compare top-level scalar metrics
    np.testing.assert_allclose(
        res_tdm["loss"], res_static["loss"], rtol=rtol, atol=atol,
        err_msg="Loss value mismatch between TDM (steps=1) and Static runner."
    )
    np.testing.assert_allclose(
        res_tdm["expected_fidelity"], res_static["expected_fidelity"], rtol=rtol, atol=atol,
        err_msg="Expected fidelity mismatch between TDM (steps=1) and Static runner."
    )
    np.testing.assert_allclose(
        res_tdm["total_probability"], res_static["total_probability"], rtol=rtol, atol=atol,
        err_msg="Total probability mismatch between TDM (steps=1) and Static runner."
    )

    # 2. Compare branch details
    branches_tdm = sorted(res_tdm["branches"], key=lambda b: b["outcome"])
    branches_static = sorted(res_static["branches"], key=lambda b: b["outcome"])

    assert len(branches_tdm) == len(branches_static), (
        f"Branch count mismatch: TDM produced {len(branches_tdm)} branches, Static produced {len(branches_static)}"
    )

    for idx, (b_tdm, b_static) in enumerate(zip(branches_tdm, branches_static)):
        assert b_tdm["outcome"] == b_static["outcome"], (
            f"Branch {idx} outcome mismatch: TDM={b_tdm['outcome']}, Static={b_static['outcome']}"
        )
        assert b_tdm["target_idx"] == b_static["target_idx"], (
            f"Branch {idx} target_idx mismatch: TDM={b_tdm['target_idx']}, Static={b_static['target_idx']}"
        )
        np.testing.assert_allclose(
            b_tdm["prob"], b_static["prob"], rtol=rtol, atol=atol,
            err_msg=f"Branch {idx} ({b_tdm['outcome']}) probability mismatch."
        )
        np.testing.assert_allclose(
            b_tdm["fidelity"], b_static["fidelity"], rtol=rtol, atol=atol,
            err_msg=f"Branch {idx} ({b_tdm['outcome']}) fidelity mismatch."
        )


class TestEvaluatorsEquivalence:
    """Test suite comparing evaluate_time_domain_circuit (steps=1) and evaluate_circuit."""

    @pytest.mark.parametrize("mode", ["beam_search", "fixed_pattern"])
    @pytest.mark.parametrize("return_details", [False, True])
    def test_two_mode_squeeze_only_equivalence(self, mode: str, return_details: bool):
        """Tests 2-mode squeeze-only circuit equivalence between TDM (steps=1) and Static models."""
        cutoff_dim = 15
        beam_width = 10
        penalty_strength = 1.0

        # Define 2-mode targets
        targets = [
            SqueezedCatTarget(alpha=1.5, r=0.3, p=0),
            CatTarget(alpha=2.0, p=1)
        ]
        target_kets = [t.get_target_ket(cutoff_dim) for t in targets]

        # Initialize circuits with measure_fock_cutoff set to cutoff_dim
        tdm_circuit = TwoModeTimeDomainSqueezeOnly(steps=1, time_invariant=False, measure_fock_cutoff=cutoff_dim)
        static_circuit = TwoModeStaticSqueezeOnly(measure_fock_cutoff=cutoff_dim)

        # Generate test parameters
        np.random.seed(42)
        tdm_params = np.random.uniform(-0.4, 0.4, size=6)
        static_params = map_2mode_tdm_to_static(tdm_params)

        if mode == "beam_search":
            patterns_tdm = None
            patterns_static = None
            loss_fn = beam_search_loss_fn
        else:
            # Fixed pattern mode: outcome n=4 and n=5 on mode 1
            patterns_tdm = np.array([[[4]], [[5]]])       # shape (2, 1, 1) -> (sequences, steps, meas_modes)
            patterns_static = np.array([[4], [5]])         # shape (2, 1) -> (sequences, meas_modes)
            loss_fn = fixed_pattern_capped_loss_fn

        # Run time-domain evaluator (steps = 1)
        res_tdm = evaluate_time_domain_circuit(
            flat_params=tdm_params,
            circuit=tdm_circuit,
            target_kets=target_kets,
            cutoff_dim=cutoff_dim,
            beam_width=beam_width,
            penalty_strength=penalty_strength,
            measurement_patterns=patterns_tdm,
            return_details=return_details,
            loss_fn=loss_fn
        )

        # Run static evaluator
        res_static = evaluate_circuit(
            params=static_params,
            circuit=static_circuit,
            target_kets=target_kets,
            cutoff_dim=cutoff_dim,
            beam_width=beam_width,
            penalty_strength=penalty_strength,
            measurement_patterns=patterns_static,
            return_details=return_details,
            loss_fn=loss_fn
        )

        if not return_details:
            assert isinstance(res_tdm, float)
            assert isinstance(res_static, float)
            np.testing.assert_allclose(
                res_tdm, res_static, rtol=1e-6, atol=1e-6,
                err_msg=f"Scalar loss mismatch in 2-mode {mode} evaluation."
            )
        else:
            assert_evaluation_details_match(res_tdm, res_static)

    @pytest.mark.parametrize("mode", ["beam_search", "fixed_pattern"])
    @pytest.mark.parametrize("loss_fn_type", ["capped", "free"])
    def test_three_mode_squeeze_only_equivalence(self, mode: str, loss_fn_type: str):
        """Tests 3-mode squeeze-only circuit equivalence between TDM (steps=1) and Static models."""
        cutoff_dim = 12
        beam_width = 15
        penalty_strength = 2.0

        # Define 3-mode targets
        targets = [
            BinomialCodeTarget(N=2, S=2, mu=0),
            CubicPhaseTarget(gamma=-0.1, r=-0.5, alpha=1.0)
        ]
        target_kets = [t.get_target_ket(cutoff_dim) for t in targets]

        # Initialize circuits with measure_fock_cutoff set to cutoff_dim
        tdm_circuit = ThreeModeTimeDomainSqueezeOnly(steps=1, time_invariant=False, measure_fock_cutoff=cutoff_dim)
        static_circuit = ThreeModeStaticSqueezeOnly(measure_fock_cutoff=cutoff_dim)

        # Generate test parameters
        np.random.seed(123)
        tdm_params = np.random.uniform(-0.3, 0.3, size=12)
        static_params = map_3mode_tdm_to_static(tdm_params)

        if mode == "beam_search":
            patterns_tdm = None
            patterns_static = None
            loss_fn = beam_search_loss_fn
        else:
            # Fixed patterns: (1, 3) and (2, 2) across measured modes 1 and 2
            patterns_tdm = np.array([[[1, 3]], [[2, 2]]])   # shape (2, 1, 2)
            patterns_static = np.array([[1, 3], [2, 2]])     # shape (2, 2)
            loss_fn = fixed_pattern_capped_loss_fn if loss_fn_type == "capped" else fixed_pattern_free_loss_fn

        # Evaluate both models with return_details=True
        res_tdm = evaluate_time_domain_circuit(
            flat_params=tdm_params,
            circuit=tdm_circuit,
            target_kets=target_kets,
            cutoff_dim=cutoff_dim,
            beam_width=beam_width,
            penalty_strength=penalty_strength,
            measurement_patterns=patterns_tdm,
            return_details=True,
            loss_fn=loss_fn
        )

        res_static = evaluate_circuit(
            params=static_params,
            circuit=static_circuit,
            target_kets=target_kets,
            cutoff_dim=cutoff_dim,
            beam_width=beam_width,
            penalty_strength=penalty_strength,
            measurement_patterns=patterns_static,
            return_details=True,
            loss_fn=loss_fn
        )

        assert_evaluation_details_match(res_tdm, res_static)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
