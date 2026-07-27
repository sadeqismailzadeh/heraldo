"""
Verification script to thoroughly test and assert numerical and logical equivalence
between `evaluate_time_domain_circuit_old` and `evaluate_time_domain_circuit`
across various circuit architectures, target states, beam search widths, and fixed measurement patterns.
"""

import os
import sys
import time
import warnings
from pathlib import Path
import numpy as np

# Set single thread environment variables
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

warnings.filterwarnings("ignore", category=RuntimeWarning, module="scipy.optimize")

import quantum_agent.optimization.time_circuits as circuit_module
import quantum_agent.components.targets as target_module
from quantum_agent.optimization.time_circuits import *
from quantum_agent.optimization.time_runner import (
    BasinHoppingRunner,
    evaluate_time_domain_circuit,
    evaluate_time_domain_circuit_old
)
from quantum_agent.components.targets import *
from quantum_agent.utils import *
from quantum_agent.factory import create_from_config


def prepare_measurement_patterns(patterns):
    """Convert input list patterns to numpy structure expected by runner."""
    if patterns is None:
        return None
    if isinstance(patterns, np.ndarray):
        if patterns.ndim == 2:
            return patterns[None, ...]
        return patterns
    try:
        patterns_np = np.array(patterns, dtype=int)
        if patterns_np.ndim == 2:
            patterns_np = patterns_np[None, ...]
        return patterns_np
    except Exception:
        return patterns


def compare_evaluations(res_old, res_new, rtol=1e-5, atol=1e-7):
    """Compares dictionary or float outputs from old and new evaluation functions."""
    if isinstance(res_old, (float, int, np.number)):
        np.testing.assert_allclose(res_old, res_new, rtol=rtol, atol=atol)
        return abs(res_old - res_new)

    # Detailed dict comparison
    loss_diff = abs(res_old["loss"] - res_new["loss"])
    fid_diff = abs(res_old["expected_fidelity"] - res_new["expected_fidelity"])
    prob_diff = abs(res_old["total_probability"] - res_new["total_probability"])

    np.testing.assert_allclose(res_old["loss"], res_new["loss"], rtol=rtol, atol=atol)
    np.testing.assert_allclose(res_old["expected_fidelity"], res_new["expected_fidelity"], rtol=rtol, atol=atol)
    np.testing.assert_allclose(res_old["total_probability"], res_new["total_probability"], rtol=rtol, atol=atol)

    branches_old = res_old.get("branches", [])
    branches_new = res_new.get("branches", [])
    assert len(branches_old) == len(branches_new), f"Branch count mismatch: {len(branches_old)} vs {len(branches_new)}"

    for b_old, b_new in zip(branches_old, branches_new):
        assert b_old["outcome"] == b_new["outcome"], f"Outcome mismatch: {b_old['outcome']} vs {b_new['outcome']}"
        assert b_old["target_idx"] == b_new["target_idx"], f"Target idx mismatch: {b_old['target_idx']} vs {b_new['target_idx']}"
        np.testing.assert_allclose(b_old["prob"], b_new["prob"], rtol=rtol, atol=atol)
        np.testing.assert_allclose(b_old["fidelity"], b_new["fidelity"], rtol=rtol, atol=atol)

    return loss_diff, fid_diff, prob_diff


def main():
    print("=" * 80)
    print("STARTING EQUIVALENCE TEST: evaluate_time_domain_circuit_old vs evaluate_time_domain_circuit")
    print("=" * 80)

    CUTOFF_DIM = 25
    STEPS = 1
    squeezing = db_to_r(12)
    csv_path_abs = str(Path(__file__).resolve().parent.parent.parent / "data" / "GKP_core_coefficients.csv")
    has_gkp_csv = os.path.exists(csv_path_abs)

    # Define test suite configurations
    test_cases = [
        # --- BEAM SEARCH CASES ---
        {
            "name": "Beam Search - 2 Mode - Squeezed Cat Target (B=10)",
            "circuit_config": {
                'class_name': 'TwoModeTimeDomainSqueezeOnly',
                'params': {
                    'steps': STEPS,
                    'time_invariant': False,
                    'clip_size': squeezing,
                    'measure_fock_cutoff': CUTOFF_DIM,
                    'num_single_photon': 0,
                    'train_initial_state': True,
                    'initial_r': squeezing,
                    'initial_fock_one': False
                }
            },
            "target_configs": [
                {'class_name': 'SqueezedCatTarget', 'params': {'alpha': np.sqrt(6), 'r': 0.5, 'p': 0}}
            ],
            "beam_width": 10,
            "patterns": None
        },
        {
            "name": "Beam Search - 3 Mode - Multi Cat Targets (B=20)",
            "circuit_config": {
                'class_name': 'ThreeModeTimeDomainSqueezeOnly',
                'params': {
                    'steps': STEPS,
                    'time_invariant': False,
                    'clip_size': squeezing,
                    'measure_fock_cutoff': CUTOFF_DIM,
                    'num_single_photon': 0,
                    'train_initial_state': True,
                    'initial_r': squeezing
                }
            },
            "target_configs": [
                {'class_name': 'SqueezedCatTarget', 'params': {'alpha': np.sqrt(6), 'r': 0.5, 'p': 0}},
                {'class_name': 'SqueezedCatTarget', 'params': {'alpha': np.sqrt(6), 'r': 0.5, 'p': 1}}
            ],
            "beam_width": 20,
            "patterns": None
        },
        {
            "name": "Beam Search - 3 Mode - Binomial Target (B=15)",
            "circuit_config": {
                'class_name': 'ThreeModeTimeDomainSqueezeOnly',
                'params': {
                    'steps': STEPS,
                    'time_invariant': False,
                    'clip_size': squeezing,
                    'measure_fock_cutoff': CUTOFF_DIM,
                    'num_single_photon': 0,
                    'train_initial_state': True,
                    'initial_r': squeezing
                }
            },
            "target_configs": [
                {'class_name': 'BinomialCodeTarget', 'params': {'N': 2, 'S': 2, 'mu': 0}}
            ],
            "beam_width": 15,
            "patterns": None
        },

        # --- FIXED PATTERN CASES ---
        {
            "name": "Fixed Pattern - 2 Mode - Squeezed Cat - Single [(4,)]",
            "circuit_config": {
                'class_name': 'TwoModeTimeDomainSqueezeOnly',
                'params': {
                    'steps': STEPS,
                    'time_invariant': False,
                    'clip_size': squeezing,
                    'measure_fock_cutoff': CUTOFF_DIM,
                    'num_single_photon': 0,
                    'train_initial_state': True,
                    'initial_r': squeezing,
                    'initial_fock_one': False
                }
            },
            "target_configs": [
                {'class_name': 'SqueezedCatTarget', 'params': {'alpha': np.sqrt(6), 'r': 0.5, 'p': 0}}
            ],
            "beam_width": 200,
            "patterns": [[(4,)]]
        },
        {
            "name": "Fixed Pattern - 2 Mode - Squeezed Cat - Multi [(4,)], [(5,)]",
            "circuit_config": {
                'class_name': 'TwoModeTimeDomainSqueezeOnly',
                'params': {
                    'steps': STEPS,
                    'time_invariant': False,
                    'clip_size': squeezing,
                    'measure_fock_cutoff': CUTOFF_DIM,
                    'num_single_photon': 0,
                    'train_initial_state': True,
                    'initial_r': squeezing,
                    'initial_fock_one': False
                }
            },
            "target_configs": [
                {'class_name': 'SqueezedCatTarget', 'params': {'alpha': np.sqrt(6), 'r': 0.5, 'p': 0}},
                {'class_name': 'SqueezedCatTarget', 'params': {'alpha': np.sqrt(6), 'r': 0.5, 'p': 1}}
            ],
            "beam_width": 200,
            "patterns": [[(4,)], [(5,)]]
        },
        {
            "name": "Fixed Pattern - 3 Mode - Binomial - Multi [(2,4)], [(4,2)]",
            "circuit_config": {
                'class_name': 'ThreeModeTimeDomainSqueezeOnly',
                'params': {
                    'steps': STEPS,
                    'time_invariant': False,
                    'clip_size': squeezing,
                    'measure_fock_cutoff': CUTOFF_DIM,
                    'num_single_photon': 0,
                    'train_initial_state': True,
                    'initial_r': squeezing
                }
            },
            "target_configs": [
                {'class_name': 'BinomialCodeTarget', 'params': {'N': 2, 'S': 2, 'mu': 0}}
            ],
            "beam_width": 200,
            "patterns": [[(2, 4)], [(4, 2)]]
        },
    ]

    if has_gkp_csv:
        test_cases.append({
            "name": "Beam Search - 3 Mode - CoreGKP Target (B=25)",
            "circuit_config": {
                'class_name': 'ThreeModeTimeDomainSqueezeOnly',
                'params': {
                    'steps': STEPS,
                    'time_invariant': False,
                    'clip_size': squeezing,
                    'measure_fock_cutoff': CUTOFF_DIM,
                    'num_single_photon': 0,
                    'train_initial_state': True,
                    'initial_r': squeezing
                }
            },
            "target_configs": [
                {'class_name': 'CoreGKPTarget', 'params': {'csv_path': csv_path_abs, 'n_max': 4, 'delta_db': 10, 'mu': 0}}
            ],
            "beam_width": 25,
            "patterns": None
        })
        test_cases.append({
            "name": "Fixed Pattern - 3 Mode - CoreGKP - Multi [(1,3)], [(3,1)]",
            "circuit_config": {
                'class_name': 'ThreeModeTimeDomainSqueezeOnly',
                'params': {
                    'steps': STEPS,
                    'time_invariant': False,
                    'clip_size': squeezing,
                    'measure_fock_cutoff': CUTOFF_DIM,
                    'num_single_photon': 0,
                    'train_initial_state': True,
                    'initial_r': squeezing
                }
            },
            "target_configs": [
                {'class_name': 'CoreGKPTarget', 'params': {'csv_path': csv_path_abs, 'n_max': 4, 'delta_db': 10, 'mu': 0}}
            ],
            "beam_width": 200,
            "patterns": [[(1, 3)], [(3, 1)]]
        })

    N_RANDOM_EVALS = 10
    total_passed = 0
    total_tests = len(test_cases)

    for case_idx, case in enumerate(test_cases):
        print(f"\n--- [Case {case_idx + 1}/{total_tests}] {case['name']} ---")

        circuit = create_from_config(case["circuit_config"], circuit_module)
        targets = [create_from_config(cfg, target_module) for cfg in case["target_configs"]]
        target_kets = [t.get_target_ket(CUTOFF_DIM) for t in targets]
        patterns = prepare_measurement_patterns(case["patterns"])

        bounds = circuit.initial_parameter_bounds + circuit.per_step_parameter_bounds * circuit.steps
        np.random.seed(100 + case_idx)

        # 1. Direct Random Parameter Sweep Evaluation
        print(f"  -> Testing {N_RANDOM_EVALS} random parameter evaluations (return_details=False & True)...")
        max_loss_diff = 0.0
        max_fid_diff = 0.0
        max_prob_diff = 0.0

        for eval_idx in range(N_RANDOM_EVALS):
            x = np.array([np.random.uniform(low, high) for low, high in bounds])

            # Test return_details=False
            loss_old = evaluate_time_domain_circuit_old(
                x, circuit, target_kets, CUTOFF_DIM, case["beam_width"], penalty_strength=1.0,
                measurement_patterns=patterns, return_details=False
            )
            loss_new = evaluate_time_domain_circuit(
                x, circuit, target_kets, CUTOFF_DIM, case["beam_width"], penalty_strength=1.0,
                measurement_patterns=patterns, return_details=False
            )
            l_diff = compare_evaluations(loss_old, loss_new)
            max_loss_diff = max(max_loss_diff, l_diff)

            # Test return_details=True
            res_old = evaluate_time_domain_circuit_old(
                x, circuit, target_kets, CUTOFF_DIM, case["beam_width"], penalty_strength=1.0,
                measurement_patterns=patterns, return_details=True
            )
            res_new = evaluate_time_domain_circuit(
                x, circuit, target_kets, CUTOFF_DIM, case["beam_width"], penalty_strength=1.0,
                measurement_patterns=patterns, return_details=True
            )
            ld, fd, pd = compare_evaluations(res_old, res_new)
            max_loss_diff = max(max_loss_diff, ld)
            max_fid_diff = max(max_fid_diff, fd)
            max_prob_diff = max(max_prob_diff, pd)

        print(f"     Max Loss Diff: {max_loss_diff:.2e} | Max Fid Diff: {max_fid_diff:.2e} | Max Prob Diff: {max_prob_diff:.2e}")

        # 2. Short Optimization Run Test via BasinHoppingRunner
        print("  -> Running short BasinHopping optimization (n_iter=2)...")
        runner = BasinHoppingRunner(
            circuit=circuit,
            target_gens=targets,
            cutoff_dim=CUTOFF_DIM,
            beam_width=case["beam_width"],
            penalty_strength=1.0,
            measurement_patterns=patterns,
            num_parallel_runs=1,
            num_processes=1
        )

        res = runner.run(n_iter=2, base_seed=42 + case_idx)
        print(f"     Optimization completed cleanly. Final Loss: {res['loss']:.6f} | Fidelity: {res['expected_fidelity']:.4f}")

        total_passed += 1
        print(f"  [PASS] Case {case_idx + 1} passed all equivalence checks.")

    print("\n" + "=" * 80)
    print(f"SUMMARY: {total_passed}/{total_tests} TEST CASES PASSED SUCCESSFULLY.")
    print("=" * 80)


if __name__ == "__main__":
    main()