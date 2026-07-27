"""
Test script to verify exact equivalence between evaluate_time_domain_circuit_old
and evaluate_time_domain_circuit across multiple circuit architectures, target states,
measurement modes (Beam Search and Fixed Patterns), and parameter vectors.
"""

import sys
from pathlib import Path
import numpy as np
import pytest

# Ensure project root is on Python path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import quantum_agent.optimization.time_circuits as circuit_module
import quantum_agent.components.targets as target_module
from quantum_agent.factory import create_from_config
from quantum_agent.optimization.time_runner import (
    evaluate_time_domain_circuit,
    evaluate_time_domain_circuit_old,
)
from quantum_agent.utils import db_to_r


def sample_random_flat_params(circuit, seed=42):
    """Generates a valid flat_params array within the parameter bounds of the given circuit."""
    rng = np.random.default_rng(seed)
    bounds = circuit.initial_parameter_bounds
    if circuit.time_invariant:
        bounds = bounds + circuit.per_step_parameter_bounds
    else:
        bounds = bounds + (circuit.per_step_parameter_bounds * circuit.steps)
    
    flat_params = np.array([rng.uniform(low, high) for low, high in bounds])
    return flat_params


def compare_branch_details(branches_old, branches_new, rtol=1e-5, atol=1e-7):
    """Asserts equivalence between two lists of branch dictionaries."""
    assert len(branches_old) == len(branches_new), (
        f"Branch count mismatch: {len(branches_old)} (old) vs {len(branches_new)} (new)"
    )

    # Sort branches deterministically by outcome tuple and target_idx
    sorted_old = sorted(branches_old, key=lambda b: (b["outcome"], b["target_idx"]))
    sorted_new = sorted(branches_new, key=lambda b: (b["outcome"], b["target_idx"]))

    for idx, (b_old, b_new) in enumerate(zip(sorted_old, sorted_new)):
        assert b_old["outcome"] == b_new["outcome"], (
            f"Branch {idx} outcome mismatch: {b_old['outcome']} vs {b_new['outcome']}"
        )
        assert b_old["target_idx"] == b_new["target_idx"], (
            f"Branch {idx} target_idx mismatch: {b_old['target_idx']} vs {b_new['target_idx']}"
        )
        np.testing.assert_allclose(
            b_old["prob"], b_new["prob"], rtol=rtol, atol=atol,
            err_msg=f"Branch {idx} prob mismatch for outcome {b_old['outcome']}"
        )
        np.testing.assert_allclose(
            b_old["fidelity"], b_new["fidelity"], rtol=rtol, atol=atol,
            err_msg=f"Branch {idx} fidelity mismatch for outcome {b_old['outcome']}"
        )


def compare_evaluations(res_old, res_new, rtol=1e-5, atol=1e-7):
    """Asserts equivalence between old and new evaluation outputs."""
    if isinstance(res_old, (float, int, np.number)):
        assert isinstance(res_new, (float, int, np.number)), (
            f"Type mismatch: res_old is {type(res_old)}, res_new is {type(res_new)}"
        )
        np.testing.assert_allclose(
            res_old, res_new, rtol=rtol, atol=atol, err_msg="Loss scalar mismatch"
        )
    else:
        assert isinstance(res_old, dict) and isinstance(res_new, dict), (
            f"Type mismatch: {type(res_old)} vs {type(res_new)}"
        )
        np.testing.assert_allclose(
            res_old["loss"], res_new["loss"], rtol=rtol, atol=atol,
            err_msg="Dict loss mismatch"
        )
        np.testing.assert_allclose(
            res_old["expected_fidelity"], res_new["expected_fidelity"], rtol=rtol, atol=atol,
            err_msg="Expected fidelity mismatch"
        )
        np.testing.assert_allclose(
            res_old["total_probability"], res_new["total_probability"], rtol=rtol, atol=atol,
            err_msg="Total probability mismatch"
        )
        compare_branch_details(res_old["branches"], res_new["branches"], rtol=rtol, atol=atol)


# Helper target setup functions
def get_cubic_phase_target():
    cfg = {'class_name': 'CubicPhaseTarget', 'params': {'gamma': -0.2, 'r': -0.7, 'alpha': 1.25}}
    return [create_from_config(cfg, target_module)]

def get_cat_target():
    cfg = {'class_name': 'CatTarget', 'params': {'alpha': 1.5, 'p': 0}}
    return [create_from_config(cfg, target_module)]

def get_multi_sq_cat_targets():
    cfg_even = {'class_name': 'SqueezedCatTarget', 'params': {'alpha': 2.0, 'r': 0.5, 'p': 0}}
    cfg_odd = {'class_name': 'SqueezedCatTarget', 'params': {'alpha': 2.0, 'r': 0.5, 'p': 1}}
    return [create_from_config(cfg_even, target_module), create_from_config(cfg_odd, target_module)]

def get_binomial_target():
    cfg = {'class_name': 'BinomialCodeTarget', 'params': {'N': 2, 'S': 2, 'mu': 0}}
    return [create_from_config(cfg, target_module)]


# =============================================================================
# Pytest Test Functions
# =============================================================================

def test_3mode_gadget_beam_search():
    """Test 3-mode Gadget circuit using Beam Search."""
    squeezing = db_to_r(10)
    cutoff_dim = 20
    circuit_cfg = {
        'class_name': 'ThreeModeTimeDomainGadget',
        'params': {
            'steps': 1,
            'time_invariant': False,
            'clip_size': squeezing,
            'measure_fock_cutoff': cutoff_dim,
            'train_initial_state': True,
            'initial_r': squeezing
        }
    }
    circuit = create_from_config(circuit_cfg, circuit_module)
    targets = get_cubic_phase_target()
    target_kets = [t.get_target_ket(cutoff_dim) for t in targets]

    for seed in [1, 42, 100]:
        flat_params = sample_random_flat_params(circuit, seed=seed)
        for return_details in [False, True]:
            res_old = evaluate_time_domain_circuit_old(
                flat_params, circuit, target_kets, cutoff_dim,
                beam_width=10, penalty_strength=1.0,
                measurement_patterns=None, return_details=return_details
            )
            res_new = evaluate_time_domain_circuit(
                flat_params, circuit, target_kets, cutoff_dim,
                beam_width=10, penalty_strength=1.0,
                measurement_patterns=None, return_details=return_details
            )
            compare_evaluations(res_old, res_new)


def test_3mode_gadget_fixed_patterns():
    """Test 3-mode Gadget circuit using Fixed Measurement Patterns."""
    squeezing = db_to_r(10)
    cutoff_dim = 20
    circuit_cfg = {
        'class_name': 'ThreeModeTimeDomainGadget',
        'params': {
            'steps': 1,
            'time_invariant': False,
            'clip_size': squeezing,
            'measure_fock_cutoff': cutoff_dim,
            'train_initial_state': True,
            'initial_r': squeezing
        }
    }
    circuit = create_from_config(circuit_cfg, circuit_module)
    targets = get_cubic_phase_target()
    target_kets = [t.get_target_ket(cutoff_dim) for t in targets]

    patterns = np.array([
        [(0, 6)],
        [(2, 6)],
        [(4, 6)],
        [(8, 6)]
    ], dtype=int)  # shape: (4, 1, 2)

    for seed in [1, 42, 100]:
        flat_params = sample_random_flat_params(circuit, seed=seed)
        for return_details in [False, True]:
            res_old = evaluate_time_domain_circuit_old(
                flat_params, circuit, target_kets, cutoff_dim,
                beam_width=5, penalty_strength=1.0,
                measurement_patterns=patterns, return_details=return_details
            )
            res_new = evaluate_time_domain_circuit(
                flat_params, circuit, target_kets, cutoff_dim,
                beam_width=5, penalty_strength=1.0,
                measurement_patterns=patterns, return_details=return_details
            )
            compare_evaluations(res_old, res_new)


def test_2mode_squeeze_only_beam_search_and_fixed():
    """Test 2-mode Squeeze-Only circuit with Cat target state."""
    squeezing = db_to_r(8)
    cutoff_dim = 20
    circuit_cfg = {
        'class_name': 'TwoModeTimeDomainSqueezeOnly',
        'params': {
            'steps': 1,
            'time_invariant': False,
            'clip_size': squeezing,
            'measure_fock_cutoff': cutoff_dim,
            'num_single_photon': 0,
            'train_initial_state': True,
            'initial_r': squeezing,
            'initial_fock_one': False
        }
    }
    circuit = create_from_config(circuit_cfg, circuit_module)
    targets = get_cat_target()
    target_kets = [t.get_target_ket(cutoff_dim) for t in targets]

    flat_params = sample_random_flat_params(circuit, seed=77)

    # 1. Beam Search Mode
    res_old_beam = evaluate_time_domain_circuit_old(
        flat_params, circuit, target_kets, cutoff_dim,
        beam_width=15, penalty_strength=2.0, return_details=True
    )
    res_new_beam = evaluate_time_domain_circuit(
        flat_params, circuit, target_kets, cutoff_dim,
        beam_width=15, penalty_strength=2.0, return_details=True
    )
    compare_evaluations(res_old_beam, res_new_beam)

    # 2. Fixed Pattern Mode (2-mode circuit -> 1 measured mode per step)
    patterns = np.array([
        [(2,)],
        [(4,)],
        [(6,)]
    ], dtype=int)  # shape (3, 1, 1)

    res_old_fixed = evaluate_time_domain_circuit_old(
        flat_params, circuit, target_kets, cutoff_dim,
        beam_width=15, penalty_strength=2.0, measurement_patterns=patterns, return_details=True
    )
    res_new_fixed = evaluate_time_domain_circuit(
        flat_params, circuit, target_kets, cutoff_dim,
        beam_width=15, penalty_strength=2.0, measurement_patterns=patterns, return_details=True
    )
    compare_evaluations(res_old_fixed, res_new_fixed)


def test_3mode_squeeze_only_multi_target():
    """Test 3-mode Squeeze-Only circuit with multiple targets (Even & Odd Squeezed Cats)."""
    squeezing = db_to_r(10)
    cutoff_dim = 22
    circuit_cfg = {
        'class_name': 'ThreeModeTimeDomainSqueezeOnly',
        'params': {
            'steps': 1,
            'time_invariant': False,
            'clip_size': squeezing,
            'measure_fock_cutoff': cutoff_dim,
            'num_single_photon': 0,
            'train_initial_state': True,
            'initial_r': squeezing
        }
    }
    circuit = create_from_config(circuit_cfg, circuit_module)
    targets = get_multi_sq_cat_targets()
    target_kets = [t.get_target_ket(cutoff_dim) for t in targets]

    flat_params = sample_random_flat_params(circuit, seed=99)

    res_old = evaluate_time_domain_circuit_old(
        flat_params, circuit, target_kets, cutoff_dim,
        beam_width=25, penalty_strength=1.5, return_details=True
    )
    res_new = evaluate_time_domain_circuit(
        flat_params, circuit, target_kets, cutoff_dim,
        beam_width=25, penalty_strength=1.5, return_details=True
    )
    compare_evaluations(res_old, res_new)


def test_multistep_circuit_binomial_target():
    """Test multi-step (steps=2) 3-mode Gadget circuit with Binomial Target."""
    squeezing = db_to_r(10)
    cutoff_dim = 20
    circuit_cfg = {
        'class_name': 'ThreeModeTimeDomainGadget',
        'params': {
            'steps': 2,
            'time_invariant': False,
            'clip_size': squeezing,
            'measure_fock_cutoff': cutoff_dim,
            'train_initial_state': True,
            'initial_r': squeezing
        }
    }
    circuit = create_from_config(circuit_cfg, circuit_module)
    targets = get_binomial_target()
    target_kets = [t.get_target_ket(cutoff_dim) for t in targets]

    flat_params = sample_random_flat_params(circuit, seed=123)

    # 1. Beam Search (steps=2)
    res_old_beam = evaluate_time_domain_circuit_old(
        flat_params, circuit, target_kets, cutoff_dim,
        beam_width=10, penalty_strength=1.0, return_details=True
    )
    res_new_beam = evaluate_time_domain_circuit(
        flat_params, circuit, target_kets, cutoff_dim,
        beam_width=10, penalty_strength=1.0, return_details=True
    )
    compare_evaluations(res_old_beam, res_new_beam)

    # 2. Fixed Patterns (2 steps, 2 measured modes per step -> shape: (3, 2, 2))
    patterns = np.array([
        [[(0, 2), (1, 2)]],
        [[(2, 2), (0, 0)]],
        [[(0, 0), (2, 2)]]
    ], dtype=int).reshape(3, 2, 2)

    res_old_fixed = evaluate_time_domain_circuit_old(
        flat_params, circuit, target_kets, cutoff_dim,
        beam_width=10, penalty_strength=1.0, measurement_patterns=patterns, return_details=True
    )
    res_new_fixed = evaluate_time_domain_circuit(
        flat_params, circuit, target_kets, cutoff_dim,
        beam_width=10, penalty_strength=1.0, measurement_patterns=patterns, return_details=True
    )
    compare_evaluations(res_old_fixed, res_new_fixed)


def test_random_parameter_sweep():
    """Stress test equivalence over multiple random seeds and parameter settings."""
    squeezing = db_to_r(12)
    cutoff_dim = 20
    circuit_cfg = {
        'class_name': 'ThreeModeTimeDomainGadget',
        'params': {
            'steps': 1,
            'time_invariant': False,
            'clip_size': squeezing,
            'measure_fock_cutoff': cutoff_dim,
            'train_initial_state': True,
            'initial_r': squeezing
        }
    }
    circuit = create_from_config(circuit_cfg, circuit_module)
    targets = get_cubic_phase_target()
    target_kets = [t.get_target_ket(cutoff_dim) for t in targets]

    for seed in range(10):
        flat_params = sample_random_flat_params(circuit, seed=1000 + seed)
        res_old = evaluate_time_domain_circuit_old(
            flat_params, circuit, target_kets, cutoff_dim,
            beam_width=20, penalty_strength=1.0, return_details=True
        )
        res_new = evaluate_time_domain_circuit(
            flat_params, circuit, target_kets, cutoff_dim,
            beam_width=20, penalty_strength=1.0, return_details=True
        )
        compare_evaluations(res_old, res_new)


# Direct execution entry point
if __name__ == "__main__":
    print("=" * 80)
    print(" RUNNING EQUIVALENCE TESTS FOR EVALUATE_TIME_DOMAIN_CIRCUIT ")
    print("=" * 80)

    tests = [
        ("3-Mode Gadget Beam Search", test_3mode_gadget_beam_search),
        ("3-Mode Gadget Fixed Patterns", test_3mode_gadget_fixed_patterns),
        ("2-Mode Squeeze-Only Beam Search & Fixed", test_2mode_squeeze_only_beam_search_and_fixed),
        ("3-Mode Squeeze-Only Multi-Target", test_3mode_squeeze_only_multi_target),
        ("Multi-step (steps=2) Circuit Binomial Target", test_multistep_circuit_binomial_target),
        ("Random Parameter Sweep (10 Seeds)", test_random_parameter_sweep),
    ]

    passed = 0
    failed = 0

    for name, test_func in tests:
        print(f"\nRunning test: {name}...")
        try:
            test_func()
            print(f"  --> PASSED [✓]")
            passed += 1
        except Exception as e:
            print(f"  --> FAILED [✗]: {e}")
            failed += 1

    print("\n" + "=" * 80)
    print(f" SUMMARY: {passed} Passed, {failed} Failed")
    print("=" * 80)

    if failed > 0:
        sys.exit(1)
