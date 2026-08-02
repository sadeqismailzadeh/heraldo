"""Fixed-Pattern Optimization Sweeps for Static Spatial Circuits (Tables II and III Reproducibility).

Automates parameter optimization sweeps for static Continuous-Variable (CV) spatial photonic circuits
under pre-determined photon-number-resolving (PNR) detection pattern sequences.
This script reproduces the numerical optimization results presented in Table II
(Resource Multiplexing) and Table III (Single-Target Probability Harvesting) of the paper:
"Multi-Outcome Circuit Optimization for Enhanced Non-Gaussian State Generation"
(Ismailzadeh & Abedi Ravan, 2026).

Overview & Methodological Framework
------------------------------------
While Phase 1 (Beam Search) discovers viable measurement outcomes without prior assumptions,
Phase 2 (Fixed-Pattern Optimization, executed here) locks in specific sets of heralding
patterns S = {n_k} and optimizes classical circuit parameters theta (squeezing magnitudes/phases
and beam-splitter angles) to maximize state fidelity and generation probability.

This script directly evaluates static spatial architectures (T = 1) using the `StaticCircuit`
and `static_runner` components.

Generated Output & Artifacts
----------------------------
Results are written to: `results/sweeps_static_fid099_<Timestamp>/`

Execution
---------
Run directly via Python from the repository root:

.. code-block:: bash

    python scripts/optimize/run_fixed_pattern_sweeps_static.py
"""

import os

# --- Set thread limits for NumPy/OpenBLAS/MKL before importing libraries ---
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

import sys
import json
import pickle
import time
import warnings
import shutil
import multiprocessing as mp
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple, Any, Optional, Union
import numpy as np

warnings.filterwarnings("ignore", category=RuntimeWarning, module="scipy.optimize")

import heraldo.components.static_circuits as circuit_module
import heraldo.components.targets as target_module
from heraldo.components.static_circuits import *
from heraldo.components.static_runner import BasinHoppingRunner
from heraldo.components.objectives import fixed_pattern_capped_loss_fn
from heraldo.components.targets import *
from heraldo.utils import *
from heraldo.factory import create_from_config


def prepare_measurement_patterns(patterns: Optional[Union[List, np.ndarray]]) -> Optional[np.ndarray]:
    """Converts user-specified measurement pattern lists into a normalized 2D NumPy array.

    Args:
        patterns (list or np.ndarray, optional): List or array of measurement pattern sequences.
            Each pattern specifies photon counts for measured modes,
            e.g. ``[[(4,)], [(5,)]]`` or ``[[(1, 3)], [(3, 1)]]``.

    Returns:
        np.ndarray or None: Dense 2D NumPy array of shape ``(n_sequences, n_meas_modes)``,
        or ``None`` if ``patterns`` is None.
    """
    if patterns is None:
        return None
    if isinstance(patterns, np.ndarray):
        if patterns.ndim == 3 and patterns.shape[1] == 1:
            return patterns.squeeze(axis=1)
        if patterns.ndim == 1:
            return patterns[None, :]
        return patterns
    try:
        patterns_np = np.array(patterns, dtype=int)
        if patterns_np.ndim == 3 and patterns_np.shape[1] == 1:
            patterns_np = patterns_np.squeeze(axis=1)
        elif patterns_np.ndim == 1:
            patterns_np = patterns_np[None, :]
        return patterns_np
    except Exception:
        return patterns


def format_branches_report(branches: List[Dict[str, Any]], 
                           target_names: List[str], 
                           success_threshold: float) -> str:
    """Formats branch probabilities, state fidelities, and target assignments into a plain-text table.

    Args:
        branches (List[dict]): List of branch dictionaries containing ``"outcome"``, ``"prob"``,
            ``"fidelity"``, and ``"target_idx"``.
        target_names (List[str]): Human-readable labels for the target states.
        success_threshold (float): Minimum state fidelity threshold for success analysis.

    Returns:
        str: Formatted plain-text table summarizing branch performance.
    """
    lines = []
    lines.append("-" * 80)
    lines.append(f"{'Outcome':<20} {'Prob':<10} {'Fidelity':<10} {'1-Fid':<10} {'Best Target':<15}")
    lines.append("-" * 80)

    sorted_branches = sorted(branches, key=lambda x: x['prob'], reverse=True)
    total_prob = 0.0
    for b in sorted_branches:
        total_prob += b['prob']
        outcome_str = str(b['outcome'])
        t_idx = b.get('target_idx', 0)
        tgt_name = target_names[t_idx] if t_idx < len(target_names) else f"Target_{t_idx}"
        lines.append(f"{outcome_str:<20} {b['prob']:<10.4f} {b['fidelity']:<10.4f} {(1-b['fidelity']):<10.1e} {tgt_name:<15}")

    lines.append(f"\nTotal Probability captured: {total_prob:.5f}")

    # Target Distribution Analysis
    lines.append("-" * 60)
    lines.append(f"Target Distribution Analysis (Success > {success_threshold}):")
    lines.append(f"{'Rank':<5} {'Target Name':<20} {'Tot. Prob':<10} {'Outcomes (Top 3)'}")
    lines.append("-" * 60)

    target_stats = {}
    for b in branches:
        if b['fidelity'] > success_threshold:
            idx = b.get('target_idx', 0)
            if idx not in target_stats:
                target_stats[idx] = {'prob': 0.0, 'outcomes': []}
            target_stats[idx]['prob'] += b['prob']
            target_stats[idx]['outcomes'].append((b['outcome'], b['prob']))

    sorted_targets = sorted(target_stats.items(), key=lambda x: x[1]['prob'], reverse=True)

    if not sorted_targets:
        lines.append("No branches met the success threshold.")
    else:
        for rank, (idx, stats) in enumerate(sorted_targets):
            stats['outcomes'].sort(key=lambda x: x[1], reverse=True)
            top_outcomes = [str(o[0]) for o in stats['outcomes'][:3]]
            outcome_str = ", ".join(top_outcomes)
            if len(stats['outcomes']) > 3:
                outcome_str += ", ..."
            t_name = target_names[idx] if idx < len(target_names) else f"Target_{idx}"
            lines.append(f"{rank+1:<5} {t_name:<20} {stats['prob']:<10.4f} {outcome_str}")

    return "\n".join(lines)


def get_target_name_brief(cfg: Dict[str, Any]) -> str:
    """Formats a target configuration dictionary into a concise string label."""
    c_name = cfg.get('class_name', 'Unknown')
    p = cfg.get('params', {})
    if "CoreGKP" in c_name:
        return f"GKP_n{p.get('n_max')}_mu{p.get('mu')}"
    elif "SqueezedCat" in c_name:
        return f"SqCat_a{p.get('alpha'):.2f}_r{p.get('r'):.2f}_p{p.get('p')}"
    elif "Cat" in c_name:
        return f"Cat_a{p.get('alpha'):.2f}_p{p.get('p')}"
    elif "Binomial" in c_name:
        return f"Bin_N{p.get('N')}_S{p.get('S')}_mu{p.get('mu')}"
    return c_name


def map_branch_to_target(b: Dict[str, Any], family: str, target_configs: List[Dict[str, Any]]) -> int:
    """Identifies which target state index a given branch belongs to based on photon count rules."""
    if 'target_idx' in b and b['target_idx'] is not None:
        return b['target_idx']
    
    pat = b.get('pattern')
    if pat is None:
        pat = b.get('outcome')
    if pat is None:
        return 0
        
    pat_sum = sum(pat) if isinstance(pat, (list, tuple, np.ndarray)) else int(pat)
    
    if "GKP mu=0" in family or "GKP mu=1" in family:
        for idx, t_cfg in enumerate(target_configs):
            n_max = t_cfg.get('params', {}).get('n_max')
            if n_max is not None and pat_sum == n_max:
                return idx
                
    elif "Cat" in family:
        for idx, t_cfg in enumerate(target_configs):
            p = t_cfg.get('params', {}).get('p', 0)
            if pat_sum == (p + 4):
                return idx
                
    elif "Binomial" in family:
        for idx, t_cfg in enumerate(target_configs):
            S = t_cfg.get('params', {}).get('S', 2)
            if pat_sum == (2 * S + 2):
                return idx
                
    return 0


def get_patterns_for_target(target_idx: int, job_patterns: Optional[List], family: str, target_configs: List[Dict[str, Any]]) -> List:
    """Extracts the subset of measurement patterns associated with a specific target index."""
    matched = []
    if job_patterns is None:
        return matched
    for group in job_patterns:
        for pat in group:
            pat_item = pat[0] if isinstance(pat, (list, tuple)) and len(pat) > 0 and isinstance(pat[0], (list, tuple)) else pat
            b_mock = {'pattern': pat_item}
            idx = map_branch_to_target(b_mock, family, target_configs)
            if idx == target_idx:
                matched.append(pat_item)
    return matched


def format_target_latex(cfg: Dict[str, Any]) -> str:
    """Converts a target configuration dictionary into a LaTeX mathematical representation string."""
    c_name = cfg.get('class_name', 'Unknown')
    p = cfg.get('params', {})
    if "CoreGKP" in c_name:
        mu = p.get('mu', 0)
        n_max = p.get('n_max', 4)
        return f"$\\ket{{{mu}_{{A{n_max}}}}}$"
    elif "SqueezedCat" in c_name or "Cat" in c_name:
        p_val = p.get('p', 0)
        sign = "+" if p_val == 0 else "-"
        return f"$\\ket{{\\text{{cat}}_{{{sign}}}}}$"
    elif "Binomial" in c_name or "BinomialCode" in c_name:
        S = p.get('S', 2)
        return f"$\\ket{{0_{{S={S}}}}}$"
    return c_name


def format_patterns(pats: List) -> str:
    """Formats heralding pattern tuples into a readable string representation."""
    if not pats:
        return ""
    pats_tuples = [tuple(p) if isinstance(p, (list, tuple, np.ndarray)) else (p,) for p in pats]
    if len(pats_tuples) == 1:
        return str(pats_tuples[0])
    if len(pats_tuples) <= 4:
        return ", ".join(str(p) for p in pats_tuples)
        
    sums = [sum(p) for p in pats_tuples]
    common_sum = sums[0] if all(s == sums[0] for s in sums) else None
    if common_sum is not None:
        examples = pats_tuples[:2]
        examples_str = ", ".join(str(e) for e in examples)
        return f"$\\sum n_i = {common_sum}$ (e.g., {examples_str})"
    return ", ".join(str(p) for p in pats_tuples[:3]) + ", ..."


def main() -> None:
    """Executes fixed-pattern optimization sweeps for static circuits reproducing Tables II and III."""
    CUTOFF_DIM = 30
    BEAM_WIDTH = 200
    MEASURE_CUTOFF = CUTOFF_DIM
    SUCCESS_THRESHOLD = 0.93  # Baseline quality threshold
    N_GENERATIONS = 30

    squeezing = db_to_r(12)
    csv_path_abs = str(Path(__file__).resolve().parent.parent.parent / "data" / "GKP_core_coefficients.csv")

    print("--- Defining Static Circuit Fixed-Pattern Optimization Jobs (Tables II and III) ---")

    sweep_jobs = []

    # =========================================================================
    # TABLE II CONFIGURATIONS: RESOURCE MULTIPLEXING
    # =========================================================================

    # # 1. GKP mu=0, 3 modes, Single (1,3)
    # sweep_jobs.append({
    #     "table": "Table II",
    #     "family": "GKP mu=0",
    #     "strategy": "Single (1,3)",
    #     "modes": 3,
    #     "circuit_config": {
    #         'class_name': 'ThreeModeStaticSqueezeOnly',
    #         'params': {
    #             'clip_size': squeezing,
    #             'measure_fock_cutoff': MEASURE_CUTOFF,
    #             'num_single_photon': 0
    #         }
    #     },
    #     "target_configs": [
    #         {'class_name': 'CoreGKPTarget', 'params': {'csv_path': csv_path_abs, 'n_max': 4, 'delta_db': 10, 'mu': 0}}
    #     ],
    #     "patterns": [[(1, 3)]]
    # })

    # # 2. GKP mu=0, 3 modes, Single (2,2)
    # sweep_jobs.append({
    #     "table": "Table II",
    #     "family": "GKP mu=0",
    #     "strategy": "Single (2,2)",
    #     "modes": 3,
    #     "circuit_config": {
    #         'class_name': 'ThreeModeStaticSqueezeOnly',
    #         'params': {
    #             'clip_size': squeezing,
    #             'measure_fock_cutoff': MEASURE_CUTOFF,
    #             'num_single_photon': 0
    #         }
    #     },
    #     "target_configs": [
    #         {'class_name': 'CoreGKPTarget', 'params': {'csv_path': csv_path_abs, 'n_max': 4, 'delta_db': 10, 'mu': 0}}
    #     ],
    #     "patterns": [[(2, 2)]]
    # })

    # # 3. GKP mu=0, 3 modes, Multi (Multiplexing n_max=4, 8, 12)
    # sweep_jobs.append({
    #     "table": "Table II",
    #     "family": "GKP mu=0",
    #     "strategy": "Multi",
    #     "modes": 3,
    #     "circuit_config": {
    #         'class_name': 'ThreeModeStaticSqueezeOnly',
    #         'params': {
    #             'clip_size': squeezing,
    #             'measure_fock_cutoff': MEASURE_CUTOFF,
    #             'num_single_photon': 0
    #         }
    #     },
    #     "target_configs": [
    #         {'class_name': 'CoreGKPTarget', 'params': {'csv_path': csv_path_abs, 'n_max': 4, 'delta_db': 10, 'mu': 0}},
    #         {'class_name': 'CoreGKPTarget', 'params': {'csv_path': csv_path_abs, 'n_max': 8, 'delta_db': 10, 'mu': 0}},
    #         {'class_name': 'CoreGKPTarget', 'params': {'csv_path': csv_path_abs, 'n_max': 12, 'delta_db': 10, 'mu': 0}}
    #     ],
    #     "patterns": [[(2, 2)], [(4, 4)], [(6, 6)]]
    # })

    # 4. Cat, 2 modes, Single (+)
    sweep_jobs.append({
        "table": "Table II",
        "family": "Cat",
        "strategy": "Single (+)",
        "modes": 2,
        "circuit_config": {
            'class_name': 'TwoModeStaticSqueezeOnly',
            'params': {
                'clip_size': squeezing,
                'measure_fock_cutoff': MEASURE_CUTOFF,
                'num_single_photon': 0
            }
        },
        "target_configs": [
            {'class_name': 'SqueezedCatTarget', 'params': {'alpha': np.sqrt(6), 'r': 0.5, 'p': 0}}
        ],
        "patterns": [[(4,)]]
    })

    # 5. Cat, 2 modes, Single (-)
    sweep_jobs.append({
        "table": "Table II",
        "family": "Cat",
        "strategy": "Single (-)",
        "modes": 2,
        "circuit_config": {
            'class_name': 'TwoModeStaticSqueezeOnly',
            'params': {
                'clip_size': squeezing,
                'measure_fock_cutoff': MEASURE_CUTOFF,
                'num_single_photon': 0
            }
        },
        "target_configs": [
            {'class_name': 'SqueezedCatTarget', 'params': {'alpha': np.sqrt(6), 'r': 0.5, 'p': 1}}
        ],
        "patterns": [[(5,)]]
    })

    # 6. Cat, 2 modes, Multi (Multiplexing even n=4 and odd n=5)
    sweep_jobs.append({
        "table": "Table II",
        "family": "Cat",
        "strategy": "Multi (2 modes)",
        "modes": 2,
        "circuit_config": {
            'class_name': 'TwoModeStaticSqueezeOnly',
            'params': {
                'clip_size': squeezing,
                'measure_fock_cutoff': MEASURE_CUTOFF,
                'num_single_photon': 0
            }
        },
        "target_configs": [
            {'class_name': 'SqueezedCatTarget', 'params': {'alpha': np.sqrt(6), 'r': 0.5, 'p': 0}},
            {'class_name': 'SqueezedCatTarget', 'params': {'alpha': np.sqrt(6), 'r': 0.5, 'p': 1}}
        ],
        "patterns": [[(4,)], [(5,)]]
    })

    # # 7. Cat, 3 modes, Multi
    # sweep_jobs.append({
    #     "table": "Table II",
    #     "family": "Cat",
    #     "strategy": "Multi (3 modes)",
    #     "modes": 3,
    #     "circuit_config": {
    #         'class_name': 'ThreeModeStaticSqueezeOnly',
    #         'params': {
    #             'clip_size': squeezing,
    #             'measure_fock_cutoff': MEASURE_CUTOFF,
    #             'num_single_photon': 0
    #         }
    #     },
    #     "target_configs": [
    #         {'class_name': 'SqueezedCatTarget', 'params': {'alpha': np.sqrt(6), 'r': 0.5, 'p': 0}},
    #         {'class_name': 'SqueezedCatTarget', 'params': {'alpha': np.sqrt(6), 'r': 0.5, 'p': 1}}
    #     ],
    #     "patterns": [[(i, 4-i)] for i in range(5)] + [[(i, 5-i)] for i in range(6)]
    # })

    # 8. GKP mu=1, 2 modes, Single
    sweep_jobs.append({
        "table": "Table II",
        "family": "GKP mu=1",
        "strategy": "Single (4)",
        "modes": 2,
        "circuit_config": {
            'class_name': 'TwoModeStaticSqueezeOnly',
            'params': {
                'clip_size': squeezing,
                'measure_fock_cutoff': MEASURE_CUTOFF,
                'num_single_photon': 0
            }
        },
        "target_configs": [
            {'class_name': 'CoreGKPTarget', 'params': {'csv_path': csv_path_abs, 'n_max': 4, 'delta_db': 10, 'mu': 1}}
        ],
        "patterns": [[(4,)]]
    })

    # 9. GKP mu=1, 2 modes, Multi
    sweep_jobs.append({
        "table": "Table II",
        "family": "GKP mu=1",
        "strategy": "Multi (2 modes)",
        "modes": 2,
        "circuit_config": {
            'class_name': 'TwoModeStaticSqueezeOnly',
            'params': {
                'clip_size': squeezing,
                'measure_fock_cutoff': MEASURE_CUTOFF,
                'num_single_photon': 0
            }
        },
        "target_configs": [
            {'class_name': 'CoreGKPTarget', 'params': {'csv_path': csv_path_abs, 'n_max': 4, 'delta_db': 10, 'mu': 1}},
            {'class_name': 'CoreGKPTarget', 'params': {'csv_path': csv_path_abs, 'n_max': 6, 'delta_db': 10, 'mu': 1}},
            {'class_name': 'CoreGKPTarget', 'params': {'csv_path': csv_path_abs, 'n_max': 8, 'delta_db': 10, 'mu': 1}},
            {'class_name': 'CoreGKPTarget', 'params': {'csv_path': csv_path_abs, 'n_max': 10, 'delta_db': 10, 'mu': 1}}
        ],
        "patterns": [[(4,)], [(6,)], [(8,)], [(10,)]]
    })

    # 10. GKP mu=1, 3 modes, Single
    sweep_jobs.append({
        "table": "Table II",
        "family": "GKP mu=1",
        "strategy": "Single (4,0)",
        "modes": 3,
        "circuit_config": {
            'class_name': 'ThreeModeStaticSqueezeOnly',
            'params': {
                'clip_size': squeezing,
                'measure_fock_cutoff': MEASURE_CUTOFF,
                'num_single_photon': 0
            }
        },
        "target_configs": [
            {'class_name': 'CoreGKPTarget', 'params': {'csv_path': csv_path_abs, 'n_max': 4, 'delta_db': 10, 'mu': 1}}
        ],
        "patterns": [[(4, 0)]]
    })

    # 11. GKP mu=1, 3 modes, Multi
    gkp_mu1_3m_multi_patterns = []
    for s in [4, 6, 8, 10]:
        gkp_mu1_3m_multi_patterns.extend([[(i, s-i)] for i in range(s + 1)])

    sweep_jobs.append({
        "table": "Table II",
        "family": "GKP mu=1",
        "strategy": "Multi (3 modes)",
        "modes": 3,
        "circuit_config": {
            'class_name': 'ThreeModeStaticSqueezeOnly',
            'params': {
                'clip_size': squeezing,
                'measure_fock_cutoff': MEASURE_CUTOFF,
                'num_single_photon': 0
            }
        },
        "target_configs": [
            {'class_name': 'CoreGKPTarget', 'params': {'csv_path': csv_path_abs, 'n_max': 4, 'delta_db': 10, 'mu': 1}},
            {'class_name': 'CoreGKPTarget', 'params': {'csv_path': csv_path_abs, 'n_max': 6, 'delta_db': 10, 'mu': 1}},
            {'class_name': 'CoreGKPTarget', 'params': {'csv_path': csv_path_abs, 'n_max': 8, 'delta_db': 10, 'mu': 1}},
            {'class_name': 'CoreGKPTarget', 'params': {'csv_path': csv_path_abs, 'n_max': 10, 'delta_db': 10, 'mu': 1}}
        ],
        "patterns": gkp_mu1_3m_multi_patterns
    })

    # 12. Binomial, 3 modes, Single
    sweep_jobs.append({
        "table": "Table II",
        "family": "Binomial",
        "strategy": "Single (2,4)",
        "modes": 3,
        "circuit_config": {
            'class_name': 'ThreeModeStaticSqueezeOnly',
            'params': {
                'clip_size': squeezing,
                'measure_fock_cutoff': MEASURE_CUTOFF,
                'num_single_photon': 0
            }
        },
        "target_configs": [
            {'class_name': 'BinomialCodeTarget', 'params': {'N': 2, 'S': 2, 'mu': 0}}
        ],
        "patterns": [[(2, 4)]]
    })

    # 13. Binomial, 3 modes, Multi
    sweep_jobs.append({
        "table": "Table II",
        "family": "Binomial",
        "strategy": "Multi",
        "modes": 3,
        "circuit_config": {
            'class_name': 'ThreeModeStaticSqueezeOnly',
            'params': {
                'clip_size': squeezing,
                'measure_fock_cutoff': MEASURE_CUTOFF,
                'num_single_photon': 0
            }
        },
        "target_configs": [
            {'class_name': 'BinomialCodeTarget', 'params': {'N': 2, 'S': 2, 'mu': 0}},
            {'class_name': 'BinomialCodeTarget', 'params': {'N': 2, 'S': 3, 'mu': 0}}
        ],
        "patterns": [[(2, 4)], [(4, 2)], [(3, 5)], [(5, 3)]]
    })

    # =========================================================================
    # TABLE III CONFIGURATIONS: SINGLE-TARGET PROBABILITY HARVESTING
    # =========================================================================

    # 14. Binomial, 3 modes, Harvest (2,4), (4,2)
    sweep_jobs.append({
        "table": "Table III",
        "family": "Binomial",
        "strategy": "Harvest (2,4), (4,2)",
        "modes": 3,
        "circuit_config": {
            'class_name': 'ThreeModeStaticSqueezeOnly',
            'params': {
                'clip_size': squeezing,
                'measure_fock_cutoff': MEASURE_CUTOFF,
                'num_single_photon': 0
            }
        },
        "target_configs": [
            {'class_name': 'BinomialCodeTarget', 'params': {'N': 2, 'S': 2, 'mu': 0}}
        ],
        "patterns": [[(2, 4)], [(4, 2)]]
    })

    # 15. GKP mu=0, 3 modes, Harvesting (1,3) + (3,1)
    sweep_jobs.append({
        "table": "Table III",
        "family": "GKP mu=0",
        "strategy": "Harvesting (1,3), (3,1)",
        "modes": 3,
        "circuit_config": {
            'class_name': 'ThreeModeStaticSqueezeOnly',
            'params': {
                'clip_size': squeezing,
                'measure_fock_cutoff': MEASURE_CUTOFF,
                'num_single_photon': 0
            }
        },
        "target_configs": [
            {'class_name': 'CoreGKPTarget', 'params': {'csv_path': csv_path_abs, 'n_max': 4, 'delta_db': 10, 'mu': 0}}
        ],
        "patterns": [[(1, 3)], [(3, 1)]]
    })

    # 16. GKP mu=0, 3 modes, Harvesting (1,3) + (3,1) + (2,2)
    sweep_jobs.append({
        "table": "Table III",
        "family": "GKP mu=0",
        "strategy": "Harvesting (1,3), (3,1), (2,2)",
        "modes": 3,
        "circuit_config": {
            'class_name': 'ThreeModeStaticSqueezeOnly',
            'params': {
                'clip_size': squeezing,
                'measure_fock_cutoff': MEASURE_CUTOFF,
                'num_single_photon': 0
            }
        },
        "target_configs": [
            {'class_name': 'CoreGKPTarget', 'params': {'csv_path': csv_path_abs, 'n_max': 4, 'delta_db': 10, 'mu': 0}}
        ],
        "patterns": [[(1, 3)], [(3, 1)], [(2, 2)]]
    })

    # 17. GKP mu=1, 3 modes, Harvesting (2,2)
    sweep_jobs.append({
        "table": "Table III",
        "family": "GKP mu=1",
        "strategy": "Harvesting (2,2)",
        "modes": 3,
        "circuit_config": {
            'class_name': 'ThreeModeStaticSqueezeOnly',
            'params': {
                'clip_size': squeezing,
                'measure_fock_cutoff': MEASURE_CUTOFF,
                'num_single_photon': 0
            }
        },
        "target_configs": [
            {'class_name': 'CoreGKPTarget', 'params': {'csv_path': csv_path_abs, 'n_max': 4, 'delta_db': 10, 'mu': 1}}
        ],
        "patterns": [[(2, 2)]]
    })

    # 18. GKP mu=1, 3 modes, Harvesting 5-patterns
    sweep_jobs.append({
        "table": "Table III",
        "family": "GKP mu=1",
        "strategy": "Harvesting 5-patterns",
        "modes": 3,
        "circuit_config": {
            'class_name': 'ThreeModeStaticSqueezeOnly',
            'params': {
                'clip_size': squeezing,
                'measure_fock_cutoff': MEASURE_CUTOFF,
                'num_single_photon': 0
            }
        },
        "target_configs": [
            {'class_name': 'CoreGKPTarget', 'params': {'csv_path': csv_path_abs, 'n_max': 4, 'delta_db': 10, 'mu': 1}}
        ],
        "patterns": [[(3, 1)], [(2, 2)], [(4, 0)], [(1, 3)], [(0, 4)]]
    })

    # 19. Cat, 3 modes, Harvesting (2,2)
    sweep_jobs.append({
        "table": "Table III",
        "family": "Cat",
        "strategy": "Harvesting (2,2)",
        "modes": 3,
        "circuit_config": {
            'class_name': 'ThreeModeStaticSqueezeOnly',
            'params': {
                'clip_size': squeezing,
                'measure_fock_cutoff': MEASURE_CUTOFF,
                'num_single_photon': 0
            }
        },
        "target_configs": [
            {'class_name': 'SqueezedCatTarget', 'params': {'alpha': np.sqrt(6), 'r': 0.5, 'p': 0}}
        ],
        "patterns": [[(2, 2)]]
    })

    # 20. Cat, 3 modes, Harvesting 5-patterns
    sweep_jobs.append({
        "table": "Table III",
        "family": "Cat",
        "strategy": "Harvesting 5-patterns",
        "modes": 3,
        "circuit_config": {
            'class_name': 'ThreeModeStaticSqueezeOnly',
            'params': {
                'clip_size': squeezing,
                'measure_fock_cutoff': MEASURE_CUTOFF,
                'num_single_photon': 0
            }
        },
        "target_configs": [
            {'class_name': 'SqueezedCatTarget', 'params': {'alpha': np.sqrt(6), 'r': 0.5, 'p': 0}}
        ],
        "patterns": [[(2, 2)], [(1, 3)], [(3, 1)], [(0, 4)], [(4, 0)]]
    })

    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    results_dir = Path(__file__).resolve().parent.parent.parent / "results" / f"sweeps_static_fid099_{timestamp}"
    results_dir.mkdir(parents=True, exist_ok=True)
    print(f"Sweep results will be written to: {results_dir}\n")

    num_procs = min(4, os.cpu_count() or 1)
    results_summary = []

    for idx, job in enumerate(sweep_jobs):
        print("=" * 80)
        print(f"JOB {idx+1}/{len(sweep_jobs)}: {job['table']} | {job['family']} | {job['strategy']}")
        print("=" * 80)

        safe_strategy_name = "".join([c if c.isalnum() else "_" for c in job['strategy']])
        job_dir = results_dir / f"job_{idx+1:02d}_{job['family'].replace(' ', '_')}_{safe_strategy_name}"
        job_dir.mkdir(exist_ok=True)

        try:
            circuit = create_from_config(job['circuit_config'], circuit_module)
            targets = [create_from_config(cfg, target_module) for cfg in job['target_configs']]
            patterns = prepare_measurement_patterns(job['patterns'])

            runner = BasinHoppingRunner(
                num_processes=num_procs,
                circuit=circuit,
                target_gens=targets,
                cutoff_dim=CUTOFF_DIM,
                beam_width=BEAM_WIDTH,
                penalty_strength=1.0,
                measurement_patterns=patterns,
                loss_fn=fixed_pattern_capped_loss_fn
            )

            start_time = time.time()
            res = runner.run(n_iter=N_GENERATIONS, method="L-BFGS-B")
            duration = time.time() - start_time

            branches = res.get('branches', [])
            
            target_results = []
            for k, tgt_cfg in enumerate(job['target_configs']):
                tgt_branches = [b for b in branches if map_branch_to_target(b, job['family'], job['target_configs']) == k]
                tgt_pats = get_patterns_for_target(k, job['patterns'], job['family'], job['target_configs'])
                n_pat_total = len(tgt_pats)
                
                if len(tgt_branches) > 0:
                    tgt_fidelities = [b['fidelity'] for b in tgt_branches]
                    tgt_min_fid = min(tgt_fidelities)
                    tgt_max_infid = 1.0 - tgt_min_fid
                    tgt_agg_prob = sum(b['prob'] for b in tgt_branches if b['fidelity'] >= SUCCESS_THRESHOLD)
                    tgt_n_pat_success = sum(1 for b in tgt_branches if b['fidelity'] >= SUCCESS_THRESHOLD)
                else:
                    tgt_max_infid = float('nan')
                    tgt_agg_prob = 0.0
                    tgt_n_pat_success = 0
                
                target_results.append({
                    "target_idx": k,
                    "target_label": format_target_latex(tgt_cfg),
                    "patterns_str": format_patterns(tgt_pats),
                    "n_pat_total": n_pat_total,
                    "n_pat_success": tgt_n_pat_success,
                    "max_infid": tgt_max_infid,
                    "agg_prob": tgt_agg_prob
                })

            if len(branches) > 0:
                fidelities = [b['fidelity'] for b in branches]
                min_fid = min(fidelities)
                max_infid = 1.0 - min_fid
                agg_prob = sum(b['prob'] for b in branches if b['fidelity'] >= SUCCESS_THRESHOLD)
                n_pat_success = sum(1 for b in branches if b['fidelity'] >= SUCCESS_THRESHOLD)
            else:
                max_infid = float('nan')
                agg_prob = 0.0
                n_pat_success = 0

            target_labels = ", ".join([get_target_name_brief(cfg) for cfg in job['target_configs']])

            job_result = {
                "idx": idx + 1,
                "table": job["table"],
                "family": job["family"],
                "strategy": job["strategy"],
                "modes": job["modes"],
                "targets": target_labels,
                "targets_detailed": target_results,
                "n_pat_total": sum(t['n_pat_total'] for t in target_results),
                "n_pat_success": n_pat_success,
                "max_infid": max_infid,
                "agg_prob": agg_prob,
                "duration": duration,
                "status": "Success"
            }

            print(f"-> Job {idx+1} Completed in {duration:.1f}s.")
            for tgt in target_results:
                t_infid_str = f"{tgt['max_infid']:.2e}" if not np.isnan(tgt['max_infid']) else "N/A"
                print(f"   * Target {tgt['target_label']} with patterns {tgt['patterns_str']}:")
                print(f"     Worst-case Infidelity: {t_infid_str}")
                print(f"     Success Prob: {tgt['agg_prob']:.2%}")
            if len(target_results) > 1:
                print(f"   * Total Aggregated Success Prob (P_agg @ >=0.93): {agg_prob:.2%}")

            target_names = []
            for t in targets:
                if isinstance(t, CoreGKPTarget):
                    target_names.append(f"GKP_n{t.n_max}_mu{t.mu}")
                elif isinstance(t, SqueezedCatTarget):
                    target_names.append(f"Sq_cat_a{t.alpha}_r{t.r}_p{t.p}")
                elif isinstance(t, CatTarget):
                    target_names.append(f"Cat_a{t.alpha}_p{t.p}")
                elif isinstance(t, BinomialCodeTarget):
                    target_names.append(f"Binomial_N{t.N}_S{t.S}_mu{t.mu}")
                elif isinstance(t, CubicPhaseTarget):
                    target_names.append(f"CubicPh_g{t.gamma}_r{t.r}")
                else:
                    target_names.append("UnknownTarget")

            branches.sort(key=lambda x: x['prob'], reverse=True)
            expected_fidelity = sum(b['prob'] * b['fidelity'] for b in branches)
            
            res['expected_fidelity'] = expected_fidelity
            res['success_prob'] = agg_prob
            res['run_index'] = 1
            
            run_meta = {
                "run_index": 1,
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "prob_power": 1.0,
                "expected_fidelity": float(expected_fidelity),
                "success_prob": float(agg_prob),
                "circuit_config": job['circuit_config'],
                "target_configs": job['target_configs'],
                "measurement_patterns": job['patterns']
            }

            run_file = job_dir / "run_0001.pkl"
            with open(run_file, "wb") as f:
                pickle.dump({"meta": run_meta, "res": res}, f)

            summary = {
                "run_index": 1,
                "expected_fidelity": float(expected_fidelity),
                "success_prob": float(agg_prob),
                "total_probability": float(res.get("total_probability", 0.0))
            }
            with open(job_dir / "run_0001_summary.json", "w") as f:
                json.dump(summary, f, indent=2)

            branches_report = format_branches_report(branches, target_names, SUCCESS_THRESHOLD)
            with open(job_dir / "run_0001_branches.txt", "w") as f:
                f.write(branches_report)

            with open(job_dir / "results.pkl", "wb") as f:
                pickle.dump({"meta": run_meta, "res": res}, f)

            with open(job_dir / "summary.json", "w") as f:
                json.dump(job_result, f, indent=2)

            try:
                best_dir = job_dir / "best"
                best_dir.mkdir(exist_ok=True)

                shutil.copy(run_file, best_dir / "best_run_0001.pkl")
                with open(best_dir / "best_run_0001_branches.txt", "w") as f:
                    f.write(branches_report)
                with open(best_dir / "best_run_0001_summary.json", "w") as f:
                    json.dump(summary, f, indent=2)

                if 'x' in res:
                    np.save(best_dir / "best_x.npy", res['x'])
                    param_names = circuit.parameter_names
                    
                    schedule = {
                        "param_names": param_names,
                        "params": res['x'].tolist() if hasattr(res['x'], "tolist") else [float(v) for v in res['x']],
                        "circuit_config": job['circuit_config'],
                        "target_configs": job['target_configs']
                    }
                    with open(best_dir / "schedule.json", "w") as f:
                        json.dump(schedule, f, indent=2)
            except Exception as e_best:
                print(f"Warning: Failed to save best directory outputs: {e_best}")

        except Exception as e:
            print(f"Error executing job {idx+1}: {e}")
            job_result = {
                "idx": idx + 1,
                "table": job["table"],
                "family": job["family"],
                "strategy": job["strategy"],
                "modes": job["modes"],
                "targets": "Error",
                "n_pat_total": len(job["patterns"]),
                "n_pat_success": 0,
                "max_infid": float('nan'),
                "agg_prob": 0.0,
                "duration": 0.0,
                "status": f"Failed: {str(e)}"
            }

        results_summary.append(job_result)

    # =========================================================================
    # COMPILE MASTER MARKDOWN REPORT FOR TABLES II AND III
    # =========================================================================
    print("\n" + "=" * 80)
    print("ALL STATIC SWEEPS COMPLETE - COMPILING REPORTS FOR TABLES II AND III")
    print("=" * 80)

    report_path = results_dir / "sweep_report.md"
    with open(report_path, "w") as f:
        f.write("# Fixed-Pattern Optimization Sweep Report (Static Circuits, Tables II and III)\n\n")
        f.write(f"Generated on: {datetime.utcnow().isoformat()}Z\n\n")

        # Table II Section: Resource Multiplexing
        f.write("## Table II: Resource Multiplexing Summary\n\n")
        f.write("| Family | Modes | Strategy | Target | Patterns | $N_{\\mathrm{pat}}$ | Max Infidelity ($1-\\mathcal{F}_{\\mathrm{min}}$) | Success Prob ($P_{\\mathrm{agg}}$) | Status |\n")
        f.write("| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |\n")
        
        for r in results_summary:
            if r["table"] == "Table II":
                detailed = r.get("targets_detailed", [])
                if not detailed:
                    f.write(f"| {r['family']} | {r['modes']} | {r['strategy']} | Error | - | - | - | - | {r['status']} |\n")
                    continue
                
                is_multi = len(detailed) > 1
                for i, tgt in enumerate(detailed):
                    fam_str = r['family'] if i == 0 else ""
                    strat_str = r['strategy'] if i == 0 else ""
                    modes_str = str(r['modes']) if i == 0 else ""
                    
                    infid_str = f"{tgt['max_infid']:.2e}" if not np.isnan(tgt['max_infid']) else "N/A"
                    prob_str = f"{tgt['agg_prob']:.2%}"
                    f.write(f"| {fam_str} | {modes_str} | {strat_str} | {tgt['target_label']} | {tgt['patterns_str']} | {tgt['n_pat_total']} | {infid_str} | {prob_str} | {r['status'] if i == 0 else ''} |\n")
                
                if is_multi:
                    prob_total_str = f"{r['agg_prob']:.2%}"
                    f.write(f"| | | | *Total* | | {r['n_pat_total']} | -- | {prob_total_str} | |\n")
                
                f.write("| | | | | | | | | |\n")

        f.write("\n\n")

        # Table III Section: Single-Target Probability Harvesting
        f.write("## Table III: Single-Target Probability Harvesting Summary\n\n")
        f.write("| Target | Modes | Strategy/Patterns | Max Infidelity ($1-\\mathcal{F}_{\\mathrm{min}}$) | Total Success Prob ($P_{\\mathrm{total}}$) | Status |\n")
        f.write("| :--- | :--- | :--- | :--- | :--- | :--- |\n")
        for r in results_summary:
            if r["table"] == "Table III":
                detailed = r.get("targets_detailed", [])
                if detailed:
                    tgt = detailed[0]
                    infid_str = f"{tgt['max_infid']:.2e}" if not np.isnan(tgt['max_infid']) else "N/A"
                    prob_str = f"{r['agg_prob']:.2%}"
                    f.write(f"| {tgt['target_label']} | {r['modes']} | {r['strategy']} | {infid_str} | {prob_str} | {r['status']} |\n")
                else:
                    f.write(f"| {r['targets']} | {r['modes']} | {r['strategy']} | N/A | 0.0% | {r['status']} |\n")

    with open(report_path, "r") as f:
        print(f.read())

    print(f"\nAll details and summaries saved in: {results_dir}")


if __name__ == "__main__":
    main()
