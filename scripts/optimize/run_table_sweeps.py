"""
Script to automate the optimization sweeps for Table 1 and Table 2 configurations
at a fixed success fidelity threshold of 0.99 (as requested by referees).
"""

import os
import sys
import json
import pickle
import time
import warnings
import multiprocessing as mp
from pathlib import Path
from datetime import datetime
import numpy as np

# --- Set thread limits for NumPy/OpenBLAS/MKL before importing libraries ---
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

warnings.filterwarnings("ignore", category=RuntimeWarning, module="scipy.optimize")

import quantum_agent.optimization.time_circuits as circuit_module
import quantum_agent.components.targets as target_module
from quantum_agent.optimization.time_circuits import *
from quantum_agent.optimization.time_runner import BasinHoppingRunner
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


def get_target_name_brief(cfg):
    """Formats config to readable target label."""
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


def safe_db_to_r(db_val):
    """Converts dB to squeezing parameter r safely."""
    try:
        return db_to_r(db_val)
    except NameError:
        return 0.11512925464970229 * db_val


def main():
    CUTOFF_DIM = 30
    STEPS = 1
    BEAM_WIDTH = 200
    TIME_INVARIANT = False
    MEASURE_CUTOFF = CUTOFF_DIM
    SUCCESS_THRESHOLD = 0.99  # Fixed at 0.99 per referee request
    N_GENERATIONS = 20
    NITER = 1

    squeezing = db_to_r(12)
    csv_path_abs = str(Path(__file__).resolve().parent.parent.parent / "data" / "GKP_core_coefficients.csv")

    print("--- Defining Sweep Jobs for LaTeX Tables ---")

    sweep_jobs = []

    # ==========================================
    # TABLE 1 CONFIGURATIONS
    # ==========================================

    # 1. GKP mu=0, 3 modes, Single (1,3)
    sweep_jobs.append({
        "table": "Table 1",
        "family": "GKP mu=0",
        "strategy": "Single (1,3)",
        "modes": 3,
        "circuit_config": {
            'class_name': 'ThreeModeTimeDomainSqueezeOnly',
            'params': {
                'steps': STEPS,
                'time_invariant': TIME_INVARIANT,
                'clip_size': squeezing,
                'measure_fock_cutoff': MEASURE_CUTOFF,
                'num_single_photon': 0,
                'train_initial_state': True,
                'initial_r': squeezing
            }
        },
        "target_configs": [
            {'class_name': 'CoreGKPTarget', 'params': {'csv_path': csv_path_abs, 'n_max': 4, 'delta_db': 10, 'mu': 0}}
        ],
        "patterns": [[(1, 3)]]
    })

    # 2. GKP mu=0, 3 modes, Single (2,2)
    sweep_jobs.append({
        "table": "Table 1",
        "family": "GKP mu=0",
        "strategy": "Single (2,2)",
        "modes": 3,
        "circuit_config": {
            'class_name': 'ThreeModeTimeDomainSqueezeOnly',
            'params': {
                'steps': STEPS,
                'time_invariant': TIME_INVARIANT,
                'clip_size': squeezing,
                'measure_fock_cutoff': MEASURE_CUTOFF,
                'num_single_photon': 0,
                'train_initial_state': True,
                'initial_r': squeezing
            }
        },
        "target_configs": [
            {'class_name': 'CoreGKPTarget', 'params': {'csv_path': csv_path_abs, 'n_max': 4, 'delta_db': 10, 'mu': 0}}
        ],
        "patterns": [[(2, 2)]]
    })

    # 3. GKP mu=0, 3 modes, Multi
    sweep_jobs.append({
        "table": "Table 1",
        "family": "GKP mu=0",
        "strategy": "Multi",
        "modes": 3,
        "circuit_config": {
            'class_name': 'ThreeModeTimeDomainSqueezeOnly',
            'params': {
                'steps': STEPS,
                'time_invariant': TIME_INVARIANT,
                'clip_size': squeezing,
                'measure_fock_cutoff': MEASURE_CUTOFF,
                'num_single_photon': 0,
                'train_initial_state': True,
                'initial_r': squeezing
            }
        },
        "target_configs": [
            {'class_name': 'CoreGKPTarget', 'params': {'csv_path': csv_path_abs, 'n_max': 4, 'delta_db': 10, 'mu': 0}},
            {'class_name': 'CoreGKPTarget', 'params': {'csv_path': csv_path_abs, 'n_max': 8, 'delta_db': 10, 'mu': 0}},
            {'class_name': 'CoreGKPTarget', 'params': {'csv_path': csv_path_abs, 'n_max': 12, 'delta_db': 10, 'mu': 0}}
        ],
        "patterns": [[(2, 2)], [(4, 4)], [(6, 6)]]
    })

    # 4. Cat, 2 modes, Single (+)
    sweep_jobs.append({
        "table": "Table 1",
        "family": "Cat",
        "strategy": "Single (+)",
        "modes": 2,
        "circuit_config": {
            'class_name': 'TwoModeTimeDomainSqueezeOnly',
            'params': {
                'steps': STEPS,
                'time_invariant': TIME_INVARIANT,
                'clip_size': squeezing,
                'measure_fock_cutoff': MEASURE_CUTOFF,
                'num_single_photon': 0,
                'train_initial_state': True,
                'initial_r': squeezing,
                'initial_fock_one': False
            }
        },
        "target_configs": [
            {'class_name': 'SqueezedCatTarget', 'params': {'alpha': np.sqrt(6), 'r': 0.5, 'p': 0}}
        ],
        "patterns": [[(4,)]]
    })

    # 5. Cat, 2 modes, Single (-)
    sweep_jobs.append({
        "table": "Table 1",
        "family": "Cat",
        "strategy": "Single (-)",
        "modes": 2,
        "circuit_config": {
            'class_name': 'TwoModeTimeDomainSqueezeOnly',
            'params': {
                'steps': STEPS,
                'time_invariant': TIME_INVARIANT,
                'clip_size': squeezing,
                'measure_fock_cutoff': MEASURE_CUTOFF,
                'num_single_photon': 0,
                'train_initial_state': True,
                'initial_r': squeezing,
                'initial_fock_one': False
            }
        },
        "target_configs": [
            {'class_name': 'SqueezedCatTarget', 'params': {'alpha': np.sqrt(6), 'r': 0.5, 'p': 1}}
        ],
        "patterns": [[(5,)]]
    })

    # 6. Cat, 2 modes, Multi
    sweep_jobs.append({
        "table": "Table 1",
        "family": "Cat",
        "strategy": "Multi (2 modes)",
        "modes": 2,
        "circuit_config": {
            'class_name': 'TwoModeTimeDomainSqueezeOnly',
            'params': {
                'steps': STEPS,
                'time_invariant': TIME_INVARIANT,
                'clip_size': squeezing,
                'measure_fock_cutoff': MEASURE_CUTOFF,
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
        "patterns": [[(4,)], [(5,)]]
    })

    # 7. Cat, 3 modes, Multi
    sweep_jobs.append({
        "table": "Table 1",
        "family": "Cat",
        "strategy": "Multi (3 modes)",
        "modes": 3,
        "circuit_config": {
            'class_name': 'ThreeModeTimeDomainSqueezeOnly',
            'params': {
                'steps': STEPS,
                'time_invariant': TIME_INVARIANT,
                'clip_size': squeezing,
                'measure_fock_cutoff': MEASURE_CUTOFF,
                'num_single_photon': 0,
                'train_initial_state': True,
                'initial_r': squeezing
            }
        },
        "target_configs": [
            {'class_name': 'SqueezedCatTarget', 'params': {'alpha': np.sqrt(6), 'r': 0.5, 'p': 0}},
            {'class_name': 'SqueezedCatTarget', 'params': {'alpha': np.sqrt(6), 'r': 0.5, 'p': 1}}
        ],
        "patterns": [[(i, 4-i)] for i in range(5)] + [[(i, 5-i)] for i in range(6)]
    })

    # 8. GKP mu=1, 2 modes, Single
    sweep_jobs.append({
        "table": "Table 1",
        "family": "GKP mu=1",
        "strategy": "Single (4)",
        "modes": 2,
        "circuit_config": {
            'class_name': 'TwoModeTimeDomainSqueezeOnly',
            'params': {
                'steps': STEPS,
                'time_invariant': TIME_INVARIANT,
                'clip_size': squeezing,
                'measure_fock_cutoff': MEASURE_CUTOFF,
                'num_single_photon': 0,
                'train_initial_state': True,
                'initial_r': squeezing,
                'initial_fock_one': False
            }
        },
        "target_configs": [
            {'class_name': 'CoreGKPTarget', 'params': {'csv_path': csv_path_abs, 'n_max': 4, 'delta_db': 10, 'mu': 1}}
        ],
        "patterns": [[(4,)]]
    })

    # 9. GKP mu=1, 2 modes, Multi
    sweep_jobs.append({
        "table": "Table 1",
        "family": "GKP mu=1",
        "strategy": "Multi (2 modes)",
        "modes": 2,
        "circuit_config": {
            'class_name': 'TwoModeTimeDomainSqueezeOnly',
            'params': {
                'steps': STEPS,
                'time_invariant': TIME_INVARIANT,
                'clip_size': squeezing,
                'measure_fock_cutoff': MEASURE_CUTOFF,
                'num_single_photon': 0,
                'train_initial_state': True,
                'initial_r': squeezing,
                'initial_fock_one': False
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
        "table": "Table 1",
        "family": "GKP mu=1",
        "strategy": "Single (4,0)",
        "modes": 3,
        "circuit_config": {
            'class_name': 'ThreeModeTimeDomainSqueezeOnly',
            'params': {
                'steps': STEPS,
                'time_invariant': TIME_INVARIANT,
                'clip_size': squeezing,
                'measure_fock_cutoff': MEASURE_CUTOFF,
                'num_single_photon': 0,
                'train_initial_state': True,
                'initial_r': squeezing
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
        "table": "Table 1",
        "family": "GKP mu=1",
        "strategy": "Multi (3 modes)",
        "modes": 3,
        "circuit_config": {
            'class_name': 'ThreeModeTimeDomainSqueezeOnly',
            'params': {
                'steps': STEPS,
                'time_invariant': TIME_INVARIANT,
                'clip_size': squeezing,
                'measure_fock_cutoff': MEASURE_CUTOFF,
                'num_single_photon': 0,
                'train_initial_state': True,
                'initial_r': squeezing
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
        "table": "Table 1",
        "family": "Binomial",
        "strategy": "Single (2,4)",
        "modes": 3,
        "circuit_config": {
            'class_name': 'ThreeModeTimeDomainSqueezeOnly',
            'params': {
                'steps': STEPS,
                'time_invariant': TIME_INVARIANT,
                'clip_size': squeezing,
                'measure_fock_cutoff': MEASURE_CUTOFF,
                'num_single_photon': 0,
                'train_initial_state': True,
                'initial_r': squeezing
            }
        },
        "target_configs": [
            {'class_name': 'BinomialCodeTarget', 'params': {'N': 2, 'S': 2, 'mu': 0}}
        ],
        "patterns": [[(2, 4)]]
    })

    # 13. Binomial, 3 modes, Multi
    sweep_jobs.append({
        "table": "Table 1",
        "family": "Binomial",
        "strategy": "Multi",
        "modes": 3,
        "circuit_config": {
            'class_name': 'ThreeModeTimeDomainSqueezeOnly',
            'params': {
                'steps': STEPS,
                'time_invariant': TIME_INVARIANT,
                'clip_size': squeezing,
                'measure_fock_cutoff': MEASURE_CUTOFF,
                'num_single_photon': 0,
                'train_initial_state': True,
                'initial_r': squeezing
            }
        },
        "target_configs": [
            {'class_name': 'BinomialCodeTarget', 'params': {'N': 2, 'S': 2, 'mu': 0}},
            {'class_name': 'BinomialCodeTarget', 'params': {'N': 2, 'S': 3, 'mu': 0}}
        ],
        "patterns": [[(2, 4)], [(4, 2)], [(3, 5)], [(5, 3)]]
    })

    # ==========================================
    # TABLE 2 CONFIGURATIONS (Non-redundant entries)
    # ==========================================

    # 14. Table 2: GKP mu=0, 3 modes, (1,3) + (3,1)
    sweep_jobs.append({
        "table": "Table 2",
        "family": "GKP mu=0",
        "strategy": "Harvesting (1,3), (3,1)",
        "modes": 3,
        "circuit_config": {
            'class_name': 'ThreeModeTimeDomainSqueezeOnly',
            'params': {
                'steps': STEPS,
                'time_invariant': TIME_INVARIANT,
                'clip_size': squeezing,
                'measure_fock_cutoff': MEASURE_CUTOFF,
                'num_single_photon': 0,
                'train_initial_state': True,
                'initial_r': squeezing
            }
        },
        "target_configs": [
            {'class_name': 'CoreGKPTarget', 'params': {'csv_path': csv_path_abs, 'n_max': 4, 'delta_db': 10, 'mu': 0}}
        ],
        "patterns": [[(1, 3)], [(3, 1)]]
    })

    # 15. Table 2: GKP mu=0, 3 modes, (1,3) + (3,1) + (2,2)
    sweep_jobs.append({
        "table": "Table 2",
        "family": "GKP mu=0",
        "strategy": "Harvesting (1,3), (3,1), (2,2)",
        "modes": 3,
        "circuit_config": {
            'class_name': 'ThreeModeTimeDomainSqueezeOnly',
            'params': {
                'steps': STEPS,
                'time_invariant': TIME_INVARIANT,
                'clip_size': squeezing,
                'measure_fock_cutoff': MEASURE_CUTOFF,
                'num_single_photon': 0,
                'train_initial_state': True,
                'initial_r': squeezing
            }
        },
        "target_configs": [
            {'class_name': 'CoreGKPTarget', 'params': {'csv_path': csv_path_abs, 'n_max': 4, 'delta_db': 10, 'mu': 0}}
        ],
        "patterns": [[(1, 3)], [(3, 1)], [(2, 2)]]
    })

    # 16. Table 2: GKP mu=1, 3 modes, (2,2)
    sweep_jobs.append({
        "table": "Table 2",
        "family": "GKP mu=1",
        "strategy": "Harvesting (2,2)",
        "modes": 3,
        "circuit_config": {
            'class_name': 'ThreeModeTimeDomainSqueezeOnly',
            'params': {
                'steps': STEPS,
                'time_invariant': TIME_INVARIANT,
                'clip_size': squeezing,
                'measure_fock_cutoff': MEASURE_CUTOFF,
                'num_single_photon': 0,
                'train_initial_state': True,
                'initial_r': squeezing
            }
        },
        "target_configs": [
            {'class_name': 'CoreGKPTarget', 'params': {'csv_path': csv_path_abs, 'n_max': 4, 'delta_db': 10, 'mu': 1}}
        ],
        "patterns": [[(2, 2)]]
    })

    # 17. Table 2: GKP mu=1, 3 modes, (3,1), (2,2), (4,0), (1,3), (0,4)
    sweep_jobs.append({
        "table": "Table 2",
        "family": "GKP mu=1",
        "strategy": "Harvesting 5-patterns",
        "modes": 3,
        "circuit_config": {
            'class_name': 'ThreeModeTimeDomainSqueezeOnly',
            'params': {
                'steps': STEPS,
                'time_invariant': TIME_INVARIANT,
                'clip_size': squeezing,
                'measure_fock_cutoff': MEASURE_CUTOFF,
                'num_single_photon': 0,
                'train_initial_state': True,
                'initial_r': squeezing
            }
        },
        "target_configs": [
            {'class_name': 'CoreGKPTarget', 'params': {'csv_path': csv_path_abs, 'n_max': 4, 'delta_db': 10, 'mu': 1}}
        ],
        "patterns": [[(3, 1)], [(2, 2)], [(4, 0)], [(1, 3)], [(0, 4)]]
    })

    # 18. Table 2: Cat, 3 modes, (2,2)
    sweep_jobs.append({
        "table": "Table 2",
        "family": "Cat",
        "strategy": "Harvesting (2,2)",
        "modes": 3,
        "circuit_config": {
            'class_name': 'ThreeModeTimeDomainSqueezeOnly',
            'params': {
                'steps': STEPS,
                'time_invariant': TIME_INVARIANT,
                'clip_size': squeezing,
                'measure_fock_cutoff': MEASURE_CUTOFF,
                'num_single_photon': 0,
                'train_initial_state': True,
                'initial_r': squeezing
            }
        },
        "target_configs": [
            {'class_name': 'SqueezedCatTarget', 'params': {'alpha': np.sqrt(6), 'r': 0.5, 'p': 0}}
        ],
        "patterns": [[(2, 2)]]
    })

    # 19. Table 2: Cat, 3 modes, 5-patterns
    sweep_jobs.append({
        "table": "Table 2",
        "family": "Cat",
        "strategy": "Harvesting 5-patterns",
        "modes": 3,
        "circuit_config": {
            'class_name': 'ThreeModeTimeDomainSqueezeOnly',
            'params': {
                'steps': STEPS,
                'time_invariant': TIME_INVARIANT,
                'clip_size': squeezing,
                'measure_fock_cutoff': MEASURE_CUTOFF,
                'num_single_photon': 0,
                'train_initial_state': True,
                'initial_r': squeezing
            }
        },
        "target_configs": [
            {'class_name': 'SqueezedCatTarget', 'params': {'alpha': np.sqrt(6), 'r': 0.5, 'p': 0}}
        ],
        "patterns": [[(2, 2)], [(1, 3)], [(3, 1)], [(0, 4)], [(4, 0)]]
    })

    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    results_dir = Path(__file__).resolve().parent.parent.parent / "results" / f"sweeps_fid099_{timestamp}"
    results_dir.mkdir(parents=True, exist_ok=True)
    print(f"Sweep results will be written to: {results_dir}\n")

    num_procs = min(4, os.cpu_count() or 1)
    results_summary = []

    for idx, job in enumerate(sweep_jobs):
        print("=" * 80)
        print(f"JOB {idx+1}/{len(sweep_jobs)}: {job['table']} | {job['family']} | {job['strategy']}")
        print("=" * 80)

        # Build subfolder for job
        safe_strategy_name = "".join([c if c.isalnum() else "_" for c in job['strategy']])
        job_dir = results_dir / f"job_{idx+1:02d}_{job['family'].replace(' ', '_')}_{safe_strategy_name}"
        job_dir.mkdir(exist_ok=True)

        try:
            # Instantiate components
            circuit = create_from_config(job['circuit_config'], circuit_module)
            targets = [create_from_config(cfg, target_module) for cfg in job['target_configs']]
            patterns = prepare_measurement_patterns(job['patterns'])

            # Run Basin Hopping Optimization
            runner = BasinHoppingRunner(
                num_processes=num_procs,
                circuit=circuit,
                target_gens=targets,
                cutoff_dim=CUTOFF_DIM,
                beam_width=BEAM_WIDTH,
                penalty_strength=1,
                success_threshold=SUCCESS_THRESHOLD,
                success_weight=0.0,
                ng_weight=0.,
                ng_threshold=0,
                sigma0=1,
                photon_dist_weight=0,
                max_photon_dist=CUTOFF_DIM,
                measurement_patterns=patterns,
                popsize=15
            )

            start_time = time.time()
            res = runner.run(n_generations=N_GENERATIONS, prob_power=1.0)
            duration = time.time() - start_time

            branches = res.get('branches', [])
            
            # Analyze results within the targeted set
            if len(branches) > 0:
                fidelities = [b['fidelity'] for b in branches]
                min_fid = min(fidelities)
                max_infid = 1.0 - min_fid
                
                # Success probability is the sum of probabilities of designated patterns
                # achieving our requested fidelity of 0.99
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
                "n_pat_total": len(job["patterns"]),
                "n_pat_success": n_pat_success,
                "max_infid": max_infid,
                "agg_prob": agg_prob,
                "duration": duration,
                "status": "Success"
            }

            print(f"-> Job {idx+1} Completed in {duration:.1f}s.")
            print(f"-> Worst-case Infidelity (1-F_min): {max_infid:.2e}")
            print(f"-> Aggregated Probability (P_agg @ >=0.99): {agg_prob:.2%}")

            # Save raw pickle metadata
            with open(job_dir / "results.pkl", "wb") as f:
                pickle.dump({"config": job, "results": res}, f)

            with open(job_dir / "summary.json", "w") as f:
                json.dump(job_result, f, indent=2)

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

    # Save aggregated table summary files
    print("\n" + "=" * 80)
    print("ALL SWEEPS COMPLETE - COMPILING REPORTS")
    print("=" * 80)

    # Write Markdown Report
    report_path = results_dir / "sweep_report.md"
    with open(report_path, "w") as f:
        f.write("# Referee-Requested Sweep Report (Fixed Fidelity Threshold = 0.99)\n\n")
        f.write(f"Generated on: {datetime.utcnow().isoformat()}Z\n\n")

        # Table 1 Section
        f.write("## Table 1 Comparison Summary\n\n")
        f.write("| Family | Strategy | Modes | Target State(s) | N_pat | Max Infidelity ($1-\\mathcal{F}_{\\mathrm{min}}$) | Aggregated Success Prob ($P_{\\mathrm{agg}}$) | Status |\n")
        f.write("| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |\n")
        for r in results_summary:
            if r["table"] == "Table 1":
                infid_str = f"{r['max_infid']:.2e}" if not np.isnan(r['max_infid']) else "N/A"
                prob_str = f"{r['agg_prob']:.2%}"
                f.write(f"| {r['family']} | {r['strategy']} | {r['modes']} | {r['targets']} | {r['n_pat_total']} | {infid_str} | {prob_str} | {r['status']} |\n")

        f.write("\n\n")

        # Table 2 Section
        f.write("## Table 2 Harvesting Summary\n\n")
        f.write("| Target | Modes | Strategy/Patterns | Max Infidelity ($1-\\mathcal{F}_{\\mathrm{min}}$) | Total Success Prob ($P_{\\mathrm{total}}$) | Status |\n")
        f.write("| :--- | :--- | :--- | :--- | :--- | :--- |\n")
        for r in results_summary:
            if r["table"] == "Table 2":
                infid_str = f"{r['max_infid']:.2e}" if not np.isnan(r['max_infid']) else "N/A"
                prob_str = f"{r['agg_prob']:.2%}"
                f.write(f"| {r['targets']} | {r['modes']} | {r['strategy']} | {infid_str} | {prob_str} | {r['status']} |\n")

    # Print markdown structure to console for easy viewing
    with open(report_path, "r") as f:
        print(f.read())

    print(f"\nAll details and summaries saved in: {results_dir}")


if __name__ == "__main__":
    main()
