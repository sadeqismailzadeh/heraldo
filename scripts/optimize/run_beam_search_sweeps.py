"""
Script to automate the Beam Search pattern discovery stage (Table I configurations)
for unconstrained measurement pattern discovery in photonic continuous-variable circuits.
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
import numpy as np

warnings.filterwarnings("ignore", category=RuntimeWarning, module="scipy.optimize")

import heraldo.optimization.circuits as circuit_module
import heraldo.components.targets as target_module
from heraldo.optimization.circuits import *
from heraldo.optimization.runner import BasinHoppingRunner
from heraldo.components.targets import *
from heraldo.utils import *
from heraldo.factory import create_from_config


def format_branches_report(branches, target_names, success_threshold):
    """Returns a formatted string of branch statistics discovered by beam search."""
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

    # Target Analysis
    lines.append("-" * 60)
    lines.append(f"Target Distribution Analysis (Success > {success_threshold}):")
    lines.append(f"{'Rank':<5} {'Target Name':<20} {'Tot. Prob':<10} {'Outcomes (Top 5)'}")
    lines.append("-" * 60)

    target_stats = {}
    for b in branches:
        if b['fidelity'] >= success_threshold:
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
            top_outcomes = [str(o[0]) for o in stats['outcomes'][:5]]
            outcome_str = ", ".join(top_outcomes)
            if len(stats['outcomes']) > 5:
                outcome_str += ", ..."
            t_name = target_names[idx] if idx < len(target_names) else f"Target_{idx}"
            lines.append(f"{rank+1:<5} {t_name:<20} {stats['prob']:<10.4f} {outcome_str}")

    return "\n".join(lines)


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
    elif "Cubic" in c_name:
        return f"Cubic_g{p.get('gamma')}_r{p.get('r')}"
    return c_name


def format_target_latex(cfg):
    """Converts target configuration to LaTeX representation."""
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
    elif "Cubic" in c_name:
        return "$\\ket{\\text{Cubic}}$"
    return c_name


def format_patterns(outcomes):
    """Formats discovered heralding patterns into string representation."""
    if not outcomes:
        return "None"
    out_tuples = [tuple(o) for o in outcomes]
    if len(out_tuples) == 1:
        return str(out_tuples[0])
    if len(out_tuples) <= 4:
        return ", ".join(str(o) for o in out_tuples)
    
    # Check if outcomes share a constant photon sum
    sums = [sum(o) for o in out_tuples]
    if all(s == sums[0] for s in sums):
        return f"$\\sum n_i = {sums[0]}$ ({len(out_tuples)} patterns)"
    return f"{len(out_tuples)} patterns (e.g., {out_tuples[0]}, {out_tuples[1]}, ...)"


def main():
    CUTOFF_DIM = 30
    STEPS = 1
    BEAM_WIDTH = 200  # Beam width B = 200 for pattern discovery
    TIME_INVARIANT = False
    MEASURE_CUTOFF = CUTOFF_DIM
    SUCCESS_THRESHOLD = 0.93  # Threshold for considering a branch a valid target candidate
    N_GENERATIONS = 200

    squeezing = db_to_r(12)
    csv_path_abs = str(Path(__file__).resolve().parent.parent.parent / "data" / "GKP_core_coefficients.csv")

    print("--- Defining Beam Search Discovery Jobs (Table I) ---")

    sweep_jobs = []

    # =========================================================================
    # BEAM SEARCH PATTERN DISCOVERY JOBS
    # =========================================================================

    # 1. GKP mu=1, 2 modes
    sweep_jobs.append({
        "family": "GKP core mu=1",
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
        ]
    })

    # 2. GKP mu=1, 3 modes
    sweep_jobs.append({
        "family": "GKP core mu=1",
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
        ]
    })

    # 3. Cat, 2 modes
    sweep_jobs.append({
        "family": "Cat",
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
        ]
    })

    # 4. Cat, 3 modes
    sweep_jobs.append({
        "family": "Cat",
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
        ]
    })

    # 5. GKP mu=0, 3 modes
    sweep_jobs.append({
        "family": "GKP core mu=0",
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
        ]
    })

    # 6. Binomial, 3 modes
    sweep_jobs.append({
        "family": "Binomial",
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
        ]
    })

    # 7. Cubic Phase, 3 modes
    sweep_jobs.append({
        "family": "Cubic",
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
            {'class_name': 'CubicPhaseTarget', 'params': {'gamma': -0.2, 'r': -0.7, 'alpha': 1.25}}
        ]
    })

    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    results_dir = Path(__file__).resolve().parent.parent.parent / "results" / f"sweeps_beam_search_{timestamp}"
    results_dir.mkdir(parents=True, exist_ok=True)
    print(f"Beam Search sweep results will be written to: {results_dir}\n")

    num_procs = min(4, os.cpu_count() or 1)
    results_summary = []

    for idx, job in enumerate(sweep_jobs):
        print("=" * 80)
        print(f"BEAM SEARCH JOB {idx+1}/{len(sweep_jobs)}: Family={job['family']} | Modes={job['modes']}")
        print("=" * 80)

        job_dir = results_dir / f"job_{idx+1:02d}_{job['family'].replace(' ', '_')}_{job['modes']}mode"
        job_dir.mkdir(exist_ok=True)

        try:
            circuit = create_from_config(job['circuit_config'], circuit_module)
            targets = [create_from_config(cfg, target_module) for cfg in job['target_configs']]

            # Instantiate BasinHoppingRunner with measurement_patterns=None to run BEAM SEARCH
            runner = BasinHoppingRunner(
                num_processes=num_procs,
                circuit=circuit,
                target_gens=targets,
                cutoff_dim=CUTOFF_DIM,
                beam_width=BEAM_WIDTH,
                penalty_strength=1.0,
                measurement_patterns=None  # Triggers Beam Search discovery mode!
            )

            start_time = time.time()
            res = runner.run(n_generations=N_GENERATIONS, prob_power=1.0)
            duration = time.time() - start_time

            branches = res.get('branches', [])

            # Group discovered patterns by target index
            target_results = []
            for k, tgt_cfg in enumerate(job['target_configs']):
                tgt_branches = [b for b in branches if b.get('target_idx', 0) == k and b['fidelity'] >= SUCCESS_THRESHOLD]
                discovered_outcomes = [b['outcome'] for b in tgt_branches]
                n_pat = len(discovered_outcomes)

                if n_pat > 0:
                    tgt_fidelities = [b['fidelity'] for b in tgt_branches]
                    min_fid = min(tgt_fidelities)
                    max_fid = max(tgt_fidelities)
                    agg_prob = sum(b['prob'] for b in tgt_branches)
                else:
                    min_fid = float('nan')
                    max_fid = float('nan')
                    agg_prob = 0.0

                target_results.append({
                    "target_idx": k,
                    "target_label": format_target_latex(tgt_cfg),
                    "outcomes": discovered_outcomes,
                    "patterns_str": format_patterns(discovered_outcomes),
                    "n_pat": n_pat,
                    "min_fidelity": min_fid,
                    "max_fidelity": max_fid,
                    "agg_prob": agg_prob
                })

            valid_branches = [b for b in branches if b['fidelity'] >= SUCCESS_THRESHOLD]
            total_agg_prob = sum(b['prob'] for b in valid_branches)
            total_n_pat = len(valid_branches)

            target_labels = ", ".join([get_target_name_brief(cfg) for cfg in job['target_configs']])

            job_result = {
                "idx": idx + 1,
                "family": job["family"],
                "modes": job["modes"],
                "targets": target_labels,
                "targets_detailed": target_results,
                "total_n_pat": total_n_pat,
                "total_agg_prob": total_agg_prob,
                "duration": duration,
                "status": "Success"
            }

            print(f"-> Job {idx+1} Completed in {duration:.1f}s.")
            print(f"   * Discovered {total_n_pat} total measurement patterns with Fidelity >= {SUCCESS_THRESHOLD}:")
            for tgt in target_results:
                f_str = f"{tgt['min_fidelity']:.1%} - {tgt['max_fidelity']:.1%}" if not np.isnan(tgt['min_fidelity']) else "N/A"
                print(f"     - Target {tgt['target_label']}: {tgt['n_pat']} patterns ({tgt['patterns_str']}), Fid: {f_str}, Prob: {tgt['agg_prob']:.2%}")
            print(f"   * Total Captured Success Probability: {total_agg_prob:.2%}")

            # Build target names list
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
            res['success_prob'] = total_agg_prob
            res['run_index'] = 1

            run_meta = {
                "run_index": 1,
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "prob_power": 1.0,
                "beam_width": BEAM_WIDTH,
                "expected_fidelity": float(expected_fidelity),
                "success_prob": float(total_agg_prob),
                "circuit_config": job['circuit_config'],
                "target_configs": job['target_configs'],
                "measurement_patterns": None
            }

            # Save artifacts
            run_file = job_dir / "run_0001.pkl"
            with open(run_file, "wb") as f:
                pickle.dump({"meta": run_meta, "res": res}, f)

            summary = {
                "run_index": 1,
                "expected_fidelity": float(expected_fidelity),
                "success_prob": float(total_agg_prob),
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

            # Best directory
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
                    mapped_params = circuit.map_parameters(res['x'])
                    param_names = circuit.per_step_parameter_names
                    np.savez(best_dir / "mapped_params.npz", mapped_params=mapped_params)

                    schedule = {
                        "param_names": param_names,
                        "mapped_params": mapped_params.tolist() if hasattr(mapped_params, "tolist") else [[float(v) for v in row] for row in mapped_params],
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
                "family": job["family"],
                "modes": job["modes"],
                "targets": "Error",
                "total_n_pat": 0,
                "total_agg_prob": 0.0,
                "duration": 0.0,
                "status": f"Failed: {str(e)}"
            }

        results_summary.append(job_result)

    # =========================================================================
    # COMPILE MASTER BEAM SEARCH REPORT
    # =========================================================================

    print("\n" + "=" * 80)
    print("ALL BEAM SEARCH JOBS COMPLETE - COMPILING REPORT")
    print("=" * 80)

    report_path = results_dir / "beam_search_report.md"
    with open(report_path, "w") as f:
        f.write("# Beam Search Pattern Discovery Report (Table I)\n\n")
        f.write(f"Generated on: {datetime.utcnow().isoformat()}Z\n")
        f.write(f"Beam Width (B): {BEAM_WIDTH} | Threshold: {SUCCESS_THRESHOLD}\n\n")

        f.write("## Summary of Discovered Measurement Patterns\n\n")
        f.write("| Target Family | Modes | Target State(s) | Heralding Pattern(s) | $N_{\\mathrm{pat}}$ | Fidelity Range | Total Success Prob ($P$) | Status |\n")
        f.write("| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |\n")

        for r in results_summary:
            detailed = r.get("targets_detailed", [])
            if not detailed:
                f.write(f"| {r['family']} | {r['modes']} | Error | - | - | - | - | {r['status']} |\n")
                continue

            for i, tgt in enumerate(detailed):
                fam_str = r['family'] if i == 0 else ""
                modes_str = str(r['modes']) if i == 0 else ""

                f_str = f"{tgt['min_fidelity']:.1%} - {tgt['max_fidelity']:.1%}" if not np.isnan(tgt['min_fidelity']) else "N/A"
                prob_str = f"{tgt['agg_prob']:.2%}"
                f.write(f"| {fam_str} | {modes_str} | {tgt['target_label']} | {tgt['patterns_str']} | {tgt['n_pat']} | {f_str} | {prob_str} | {r['status'] if i == 0 else ''} |\n")

            total_prob_str = f"{r['total_agg_prob']:.2%}"
            f.write(f"| | | *Total* | | {r['total_n_pat']} | -- | {total_prob_str} | |\n")
            f.write("| | | | | | | | |\n")

    with open(report_path, "r") as f:
        print(f.read())

    print(f"\nAll details and summaries saved in: {results_dir}")


if __name__ == "__main__":
    main()
