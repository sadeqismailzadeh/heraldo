"""
Script to filter out low-fidelity branches from completed optimization sweeps
and compile an aggregated filtered sweep report matching Table 1 and Table 2 formats.
"""

import os
import json
import pickle
import numpy as np
from pathlib import Path
from datetime import datetime

# --- User Configuration (No CLI arguments required) ---
# Set the threshold below which outcomes are excluded from the summary.
FIDELITY_THRESHOLD = 0.95

# Specify the results folder path relative to the repository root.
# Leaving this as None will automatically find the latest sweep folder.
TARGET_RESULTS_DIR = r"E:\Quantum\reports\paper\results1\fixed fid\nonlinear\temp\sweeps_fid099_merged"


def get_target_name_brief(cfg):
    """Formats target configuration to a readable brief label."""
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


def map_branch_to_target(b, family, target_configs):
    """Identifies which target index a given branch belongs to."""
    if 'target_idx' in b and b['target_idx'] is not None:
        return b['target_idx']
    
    pat = b.get('pattern')
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


def get_patterns_for_target(target_idx, job_patterns, family, target_configs):
    """Extracts the subset of patterns associated with a given target index."""
    matched = []
    if job_patterns is None:
        return matched
    for group in job_patterns:
        for pat in group:
            b_mock = {'pattern': pat}
            idx = map_branch_to_target(b_mock, family, target_configs)
            if idx == target_idx:
                matched.append(pat)
    return matched


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
    return c_name


def format_patterns(pats):
    """Formats heralding patterns to be human-readable."""
    if not pats:
        return ""
    pats_tuples = [tuple(p) for p in pats]
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


def format_filtered_branches_report(branches, target_names, threshold):
    """Formats the table showing only branches that meet the threshold."""
    lines = []
    lines.append("-" * 80)
    lines.append(f"Filtered Outcomes (Fidelity >= {threshold}):")
    lines.append("-" * 80)
    lines.append(f"{'Outcome':<20} {'Prob':<10} {'Fidelity':<10} {'1-Fid':<10} {'Best Target':<15}")
    lines.append("-" * 80)

    sorted_branches = sorted(branches, key=lambda x: x['prob'], reverse=True)
    total_prob = 0.0
    filtered_count = 0
    
    for b in sorted_branches:
        if b['fidelity'] >= threshold:
            filtered_count += 1
            total_prob += b['prob']
            outcome_str = str(b['outcome'])
            t_idx = b.get('target_idx', 0)
            tgt_name = target_names[t_idx] if t_idx < len(target_names) else f"Target_{t_idx}"
            lines.append(f"{outcome_str:<20} {b['prob']:<10.4f} {b['fidelity']:<10.4f} {(1-b['fidelity']):<10.1e} {tgt_name:<15}")

    if filtered_count == 0:
        lines.append("No branches met the threshold.")
    
    lines.append(f"\nTotal Probability captured (filtered): {total_prob:.5f}")
    return "\n".join(lines)


def decode_folder_details(job_path):
    """Extracts job attributes using summary.json or folder name fallback."""
    summary_path = job_path / "summary.json"
    if summary_path.exists():
        try:
            with open(summary_path, "r") as f:
                data = json.load(f)
            return data.get("table", "Table 1"), data.get("family", "Unknown"), data.get("strategy", "Unknown"), data.get("modes", 3)
        except Exception:
            pass

    name = job_path.name.lower()
    table = "Table 2" if "table_2" in name or "table2" in name else "Table 1"
    
    family = "Unknown"
    if "gkp_mu_0" in name or "gkp_mu0" in name or "gkp_mu=0" in name:
        family = "GKP mu=0"
    elif "gkp_mu_1" in name or "gkp_mu1" in name or "gkp_mu=1" in name:
        family = "GKP mu=1"
    elif "cat" in name:
        family = "Cat"
    elif "binomial" in name:
        family = "Binomial"

    strategy = "Unknown"
    parts = job_path.name.split("_", 3)
    if len(parts) >= 4:
        strategy = parts[3].replace("_", " ")

    return table, family, strategy, 3


def filter_batch_results():
    root_dir = Path(__file__).resolve().parent.parent.parent
    results_parent = root_dir / "results"

    if TARGET_RESULTS_DIR is not None:
        run_dir = Path(TARGET_RESULTS_DIR)
        if not run_dir.is_absolute():
            run_dir = root_dir / run_dir
    else:
        # Automatically find the latest sweeps directory matching the prefix
        candidates = sorted(results_parent.glob("sweeps_fid099_*"), key=os.path.getmtime)
        if not candidates:
            print(f"No sweep folders found in {results_parent}")
            return
        run_dir = candidates[-1]

    print(f"Filtering runs in: {run_dir}")
    print(f"Applying fidelity threshold: >= {FIDELITY_THRESHOLD}")

    job_folders = sorted(run_dir.glob("job_*"))
    if not job_folders:
        print("No job directories found in target folder.")
        return

    results_summary = []

    for job_path in job_folders:
        pkl_files = list(job_path.glob("*.pkl"))
        if not pkl_files:
            continue
        
        pkl_path = job_path / "run_0001.pkl"
        if not pkl_path.exists():
            pkl_path = pkl_files[0]

        try:
            with open(pkl_path, "rb") as f:
                data = pickle.load(f)
            
            res = data.get("res", {})
            meta = data.get("meta", {})
            branches = res.get("branches", [])
            target_configs = meta.get("target_configs", [])
            patterns_raw = meta.get("measurement_patterns", [])
            
            table, family, strategy, modes = decode_folder_details(job_path)
            target_names = [get_target_name_brief(cfg) for cfg in target_configs]

            # Generate filtered branch list
            filtered_branches = [b for b in branches if b.get('fidelity', 0.0) >= FIDELITY_THRESHOLD]
            total_prob = sum(b.get('prob', 0.0) for b in filtered_branches)

            # Write filtered text report
            report_str = format_filtered_branches_report(branches, target_names, FIDELITY_THRESHOLD)
            report_name = f"run_0001_branches_filtered_{FIDELITY_THRESHOLD}.txt"
            with open(job_path / report_name, "w") as f:
                f.write(report_str)

            # Write filtered json summary
            summary_data = {
                "threshold": FIDELITY_THRESHOLD,
                "total_branches_above_threshold": len(filtered_branches),
                "total_probability_above_threshold": total_prob,
                "branches": [
                    {
                        "outcome": b.get("outcome"),
                        "prob": b.get("prob"),
                        "fidelity": b.get("fidelity"),
                        "target_idx": b.get("target_idx", 0)
                    }
                    for b in filtered_branches
                ]
            }
            summary_name = f"run_0001_summary_filtered_{FIDELITY_THRESHOLD}.json"
            with open(job_path / summary_name, "w") as f:
                json.dump(summary_data, f, indent=2)

            # Detailed target metrics recalculated for the filtered sweep report
            target_results = []
            for k, tgt_cfg in enumerate(target_configs):
                tgt_branches = [b for b in branches if map_branch_to_target(b, family, target_configs) == k]
                tgt_pats = get_patterns_for_target(k, patterns_raw, family, target_configs)
                n_pat_total = len(tgt_pats)

                filtered_tgt_branches = [b for b in tgt_branches if b['fidelity'] >= FIDELITY_THRESHOLD]

                if len(filtered_tgt_branches) > 0:
                    tgt_fidelities = [b['fidelity'] for b in filtered_tgt_branches]
                    tgt_min_fid = min(tgt_fidelities)
                    tgt_max_infid = 1.0 - tgt_min_fid
                    tgt_agg_prob = sum(b['prob'] for b in filtered_tgt_branches)
                    tgt_n_pat_success = len(filtered_tgt_branches)
                else:
                    tgt_max_infid = float('nan')
                    tgt_agg_prob = 0.0
                    tgt_n_pat_success = 0

                # Gather only the successful patterns for this target
                tgt_pats_filtered = []
                for b in filtered_tgt_branches:
                    pat = b.get('outcome') or b.get('pattern')
                    if pat is not None:
                        tgt_pats_filtered.append(pat)

                target_results.append({
                    "target_idx": k,
                    "target_label": format_target_latex(tgt_cfg),
                    "patterns_str": format_patterns(tgt_pats_filtered),
                    "n_pat_total": n_pat_total,
                    "n_pat_success": tgt_n_pat_success,
                    "max_infid": tgt_max_infid,
                    "agg_prob": tgt_agg_prob
                })

            if len(filtered_branches) > 0:
                fidelities = [b['fidelity'] for b in filtered_branches]
                min_fid = min(fidelities)
                max_infid = 1.0 - min_fid
                agg_prob = sum(b['prob'] for b in filtered_branches)
                n_pat_success = len(filtered_branches)
            else:
                max_infid = float('nan')
                agg_prob = 0.0
                n_pat_success = 0

            results_summary.append({
                "idx": job_path.name,
                "table": table,
                "family": family,
                "strategy": strategy,
                "modes": modes,
                "targets_detailed": target_results,
                "n_pat_total": sum(t['n_pat_total'] for t in target_results),
                "n_pat_success": n_pat_success,
                "max_infid": max_infid,
                "agg_prob": agg_prob,
                "status": "Success"
            })

            print(f"Processed {job_path.name}: Saved filtered summary (total prob: {total_prob:.5f})")

        except Exception as e:
            print(f"Failed to process {job_path.name}: {e}")

    # Generate aggregated markdown report with new threshold
    report_path = run_dir / f"sweep_report_filtered_{FIDELITY_THRESHOLD}.md"
    with open(report_path, "w") as f:
        f.write(f"# Filtered Sweep Report (Fidelity Threshold = {FIDELITY_THRESHOLD})\n\n")
        f.write(f"Generated on: {datetime.utcnow().isoformat()}Z\n\n")

        # Table 1 Section
        f.write("## Table 1 Comparison Summary\n\n")
        f.write("| Family | Modes | Strategy | Target | Patterns | $N_{\\mathrm{pat}}$ | Max Infidelity ($1-\\mathcal{F}_{\\mathrm{min}}$) | Success Prob ($P_{\\mathrm{agg}}$) | Status |\n")
        f.write("| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |\n")
        
        for r in results_summary:
            if r["table"] == "Table 1":
                detailed = r.get("targets_detailed", [])
                if not detailed:
                    f.write(f"| {r['family']} | {r['modes']} | {r['strategy']} | - | - | - | - | - | {r['status']} |\n")
                    continue
                
                is_multi = len(detailed) > 1
                for i, tgt in enumerate(detailed):
                    fam_str = r['family'] if i == 0 else ""
                    strat_str = r['strategy'] if i == 0 else ""
                    modes_str = str(r['modes']) if i == 0 else ""
                    
                    infid_str = f"{tgt['max_infid']:.2e}" if not np.isnan(tgt['max_infid']) else "N/A"
                    prob_str = f"{tgt['agg_prob']:.2%}"
                    f.write(f"| {fam_str} | {modes_str} | {strat_str} | {tgt['target_label']} | {tgt['patterns_str']} | {tgt['n_pat_success']} | {infid_str} | {prob_str} | {r['status'] if i == 0 else ''} |\n")
                
                if is_multi:
                    prob_total_str = f"{r['agg_prob']:.2%}"
                    f.write(f"| | | | *Total* | | {r['n_pat_success']} | -- | {prob_total_str} | |\n")
                
                f.write("| | | | | | | | | |\n")

        f.write("\n\n")

        # Table 2 Section
        f.write("## Table 2 Harvesting Summary\n\n")
        f.write("| Target | Modes | Strategy/Patterns | Max Infidelity ($1-\\mathcal{F}_{\\mathrm{min}}$) | Total Success Prob ($P_{\\mathrm{total}}$) | Status |\n")
        f.write("| :--- | :--- | :--- | :--- | :--- | :--- |\n")
        for r in results_summary:
            if r["table"] == "Table 2":
                detailed = r.get("targets_detailed", [])
                if detailed:
                    tgt = detailed[0]
                    infid_str = f"{tgt['max_infid']:.2e}" if not np.isnan(tgt['max_infid']) else "N/A"
                    prob_str = f"{r['agg_prob']:.2%}"
                    f.write(f"| {tgt['target_label']} | {r['modes']} | {r['strategy']} | {infid_str} | {prob_str} | {r['status']} |\n")
                else:
                    f.write(f"| - | {r['modes']} | {r['strategy']} | N/A | 0.0% | {r['status']} |\n")

    print("\n" + "=" * 80)
    print(f"FILTERED SWEEP REPORT COMPILED AT: {report_path}")
    print("=" * 80)
    with open(report_path, "r") as f:
        print(f.read())


if __name__ == "__main__":
    filter_batch_results()
