"""
Script to merge the results of multiple partial/interrupted Table Sweep runs.
Scans parent and workspace directories to collect incomplete runs, ignores
interrupted empty folders, resolves duplicates by selecting the run with the 
best success probability, sequentially numbers the jobs, and compiles a clean,
final merged Markdown sweep report.
"""

import os
import sys
import json
import shutil
import math
from pathlib import Path
from datetime import datetime

# Exact mapping from (table, family, strategy) to canonical index (1-20)
CANONICAL_MAP = {
    ("Table 1", "GKP mu=0", "Single (1,3)"): 1,
    ("Table 1", "GKP mu=0", "Single (2,2)"): 2,
    ("Table 1", "GKP mu=0", "Multi"): 3,
    ("Table 1", "Cat", "Single (+)"): 4,
    ("Table 1", "Cat", "Single (-)"): 5,
    ("Table 1", "Cat", "Multi (2 modes)"): 6,
    ("Table 1", "Cat", "Multi (3 modes)"): 7,
    ("Table 1", "GKP mu=1", "Single (4)"): 8,
    ("Table 1", "GKP mu=1", "Multi (2 modes)"): 9,
    ("Table 1", "GKP mu=1", "Single (4,0)"): 10,
    ("Table 1", "GKP mu=1", "Multi (3 modes)"): 11,
    ("Table 1", "Binomial", "Single (2,4)"): 12,
    ("Table 1", "Binomial", "Multi"): 13,
    ("Table 2", "Binomial", "Harvest"): 14,
    ("Table 2", "GKP mu=0", "Harvesting (1,3), (3,1)"): 15,
    ("Table 2", "GKP mu=0", "Harvesting (1,3), (3,1), (2,2)"): 16,
    ("Table 2", "GKP mu=1", "Harvesting (2,2)"): 17,
    ("Table 2", "GKP mu=1", "Harvesting 5-patterns"): 18,
    ("Table 2", "Cat", "Harvesting (2,2)"): 19,
    ("Table 2", "Cat", "Harvesting 5-patterns"): 20,
}


def find_sweep_folders():
    """Locates directories containing sweeps_fid099_* by scanning standard locations."""
    search_paths = [
        Path(r"E:\Quantum\reports\paper\results1\fixed fid\nonlinear\temp")
    ]

    sweep_dirs = []
    seen = set()

    for base in search_paths:
        if not base.exists():
            continue

        # Check if the base path itself matches the pattern
        if base.is_dir() and base.name.startswith("sweeps_fid099_") and "merged" not in base.name:
            resolved = base.resolve()
            if resolved not in seen:
                sweep_dirs.append(resolved)
                seen.add(resolved)

        # Scan child subdirectories
        try:
            for item in base.iterdir():
                if item.is_dir() and item.name.startswith("sweeps_fid099_") and "merged" not in item.name:
                    resolved = item.resolve()
                    if resolved not in seen:
                        sweep_dirs.append(resolved)
                        seen.add(resolved)
        except Exception:
            pass

    # Sort lexicographically so timestamps are ordered chronologically
    return sorted(sweep_dirs)


def is_nan(val):
    """Safely checks if a value represents NaN."""
    if val is None:
        return True
    try:
        if math.isnan(float(val)):
            return True
    except (ValueError, TypeError):
        pass
    return str(val).lower() == "nan"


def format_infid(val):
    """Safely formats infidelity values."""
    if is_nan(val):
        return "N/A"
    try:
        return f"{float(val):.2e}"
    except (ValueError, TypeError):
        return str(val)


def main():
    print("--- Starting Sweep Results Merging ---")

    sweep_dirs = find_sweep_folders()
    if not sweep_dirs:
        print("Error: No partial directories matching 'sweeps_fid099_*' were found.")
        sys.exit(1)

    print(f"Found {len(sweep_dirs)} directories to consolidate:")
    for sd in sweep_dirs:
        print(f"  * {sd}")

    # Determine parent and merged folder destinations
    parent_dir = sweep_dirs[0].parent
    merged_dir = parent_dir / "sweeps_fid099_merged"

    print(f"\nTarget consolidated output directory: {merged_dir}")

    # Gather jobs from each directory
    canonical_jobs = {}

    for sd in sweep_dirs:
        print(f"\nScanning run folder: {sd.name}")
        for item in sd.iterdir():
            if not item.is_dir():
                continue

            summary_path = item / "summary.json"
            if not summary_path.exists():
                # Unfinished / interrupted directory lacks summary.json, skip it
                print(f"  [Skipping] Empty/Interrupted folder: {item.name}")
                continue

            try:
                with open(summary_path, "r") as f:
                    job_data = json.load(f)
            except Exception as e:
                print(f"  [Warning] Could not read {summary_path}: {e}")
                continue

            table = job_data.get("table")
            family = job_data.get("family")
            strategy = job_data.get("strategy")
            key = (table, family, strategy)

            if key not in CANONICAL_MAP:
                print(f"  [Warning] Unmapped job parameters in {item.name}: {key}")
                continue

            canonical_idx = CANONICAL_MAP[key]
            agg_prob = job_data.get("agg_prob", 0.0)

            if canonical_idx in canonical_jobs:
                existing_data = canonical_jobs[canonical_idx]["job_data"]
                existing_prob = existing_data.get("agg_prob", 0.0)

                # Keep the run with the better success probability
                if agg_prob > existing_prob:
                    print(f"  [Override] Canonical Job {canonical_idx:02d} ({family} | {strategy}): "
                          f"Using run from {sd.name} (agg_prob: {agg_prob:.2%} > previous: {existing_prob:.2%})")
                    canonical_jobs[canonical_idx] = {
                        "src_dir": item,
                        "job_data": job_data,
                        "run_folder": sd.name
                    }
                else:
                    print(f"  [Keep Current] Canonical Job {canonical_idx:02d} ({family} | {strategy}): "
                          f"Keeping run from {canonical_jobs[canonical_idx]['run_folder']} "
                          f"(agg_prob: {existing_prob:.2%} >= new: {agg_prob:.2%})")
            else:
                print(f"  [Registered] Canonical Job {canonical_idx:02d} ({family} | {strategy}) "
                      f"from {sd.name} (agg_prob: {agg_prob:.2%})")
                canonical_jobs[canonical_idx] = {
                    "src_dir": item,
                    "job_data": job_data,
                    "run_folder": sd.name
                }

    if not canonical_jobs:
        print("\nNo valid completed jobs with 'summary.json' found across folders.")
        sys.exit(1)

    # Clean existing merged folder
    if merged_dir.exists():
        print(f"\nClearing previous consolidated directory...")
        shutil.rmtree(merged_dir)
    merged_dir.mkdir(parents=True, exist_ok=True)

    print("\n--- Copying Selected Runs and Re-Indexing ---")
    sorted_indices = sorted(canonical_jobs.keys())
    final_summaries = []

    for idx, canonical_idx in enumerate(sorted_indices, start=1):
        job_info = canonical_jobs[canonical_idx]
        src_dir = job_info["src_dir"]
        job_data = job_info["job_data"]

        # Re-generate safe directory names using the original structure
        family_part = job_data['family'].replace(' ', '_')
        strategy_part = "".join([c if c.isalnum() else "_" for c in job_data['strategy']])
        dest_folder_name = f"job_{canonical_idx:02d}_{family_part}_{strategy_part}"
        dest_dir = merged_dir / dest_folder_name

        print(f"  * Canonical {canonical_idx:02d} -> {dest_folder_name} (from {job_info['run_folder']})")
        shutil.copytree(src_dir, dest_dir)

        # Update metadata index to match canonical order
        dest_summary_path = dest_dir / "summary.json"
        if dest_summary_path.exists():
            try:
                with open(dest_summary_path, "r") as f:
                    data = json.load(f)
                data["idx"] = canonical_idx
                with open(dest_summary_path, "w") as f:
                    json.dump(data, f, indent=2)
                # Keep reference of updated dataset for the report generator
                final_summaries.append(data)
            except Exception as e:
                print(f"    [Warning] Could not update index inside summary.json: {e}")
                final_summaries.append(job_data)
        else:
            final_summaries.append(job_data)

    # Write Markdown Report
    report_path = merged_dir / "sweep_report.md"
    print(f"\nCompiling consolidated report to: {report_path}")

    with open(report_path, "w") as f:
        f.write("# Referee-Requested Sweep Report (Fixed Fidelity Threshold = 0.99)\n\n")
        f.write(f"Consolidated and Generated on: {datetime.utcnow().isoformat()}Z\n\n")

        # Table 1 Section
        f.write("## Table 1 Comparison Summary\n\n")
        f.write("| Family | Modes | Strategy | Target | Patterns | $N_{\\mathrm{pat}}$ | Max Infidelity ($1-\\mathcal{F}_{\\mathrm{min}}$) | Success Prob ($P_{\\mathrm{agg}}$) | Status |\n")
        f.write("| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |\n")
        
        for r in final_summaries:
            if r.get("table") == "Table 1":
                detailed = r.get("targets_detailed", [])
                if not detailed:
                    f.write(f"| {r.get('family', '')} | {r.get('modes', '')} | {r.get('strategy', '')} | Error | - | - | - | - | {r.get('status', 'Failed')} |\n")
                    continue
                
                is_multi = len(detailed) > 1
                for i, tgt in enumerate(detailed):
                    fam_str = r.get('family', '') if i == 0 else ""
                    strat_str = r.get('strategy', '') if i == 0 else ""
                    modes_str = str(r.get('modes', '')) if i == 0 else ""
                    
                    infid_str = format_infid(tgt.get('max_infid'))
                    prob_str = f"{tgt.get('agg_prob', 0.0):.2%}"
                    f.write(f"| {fam_str} | {modes_str} | {strat_str} | {tgt.get('target_label', '')} | {tgt.get('patterns_str', '')} | {tgt.get('n_pat_total', 0)} | {infid_str} | {prob_str} | {r.get('status', 'Success') if i == 0 else ''} |\n")
                
                if is_multi:
                    prob_total_str = f"{r.get('agg_prob', 0.0):.2%}"
                    f.write(f"| | | | *Total* | | {r.get('n_pat_total', 0)} | -- | {prob_total_str} | |\n")
                
                f.write("| | | | | | | | | |\n")

        f.write("\n\n")

        # Table 2 Section
        f.write("## Table 2 Harvesting Summary\n\n")
        f.write("| Target | Modes | Strategy/Patterns | Max Infidelity ($1-\\mathcal{F}_{\\mathrm{min}}$) | Total Success Prob ($P_{\\mathrm{total}}$) | Status |\n")
        f.write("| :--- | :--- | :--- | :--- | :--- | :--- |\n")
        for r in final_summaries:
            if r.get("table") == "Table 2":
                detailed = r.get("targets_detailed", [])
                if detailed:
                    tgt = detailed[0]
                    infid_str = format_infid(tgt.get('max_infid'))
                    prob_str = f"{r.get('agg_prob', 0.0):.2%}"
                    f.write(f"| {tgt.get('target_label', '')} | {r.get('modes', '')} | {r.get('strategy', '')} | {infid_str} | {prob_str} | {r.get('status', 'Success')} |\n")
                else:
                    f.write(f"| {r.get('targets', '')} | {r.get('modes', '')} | {r.get('strategy', '')} | N/A | 0.0% | {r.get('status', 'Failed')} |\n")

    # Read and print report to terminal
    print("\n" + "=" * 80)
    print("CONSOLIDATED MARKDOWN REPORT")
    print("=" * 80)
    with open(report_path, "r") as f:
        print(f.read())
    print("=" * 80)

    print(f"\nSuccessfully merged {len(sorted_indices)} valid jobs. Output saved in:\n{merged_dir}")


if __name__ == "__main__":
    main()
