import os

# --- Set thread limits for NumPy/OpenBLAS/MKL before importing heavy backend libraries ---
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'


from pathlib import Path

from heraldo.components.circuits import ThreeModeStaticSqueezeOnly
from heraldo.components.targets import CoreGKPTarget
from heraldo.components.runner import BasinHoppingRunner
from heraldo.components.objectives import fixed_pattern_free_loss_fn
from heraldo.utils import db_to_r
from heraldo.analyze import (
    save_results, load_results, print_results, plot_outcomes,
    analyze_rotations, analyze_loss, analyze_cutoff
)


def main():
    # 1. Initialize 3-mode static spatial circuit with 12 dB squeezing
    circuit = ThreeModeStaticSqueezeOnly(clip_size=db_to_r(12.0), measure_fock_cutoff=30)

    # 2. Define target: Gottesman-Kitaev-Preskill (GKP) core state |0_A4> (mu=0, n_max=4)
    csv_path = Path(__file__).resolve().parent.parent.parent / "data" / "GKP_core_coefficients.csv"
    targets = [
        CoreGKPTarget(csv_path=csv_path, n_max=4, mu=0, delta_db=10.0),
    ]

    # 3. Define fixed measurement patterns (n1, n2) for ancillary modes 1 and 2
    patterns = [(1, 3), (3, 1), (2, 2)]

    # 4. Configure BasinHoppingRunner for fixed-pattern harvesting optimization
    runner = BasinHoppingRunner(
        circuit=circuit,
        target_gens=targets,
        cutoff_dim=30,
        measurement_patterns=patterns,
        loss_fn=fixed_pattern_free_loss_fn,
        penalty_strength=0.1
    )

    # 5. Run the optimization
    result = runner.run(n_iter=5)

    # 6. Save results to pickle file in the script directory
    save_path = Path(__file__).parent / "example_3mode_gkp_results.pkl"
    saved_file = save_results(result, filepath=save_path)

    # 7. Load results with object reconstruction
    print("\n--- Loading Saved Results ---")
    loaded_result = load_results(saved_file, reconstruct=True)

    # 8. Display loaded results summary
    print_results(loaded_result)

    # 9. Analyze phase rotations across outcomes (1,3), (3,1), and (2,2)
    print("--- Rotation Analysis ---")
    analyze_rotations(loaded_result, outcomes=patterns)

    # 10. Analyze performance under photon loss (0%, 1%, 10%)
    print("--- Photon Loss Analysis ---")
    analyze_loss(loaded_result, outcomes=patterns)

    # 11. Evaluate Fock cutoff truncation fidelity (cutoff 30 vs 45)
    print("--- Cutoff Truncation Analysis ---")
    analyze_cutoff(loaded_result, low_cutoff=30, high_cutoff=45, outcomes=patterns)

    # 12. Plot Wigner functions and Fock probabilities for all harvested outcomes
    print("--- Plotting Outcomes ---")
    plot_outcomes(
        loaded_result,
        outcomes=patterns,
        save_prefix=Path(__file__).parent / "example_3mode_gkp_plot",
        show=True
    )


if __name__ == "__main__":
    main()
