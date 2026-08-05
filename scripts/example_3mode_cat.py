import os

# --- Set thread limits for NumPy/OpenBLAS/MKL before importing heavy backend libraries ---
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'


from pathlib import Path
import numpy as np

from heraldo.components.circuits import ThreeModeStaticSqueezeOnly
from heraldo.components.targets import SqueezedCatTarget
from heraldo.components.runner import BasinHoppingRunner
from heraldo.components.objectives import fixed_pattern_free_loss_fn,fixed_pattern_capped_loss_fn
from heraldo.utils import db_to_r
from heraldo.serialization import save_results, load_results
from heraldo.analyze import (
    print_results, plot_outcomes,
    analyze_rotations, analyze_loss, analyze_cutoff
)


def main():
    # 1. Initialize 3-mode static spatial circuit with 12 dB squeezing
    circuit = ThreeModeStaticSqueezeOnly(clip_size=db_to_r(12.0), measure_fock_cutoff=30)

    # 2. Define targets: Even (|cat_+>) and Odd (|cat_->) Squeezed Cat states (alpha=sqrt(6), r=0.5)
    targets = [
        SqueezedCatTarget(alpha=np.sqrt(6), r=0.5, p=0),  # Even cat state (|cat_+>)
        SqueezedCatTarget(alpha=np.sqrt(6), r=0.5, p=1),  # Odd cat state (|cat_->)
    ]

    # 3. Define fixed measurement patterns (n1, n2) for ancillary modes 1 and 2
    # Patterns with sum(n_i) = 4 map to |cat_+>, patterns with sum(n_i) = 5 map to |cat_->
    patterns_sum_4 = [(0, 4), (1, 3), (2, 2), (3, 1), (4, 0)]          # sum = 4 (Even cat)
    patterns_sum_5 = [(0, 5), (1, 4), (2, 3), (3, 2), (4, 1), (5, 0)]  # sum = 5 (Odd cat)
    patterns = patterns_sum_4 + patterns_sum_5

    # 4. Configure BasinHoppingRunner for fixed-pattern resource multiplexing optimization
    runner = BasinHoppingRunner(
        circuit=circuit,
        target_gens=targets,
        cutoff_dim=30,
        measurement_patterns=patterns,
        loss_fn=fixed_pattern_capped_loss_fn,
        penalty_strength=0.1,
        phase_lock=True,
    )

    # 5. Run the optimization
    result = runner.run(n_iter=5)

    # 6. Save results to pickle file in the script directory
    save_path = Path(__file__).parent / "example_3mode_cat_results.pkl"
    saved_file = save_results(result, filepath=save_path)

    # 7. Load results with object reconstruction
    print("\n--- Loading Saved Results ---")
    loaded_result = load_results(saved_file, reconstruct=True)

    # 8. Display loaded results summary
    print_results(loaded_result)

    # 9. Analyze phase rotations across outcomes
    print("--- Rotation Analysis ---")
    analyze_rotations(loaded_result, outcomes=patterns)

    # 10. Analyze performance under photon loss (0%, 1%, 10%)
    print("--- Photon Loss Analysis ---")
    analyze_loss(loaded_result, outcomes=patterns)

    # 11. Evaluate Fock cutoff truncation fidelity (cutoff 30 vs 45)
    print("--- Cutoff Truncation Analysis ---")
    analyze_cutoff(loaded_result, low_cutoff=30, high_cutoff=45, outcomes=patterns)

    # 12. Plot Wigner functions and Fock probabilities for all multiplexed outcomes
    print("--- Plotting Outcomes ---")
    plot_outcomes(
        loaded_result,
        outcomes=patterns,
        save_prefix=Path(__file__).parent / "example_3mode_cat_plot",
        show=True
    )


if __name__ == "__main__":
    main()
