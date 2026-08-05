from pathlib import Path
import numpy as np

from heraldo.components.circuits import TwoModeStaticSqueezeOnly
from heraldo.components.targets import SqueezedCatTarget
from heraldo.components.runner import BasinHoppingRunner
from heraldo.components.objectives import fixed_pattern_capped_loss_fn
from heraldo.utils import db_to_r
from heraldo.serialization import save_results, load_results, reconstruct_objects
from heraldo.analyze import (
    print_results, plot_outcomes,
    analyze_rotations, analyze_loss, analyze_cutoff
)


def main():
    # 1. Initialize 2-mode static spatial circuit with 12 dB squeezing
    circuit = TwoModeStaticSqueezeOnly(clip_size=db_to_r(12.0), measure_fock_cutoff=30)

    # 2. Define targets: Even (|cat_+>) and Odd (|cat_->) Schrödinger cat states
    targets = [
        SqueezedCatTarget(alpha=np.sqrt(6), r=0.5, p=0),
        SqueezedCatTarget(alpha=np.sqrt(6), r=0.5, p=1),
    ]

    # 3. Optimize for fixed patterns n=4 (even) and n=5 (odd)
    runner = BasinHoppingRunner(
        circuit=circuit,
        target_gens=targets,
        cutoff_dim=30,
        # measurement_patterns=[[4], [5]],
        # loss_fn=fixed_pattern_capped_loss_fn,
        penalty_strength=0.1,
        # phase_lock=True,
    )

    result = runner.run(n_iter=5)

    # 5. Save results to pickle file in the example script directory
    save_path = Path(__file__).parent / "example_results.pkl"
    saved_file = save_results(result, filepath=save_path)

    # 6. Load results from the saved file with object reconstruction
    print("\n--- Loading Saved Results ---")
    loaded_result = load_results(saved_file, reconstruct=True)

    # 7. Display loaded results using print_results
    print_results(loaded_result)

    # 8. Analyze phase rotations for specific outcomes (n=4 and n=5)
    print("--- Rotation Analysis ---")
    analyze_rotations(loaded_result, outcomes=[1,2, 3, 4, 5, 6, 7, 8])

    # 9. Analyze impact of photon loss (e.g., ideal 100%, 1% loss, 10% loss)
    print("--- Photon Loss Analysis ---")
    analyze_loss(loaded_result, outcomes=[4, 5])

    # 10. Evaluate Fock cutoff truncation fidelity (e.g., low cutoff 30 vs high cutoff 45)
    print("--- Cutoff Truncation Analysis ---")
    analyze_cutoff(loaded_result, low_cutoff=30, high_cutoff=45, outcomes=[4, 5])

    # 11. Plot Wigner functions and Fock probabilities for specific outcomes (n=4 and n=5)
    print("--- Plotting Outcomes ---")
    plot_outcomes(
        loaded_result,
        outcomes=[4, 5],
        save_prefix=Path(__file__).parent / "example_plot",
        show=True
    )


if __name__ == "__main__":
    main()
