from pathlib import Path
import numpy as np

from heraldo.components.circuits import TwoModeStaticSqueezeOnly
from heraldo.components.targets import SqueezedCatTarget
from heraldo.components.runner import BasinHoppingRunner
from heraldo.components.objectives import fixed_pattern_capped_loss_fn
from heraldo.utils import db_to_r
from heraldo.analyze import save_results, load_results, print_results


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
        penalty_strength=0
    )

    result = runner.run(n_iter=5)

    # 5. Save results to pickle file in the example script directory
    save_path = Path(__file__).parent / "example_results.pkl"
    saved_file = save_results(result, filepath=save_path)

    # 6. Load results from the saved file
    print("\n--- Loading Saved Results ---")
    loaded_result = load_results(saved_file)

    # 7. Display loaded results using print_results
    print_results(loaded_result)


if __name__ == "__main__":
    main()
