import os
# Set thread limits before importing heavy libraries
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

from pathlib import Path
import numpy as np
from heraldo.components.circuits import TwoModeStaticSqueezeOnly
from heraldo.components.runner import BasinHoppingRunner
from heraldo.components.objectives import fixed_pattern_free_loss_fn
from heraldo.utils import db_to_r
from heraldo.analyze import save_results, load_results, print_results
from custom_targets import CustomFockTarget  # import custom target

def main():
    # 1. 2‑mode static squeeze‑only circuit with 12 dB squeezing
    circuit = TwoModeStaticSqueezeOnly(clip_size=db_to_r(12.0), measure_fock_cutoff=10)

    # 2. Custom target: normalized superposition |0> + |2>
    target = CustomFockTarget(coeffs=[1.0, 0.0, 1.0])

    # 3. Fixed measurement pattern: detect n=2 on the single ancilla mode (mode 1)
    runner = BasinHoppingRunner(
        circuit=circuit,
        target_gens=[target],
        cutoff_dim=10,
        measurement_patterns=[(2,)],
        loss_fn=fixed_pattern_free_loss_fn,
        penalty_strength=0.1
    )

    # 4. Run a brief optimization (2 iterations for demonstration)
    result = runner.run(n_iter=2)

    # 5. Save results
    save_path = Path(__file__).parent / "custom_target_results.pkl"
    save_results(result, filepath=save_path)

    # 6. Load with object reconstruction
    loaded = load_results(save_path, reconstruct=True)

    # 7. Verify that the loaded target is correctly reconstructed
    loaded_target = loaded.get("targets")
    if isinstance(loaded_target, list):
        loaded_target = loaded_target[0]

    cutoff = 10
    ket_original = target.get_target_ket(cutoff)
    ket_loaded = loaded_target.get_target_ket(cutoff)

    overlap = np.vdot(ket_original, ket_loaded)
    fidelity = np.abs(overlap) ** 2
    print(f"Fidelity between original and loaded target ket: {fidelity:.6f}")
    assert np.isclose(fidelity, 1.0, atol=1e-6), "Reconstruction failed!"

    print("Custom target reconstruction verified successfully.")
    print_results(loaded)

if __name__ == "__main__":
    main()