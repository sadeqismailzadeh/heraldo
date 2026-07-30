"""
Canonical quick-start example, referenced from index.rst, user_guide.rst,
and quickstart_tutorial.rst via `.. literalinclude::`. Edit only here —
the docs pull their code samples from this file, so it's the only place
that needs to change when the API shifts.
"""
import numpy as np

from heraldo.components.circuits import TwoModeTimeDomainSqueezeOnly
from heraldo.components.runner import (
    BasinHoppingRunner,
    beam_search_loss_fn,
    fixed_pattern_capped_loss_fn,
)
from heraldo.components.targets import SqueezedCatTarget
from heraldo.utils import db_to_r

def main():
    # --8<-- [start:setup]
    # 1. Convert 12 dB of source squeezing into the squeezing parameter r
    squeezing = db_to_r(12)

    # 2. Build the circuit: 1 loop mode + 1 ancilla, single spatial stage (T=1)
    circuit = TwoModeTimeDomainSqueezeOnly(
        steps=1,
        time_invariant=False,
        clip_size=squeezing,
        measure_fock_cutoff=30,
    )

    # 3. Define the target(s) to herald
    targets = [
        SqueezedCatTarget(alpha=np.sqrt(6), r=0.5, p=0),  # even cat
        SqueezedCatTarget(alpha=np.sqrt(6), r=0.5, p=1),  # odd cat
    ]
    # --8<-- [end:setup]

    # --8<-- [start:beam-search]
    # Phase 1: unconstrained pattern discovery.
    # measurement_patterns=None triggers Beam Search discovery mode.
    runner_discovery = BasinHoppingRunner(
        circuit=circuit,
        target_gens=targets,
        cutoff_dim=30,
        beam_width=200,  # keep the 200 most-probable branches per step
        penalty_strength=1.0,
        measurement_patterns=None,
        loss_fn=beam_search_loss_fn,
    )
    result_discovery = runner_discovery.run(n_iter=200)

    for branch in result_discovery["branches"]:
        print(branch)
    # --8<-- [end:beam-search]

    # --8<-- [start:fixed-pattern]
    # Phase 2: fix the heralding patterns discovered above (n=4 -> even cat,
    # n=5 -> odd cat) and refine under the capped-fidelity regime.
    patterns = [[(4,)], [(5,)]]

    runner_fixed = BasinHoppingRunner(
        circuit=circuit,
        target_gens=targets,
        cutoff_dim=30,
        beam_width=200,
        penalty_strength=1.0,
        measurement_patterns=patterns,
        loss_fn=fixed_pattern_capped_loss_fn,
    )
    result_fixed = runner_fixed.run(n_iter=20, method="L-BFGS-B")

    print(f"Loss: {result_fixed['loss']:.5f}")
    print(f"Expected fidelity: {result_fixed['expected_fidelity']:.5f}")
    print(f"Total success prob: {result_fixed['total_probability']:.2%}")
    for branch in result_fixed["branches"]:
        print(branch)
    # --8<-- [end:fixed-pattern]


if __name__ == "__main__":
    main()
