# 1. Import the module we need to patch
import scipy.integrate

# 2. Check if the patch is needed to avoid errors
if not hasattr(scipy.integrate, 'simps'):
    scipy.integrate.simps = scipy.integrate.simpson

import os
import numpy as np
from pathlib import Path
import strawberryfields as sf
from strawberryfields import ops

from quantum_agent.components.targets import CoreGKPTarget


def simulate_with_loss(ket, loss_rate, cutoff_dim):
    """
    Simulates the state preparation with a given loss rate and returns the state density matrix.
    """
    transmission = 1.0 - loss_rate
    
    prog = sf.Program(1)
    with prog.context as q:
        ops.Ket(ket) | q[0]
        if loss_rate > 0.0:
            ops.LossChannel(transmission) | q[0]
            
    eng = sf.Engine("fock", backend_options={"cutoff_dim": cutoff_dim})
    result = eng.run(prog)
    return result.state.dm()


def calculate_infidelity(ket, dm):
    """
    Calculates the infidelity between a pure ket and a density matrix.
    Fidelity F = <ket| dm |ket>
    Infidelity = 1 - F
    """
    fidelity = np.real(np.conj(ket) @ dm @ ket)
    return 1.0 - fidelity


def main():
    cutoff_dim = 30
    
    # Locate the CSV path
    csv_path = Path(__file__).resolve().parent.parent.parent / "data" / "GKP_core_coefficients.csv"
    
    print(f"Loading GKP Core coefficients from: {csv_path}")
    
    # Initialize CoreGKPTarget
    target = CoreGKPTarget(
        csv_path=csv_path,
        n_max=4,
        delta_db=10,
        mu=1
    )
    
    # Get the target ket vector
    ket = target.get_target_ket(cutoff_dim)
    
    # Normalize ket
    norm = np.linalg.norm(ket)
    if abs(norm - 1.0) > 1e-3:
        ket = ket / norm
        
    # Define conditions to simulate
    conditions = [
        {"name": "Ideal", "loss_rate": 0.0},
        {"name": "0.001% Loss", "loss_rate": 1e-5},
        {"name": "0.01% Loss", "loss_rate": 1e-4},
        {"name": "0.1% Loss", "loss_rate": 1e-3},
        {"name": "1% Loss", "loss_rate": 1e-2},
        {"name": "10% Loss", "loss_rate": 0.1}
    ]
    
    results = []
    
    print("Running simulations...")
    for cond in conditions:
        loss_rate = cond["loss_rate"]
        name = cond["name"]
        
        dm = simulate_with_loss(ket, loss_rate, cutoff_dim)
        infidelity = calculate_infidelity(ket, dm)
        
        results.append({
            "Condition": name,
            "Loss Rate": loss_rate,
            "Transmission": 1.0 - loss_rate,
            "Infidelity": infidelity
        })
        
    # Print the markdown table
    print("\n### Simulation Results: CoreGKPTarget under Loss Conditions")
    print("| Condition | Loss Rate | Transmission | Infidelity |")
    print("| :--- | :--- | :--- | :--- |")
    for r in results:
        loss_str = f"{r['Loss Rate']:.5f}" if r['Loss Rate'] > 0 else "0.0"
        trans_str = f"{r['Transmission']:.5f}"
        # Format infidelity beautifully (scientific notation if small)
        inf_val = r['Infidelity']
        if inf_val < 1e-9:
            inf_str = "0.0"
        elif inf_val < 1e-3:
            inf_str = f"{inf_val:.2e}"
        else:
            inf_str = f"{inf_val:.6f}"
            
        print(f"| {r['Condition']} | {loss_str} | {trans_str} | {inf_str} |")


if __name__ == "__main__":
    main()
