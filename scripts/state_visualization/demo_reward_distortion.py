# 1. Import the module we need to patch
import scipy.integrate

# 2. Check if the patch is needed to avoid errors
if not hasattr(scipy.integrate, 'simps'):
    print("Monkey patching scipy.integrate: 'simps' not found. Pointing to 'simpson'.")
    # 3. Create the 'simps' attribute and point it to the existing 'simpson' function.
    scipy.integrate.simps = scipy.integrate.simpson
else:
    print("'simps' already exists in scipy.integrate. No patch needed.")

import numpy as np
import qutip as qt
import matplotlib.pyplot as plt
from pathlib import Path

# Import all available targets
from quantum_agent.components.targets import (
    GKPTarget, 
    SqueezedCatTarget, 
    CubicPhaseTarget, 
    QuarticPhaseTarget, 
    CubicResourceTarget,
    CoreGKPTarget
)
from quantum_agent.components.rewards import WignerWeightedReward

def plot_wigner(state_vec, ax, title):
    """Helper to plot Wigner function of a state vector."""
    rho = qt.Qobj(state_vec)
    # Grid range covers most features for typical states
    xvec = np.linspace(-6, 6, 100)
    pvec = np.linspace(-6, 6, 100)
    W = qt.wigner(rho, xvec, pvec)
    
    # Plot
    wlim = np.max(np.abs(W))
    c = ax.contourf(xvec, pvec, W, 100, cmap='RdBu_r', vmin=-wlim, vmax=wlim)
    ax.set_title(title, fontsize=10)
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])

def diagnose_target(target_instance, name, cutoff_dim=25, test_weights=None, visualize=True):
    """
    Generates the target state, constructs WignerWeightedReward operators for various 
    negative weights, and checks if the eigenvector with the maximum eigenvalue 
    (the 'optimal' state) still overlaps well with the intended target.
    """
    if test_weights is None:
        test_weights = [1.0, 2.0, 3.0, 5.0, 10.0, 20.0, 50.0]

    print(f"\n{'='*80}")
    print(f" DIAGNOSTIC: Reward Distortion for '{name}'")
    print(f" Cutoff: {cutoff_dim}")
    print(f"{'='*80}\n")

    # 1. Get Target Ket
    try:
        target_ket = target_instance.get_target_ket(cutoff_dim)
        # Ensure normalization
        target_ket = target_ket / np.linalg.norm(target_ket)
    except Exception as e:
        print(f"Error generating target: {e}")
        return

    # Setup Plotting
    if visualize:
        # 1 target plot + len(weights) plots
        n_plots = 1 + len(test_weights)
        cols = 4
        rows = (n_plots + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
        axes = axes.flatten()
        
        # Hide unused axes
        for i in range(n_plots, len(axes)):
            axes[i].axis('off')

        plot_wigner(target_ket, axes[0], f"Target: {name}")
    
    print(f"{'Neg Weight':<12} | {'Max Eigenval':<12} | {'Target Score':<12} | {'Overlap (Fidelity)':<20} | {'Status'}")
    print("-" * 85)

    for i, w in enumerate(test_weights):
        # 2. Build Reward Operator
        # Note: grid_range/points might need adjustment for very spread-out states like GKP
        rewarder = WignerWeightedReward(
            target_ket, cutoff_dim, 
            neg_weight=w, 
            pos_weight=1.0, 
            grid_points=201, 
            grid_range=12.0
        )
        op = rewarder.reward_operator
        
        # 3. Find Global Max (Max Eigenvector)
        # numpy.linalg.eigh returns eigenvalues in ascending order
        vals, vecs = np.linalg.eigh(op)
        max_eigenval = vals[-1]
        max_eigenvec = vecs[:, -1]
        
        # 4. Score Target
        # <target | O | target>
        target_score = np.real(np.vdot(target_ket, op.dot(target_ket)))
        
        # 5. Calculate Overlap
        overlap = np.abs(np.vdot(target_ket, max_eigenvec))**2
        
        # 6. Status
        if overlap > 0.99: status = "SAFE"
        elif overlap > 0.95: status = "ACCEPTABLE"
        elif overlap > 0.90: status = "DISTORTED"
        else: status = "BROKEN"
            
        print(f"{w:<12.1f} | {max_eigenval:<12.4f} | {target_score:<12.4f} | {overlap:<20.4f} | {status}")
        
        if visualize:
            plot_wigner(max_eigenvec, axes[i+1], f"w={w}, F={overlap:.3f}")

    if visualize:
        plt.tight_layout()
        plt.show()

def main():
    # --- 1. Cubic Phase ---
    # Standard Cubic Phase State
    # cubic = CubicPhaseTarget(gamma=-0.2, r=-0.7, alpha=1.25)
    # diagnose_target(cubic, "Cubic Phase", cutoff_dim=25)

    # --- 2. Squeezed Cat ---
    # Squeezed Cat State (Odd parity)
    # cat = SqueezedCatTarget(alpha=3.0, r=1.38, p=1)
    # diagnose_target(cat, "Squeezed Cat (alpha=3, r=1.38)", cutoff_dim=30)

    # --- 3. Square GKP ---
    # Note: GKP usually requires higher cutoff dimensions to be well-defined
    # gkp = GKPTarget(gkp_type='square', mu=0, delta=0.35)
    # diagnose_target(gkp, "Square GKP |0>", cutoff_dim=40)

    # --- 4. Core GKP (from CSV) ---
    # Requires the GKP_core_coefficients.csv file in the data folder
    csv_path = Path(__file__).resolve().parent.parent.parent / "data" / "GKP_core_coefficients.csv"
    if csv_path.exists():
        core_gkp = CoreGKPTarget(csv_path=str(csv_path), n_max=4, delta_db=10.0, mu=0, apply_squeezing=True)
        diagnose_target(core_gkp, "Core GKP (Delta=10dB)", cutoff_dim=6)
    else:
        print(f"Skipping Core GKP: CSV not found at {csv_path}")

    # --- 5. Cubic Resource ---
    # resource = CubicResourceTarget(a=0.61)
    # diagnose_target(resource, "Cubic Resource", cutoff_dim=15)

if __name__ == "__main__":
    main()
