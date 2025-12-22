import numpy as np
import qutip as qt
import matplotlib.pyplot as plt
from quantum_agent.components.rewards import WignerWeightedReward

def plot_wigner(state_vec, ax, title):
    """Helper to plot Wigner function of a state vector."""
    rho = qt.Qobj(state_vec)
    xvec = np.linspace(-5, 5, 100)
    pvec = np.linspace(-5, 5, 100)
    W = qt.wigner(rho, xvec, pvec)
    
    # Plot
    wlim = abs(W).max()
    c = ax.contourf(xvec, pvec, W, 100, cmap='RdBu_r', vmin=-wlim, vmax=wlim)
    ax.set_title(title, fontsize=10)
    ax.set_aspect('equal')
    # Remove ticks for cleaner look
    ax.set_xticks([])
    ax.set_yticks([])

def get_cat_state(alpha, dim, parity='odd'):
    """Returns normalized ket for Cat state."""
    coh_p = qt.coherent(dim, alpha)
    coh_m = qt.coherent(dim, -alpha)
    cat = (coh_p - coh_m) if parity == 'odd' else (coh_p + coh_m)
    return cat.unit().full().flatten()

def check_reward_distortion(dim=20, alpha=2.0, visualize=True):
    print(f"\n{'='*80}")
    print(f" DIAGNOSTIC: FINDING THE 'BREAKING POINT' OF NEG_WEIGHT")
    print(f" Target: Odd Cat State (alpha={alpha}, dim={dim})")
    print(f"{'='*80}\n")
    
    target = get_cat_state(alpha, dim)
    
    # We will test increasing weights to see when the optimal state diverges from the target
    test_weights = [1.0, 2.0, 3.0, 5.0, 10.0, 20.0, 50.0]
    
    if visualize:
        # Setup grid: Target + 7 weights -> 8 plots (2 rows x 4 cols)
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        axes = axes.flatten()
        
        # Plot Target
        plot_wigner(target, axes[0], "Target State (Odd Cat)")

    print(f"{'Neg Weight':<12} | {'Max Eigenval':<12} | {'Target Score':<12} | {'Overlap (Fidelity)':<20} | {'Status'}")
    print("-" * 85)
    
    for i, w in enumerate(test_weights):
        # 1. Build the Operator with ROBUST settings
        # Grid range must be large enough to cover the basis states (Fock ~20 needs ~6.5+)
        # Grid points must be odd and dense.
        rewarder = WignerWeightedReward(
            target, dim, 
            neg_weight=w, 
            pos_weight=1.0, 
            grid_points=201, 
            grid_range=12.0
        )
        op = rewarder.reward_operator
        
        # 2. Find the THEORETICAL GLOBAL MAXIMUM (Max Eigenvector)
        vals, vecs = np.linalg.eigh(op)
        max_eigenval = vals[-1]
        max_eigenvec = vecs[:, -1] # The state that gets the highest possible reward
        
        # 3. Score the Actual Target
        # <target | O | target>
        target_score = np.real(np.vdot(target, op.dot(target)))
        
        # 4. Calculate Overlap (Fidelity)
        # How close is the "Mathematical Max" to our "Desired Target"?
        overlap = np.abs(np.vdot(target, max_eigenvec))**2
        
        # 5. Determine Status
        if overlap > 0.99:
            status = "SAFE"
        elif overlap > 0.95:
            status = "ACCEPTABLE"
        elif overlap > 0.90:
            status = "DISTORTED"
        else:
            status = "BROKEN (Dangerous)"
            
        print(f"{w:<12.1f} | {max_eigenval:<12.4f} | {target_score:<12.4f} | {overlap:<20.4f} | {status}")

        if visualize:
            ax_idx = i + 1
            if ax_idx < len(axes):
                plot_wigner(max_eigenvec, axes[ax_idx], f"Max Eigvec (w={w}, F={overlap:.2f})")

    print("\nCONCLUSION:")
    print("If Status is 'BROKEN', the agent will learn a state that scores high but looks wrong.")
    print("Stick to weights where Overlap > 0.95.\n")
    
    if visualize:
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    check_reward_distortion(visualize=True)
