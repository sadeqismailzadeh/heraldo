# 1. Import the module we need to patch
import scipy.integrate

# 2. Check if the patch is needed to avoid errors
if not hasattr(scipy.integrate, 'simps'):
    print("Monkey patching scipy.integrate: 'simps' not found. Pointing to 'simpson'.")
    # 3. Create the 'simps' attribute and point it to the existing 'simpson' function.
    scipy.integrate.simps = scipy.integrate.simpson
else:
    print("'simps' already exists in scipy.integrate. No patch needed.")

import os
import numpy as np
import matplotlib.pyplot as plt
import strawberryfields as sf
from strawberryfields import ops

from targets import *


def plot_target_wigner(target_instance, name, cutoff_dim=35, grid_size=200, x_limit=5):
    """
    Generates the target state and plots its Wigner function and Fock probabilities.
    Adheres to the style of demo_cubic.py.
    """
    print(f"\n--- Generating Target: {name} ---")
    print(f"Parameters: cutoff_dim={cutoff_dim}, x_limit={x_limit}")
    
    # 1. Generate the Ket Vector
    try:
        ket = target_instance.get_target_ket(cutoff_dim)
    except Exception as e:
        print(f"Error generating target {name}: {e}")
        return

    # Verify normalization
    norm = np.linalg.norm(ket)
    print(f"State Norm: {norm:.4f}")
    if abs(norm - 1.0) > 1e-3:
        print("Warning: State not normalized. Renormalizing...")
        ket = ket / norm

    # 2. Load into Strawberry Fields
    # We use a temporary engine to utilize the built-in Wigner calculator
    prog = sf.Program(1)
    with prog.context as q:
        ops.Ket(ket) | q[0]

    eng = sf.Engine("fock", backend_options={"cutoff_dim": cutoff_dim})
    result = eng.run(prog)
    state = result.state

    # 3. Calculate Wigner Function
    print("Calculating Wigner function...")
    xvec = np.linspace(-x_limit, x_limit, grid_size)
    pvec = np.linspace(-x_limit, x_limit, grid_size)
    W = state.wigner(mode=0, xvec=xvec, pvec=pvec)

    # 4. Calculate Fock Probabilities
    probs = state.all_fock_probs(cutoff=cutoff_dim)

    # 5. Plotting
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Plot 1: Wigner Function
    # Using RdBu colormap centered at 0
    X, P = np.meshgrid(xvec, pvec)
    lim = np.max(np.abs(W))
    c = ax1.pcolormesh(X, P, W, cmap='RdBu', shading='auto', vmin=-lim, vmax=lim)
    fig.colorbar(c, ax=ax1, label='W(x, p)')
    ax1.set_title(f"Wigner Function ({name})")
    ax1.set_xlabel("x (Position)")
    ax1.set_ylabel("p (Momentum)")
    ax1.set_aspect('equal')
    
    # Add grid lines
    ax1.axhline(0, color='black', linestyle='--', alpha=0.3)
    ax1.axvline(0, color='black', linestyle='--', alpha=0.3)

    # Plot 2: Fock Distribution
    # Show only up to cutoff_dim (or limit to 25 for clarity if cutoff is huge)
    display_cutoff = min(cutoff_dim, 60)
    ax2.bar(range(display_cutoff), probs[:display_cutoff], color='teal', alpha=0.7, edgecolor='black')
    ax2.set_title("Fock State Probabilities")
    ax2.set_xlabel("Fock Number |n>")
    ax2.set_ylabel("Probability")
    ax2.set_xticks(range(display_cutoff))

    plt.tight_layout()
    plt.show()

def main():
    # 1. Cubic Phase Target
    # Demonstrating the cubic phase state generation
    # cubic = CubicPhaseTarget(gamma=-0.2, r=-0.7, alpha=1.25)
    # plot_target_wigner(cubic, "Cubic Phase", cutoff_dim=25)

    # 2. Squeezed Cat Target
    # Superposition of coherent states with squeezing
    # cat = SqueezedCatTarget(alpha=3.0, r=1.38, p=1) # p=1 for odd parity
    # plot_target_wigner(cat, "Squeezed Cat (Odd)", cutoff_dim=70)

    # # 3. Square GKP Target
    # # Gottesman-Kitaev-Preskill state (Logical 0)
    # # Note: GKP states often require high cutoff dimensions.
    # gkp_sq = GKPTarget(gkp_type='square', mu=0, delta=0.35)
    # plot_target_wigner(gkp_sq, "Square GKP |0>", cutoff_dim=40, x_limit=6)

    # # 4. Hexagonal GKP Target
    # # Hexagonal lattice GKP (Logical 1)
    # gkp_hex = GKPTarget(gkp_type='hex', mu=1, delta=0.35)
    # plot_target_wigner(gkp_hex, "Hex GKP |1>", cutoff_dim=40, x_limit=6)

    # # 5. Cubic Resource State
    # # Specific resource state for gate synthesis
    # resource = CubicResourceTarget(a=0.61)
    # plot_target_wigner(resource, "Cubic Resource", cutoff_dim=15)

    # # 6. Quartic Phase Target
    # # State with x^4 non-linearity
    # quartic = QuarticPhaseTarget(delta=0.03, s_r=0.0)
    # plot_target_wigner(quartic, "Quartic Phase", cutoff_dim=40)

    csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "GKP_core_coefficients.csv")
    target3=CoreGKPTarget(csv_path=csv_path, 
                          n_max=4, 
                          delta_db=10.4, 
                          mu=0)
    plot_target_wigner(target3, "Quartic Phase", cutoff_dim=30)

if __name__ == "__main__":
    main()