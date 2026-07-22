import quantum_agent
from quantum_agent.components.targets import CoreGKPTarget

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import strawberryfields as sf
from strawberryfields.ops import DensityMatrix
from pathlib import Path
import matplotlib.gridspec as gridspec
from quantum_agent.utils import windows_to_wsl_path

def get_wigner_from_dm(rho, grid_size=150, x_limit=5, cutoff_dim=30):
    """Calculates Wigner function from a density matrix."""
    # Ensure trace is 1
    tr = np.trace(rho)
    if abs(tr) > 1e-9:
        rho = rho / tr
        
    prog = sf.Program(1)
    with prog.context as q:
        DensityMatrix(rho) | q[0]
        
    eng = sf.Engine("fock", backend_options={"cutoff_dim": cutoff_dim})
    result = eng.run(prog)
    
    xvec = np.linspace(-x_limit, x_limit, grid_size)
    pvec = np.linspace(-x_limit, x_limit, grid_size)
    W = result.state.wigner(mode=0, xvec=xvec, pvec=pvec)
    return X_mesh(xvec, pvec), W

def X_mesh(xvec, pvec):
    return np.meshgrid(xvec, pvec)

def plot_3d_wigner(ax, X, P, W, w_abs_max, title, x_limit=5):
    """Helper function to create a 3D surface plot with a 2D projection below."""
    cmap = 'jet'
    
    # Apply a circular mask for the 3D surface to match standard phase-space plots
    R = np.sqrt(X**2 + P**2)
    W_surf = np.copy(W)
    W_surf[R > x_limit] = np.nan
    
    # 3D Surface Plot (Circular)
    surf = ax.plot_surface(X, P, W_surf, cmap=cmap, vmin=-w_abs_max, vmax=w_abs_max,
                           rstride=2, cstride=2, linewidth=0.15, edgecolor='gray', alpha=0.95)
                           
    # 2D Projection (Contour) on the bottom plane (Square base)
    z_offset = -w_abs_max * 1.4 
    ax.contourf(X, P, W, zdir='z', offset=z_offset, cmap=cmap, 
                levels=100, vmin=-w_abs_max, vmax=w_abs_max)
                
    # Set Axis Limits
    ax.set_xlim([-x_limit, x_limit])
    ax.set_ylim([-x_limit, x_limit])
    ax.set_zlim([z_offset, w_abs_max * 1.1])
    
    # Axis Labels (using bold x and p matching reference)
    ax.set_xlabel(r'$\mathbf{x}$', fontsize=26, labelpad=10)
    ax.set_ylabel(r'$\mathbf{p}$', fontsize=26, labelpad=10)
    
    # Adjust ticks to match reference bounds
    ax.set_xticks([-5, 0, 5])
    ax.set_yticks([-5, 0, 5])
    ax.set_zticks([-0.1, 0, 0.1])
    
    ax.tick_params(axis='both', which='major', labelsize=18)
    ax.tick_params(axis='z', which='major', labelsize=18, pad=8)
    
    # Customize Grid lines to be dashed
    ax.xaxis._axinfo["grid"].update({"linewidth": 0.8, "color": "black", "linestyle": "--", "alpha": 0.6})
    ax.yaxis._axinfo["grid"].update({"linewidth": 0.8, "color": "black", "linestyle": "--", "alpha": 0.6})
    ax.zaxis._axinfo["grid"].update({"linewidth": 0.8, "color": "black", "linestyle": "--", "alpha": 0.6})
    
    # Remove axis panes backgrounds for cleaner look
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    
    # Better 3D viewing angle that shows the rotation clearly without flattening
    ax.view_init(elev=28, azim=-55)
    
    # Add Subplot Title
    ax.text2D(0.02, 0.95, title, transform=ax.transAxes, fontsize=32)

def main():
    # --- CONFIGURATION ---

    results_dir = Path(windows_to_wsl_path(r"E:\Quantum\reports\paper\results1\visualize\job_15_GKP_mu=0_Harvesting__1_3____3_1_"))
    if not results_dir:
        print("Could not find results directory.")
        return

    print(f"Looking for data in: {results_dir}")

    file_13 = results_dir / "state_dm_1_3.npy"
    file_31 = results_dir / "state_dm_3_1.npy"

    if not file_13.exists() or not file_31.exists():
        print("Error: Could not find one or both state files.")
        print(f"Missing: {file_13 if not file_13.exists() else ''} {file_31 if not file_31.exists() else ''}")
        print("Please run 'scripts/optimize/eval_cma_optimized.py' twice:")
        print("  1. With measurement = '1;3'")
        print("  2. With measurement = '3;1'")
        return

    # Load Data
    rho_13 = np.load(file_13)
    rho_31 = np.load(file_31)

    # Calculate Wigner
    # Note: Reduced grid size slightly to 150 for smoother/faster 3D rendering
    print("Calculating Wigner for (1,3)...")
    (X, P), W_13 = get_wigner_from_dm(rho_13, grid_size=150)
    
    print("Calculating Wigner for (3,1)...")
    _, W_31 = get_wigner_from_dm(rho_31, grid_size=150)

    # --- TARGET STATE ---
    print("Calculating Wigner for Target...")
    csv_path = Path(__file__).resolve().parent.parent.parent / "data" / "GKP_core_coefficients.csv"
    target = CoreGKPTarget(csv_path=csv_path, n_max=4, delta_db=10, mu=0)
    ket_target = target.get_target_ket(cutoff_dim=30)
    rho_target = np.outer(ket_target, ket_target.conj())
    _, W_target = get_wigner_from_dm(rho_target, grid_size=150, cutoff_dim=30)

    # --- PLOTTING ---
    print("Generating Figure...")
    fontsize = 32
    # Publication settings
    plt.rcParams.update({
        "font.size": fontsize
    })
    
    # Fallback if LaTeX not installed
    try:
        plt.plot()
    except:
        plt.rcParams.update({"text.usetex": False, "font.family": "sans-serif"})

    # Setup the figure for 3 side-by-side 3D plots
    fig = plt.figure(figsize=(24, 7), dpi=300)
    plt.subplots_adjust(left=0.01, right=0.99, bottom=0.05, top=0.95, wspace=0.05)
    
    ax0 = fig.add_subplot(1, 3, 1, projection='3d')
    ax1 = fig.add_subplot(1, 3, 2, projection='3d')
    ax2 = fig.add_subplot(1, 3, 3, projection='3d')

    # Common max absolute value for symmetric colormap zero-point alignment
    w_max = max(np.max(np.abs(W_target)), np.max(np.abs(W_13)), np.max(np.abs(W_31)))

    # --- Draw Subplots ---
    plot_3d_wigner(ax0, X, P, W_target, w_max, 'a)')
    plot_3d_wigner(ax1, X, P, W_13, w_max, 'b)')
    plot_3d_wigner(ax2, X, P, W_31, w_max, 'c)')

    # Save
    output_path = results_dir / "Figure3_Comparison.pdf"
    output_png = results_dir / "Figure3_Comparison.png"
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.savefig(output_png, bbox_inches='tight', dpi=300)
    
    print(f"Saved PDF to: {output_path}")
    print(f"Saved PNG to: {output_png}")

if __name__ == "__main__":
    main()
