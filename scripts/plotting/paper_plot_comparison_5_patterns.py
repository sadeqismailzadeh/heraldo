import heraldo
from heraldo.components.targets import CoreGKPTarget, CatTarget, SqueezedCatTarget

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import strawberryfields as sf
from strawberryfields.ops import DensityMatrix
from pathlib import Path
import matplotlib.gridspec as gridspec
from heraldo.utils import windows_to_wsl_path
from matplotlib.colors import TwoSlopeNorm


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

def plot_3d_wigner(ax, X, P, W, title, x_limit=5, z_min=-0.20, z_max=0.10):
    """Helper function to create a 3D surface plot with a 2D contour projection below."""
    cmap = 'RdYlGn'
    
    # Diverging norm to ensure zero maps to yellow/orange, positive to green, negative to red
    norm = TwoSlopeNorm(vmin=z_min, vcenter=0.0, vmax=z_max)
    z_offset = z_min
    
    # 3D Surface Plot
    surf = ax.plot_surface(X, P, W, cmap=cmap, norm=norm,
                           rstride=2, cstride=2, linewidth=0, edgecolor='none',
                           antialiased=True, alpha=0.9)
                           
    # 2D Projection (Contour lines) on the bottom plane
    levels = np.linspace(z_min, z_max, 7)
    linestyles = ['dashed' if lvl < 0 else 'solid' for lvl in levels]
    ax.contour(X, P, W, zdir='z', offset=z_offset, cmap=cmap, norm=norm,
               levels=levels, linewidths=2, linestyles=linestyles)
                
    # Set Axis Limits
    ax.set_xlim([-x_limit, x_limit])
    ax.set_ylim([-x_limit, x_limit])
    ax.set_zlim([z_offset, z_max])
    
    # Axis Labels
    ax.set_xlabel(r'$\mathbf{q}$', fontsize=32, labelpad=10)
    ax.set_ylabel(r'$\mathbf{p}$', fontsize=32, labelpad=10)
    
    # Adjust ticks matching reference height and limits
    ax.set_xticks([-4,  0,  4])
    ax.set_yticks([-4,  0,  4])
    ax.set_zticks([-0.10,  0.00,  0.10])
    
    ax.tick_params(axis='x', which='major', labelsize=24, pad=0)
    ax.tick_params(axis='y', which='major', labelsize=24, pad=0)
    ax.tick_params(axis='z', which='major', labelsize=24, pad=8)
    
    # Grid lines styling
    ax.xaxis._axinfo["grid"].update({"linewidth": 0.6, "color": "gray", "linestyle": "--", "alpha": 0.5})
    ax.yaxis._axinfo["grid"].update({"linewidth": 0.6, "color": "gray", "linestyle": "--", "alpha": 0.5})
    ax.zaxis._axinfo["grid"].update({"linewidth": 0.6, "color": "gray", "linestyle": "--", "alpha": 0.5})
    
    # Clean axis pane backgrounds
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    
    # Orientation and Viewing Angle matching reference image
    ax.view_init(elev=28, azim=-60)
    
    # Subplot Title
    ax.text2D(0.02, 0.95, title, transform=ax.transAxes, fontsize=32)

def main():
    # --- CONFIGURATION ---    
    results_dir = Path(windows_to_wsl_path(r"E:\Quantum\reports\paper\results1\visualize\job_20_Cat_Harvesting_5_patterns"))

    if not results_dir.exists():
        print(f"Could not find results directory: {results_dir}")
        return

    patterns = ["0_4", "1_3", "2_2", "3_1", "4_0"]
    pattern_files = {p: results_dir / f"state_dm_{p}.npy" for p in patterns}

    # Verify files
    missing = [p for p, f in pattern_files.items() if not f.exists()]
    if missing:
        print(f"Error: Missing state files for patterns: {missing}")
        return

    # Calculate Wigner for Target
    print("Calculating Wigner for Target...")
    csv_path = Path(__file__).resolve().parent.parent.parent / "data" / "GKP_core_coefficients.csv"
    target = SqueezedCatTarget(alpha=np.sqrt(6), r=0.5, p=0)
    ket_target = target.get_target_ket(cutoff_dim=30)
    rho_target = np.outer(ket_target, ket_target.conj())
    (X, P), W_target = get_wigner_from_dm(rho_target, grid_size=250, x_limit=5, cutoff_dim=30)

    # Calculate Wigners for Patterns
    wigner_data = {}
    for p_name, f_path in pattern_files.items():
        print(f"Calculating Wigner for pattern ({p_name.replace('_', ',')})...")
        rho = np.load(f_path)
        _, W = get_wigner_from_dm(rho, grid_size=250, x_limit=5, cutoff_dim=30)
        wigner_data[p_name] = W

    # --- PLOTTING ---
    print("Generating Figure...")
    # fontsize = 32
    # plt.rcParams.update({
    #     "font.size": fontsize
    # })
    
    try:
        plt.plot()
    except Exception:
        plt.rcParams.update({"text.usetex": False, "font.family": "sans-serif"})

    fig = plt.figure(figsize=(22, 14))
    plt.subplots_adjust(left=0.01, right=0.98, bottom=0.02, top=0.98, wspace=0.00, hspace=0.05)

    plot_list = [
        ("(a) Target", W_target),
        ("(b) Pattern (0,4)", wigner_data["0_4"]),
        ("(c) Pattern (1,3)", wigner_data["1_3"]),
        ("(d) Pattern (2,2)", wigner_data["2_2"]),
        ("(e) Pattern (3,1)", wigner_data["3_1"]),
        ("(f) Pattern (4,0)", wigner_data["4_0"])
    ]

    for idx, (title, W) in enumerate(plot_list):
        ax = fig.add_subplot(2, 3, idx + 1, projection='3d')
        plot_3d_wigner(ax, X, P, W, title)

    # Save outputs
    output_path = results_dir / "Figure5_Comparison.pdf"
    output_eps = results_dir / "Figure5_Comparison.eps"
    output_jpg = results_dir / "Figure5_Comparison.jpg"

    # plt.savefig(output_path, bbox_inches='tight', dpi=300)
    # plt.savefig(output_eps, bbox_inches='tight', dpi=600)
    plt.savefig(output_jpg, bbox_inches='tight', dpi=300)

    print(f"Saved PDF to: {output_path}")
    print(f"Saved EPS to: {output_eps}")
    print(f"Saved JPG to: {output_jpg}")

if __name__ == "__main__":
    main()
