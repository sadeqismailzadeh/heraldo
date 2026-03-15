import quantum_agent
from quantum_agent.components.targets import CoreGKPTarget

import numpy as np
import matplotlib.pyplot as plt
import strawberryfields as sf
from strawberryfields.ops import DensityMatrix
from pathlib import Path
import matplotlib.gridspec as gridspec
from   quantum_agent.utils import windows_to_wsl_path


def get_wigner_from_dm(rho, grid_size=200, x_limit=6, cutoff_dim=30):
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

def main():
    # --- CONFIGURATION ---    
    # Update this path to point to your specific results folder
    results_dir = Path(windows_to_wsl_path(r"E:\Quantum\paper\results1\mu1\n=4\opt_Sq3_GKP_20260205T154126Z"))

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
    target = CoreGKPTarget(csv_path=csv_path, n_max=4, delta_db=10, mu=1)
    ket_target = target.get_target_ket(cutoff_dim=30)
    rho_target = np.outer(ket_target, ket_target.conj())
    (X, P), W_target = get_wigner_from_dm(rho_target, cutoff_dim=30)

    # Calculate Wigners for Patterns
    wigner_data = {}
    for p_name, f_path in pattern_files.items():
        print(f"Calculating Wigner for pattern ({p_name.replace('_', ',')})...")
        rho = np.load(f_path)
        _, W = get_wigner_from_dm(rho)
        wigner_data[p_name] = W

    # --- PLOTTING ---
    print("Generating Figure...")
    fontsize = 24
    plt.rcParams.update({"font.size": fontsize})
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 11), dpi=300, sharex=True, sharey=True)
    plt.subplots_adjust(wspace=0.1, hspace=0.4)

    # Common max value for symmetric colormap
    all_ws = [W_target] + list(wigner_data.values())
    w_max = max(np.max(np.abs(w)) for w in all_ws)
    levels = np.linspace(-w_max, w_max, 100)

    # Mapping patterns to grid positions (Row 0: Target, 0_4, 1_3; Row 1: 2_2, 3_1, 4_0)
    plot_list = [
        ("Target", W_target, "(a) Target"),
        ("0_4", wigner_data["0_4"], "(b) Pattern (0,4)"),
        ("1_3", wigner_data["1_3"], "(c) Pattern (1,3)"),
        ("2_2", wigner_data["2_2"], "(d) Pattern (2,2)"),
        ("3_1", wigner_data["3_1"], "(e) Pattern (3,1)"),
        ("4_0", wigner_data["4_0"], "(f) Pattern (4,0)")
    ]

    for idx, (name, W, title) in enumerate(plot_list):
        ax = axes[idx // 3, idx % 3]
        cf = ax.contourf(X, P, W, levels=levels, cmap='RdBu_r')
        ax.set_aspect('equal')
        
        if idx // 3 == 1: # Bottom row
            ax.set_xlabel(r'$q$', fontsize=fontsize)
        if idx % 3 == 0: # Left column
            ax.set_ylabel(r'$p$', fontsize=fontsize)
            
        ax.set_title(title, y=-0.35, fontsize=fontsize)
        ax.axhline(0, color='gray', linestyle=':', alpha=0.4, linewidth=0.8)
        ax.axvline(0, color='gray', linestyle=':', alpha=0.4, linewidth=0.8)

    # --- Shared Colorbar ---
    fig.subplots_adjust(right=0.9)
    cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
    cb = fig.colorbar(cf, cax=cbar_ax)
    cb.set_ticks([-0.15, -0.10, -0.05, 0.0, 0.05, 0.10, 0.15])

    # Save
    output_path = results_dir / "Figure_Comparison_5_Patterns.pdf"
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    print(f"Saved PDF to: {output_path}")

if __name__ == "__main__":
    main()