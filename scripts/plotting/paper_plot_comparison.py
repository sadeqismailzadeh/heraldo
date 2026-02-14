import quantum_agent

import numpy as np
import matplotlib.pyplot as plt
import strawberryfields as sf
from strawberryfields.ops import DensityMatrix
from pathlib import Path
import matplotlib.gridspec as gridspec

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
    # Path to where eval_cma_optimized.py saved the .npy files
    # Assuming this script is in scripts/plotting/ and results are in results/opt_...
    # Update this path to point to your specific results folder containing the .npy files
    base_results_path = Path(__file__).resolve().parent.parent.parent / "results"  
    
    # Auto-find latest opt run or specify manually
    results_dir = base_results_path / "opt_Sq3_GKP_20260205T134849Z" 
    # matches = sorted([p for p in base_results_path.glob("opt_*") if p.is_dir()], key=lambda p: p.stat().st_mtime)
    # results_dir = matches[-1] if matches else None
    
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
    print("Calculating Wigner for (1,3)...")
    (X, P), W_13 = get_wigner_from_dm(rho_13)
    
    print("Calculating Wigner for (3,1)...")
    _, W_31 = get_wigner_from_dm(rho_31)

    # --- PLOTTING ---
    print("Generating Figure...")
    
    # Publication settings
    plt.rcParams.update({
        # "text.usetex": True,  # Use LaTeX if available, otherwise False
        # "font.family": "serif",
        # "font.serif": ["Computer Modern Roman"],
        "font.size": 24
    })
    
    # Fallback if LaTeX not installed
    try:
        plt.plot()
    except:
        plt.rcParams.update({"text.usetex": False, "font.family": "sans-serif"})

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5), dpi=300, sharey=True)
    plt.subplots_adjust(wspace=0.1)

    # Common max value for symmetric colormap
    w_max = max(np.max(np.abs(W_13)), np.max(np.abs(W_31)))
    levels = np.linspace(-w_max, w_max, 100)

    # --- Subplot 1: (1,3) ---
    c1 = ax1.contourf(X, P, W_13, levels=levels, cmap='RdBu_r')
    ax1.set_aspect('equal')
    ax1.set_xlabel(r'$q$', fontsize=28)
    ax1.set_ylabel(r'$p$', fontsize=28)
    ax1.set_title(r'(a) Pattern (1,3)', y=-0.35, fontsize=28) 
    
    # Subtle guidelines
    ax1.axhline(0, color='gray', linestyle=':', alpha=0.4, linewidth=0.8)
    ax1.axvline(0, color='gray', linestyle=':', alpha=0.4, linewidth=0.8)
    
    # --- Subplot 2: (3,1) ---
    c2 = ax2.contourf(X, P, W_31, levels=levels, cmap='RdBu_r')
    ax2.set_aspect('equal')
    ax2.set_xlabel(r'$q$', fontsize=28)
    # ax2.set_ylabel is omitted because sharey=True
    ax2.set_title(r'(b) Pattern (3,1)', y=-0.35, fontsize=28) 
    
    # Subtle guidelines
    ax2.axhline(0, color='gray', linestyle=':', alpha=0.4, linewidth=0.8)
    ax2.axvline(0, color='gray', linestyle=':', alpha=0.4, linewidth=0.8)

    # --- Shared Colorbar ---
    # Create space on the right for one colorbar
    fig.subplots_adjust(right=0.85)
    cbar_ax = fig.add_axes([0.88, 0.15, 0.02, 0.7]) # [left, bottom, width, height]
    cb = fig.colorbar(c2, cax=cbar_ax)
    # cb.set_label(r'$W(q,p)$', rotation=270, labelpad=20, fontsize=24)
    
    # Custom ticks based on the data range (assuming 0.15 scale)
    cb.set_ticks([-0.15, -0.10, -0.05, 0.0, 0.05, 0.10, 0.15])

    # Save
    output_path = results_dir / "Figure3_Comparison.pdf"
    output_png = results_dir / "Figure3_Comparison.png"
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.savefig(output_png, bbox_inches='tight', dpi=300)
    
    print(f"Saved PDF to: {output_path}")
    print(f"Saved PNG to: {output_png}")

if __name__ == "__main__":
    main()
