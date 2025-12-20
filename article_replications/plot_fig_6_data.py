import json
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import numpy as np
import os

def plot_fig6_fixed_bar():
    # 1. Load Data
    try:
        file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'fig6_data.json')
        with open(file, 'r') as f:
            raw_data = json.load(f)
    except FileNotFoundError:
        print("Error: 'fig6_data.json' not found.")
        return

    # Ensure keys are sorted integers
    data = {int(k): v for k, v in raw_data.items()}
    all_n = sorted(data.keys())

    # 2. Setup Plot with CONSTRAINED LAYOUT
    # This is the magic keyword that prevents the colorbar from crashing into the plot
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(8, 10), sharex=True, constrained_layout=True)
    
    # Colormap: Spectral reversed (Blue=1 to Red=12)
    cmap = cm.get_cmap('Spectral_r')
    # Setup normalization from 1 to 12
    norm = mcolors.Normalize(vmin=1, vmax=12)
    
    # 3. Plot Lines
    for n in all_n:
        tau = data[n]['tau']
        fid = data[n]['fid']
        alpha = data[n]['alpha']
        r = data[n]['r']
        
        c = cmap(norm(n))
        
        ax1.plot(tau, fid, color=c, linewidth=1.8)
        ax2.plot(tau, alpha, color=c, linewidth=1.8)
        ax3.plot(tau, r, color=c, linewidth=1.8)

    # 4. Styling
    # Top: Fidelity
    ax1.set_ylabel(r"Optimal Fidelity $\mathcal{F}^{opt}$", fontsize=14)
    ax1.set_ylim(0.35, 1.02) 
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(labelsize=12, direction='in')

    # Middle: Alpha
    ax2.set_ylabel(r"Optimal Size $\alpha$", fontsize=14)
    ax2.set_ylim(-0.2, 7.0)
    ax2.grid(True, alpha=0.3)
    ax2.tick_params(labelsize=12, direction='in')

    # Bottom: Squeezing
    ax3.set_ylabel(r"Optimal Squeezing $r$", fontsize=14)
    ax3.set_xlabel(r"Transmissivity $\tau^2$", fontsize=14)
    ax3.set_ylim(-0.1, 1.7)
    ax3.grid(True, alpha=0.3)
    ax3.tick_params(labelsize=12, direction='in')

    # 5. THE FIXED COLORBAR
    # Create the ScalarMappable
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    
    # attach to [ax1, ax2, ax3]
    # aspect=30 makes it slightly thicker/shorter than before
    cbar = fig.colorbar(sm, ax=[ax1, ax2, ax3], aspect=35, pad=0.02)
    
    cbar.set_label("Number of photons (n)", fontsize=14, labelpad=10)
    
    # Force integer ticks [1, 2, ... 12]
    cbar.set_ticks(np.arange(1, 13))
    cbar.ax.tick_params(labelsize=12)

    # Note: do NOT use plt.tight_layout() here, constrained_layout handles it.
    
    plt.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)), "Figure_6.png"), dpi=300)
    print("Saved to Figure_6.png")
    plt.show()

if __name__ == "__main__":
    plot_fig6_fixed_bar()