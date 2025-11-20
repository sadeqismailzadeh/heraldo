import json
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

def plot_fig6():
    # 1. Load Data
    try:
        with open('fig6_data.json', 'r') as f:
            raw_data = json.load(f)
    except FileNotFoundError:
        print("Error: 'fig6_data.json' not found. Run step1_generate_data.py first.")
        return

    # Convert JSON keys (strings) back to integers
    data = {int(k): v for k, v in raw_data.items()}
    all_n = sorted(data.keys())

    # 2. Setup Plot
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(8, 10), sharex=True)
    colors = cm.turbo(np.linspace(0, 1, len(all_n)))

    # 3. Plot Lines
    for i, n in enumerate(all_n):
        tau = data[n]['tau']
        
        # Top: Fidelity
        ax1.plot(tau, data[n]['fid'], color=colors[i], linewidth=1.5)
        
        # Middle: Alpha
        ax2.plot(tau, data[n]['alpha'], color=colors[i], linewidth=1.5)
        
        # Bottom: Squeezing
        ax3.plot(tau, data[n]['r'], color=colors[i], linewidth=1.5)

    # 4. Styling (Tweaked for Publication Quality)
    
    # Top Panel
    ax1.set_ylabel(r"Optimal Fidelity $\mathcal{F}^{opt}$", fontsize=12)
    ax1.set_ylim(0.85, 1.005) # Focus on the high fidelity area
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(axis='both', which='major', labelsize=10)

    # Middle Panel
    ax2.set_ylabel(r"Optimal Size $\alpha$", fontsize=12)
    ax2.set_ylim(0, 4.5)
    ax2.grid(True, alpha=0.3)

    # Bottom Panel
    ax3.set_ylabel(r"Optimal Squeezing $r$", fontsize=12)
    ax3.set_xlabel(r"Transmissivity $\tau^2$", fontsize=12)
    ax3.set_ylim(0, 1.8)
    ax3.grid(True, alpha=0.3)

    # 5. Colorbar
    sm = plt.cm.ScalarMappable(cmap=cm.turbo, norm=plt.Normalize(vmin=min(all_n), vmax=max(all_n)))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=[ax1, ax2, ax3], pad=0.05, aspect=40)
    cbar.set_label("Number of photons (n)", fontsize=12)

    ax1.set_title("Figure 6: Optimized One-Step Generation", fontsize=14, pad=10)
    
    plt.tight_layout()
    plt.savefig("Figure_6_Recreation.png", dpi=300) # Saves high-res image
    plt.show()

if __name__ == "__main__":
    plot_fig6()