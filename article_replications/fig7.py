
# 1. Import the module we need to patch
import scipy.integrate

# 2. Check if the patch is needed to avoid errors
if not hasattr(scipy.integrate, 'simps'):
    print("Monkey patching scipy.integrate: 'simps' not found. Pointing to 'simpson'.")
    # 3. Create the 'simps' attribute and point it to the existing 'simpson' function.
    scipy.integrate.simps = scipy.integrate.simpson
else:
    print("'simps' already exists in scipy.integrate. No patch needed.")

import json
import numpy as np
import matplotlib.pyplot as plt
import strawberryfields as sf
from strawberryfields import ops
import os

# ==============================================================================
# SECTION 1: PHYSICS (Fast Simulation)
# ==============================================================================

def get_probabilities(tau_sq, cutoff, n_max=12):
    """
    Runs the optical circuit to find the Probability P(n) of detecting n photons.
    
    Why do we need this?
    Figure 6 tells us "How good the state is IF we detect n photons".
    Figure 7 tells us "How good the state is ON AVERAGE".
    To get the average, we need to know how likely each 'n' is.
    
    Cost: Very Cheap (One simulation step, no optimization loop).
    """
    eng = sf.Engine("fock", backend_options={"cutoff_dim": cutoff})
    prog = sf.Program(2)
    
    theta = np.arccos(np.sqrt(tau_sq))
    
    with prog.context as q:
        # Same setup as Fig 6: Two orthogonal squeezed states
        ops.Sgate(-1.38) | q[0] 
        ops.Sgate(1.38) | q[1]  
        
        # Beam Splitter interaction
        ops.BSgate(theta, 0) | (q[0], q[1])
        
    result = eng.run(prog)
    joint_ket = result.state.ket()
    
    probs = {}
    # We only care about n = 1 to 12 (the range used in Fig 6)
    for n in range(1, n_max + 1):
        # Slice the joint state to look at the condition where Mode 1 has n photons
        slice_ket = joint_ket[:, n]
        
        # The squared magnitude of this slice is the Probability P(n)
        p_n = np.sum(np.abs(slice_ket)**2)
        probs[n] = p_n
        
    return probs

# ==============================================================================
# SECTION 2: DATA PROCESSING
# ==============================================================================

def generate_fig7_data():
    print("Loading Figure 6 data...")
    try:
        file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'fig6_data.json')
        with open(file, 'r') as f:
            raw_data = json.load(f)
    except FileNotFoundError:
        print("Error: 'fig6_data.json' missing. Run the Fig 6 code first.")
        return None, None

    # Convert JSON keys to integers
    data = {int(k): v for k, v in raw_data.items()}
    
    # Extract the Tau axis (taking from n=1, assuming all n share the same x-axis)
    tau_values = data[1]['tau']
    
    # Arrays to store the final plot data
    final_taus = []
    average_fidelities = []
    
    print(f"Calculating probabilities for {len(tau_values)} points...")
    
    # Loop through every Transmissivity point in the data
    for i, tau in enumerate(tau_values):
        
        # 1. Calculate P(n) for this specific tau
        # We use cutoff=50 just like in Fig 6 to match precision
        probs_at_tau = get_probabilities(tau, cutoff=50, n_max=12)
        
        weighted_sum = 0.0
        total_prob = 0.0
        
        # 2. Sum over n=1 to 12
        for n in range(1, 13):
            # Get the Optimal Fidelity from the loaded JSON data
            f_opt = data[n]['fid'][i]
            
            # Get the Probability of this happening
            p_n = probs_at_tau[n]
            
            # Accumulate weighted sum
            weighted_sum += p_n * f_opt
            total_prob += p_n
            
        print(f"total prob = {total_prob}")

        # 3. Calculate Weighted Average
        avg_fid = weighted_sum
        
        final_taus.append(tau)
        average_fidelities.append(avg_fid)
            
    return final_taus, average_fidelities

# ==============================================================================
# SECTION 3: PLOTTING
# ==============================================================================

def plot_fig7():
    # Get the calculated data
    tau_axis, avg_fid_axis = generate_fig7_data()
    
    if tau_axis is None: return

    plt.figure(figsize=(7, 5))
    
    # 1. Plot the Main Curve
    plt.plot(tau_axis, avg_fid_axis, color='black', linewidth=2, label='Average Optimal Fidelity')
    
    # 2. Add the "Dot-Dashed Line" at 0.146 (The Secondary Maximum)
    # This marks the specific interference point where the protocol is most efficient.
    plt.axvline(x=0.146, color='red', linestyle='-.', linewidth=1.5, label=r'Secondary Max ($\tau^2=0.146$)')
    
    # 3. Styling to match the article
    plt.xlabel(r"Transmissivity $\tau^2$", fontsize=14)
    plt.ylabel(r"Average Optimal Fidelity $\bar{\mathcal{F}}$", fontsize=14)
    plt.title("Figure 7: Average Fidelity vs Transmissivity", fontsize=14, pad=15)
    
    plt.xlim(0, 0.5)
    # Start Y at 0.8 to zoom in on the relevant high-fidelity region
    plt.ylim(0.0, 1.0) 
    
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    plt.tick_params(labelsize=12, direction='in')
    
    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)), "Figure_7.png"), dpi=300)
    print("Plot saved to Figure_7.png")
    plt.show()

if __name__ == "__main__":
    plot_fig7()