# 1. Patch Scipy for compatibility
import scipy.integrate
if not hasattr(scipy.integrate, 'simps'):
    scipy.integrate.simps = scipy.integrate.simpson

import numpy as np
import matplotlib.pyplot as plt
import strawberryfields as sf
from strawberryfields import ops
import os

# ==========================================
# PART 1: MATH HELPERS
# ==========================================

def compute_matrix_sqrt(rho):
    """Compute matrix square root for safe Fidelity calculation."""
    rho = np.asarray(rho, dtype=np.complex128)
    rho = 0.5 * (rho + rho.T.conj())
    e_vals, e_vecs = np.linalg.eigh(rho)
    e_vals_clipped = np.maximum(e_vals.real, 0)
    return e_vecs @ np.diag(np.sqrt(e_vals_clipped)) @ e_vecs.T.conj()

def fidelity_with_sqrt(rho_sqrt, sigma):
    """Calculate Fidelity F = (Tr(sqrt(sqrt(rho) sigma sqrt(rho))))^2"""
    sigma = np.asarray(sigma, dtype=np.complex128)
    sigma = 0.5 * (sigma + sigma.T.conj())
    K = rho_sqrt @ sigma @ rho_sqrt
    K = 0.5 * (K + K.T.conj())
    e_vals = np.linalg.eigvalsh(K)
    trace_val = np.sum(np.sqrt(np.maximum(e_vals.real, 0)))
    return np.clip(trace_val**2, 0.0, 1.0)

# ==========================================
# PART 2: TARGET PREPARATION
# ==========================================

def get_target_sqrts(cutoff_dim):
    """
    Pre-computes the Target Squeezed Cat States.
    Crucial: Uses '3j' (Imaginary) to align with the P-axis (Vertical),
    matching the output of the interference circuit.
    """
    alpha = 3.0 
    r = 1.38
    
    target_sqrts = []
    eng = sf.Engine("fock", backend_options={"cutoff_dim": cutoff_dim})
    
    # Generate Even and Odd parity targets
    for parity in [0, 1]:
        prog = sf.Program(1)
        with prog.context as q:
            ops.Catstate(alpha, p=parity) | q[0]
            ops.Sgate(r) | q[0]
        
        result = eng.run(prog)
        dm = result.state.dm(cutoff=cutoff_dim)
        target_sqrts.append(compute_matrix_sqrt(dm))
        
    return target_sqrts

# ==========================================
# PART 3: SIMULATION (Eq 4 Calculation)
# ==========================================

def calculate_average_fidelity():
    # --- Parameters ---
    CUTOFF = 25  # High cutoff needed for energy conservation
    MAX_N = 20   # Summation limit for Eq 4 (n=1 to 20)
    
    # X-Axis: Transmissivity squared (0 to 0.5 to match Fig 5)
    tau_sq_values = np.linspace(0.0, 0.5, 60)
    
    # Store the Y-axis results
    avg_fidelities = []
    
    # 1. Precompute Targets
    targets = get_target_sqrts(CUTOFF)
    
    print(f"Simulating average fidelity over {len(tau_sq_values)} points...")

    for tau_sq in tau_sq_values:
        
        # --- A. Run Circuit to get Joint State ---
        eng = sf.Engine("fock", backend_options={"cutoff_dim": CUTOFF})
        prog = sf.Program(2)
        
        # Calculate Beam Splitter Angle
        # T = cos^2(theta) -> theta = arccos(sqrt(T))
        theta = np.arccos(np.sqrt(tau_sq))
        
        with prog.context as q:
            # ORTHOGONAL SQUEEZING (The Hidden Requirement)
            # Mode 0: Squeezed along P (Vertical)
            ops.Sgate(-1.38) | q[0]
            # Mode 1: Squeezed along X (Horizontal)
            ops.Sgate(1.38) | q[1]
            
            # Interaction
            ops.BSgate(theta, 0) | (q[0], q[1])
            
            # Note: We DO NOT measure here. We want the full state 
            # to calculate probabilities for ALL n mathematically.

        result = eng.run(prog)
        
        # Get the full two-mode ket vector |Psi>_{0,1}
        # Shape: (CUTOFF, CUTOFF)
        joint_ket = result.state.ket()
        
        # --- B. Calculate Sum(p_n * F_n) ---
        current_avg_fidelity = 0.0
        
        # Loop n from 1 to MAX_N (Discarding n=0 as per paper)
        for n in range(1, MAX_N + 1):
            
            # 1. SLICE: Project Mode 1 onto |n>
            # This extracts the column corresponding to n photons in mode 1
            # resulting in an unnormalized ket for Mode 0.
            slice_ket = joint_ket[:, n] # Shape: (CUTOFF,)
            
            # 2. PROBABILITY (p_n): Squared norm of the slice
            p_n = np.sum(np.abs(slice_ket)**2)
            
            if p_n > 1e-9: # Avoid division by zero
                # 3. NORMALIZE state
                normalized_ket = slice_ket / np.sqrt(p_n)
                
                # Convert to DM for robust fidelity calc
                rho_n = np.outer(normalized_ket, normalized_ket.conj())
                
                # 4. FIDELITY (F_n): Max over targets
                # We check both even and odd targets and take the best match
                f_n = np.max([fidelity_with_sqrt(t, rho_n) for t in targets])
                
                # 5. WEIGHTED SUM (Eq 4)
                current_avg_fidelity += p_n * f_n
                
        avg_fidelities.append(current_avg_fidelity)
        
    return tau_sq_values, avg_fidelities

# ==========================================
# PART 4: PLOTTING
# ==========================================

def plot_figure_5(x, y):
    fig, ax = plt.subplots(figsize=(7, 5))
    
    # Plot the curve
    ax.plot(x, y, color='black', linewidth=1.5, label=r"Average Fidelity $\overline{\mathcal{F}}(\tau)$")
    
    # Plot the vertical line for Agent's choice
    # The paper marks the "relevant secondary maximum" or the chosen point.
    # Fig 5 caption says: "The dot-dashed line marks the initial transmissivity (tau=0.34)"
    # 0.34^2 = 0.1156 approx 0.12
    agent_tau_sq = 0.34**2
    ax.axvline(x=agent_tau_sq, color='black', linestyle='-.', linewidth=1, 
               label=fr"Agent choice $\tau^2 \approx {agent_tau_sq:.2f}$")

    # Formatting to match paper
    ax.set_title("Fig 5: Average Fidelity vs Transmissivity")
    ax.set_xlabel(r"Transmissivity $\tau^2$")
    ax.set_ylabel(r"Average Fidelity $\overline{\mathcal{F}}$")
    
    # Set limits based on visual inspection of Fig 5
    ax.set_xlim(0.0, 0.5)
    ax.set_ylim(0.0, 0.30) # Paper peaks around 0.27
    
    ax.grid(True, which='major', alpha=0.3)
    ax.legend()
    
    plt.tight_layout()

    plt.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)), "Figure_5.png"), dpi=300)
    print("Saved to Figure_5.png")
    plt.show()
if __name__ == "__main__":
    x_vals, y_vals = calculate_average_fidelity()
    plot_figure_5(x_vals, y_vals)