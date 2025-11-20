# Notes not mentoned in the paper
# there are two targets. odd or even measurement make different parity.
# orthogonal squeezings

# 1. Import the module we need to patch
import scipy.integrate

# 2. Check if the patch is needed to avoid errors
if not hasattr(scipy.integrate, 'simps'):
    print("Monkey patching scipy.integrate: 'simps' not found. Pointing to 'simpson'.")
    # 3. Create the 'simps' attribute and point it to the existing 'simpson' function.
    scipy.integrate.simps = scipy.integrate.simpson
else:
    print("'simps' already exists in scipy.integrate. No patch needed.")

import numpy as np
import matplotlib.pyplot as plt
import strawberryfields as sf
from strawberryfields import ops
from scipy import linalg
import matplotlib.cm as cm

# ==========================================
# PART 1: ROBUST MATH HELPERS (From Agent)
# ==========================================

# --- Helper Function for Computing Matrix Square Root ---
def compute_matrix_sqrt(rho):
    """Return a numerically stable matrix square root of a density matrix.

    Args:
        rho (np.ndarray): Hermitian density matrix for which to compute
            :math:`\sqrt{\rho}`.

    Returns:
        np.ndarray: Hermitian square root of ``rho`` with negative eigenvalues
        clipped to zero.
    """
    rho = np.asarray(rho, dtype=np.complex128)
    
    # Enforce Hermiticity on input to remove numerical noise
    rho = 0.5 * (rho + rho.T.conj())
    
    # eigh is best for Hermitian matrices
    e_vals_rho, e_vecs_rho = np.linalg.eigh(rho)
    
    # Clip small negative eigenvalues to 0 due to numerical instability
    e_vals_rho_clipped = np.maximum(e_vals_rho.real, 0)
    
    # Calculate square root of eigenvalues
    sqrt_e_vals_rho = np.sqrt(e_vals_rho_clipped)
    
    # Reconstruct sqrt(rho) = U * sqrt(D) * U_dagger
    rho_sqrt = e_vecs_rho @ np.diag(sqrt_e_vals_rho) @ e_vecs_rho.T.conj()
    
    return rho_sqrt

# --- Optimized Fidelity Function (with pre-computed sqrt) ---
def fidelity_with_sqrt(rho_sqrt, sigma):
    """Return Uhlmann fidelity using a pre-computed target square root.

    Args:
        rho_sqrt (np.ndarray): Square root of a target density matrix.
        sigma (np.ndarray): Candidate density matrix produced by the agent.

    Returns:
        float: Clipped fidelity value in :math:`[0, 1]`.
    """
    sigma = np.asarray(sigma, dtype=np.complex128)
    
    # Enforce Hermiticity on sigma
    sigma = 0.5 * (sigma + sigma.T.conj())
    
    # Calculate the product matrix K and ensure it's Hermitian
    K = rho_sqrt @ sigma @ rho_sqrt
    K = 0.5 * (K + K.T.conj())
    
    # Calculate Tr(sqrt(K)) robustly
    e_vals_K = np.linalg.eigvalsh(K)
    
    # Clip before the final square root
    e_vals_K_clipped = np.maximum(e_vals_K.real, 0)
    
    # The trace of sqrt(K) is the sum of the square roots of K's eigenvalues
    trace_val = np.sum(np.sqrt(e_vals_K_clipped))
    
    # Calculate and clip final fidelity
    fidelity = trace_val**2
    
    return np.clip(fidelity, 0.0, 1.0)

# ==========================================
# PART 2: TARGET STATE GENERATION
# ==========================================

def precompute_target_sqrts(cutoff_dim):
    """
    Generates the 4 target Squeezed Cat states used by the agent
    and returns their matrix square roots.
    
    Targets:
    1. Even Parity (Standard)
    2. Odd Parity (Standard)
    3. Even Parity (Rotated 90 deg)
    4. Odd Parity (Rotated 90 deg)
    """
    alpha = 3.0
    r = 1.38
    
    target_sqrts = []
    
    # We need a temporary engine to generate these static states
    eng = sf.Engine("fock", backend_options={"cutoff_dim": cutoff_dim})
    # eng = sf.Engine("bosonic")

    # --- Define the 4 Programs ---
    programs = []
    
    # 1. Rho Plus (Even)
    p1 = sf.Program(1)
    with p1.context as q:
        ops.Catstate(alpha, p=0) | q[0]
        ops.Sgate(r) | q[0]
    programs.append(p1)

    # 2. Rho Minus (Odd)
    p2 = sf.Program(1)
    with p2.context as q:
        ops.Catstate(alpha, p=1) | q[0]
        ops.Sgate(r) | q[0]
    programs.append(p2)

    # 3. Rho Plus Rotated
    # p3 = sf.Program(1)
    # with p3.context as q:
    #     ops.Catstate(alpha, p=0) | q[0]
    #     ops.Sgate(r) | q[0]
    #     ops.Rgate(np.pi/2) | q[0]
    # programs.append(p3)

    # # 4. Rho Minus Rotated
    # p4 = sf.Program(1)
    # with p4.context as q:
    #     ops.Catstate(alpha, p=1) | q[0]
    #     ops.Sgate(r) | q[0]
    #     ops.Rgate(np.pi/2) | q[0]
    # programs.append(p4)

    # # --- Run and Compute Sqrt ---
    print("Pre-computing target states...")
    for prog in programs:
        result = eng.run(prog)
        dm = result.state.dm(cutoff=cutoff_dim)
        target_sqrts.append(compute_matrix_sqrt(dm))
        
    return target_sqrts

# ==========================================
# PART 3: SIMULATION LOOP (RECREATING FIG 4)
# ==========================================

def run_simulation():
    # Parameters from Paper/Agent
    CUTOFF_DIM = 25  # Must be high enough for alpha=3 + squeezing
    N_PHOTONS_RANGE = range(1, 13) # n = 1 to 12
    TAU_SQ_VALUES = np.linspace(0.0, 0.5, 50) # X-axis: Transmissivity Squared
    
    # 1. Get Targets
    target_sqrts = precompute_target_sqrts(CUTOFF_DIM)
    
    # Prepare Plotting Data
    results = {} # Store {n: [fidelities...]}

    print(f"Starting simulation over {len(TAU_SQ_VALUES)} points for {len(N_PHOTONS_RANGE)} photon counts.")

    # Loop over detected photon numbers (Each n is a curve)
    for n in N_PHOTONS_RANGE:
        fidelities = []
        
        # Loop over Transmissivity (X-axis)
        for tau_sq in TAU_SQ_VALUES:
            
            # --- A. CIRCUIT SETUP ---
            # Initialize Engine
            eng = sf.Engine("fock", backend_options={"cutoff_dim": CUTOFF_DIM})
            prog = sf.Program(2) # 2 Modes: q[0] (Signal), q[1] (Ancilla/PNR)
            
            # Convert Transmissivity^2 to Theta
            # BS transmission t = cos(theta). We want t^2 = tau_sq.
            # cos(theta) = sqrt(tau_sq) -> theta = arccos(sqrt(tau_sq))
            theta = np.arccos(np.sqrt(tau_sq))
            
            with prog.context as q:
                # 1. Input: Squeezed Vacuum on q[0]
                ops.Sgate(-1.38) | q[0]

                ops.Sgate(1.38) | q[1]

                # 2. Interaction: Variable Beam Splitter
                ops.BSgate(theta, 0) | (q[0], q[1])
                
                # 3. Post-Selection: Force measurement of 'n' photons on q[1]
                ops.MeasureFock(select=[n]) | q[1]

            # --- B. EXECUTION ---
            result = eng.run(prog)
            
            # Get output density matrix of q[0]
            output_dm = result.state.reduced_dm(modes=[0])

            # --- C. FIDELITY CHECK ---
            # Calculate fidelity against ALL 4 targets and take the MAX
            current_fidelities = [
                fidelity_with_sqrt(t_sqrt, output_dm) 
                for t_sqrt in target_sqrts
            ]
            best_fidelity = np.max(current_fidelities)
            fidelities.append(best_fidelity)


        results[n] = fidelities
        print(f"Finished curve for n={n}")

    return TAU_SQ_VALUES, results

# ==========================================
# PART 4: PLOTTING
# ==========================================

def plot_figure_4(x_axis, data_dict):
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Use Turbo colormap (Cool -> Warm)
    colors = cm.turbo(np.linspace(0, 1, len(data_dict)))
    
    # Plot curves
    for i, n in enumerate(data_dict.keys()):
        ax.plot(x_axis, data_dict[n], color=colors[i], linewidth=1.5, label=f"n={n}")

    # Vertical dashed line
    ax.axvline(x=0.12, color='black', linestyle='-.', linewidth=1, label=r"Initial $\tau^2 = 0.12$")

    # Styling
    ax.set_title("Fig 4: Fidelity of Single-Step Output vs Transmissivity")
    ax.set_xlabel(r"Transmissivity $\tau^2$")
    ax.set_ylabel(r"Fidelity $\mathcal{F}$")
    ax.set_ylim(0.0, 1.05)
    ax.set_xlim(0.0, 0.5)
    ax.grid(True, alpha=0.3)
    
    # --- FIX: COLORBAR SETUP ---
    # Create a ScalarMappable to generate the colorbar
    sm = plt.cm.ScalarMappable(cmap=cm.turbo, norm=plt.Normalize(vmin=1, vmax=12))
    sm.set_array([]) # Required for some matplotlib versions
    
    # Explicitly pass the axis 'ax' to steal space from
    fig.colorbar(sm, ax=ax, label="Number of photons (n)")
    
    plt.tight_layout()
    plt.show()

# ==========================================
# MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    x_vals, y_data = run_simulation()
    plot_figure_4(x_vals, y_data)