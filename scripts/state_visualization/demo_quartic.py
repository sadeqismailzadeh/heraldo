# 1. Import the module we need to patch
import scipy.integrate
import scipy.linalg  # Needed for matrix exponentiation

# 2. Check if the patch is needed
if not hasattr(scipy.integrate, 'simps'):
    print("Monkey patching scipy.integrate: 'simps' not found. Pointing to 'simpson'.")
    scipy.integrate.simps = scipy.integrate.simpson
else:
    print("'simps' already exists in scipy.integrate. No patch needed.")

import numpy as np
import matplotlib.pyplot as plt
import strawberryfields as sf
from strawberryfields import ops
from strawberryfields.ops import *

def get_position_operator_matrix(cutoff):
    """
    Constructs the position operator x = (a + a^dagger) / sqrt(2) 
    in the Fock basis up to the cutoff dimension.
    """
    # Create annihilation operator matrix a
    # a |n> = sqrt(n) |n-1>
    diag_vals = np.sqrt(np.arange(1, cutoff))
    a = np.diag(diag_vals, k=1)
    adag = a.T
    
    # Position operator
    x = (a + adag) / np.sqrt(2)
    return x

def plot_quartic_phase_target():
    # --- 1. CONFIGURATION ---
    # Quartic states require higher cutoffs (as noted in the paper, ~60+)
    cutoff_dim = 80  
    grid_size = 200
    x_limit = 6
    
    # Parameters for Quartic Phase
    # State = exp(i * delta * x^4) S(r) |0>
    squeezing = 1.38 # Squeezing parameter
    delta = 0.05     # Quarticity strength (keep small to avoid aliasing in finite cutoff)

    print(f"Generating Quartic Phase State with delta={delta}, r={squeezing}, cutoff={cutoff_dim}")

    # --- 2. GENERATE INITIAL SQUEEZED STATE ---
    # We use SF to generate the squeezed vacuum first
    prog_init = sf.Program(1)
    with prog_init.context as q:
        Sgate(-squeezing) | q[0]
        ops.Rgate(-0.) | q[0]

    eng = sf.Engine("fock", backend_options={"cutoff_dim": cutoff_dim})
    state_init = eng.run(prog_init).state
    ket_init = state_init.ket()

    # --- 3. APPLY QUARTIC OPERATOR MANUALLY ---
    # Since there is no Vgate equivalent for Quartic in SF, we compute:
    # U = exp(i * delta * x^4)
    
    x_mat = get_position_operator_matrix(cutoff_dim)
    
    # Calculate x^4
    x2 = x_mat @ x_mat
    x4 = x2 @ x2

    # Calculate unitary U = exp(i * delta * x^4) via matrix exponentiation
    U_quartic = scipy.linalg.expm(1j * delta * x4)
    
    # Apply U to the ket: |psi_new> = U |psi_old>
    ket_quartic = U_quartic @ ket_init
    
    # # Normalize (numerical safety)
    norm = np.linalg.norm(ket_quartic)
    # ket_quartic = ket_quartic / norm
    
    print(f"State prepared. Norm: {norm**2:.4f}")

    # --- 4. LOAD BACK INTO STRAWBERRY FIELDS ---
    prog_final = sf.Program(1)
    with prog_final.context as q:
        ops.Ket(ket_quartic) | q[0]
        # ops.LossChannel(0.9) | q[0]
        

    # prog_final = sf.Program(1)
    # with prog_final.context as q:
    #     ops.Fock(59) | q[0]

    eng_final = sf.Engine("fock", backend_options={"cutoff_dim": cutoff_dim})
    result = eng_final.run(prog_final)
    state = result.state
    assert state.is_pure

    # --- 5. CALCULATE WIGNER FUNCTION ---
    print("Calculating Wigner function...")
    xvec = np.linspace(-x_limit, x_limit, grid_size)
    pvec = np.linspace(-x_limit, x_limit, grid_size)
    W = state.wigner(mode=0, xvec=xvec, pvec=pvec)

    # --- 6. CALCULATE FOCK PROBABILITIES ---
    probs = state.all_fock_probs(cutoff=cutoff_dim)

    # --- 7. PLOTTING ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Plot 1: Wigner Function
    X, P = np.meshgrid(xvec, pvec)
    c = ax1.pcolormesh(X, P, W, cmap='RdBu', shading='auto', 
                       vmin=-np.max(np.abs(W)), vmax=np.max(np.abs(W)))
    fig.colorbar(c, ax=ax1, label='W(x, p)')
    ax1.set_title(f"Wigner Function (Quartic Phase $\delta={delta}$)")
    ax1.set_xlabel("x (Position)")
    ax1.set_ylabel("p (Momentum)")
    ax1.set_aspect('equal')
    
    # Grid lines
    ax1.axhline(0, color='black', linestyle='--', alpha=0.3)
    ax1.axvline(0, color='black', linestyle='--', alpha=0.3)

    # Plot 2: Fock Distribution
    # Using a bar chart but zoomed out a bit due to higher cutoff
    ax2.bar(range(cutoff_dim), probs, color='purple', alpha=0.7, edgecolor='black')
    ax2.set_title("Fock State Probabilities")
    ax2.set_xlabel("Fock Number |n>")
    ax2.set_ylabel("Probability")
    # Only show ticks every 5 due to high cutoff
    ax2.set_xticks(np.arange(0, cutoff_dim, 5)) 

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    plot_quartic_phase_target()