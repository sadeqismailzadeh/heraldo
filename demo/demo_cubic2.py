
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
from strawberryfields.ops import *

def plot_cubic_phase_target():
    # --- 1. CONFIGURATION ---
    cutoff_dim = 31
    grid_size = 200
    x_limit = 5
    
    
    prog = sf.Program(1)
    g = 1.2
    gamma = 5

    with prog.context as q:
        Vgate(-gamma  * g ** (-3/2)*2) | q[0]
        Sgate(g)  | q[0]


    eng = sf.Engine("fock", backend_options={"cutoff_dim": cutoff_dim})
    state = eng.run(prog).state
    dm = state.dm()
    # trace of dm
    tr = np.trace(dm)
    print(f"trace = {tr}")
    assert state.is_pure
    ket = state.ket()
        


    print(f"Norm: {np.linalg.norm(ket)**2:.4f}")
    
    # --- 3. LOAD INTO STRAWBERRY FIELDS ---
    # We use a temporary engine just to utilize the built-in Wigner calculator
    prog = sf.Program(1)
    with prog.context as q:
        ops.Ket(ket) | q[0]

    eng = sf.Engine("fock", backend_options={"cutoff_dim": cutoff_dim})
    result = eng.run(prog)
    state = result.state

    # --- 4. CALCULATE WIGNER FUNCTION ---
    print("Calculating Wigner function...")
    xvec = np.linspace(-x_limit, x_limit, grid_size)
    pvec = np.linspace(-x_limit, x_limit, grid_size)
    W = state.wigner(mode=0, xvec=xvec, pvec=pvec)

    # --- 5. CALCULATE FOCK PROBABILITIES ---
    probs = state.all_fock_probs(cutoff=cutoff_dim)

    # --- 6. PLOTTING ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Plot 1: Wigner Function
    # We use RdBu_r colormap: Red = Positive, Blue = Negative (Standard in Quantum Optics)
    # Negative regions indicate Non-Gaussianity.
    X, P = np.meshgrid(xvec, pvec)
    c = ax1.pcolormesh(X, P, W, cmap='RdBu', shading='auto', vmin=-np.max(np.abs(W)), vmax=np.max(np.abs(W)))
    fig.colorbar(c, ax=ax1, label='W(x, p)')
    ax1.set_title(f"Wigner Function (Cubic Phase)")
    ax1.set_xlabel("x (Position)")
    ax1.set_ylabel("p (Momentum)")
    ax1.set_aspect('equal')
    
    # Add grid lines to see the center
    ax1.axhline(0, color='black', linestyle='--', alpha=0.3)
    ax1.axvline(0, color='black', linestyle='--', alpha=0.3)

    # Plot 2: Fock Distribution
    ax2.bar(range(cutoff_dim), probs, color='teal', alpha=0.7, edgecolor='black')
    ax2.set_title("Fock State Probabilities")
    ax2.set_xlabel("Fock Number |n>")
    ax2.set_ylabel("Probability")
    ax2.set_xticks(range(cutoff_dim))
    


    plt.tight_layout()
    plt.show()
    

if __name__ == "__main__":
    plot_cubic_phase_target()