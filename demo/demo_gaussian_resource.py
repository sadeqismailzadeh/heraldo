
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
    a = 0.61  # The parameter from Article 2
    cutoff_dim = 10
    grid_size = 200
    x_limit = 5
    
    # --- 2. CONSTRUCT THE STATE VECTOR MANUALLY ---
    # Target = N * (|0> + i*a*sqrt(1.5)|1> + i*a|3>)
    # Note: The coefficient for |2> is exactly 0.
    
    # Coefficients
    c0 = 1.0 + 0j
    c1 = 0.0 + 1j * a * np.sqrt(1.5)
    c2 = 0.0 + 0j
    c3 = 0.0 + 1j * a
    
    # Build vector
    ket = np.zeros(cutoff_dim, dtype=np.complex128)
    ket[0] = c0
    ket[1] = c1
    ket[2] = c2
    ket[3] = c3
    
    # Normalize
    norm = np.linalg.norm(ket)
    ket = ket / norm

    targets = []
    # --- 2. Generate Rotated Variants ---
    # Angles: 0, 90, 180, 270 degrees
    rotation_angles = [0.0, np.pi/2, np.pi, 3*np.pi/2]
    
    # Create vector of photon numbers [0, 1, 2, ..., cutoff-1]
    n_vec = np.arange(cutoff_dim)

    for theta in rotation_angles:
        # The rotation operator R(theta) adds phase e^(-i * n * theta) to Fock state |n>
        # We construct a phase vector to multiply element-wise with the base ket
        phase_factors = np.exp(1j * n_vec * theta)
        
        # Apply rotation
        rotated_ket = ket * phase_factors
        
        targets.append(rotated_ket)

    ket = targets[0]
    print(f"State vector (first 5 terms):\n{np.round(ket[:5], 3)}")
    print(f"Norm: {np.linalg.norm(ket):.4f}")
    
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
    ax1.set_title(f"Wigner Function (Cubic Phase, a={a})")
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
    
    # Highlight the "Hole" at |2>
    ax2.annotate('The Hole\n(Must be 0)', xy=(2, probs[2]), xytext=(2, 0.3),
                 arrowprops=dict(facecolor='red', shrink=0.05),
                 ha='center', color='red', fontweight='bold')

    # Highlight the cutoff at |4+>
    ax2.axvline(3.5, color='red', linestyle='--', label='Truncation')
    ax2.text(4, 0.2, "Forbidden\nRegion", color='red')

    plt.tight_layout()
    plt.show()
    
    # --- 7. PRINT METRICS ---
    # Quick calc of the Non-Gaussianity metrics discussed earlier
    cov = state.cov()
    det_cov = np.linalg.det(cov)
    print("-" * 30)
    print("METRICS:")
    print(f"P(|2>) (Should be 0): {probs[2]:.6f}")
    print(f"Determinant of Covariance: {det_cov:.4f}")
    print(f"Genoni NG Score (log(det)/const): {np.log(det_cov):.4f}")
    if det_cov > 1.1: # Using hbar=2 units, det(vac)=1
        print("=> RESULT: State is Non-Gaussian")
    else:
        print("=> RESULT: State looks Gaussian (Unexpected for Cubic Phase)")

if __name__ == "__main__":
    plot_cubic_phase_target()