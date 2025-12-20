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


def compare_vgate_vs_approx():
    # --- PARAMETERS ---
    cutoff_dim = 20
    chi = 0.2         # Small parameter
    g_squeeze = 1.2   # Squeezing amount
    
    # --- SYSTEM A: The "Faithful" Paper Approximation ---
    # State = S(r) * (1 + i * chi * x^3) |0> (Normalized)
    
    # 1. Construct polynomial state |psi> = N * (1 + i*chi*x^3)|0>
    # Recalling x^3|0> = (3/2sqrt(2))|1> + (sqrt(3)/2)|3>
    c0 = 1.0
    c1 = 1j * chi * (3.0 / (2.0 * np.sqrt(2)))
    c3 = 1j * chi * (np.sqrt(3) / 2.0)
    
    ket_poly = np.zeros(cutoff_dim, dtype=np.complex128)
    ket_poly[0] = c0
    ket_poly[1] = c1
    ket_poly[3] = c3
    ket_poly /= np.linalg.norm(ket_poly) # Normalize manually

    prog_approx = sf.Program(1)
    with prog_approx.context as q:
        ops.Ket(ket_poly) | q[0]
        # ops.Sgate(np.log(np.sqrt(g_squeeze))) | q[0]

    # --- SYSTEM B: The "Vgate" (Ideal Unitary) ---
    # State = S(r) * exp(i * chi * x^3) |0>
    # Note: Strawberry Fields Vgate(gamma) is exp(i * gamma * x^3/hbar)
    # With default hbar=2, we need to be careful, but roughly:
    prog_vgate = sf.Program(1)
    with prog_vgate.context as q:
        # Note: We apply Vgate BEFORE squeezing to match the resource state structure
        # (The paper applies polynomial then squeezes, or squeezes then adds photons. 
        # The ordering matters for the commutation, but for simple visualization:
        ops.Vgate(chi*np.sqrt(2)) | q[0] 
        # ops.Sgate(np.log(np.sqrt(g_squeeze))) | q[0]

    # --- RUN ENGINES ---
    eng = sf.Engine("fock", backend_options={"cutoff_dim": cutoff_dim})
    
    state_approx = eng.run(prog_approx).state
    state_vgate = eng.run(prog_vgate).state

    # --- CALCULATE FIDELITY ---
    fid = state_approx.fidelity(state_vgate.ket(), mode=0)
    print(f"Fidelity between Vgate and Approximation: {fid:.6f}")
    
    # --- PLOT WIGNERS ---
    grid = 100
    limit = 5
    xvec = np.linspace(-limit, limit, grid)
    pvec = np.linspace(-limit, limit, grid)
    
    W_approx = state_approx.wigner(mode=0, xvec=xvec, pvec=pvec)
    W_vgate = state_vgate.wigner(mode=0, xvec=xvec, pvec=pvec)

    fig, ax = plt.subplots(1, 3, figsize=(18, 5))
    
    # Plot Approx
    ax[0].pcolormesh(xvec, pvec, W_approx, cmap='RdBu', shading='auto')
    ax[0].set_title("Paper Approximation\n(1 + i$\chi x^3$)")
    
    # Plot Vgate
    ax[1].pcolormesh(xvec, pvec, W_vgate, cmap='RdBu', shading='auto')
    ax[1].set_title("Strawberry Fields Vgate\n($e^{i\chi x^3}$)")

    # Plot Difference
    diff = W_vgate - W_approx
    im = ax[2].pcolormesh(xvec, pvec, diff, cmap='seismic', shading='auto') # Seismic highlights differences
    plt.colorbar(im, ax=ax[2])
    ax[2].set_title(f"Difference\n(Fidelity = {fid:.4f})")
    
    plt.show()

if __name__ == "__main__":
    compare_vgate_vs_approx()