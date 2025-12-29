import numpy as np
import matplotlib.pyplot as plt
# Importing the package triggers the scipy patch in quantum_agent/__init__.py

import quantum_agent 
import strawberryfields as sf
from strawberryfields.ops import Ket, Sgate, Dgate


from quantum_agent.utils import decompose_target_to_stellar
from quantum_agent.components.targets import GKPTarget

def plot_wigner(state, ax, title):
    """Calculates and plots the Wigner function of a Strawberry Fields state."""
    xvec = np.linspace(-6, 6, 200)
    pvec = np.linspace(-6, 6, 200)
    W = state.wigner(mode=0, xvec=xvec, pvec=pvec)
    
    X, P = np.meshgrid(xvec, pvec)
    # Use max absolute value to center colormap
    w_max = np.max(np.abs(W))
    c = ax.pcolormesh(X, P, W, cmap='RdBu', shading='auto', vmin=-w_max, vmax=w_max)
    
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("p")
    ax.set_aspect('equal')
    return c

def main():
    # Simulation parameters
    # INCREASED CUTOFF to avoid simulation artifacts during reconstruction
    cutoff = 100  
    n_max_core = 4 # Truncation for the core state (Stellar rank approximation)
    
    print("=== Stellar Representation Decomposition Test ===")
    
    # 1. Generate Target State (Square GKP)
    # mu=0 (logical 0), delta=0.3 (~10dB squeezing)
    target_gen = GKPTarget(gkp_type='square', mu=0, delta=0.3)
    target_ket = target_gen.get_target_ket(cutoff)
    
    # 2. Perform Decomposition
    print(f"\nDecomposing GKP target (mu=0, delta=0.3) with core size n_max={n_max_core}...")
    # Using the same cutoff for decomposition optimization
    result = decompose_target_to_stellar(target_ket, n_max_core, cutoff_dim=cutoff)
    
    # 3. Print Results
    print("\nOptimization Results:")
    print(f"  Fidelity:          {result['fidelity']:.6f}")
    print(f"  Squeezing r:       {result['squeezing_r']:.4f} (approx {result['squeezing_r']*8.686:.2f} dB)")
    print(f"  Squeezing phi:     {result['squeezing_phi']:.4f}")
    print(f"  Displacement:      {result['displacement_alpha']:.4f}")
    
    # Print core coefficients (first few)
    core = result['core_ket']
    print("\nCore State Coefficients (first 6):")
    for i in range(min(6, len(core))):
        val = core[i]
        if np.abs(val) > 1e-3:
            print(f"  |{i}>: {val:.4f}")
            
    # 4. Reconstruct the Approximate State
    # |psi_approx> = D(alpha) S(r, phi) |core>
    
    # Pad core to cutoff dimension
    core_padded = np.zeros(cutoff, dtype=np.complex128)
    core_padded[:len(core)] = core
    
    prog_approx = sf.Program(1)
    with prog_approx.context as q:
        Ket(core_padded) | q[0]
        Sgate(result['squeezing_r'], result['squeezing_phi']) | q[0]
        Dgate(np.abs(result['displacement_alpha']), np.angle(result['displacement_alpha'])) | q[0]
        
    eng = sf.Engine("fock", backend_options={"cutoff_dim": cutoff})
    res_approx = eng.run(prog_approx)
    approx_state = res_approx.state
    
    # Check norm
    norm_sq = res_approx.state.trace()
    print(f"\nReconstructed State Trace/Norm: {norm_sq:.4f}")
    if norm_sq > 1.1 or norm_sq < 0.9:
        print("WARNING: State norm indicates simulation cutoff truncation or explosion.")

    # 5. Get Target State Object (for visualization)
    prog_tgt = sf.Program(1)
    with prog_tgt.context as q:
        Ket(target_ket) | q[0]
    target_state = eng.run(prog_tgt).state
    
    # 6. Plot Comparison
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    plot_wigner(target_state, axes[0], "Target GKP State")
    c = plot_wigner(approx_state, axes[1], f"Stellar Approx (F={result['fidelity']:.4f})\n(n_max={n_max_core})")
    
    fig.subplots_adjust(right=0.9)
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    fig.colorbar(c, cax=cbar_ax, label="Wigner Function Value")
    
    plt.suptitle(f"Stellar Decomposition: GKP -> D(a)S(z)|Core>", fontsize=16)
    plt.show()

if __name__ == "__main__":
    main()
