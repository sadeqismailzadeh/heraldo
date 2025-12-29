"""
Utility functions for quantum state decomposition and analysis.
"""

# Patch scipy if needed (common in this codebase)
import scipy.integrate
if not hasattr(scipy.integrate, 'simps'):
    # print("Monkey patching scipy.integrate: 'simps' not found. Pointing to 'simpson'.")
    scipy.integrate.simps = scipy.integrate.simpson

import numpy as np
from scipy.optimize import basinhopping
import strawberryfields as sf
from strawberryfields.ops import Ket, Sgate, Dgate

def decompose_target_to_stellar(target_ket, n_max_core, cutoff_dim=None, method='SLSQP'):
    """
    Decomposes a target state into its stellar representation approximation:
    |psi> ~ D(alpha) S(r, phi) |core>
    
    where |core> is a state truncated at n_max_core Fock number.
    This decomposition maximizes the fidelity between the target state and the
    Gaussian-transformed core state. It uses global optimization (basinhopping)
    to find the optimal squeezing and displacement parameters.
    
    Args:
        target_ket (np.ndarray): The target state vector in Fock basis.
        n_max_core (int): The maximum photon number for the core state.
        cutoff_dim (int, optional): Simulation cutoff dimension. Defaults to len(target_ket) + padding.
        method (str): Optimization method for local search. Default 'SLSQP'.
        
    Returns:
        dict: {
            'squeezing_r': float,
            'squeezing_phi': float,
            'displacement_alpha': complex,
            'core_ket': np.ndarray, # Normalized core state of dimension n_max_core+1
            'fidelity': float       # Fidelity of the approximation
        }
    """
    # Flatten and validate target
    target_ket = np.asarray(target_ket, dtype=np.complex128).flatten()
    target_dim = len(target_ket)
    
    # Determine appropriate cutoff for Gaussian operations
    # Inverse squeezing/displacement can increase photon number significantly
    if cutoff_dim is None:
        cutoff_dim = max(target_dim, n_max_core + 25)
        
    # Prepare target in the simulation cutoff
    if len(target_ket) < cutoff_dim:
        padded = np.zeros(cutoff_dim, dtype=np.complex128)
        padded[:len(target_ket)] = target_ket
        target_ket = padded
    else:
        # If target is larger than requested cutoff, truncate and normalize
        target_ket = target_ket[:cutoff_dim]
        target_ket /= np.linalg.norm(target_ket)
        
    # Initialize Strawberry Fields engine
    eng = sf.Engine("fock", backend_options={"cutoff_dim": cutoff_dim})
    
    def negative_overlap(params):
        """
        Cost function: -1 * Probability of the inverse-transformed state 
        residing in the core subspace (Fock 0 to n_max_core).
        
        Maximizing this probability is equivalent to maximizing the achievable
        fidelity between the target and any state of form G |core>.
        """
        # Unpack parameters
        r, phi, x, y = params
        alpha = x + 1j * y
        
        # Construct the Inverse Gaussian Circuit
        # We want U_inv |target> where U = D(alpha) S(r, phi)
        # U_inv = S^dag(r, phi) D^dag(alpha)
        # Note: (AB)^-1 = B^-1 A^-1
        
        d_inv_mag = np.abs(alpha)
        d_inv_phi = np.angle(alpha) + np.pi
        
        s_inv_r = r
        s_inv_phi = phi + np.pi
        
        prog = sf.Program(1)
        with prog.context as q:
            Ket(target_ket) | q[0]
            Dgate(d_inv_mag, d_inv_phi) | q[0]
            Sgate(s_inv_r, s_inv_phi) | q[0]
            
        try:
            result = eng.run(prog)
            state_ket = result.state.ket().flatten()
            
            # Calculate probability in the core subspace
            # This is equivalent to || P_core |psi'> ||^2
            prob = np.sum(np.abs(state_ket[:n_max_core+1])**2)
            
            # Bound check (numerical errors might give > 1 slightly)
            prob = min(prob, 1.0)
            
            return -prob
        except Exception:
            return 0.0

    # Bounds for parameters
    # r: [0, 1.4] (approx 12 dB) - Restricted to avoid cutoff artifacts
    # phi: [-2pi, 2pi]
    # x, y: [-4.5, 4.5] - Restricted displacement
    bounds = [
        (0.0, 1.4),
        (-2*np.pi, 2*np.pi),
        (-4.5, 4.5),
        (-4.5, 4.5)
    ]
    
    # Initial guess
    x0 = [0.0, 0.0, 0.0, 0.0]
    
    # Optimization with Basinhopping (Global Search)
    minimizer_kwargs = {"method": method, "bounds": bounds}
    # niter=20 is usually sufficient to break symmetry, increased to 40 for robustness
    res = basinhopping(
        negative_overlap, 
        x0, 
        niter=40, 
        T=0.5, 
        stepsize=0.5, 
        minimizer_kwargs=minimizer_kwargs
    )
    
    best_params = res.x
    final_fidelity = -res.fun
    
    # Reconstruct the optimal core state
    r, phi, x, y = best_params
    alpha = x + 1j*y
    
    # Apply optimal inverse operations to target
    prog = sf.Program(1)
    with prog.context as q:
        Ket(target_ket) | q[0]
        Dgate(np.abs(alpha), np.angle(alpha) + np.pi) | q[0]
        Sgate(r, phi + np.pi) | q[0]
        
    res_final = eng.run(prog)
    final_state_ket = res_final.state.ket().flatten()
    
    # Extract, project and normalize core
    core_ket_trunc = final_state_ket[:n_max_core+1]
    norm_core = np.linalg.norm(core_ket_trunc)
    
    if norm_core > 1e-9:
        core_ket = core_ket_trunc / norm_core
    else:
        # Fallback if optimization failed completely
        core_ket = np.zeros(n_max_core+1, dtype=np.complex128)
        core_ket[0] = 1.0
        
    return {
        'squeezing_r': r,
        'squeezing_phi': phi,
        'displacement_alpha': alpha,
        'core_ket': core_ket,
        'fidelity': final_fidelity
    }
