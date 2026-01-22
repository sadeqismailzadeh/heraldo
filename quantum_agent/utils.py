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
from functools import lru_cache
import scipy.sparse as sp


@lru_cache(maxsize=8)
def _get_ng_operators(dim):
    """Cached retrieval of sparse quadrature operators."""
    sqrt_n = np.sqrt(np.arange(1, dim))
    a = sp.diags([sqrt_n], [1], shape=(dim, dim), format='csr')
    a_dag = a.T
    
    x = (a + a_dag) / np.sqrt(2)
    p = 1j * (a_dag - a) / np.sqrt(2)
    
    # Precompute powers
    x2 = x.dot(x)
    x3 = x2.dot(x)
    x4 = x2.dot(x2)
    
    p2 = p.dot(p)
    p3 = p2.dot(p)
    p4 = p2.dot(p2)
    
    return (x, x2, x3, x4), (p, p2, p3, p4)

def compute_ng_scores(kets, cutoff_dim):
    """Computes Non-Gaussianity (Negativity proxy via cumulants) for a batch of kets."""
    (xs, ps) = _get_ng_operators(cutoff_dim)
    
    scores = np.zeros(len(kets))
    
    for i, ket in enumerate(kets):
        # Helper to compute moments
        def get_moments(ops):
            # Expectation <psi|O|psi>
            m1 = np.real(np.vdot(ket, ops[0].dot(ket)))
            m2 = np.real(np.vdot(ket, ops[1].dot(ket)))
            m3 = np.real(np.vdot(ket, ops[2].dot(ket)))
            m4 = np.real(np.vdot(ket, ops[3].dot(ket)))
            return m1, m2, m3, m4
            
        def get_val(m1, m2, m3, m4):
            var = m2 - m1**2
            if var < 1e-6:
                return 0.0
            sigma = np.sqrt(var)
            m3_c = m3 - 3*m1*m2 + 2*(m1**3)
            m4_c = m4 - 4*m1*m3 + 6*(m1**2)*m2 - 3*(m1**4)
            skew = m3_c / (sigma**3)
            kurt = (m4_c / (var**2)) - 3.0
            return np.abs(skew) + np.abs(kurt)
            
        mx = get_moments(xs)
        mp = get_moments(ps)
        
        scores[i] = get_val(*mx) + get_val(*mp)
        
    return scores

# --- Pure State Fidelity Function ---
def fidelity_pure_state(target_ket, state_ket):
    """Return fidelity between two pure states.

    For two pure states |φ⟩ and |ψ⟩, the fidelity is:
    F(|φ⟩, |ψ⟩) = |⟨φ|ψ⟩|²

    Args:
        target_ket (np.ndarray): Target state vector.
        state_ket (np.ndarray): Current state vector.

    Returns:
        float: Clipped fidelity value in [0, 1].
    """
    target_ket = np.asarray(target_ket, dtype=np.complex128).flatten()
    state_ket = np.asarray(state_ket, dtype=np.complex128).flatten()

    # Fidelity: F = |⟨φ|ψ⟩|²
    overlap = np.vdot(target_ket, state_ket)  # ⟨φ|ψ⟩
    fidelity = np.abs(overlap) ** 2
    return np.clip(fidelity, 0.0, 1.0)


def fidelity_max_rotation(target_ket, state_ket, n_fft=256):
    """
    Calculates the maximum fidelity between state_ket and target_ket
    optimizing over any global phase space rotation z-rotation R(phi).

    Args:
        target_ket (np.ndarray): Target state vector (Fock basis).
        state_ket (np.ndarray): Current state vector (Fock basis).
        n_fft (int): Resolution of the angle search. Higher = more accurate.
                     2048 is usually plenty for cutoff_dim ~ 25.

    Returns:
        float: The maximum achievable fidelity.
    """
    # Ensure inputs are 1D arrays
    t = np.asarray(target_ket, dtype=np.complex128).flatten()
    s = np.asarray(state_ket, dtype=np.complex128).flatten()

    # Pad to matching lengths if necessary
    max_len = max(len(t), len(s))
    if len(t) < max_len:
        t = np.pad(t, (0, max_len - len(t)))
    if len(s) < max_len:
        s = np.pad(s, (0, max_len - len(s)))

    # 1. Calculate the element-wise product: h[n] = s[n]* . t[n]
    # We conjugate s and not t (or vice versa), the magnitude result is the same.
    h = np.conj(s) * t

    # 2. Use FFT to compute sum(h[n] * e^{-i*n*phi}) for discrete phi
    # Zero-padding (n_fft > len(h)) interpolates the spectrum, effectively
    # searching more angles for a finer resolution.
    fft_values = np.fft.fft(h, n=n_fft)

    # 3. The Fidelity is the square of the maximum magnitude of the overlap
    max_overlap = np.max(np.abs(fft_values))
    return np.clip(max_overlap**2, 0.0, 1.0)


def db_to_r(db_value):
    """Converts squeezing level from Decibels (dB) to the squeezing parameter r."""
    return db_value / (20 * np.log10(np.e))


def decompose_target_to_stellar(target_ket, n_max_core, cutoff_dim=None, method='SLSQP', even_parity=True):
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
        even_parity (bool): If True, restricts core to even Fock states (|0>, |2>...).
                            Necessary for symmetric states like GKP |0>.
        
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
        residing in the core subspace.
        
        For GKP states (even_parity=True), we only project onto the even
        Fock subspace (|0>, |2>, ...), matching the methodology in
        formalism.py.
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
            
            # Check for numerical validity (trace preservation)
            # If the trace drops significantly, we've exceeded the cutoff
            norm_sq = np.real(np.vdot(state_ket, state_ket))
            if norm_sq < 0.9 or norm_sq >1.1:
                # Penalize states that leave the Hilbert space (truncation error)
                return 1.0 + (1.0 - norm_sq)

            # Calculate probability in the core subspace
            # This is equivalent to || P_core |psi'> ||^2
            
            if even_parity:
                # Restrict to even Fock states only (|0>, |2>, |4>, ...)
                # The step '2' in the slice selects every second element starting from 0.
                # We must ensure n_max_core is even to capture the last state correctly
                eff_n_max = n_max_core if n_max_core % 2 == 0 else n_max_core - 1
                prob = np.sum(np.abs(state_ket[0:eff_n_max+1:2])**2)
            else:
                # General case: include all states |0> ... |n_max>
                prob = np.sum(np.abs(state_ket[:n_max_core+1])**2)
            
            # Bound check (numerical errors might give > 1 slightly)
            prob = min(prob, 1.0)
            
            return -prob
        except Exception:
            return 1.0

    # Bounds for parameters
    # r: [0, 1.4] (approx 12 dB) - Restricted to avoid Fock cutoff errors.
    # phi: [-2pi, 2pi]
    # x, y: [-2.5, 2.5] - Restricted to keep displacement manageable.
    bounds = [
        (0.0, 1.4),
        (-2*np.pi, 2*np.pi),
        (-2.5, 2.5),
        (-2.5, 2.5)
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
    if even_parity:
        eff_n_max = n_max_core if n_max_core % 2 == 0 else n_max_core - 1
        # Extract only even indices
        core_ket_trunc = final_state_ket[0:eff_n_max+1:2]
        
        # Reconstruct full core ket with zeros at odd indices for consistency
        core_ket = np.zeros(n_max_core+1, dtype=np.complex128)
        norm_core = np.linalg.norm(core_ket_trunc)
        if norm_core > 1e-9:
            core_ket[0:eff_n_max+1:2] = core_ket_trunc / norm_core
    else:
        core_ket_trunc = final_state_ket[:n_max_core+1]
        norm_core = np.linalg.norm(core_ket_trunc)
        if norm_core > 1e-9:
            core_ket = core_ket_trunc / norm_core
            
    if np.linalg.norm(core_ket) < 1e-9:
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
