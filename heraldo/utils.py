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
from functools import lru_cache
import scipy.sparse as sp
import re


def windows_to_wsl_path(win_path: str) -> str:
    """
    Converts an absolute Windows path to an absolute WSL path.
    Example: 'C:\\Users\\name\\folder' -> '/mnt/c/Users/name/folder'
    """
    # 1. Remove any accidental surrounding quotes
    clean_path = win_path.strip('\'"')
    
    # 2. Convert all Windows backslashes to forward slashes
    clean_path = clean_path.replace('\\', '/')
    
    # 3. Match the Windows drive letter pattern (e.g., "C:/..." or "d:/...")
    match = re.match(r'^([a-zA-Z]):/(.*)$', clean_path)
    
    if match:
        drive_letter = match.group(1).lower()
        rest_of_path = match.group(2)
        # 4. Construct the WSL /mnt/ path
        return f"/mnt/{drive_letter}/{rest_of_path}"
    
    # If it doesn't match a drive letter, return the normalized path as-is
    return clean_path

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
