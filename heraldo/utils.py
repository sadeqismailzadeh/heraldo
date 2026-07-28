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
