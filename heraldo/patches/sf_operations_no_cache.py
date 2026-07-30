"""Monkey-patch module to disable Fock backend gate matrix caching in Strawberry Fields.

Replaces LRU-cached operations in Strawberry Fields' Fock backend with uncached equivalents
to prevent uncontrolled RAM growth during long optimization loops or large Fock space dimensions.
"""

# 1. Import the module we need to patch
import scipy.integrate

# 2. Check if the patch is needed to avoid errors
if not hasattr(scipy.integrate, 'simps'):
    print("Monkey patching scipy.integrate: 'simps' not found. Pointing to 'simpson'.")
    # 3. Create the 'simps' attribute and point it to the existing 'simpson' function.
    scipy.integrate.simps = scipy.integrate.simpson
else:
    print("'simps' already exists in scipy.integrate. No patch needed.")

import strawberryfields as sf
from strawberryfields.ops import *
import numpy as np


def disable_fock_caching():
    """Disables internal LRU caching for all Strawberry Fields Fock backend operations.

    Replaces standard cached backend functions in ``strawberryfields.backends.fockbackend.ops``
    with uncached implementations. This prevents memory leaks and RAM accumulation during
    extensive optimization routines across large cutoff dimensions, trading minor recomputation
    overhead for bounded, constant memory consumption.
    """
    from strawberryfields.backends.fockbackend import ops
    from thewalrus.fock_gradients import (
        beamsplitter as beamsplitter_tw,
        squeezing as squeezing_tw,
        displacement as displacement_tw,
        two_mode_squeezing as two_mode_squeezing_tw,
        mzgate as mzgate_tw,
    )
    
    def beamsplitter_uncached(theta: float, phi: float, trunc: int) -> np.ndarray:
        """Computes the beam splitter unitary matrix in the Fock basis without caching.

        Args:
            theta (float): Transmission angle parameter :math:`\\theta`.
            phi (float): Phase angle parameter :math:`\\phi`.
            trunc (int): Fock space cutoff dimension.

        Returns:
            np.ndarray: 4D tensor of shape ``(trunc, trunc, trunc, trunc)`` representing
            the beam splitter operator.
        """
        BS_tw = beamsplitter_tw(theta, phi, cutoff=trunc)
        return BS_tw.transpose((0, 2, 1, 3))
    
    def squeezing_uncached(r: float, theta: float, trunc: int) -> np.ndarray:
        """Computes the single-mode squeezing operator matrix in the Fock basis without caching.

        Args:
            r (float): Squeezing magnitude parameter :math:`r`.
            theta (float): Squeezing phase angle parameter :math:`\\theta`.
            trunc (int): Fock space cutoff dimension.

        Returns:
            np.ndarray: 2D matrix of shape ``(trunc, trunc)`` representing the squeezing operator.
        """
        return squeezing_tw(r, theta, cutoff=trunc)
    
    def displacement_uncached(r: float, phi: float, trunc: int) -> np.ndarray:
        """Computes the displacement operator matrix in the Fock basis without caching.

        Args:
            r (float): Displacement magnitude parameter :math:`r`.
            phi (float): Displacement phase angle parameter :math:`\\phi`.
            trunc (int): Fock space cutoff dimension.

        Returns:
            np.ndarray: 2D matrix of shape ``(trunc, trunc)`` representing the displacement operator.
        """
        return displacement_tw(r, phi, cutoff=trunc)
    
    def two_mode_squeeze_uncached(r: float, theta: float, trunc: int) -> np.ndarray:
        """Computes the two-mode squeezing operator tensor in the Fock basis without caching.

        Args:
            r (float): Squeezing magnitude parameter :math:`r`.
            theta (float): Squeezing phase angle parameter :math:`\\theta`.
            trunc (int): Fock space cutoff dimension.

        Returns:
            np.ndarray: 4D tensor of shape ``(trunc, trunc, trunc, trunc)`` representing
            the two-mode squeezing operator.
        """
        ret = two_mode_squeezing_tw(r, theta, cutoff=trunc)
        return np.transpose(ret, [0, 2, 1, 3])
    
    def mzgate_uncached(phi_in: float, phi_ex: float, cutoff: int) -> np.ndarray:
        """Computes the Mach-Zehnder interferometer gate matrix in the Fock basis without caching.

        Args:
            phi_in (float): Internal phase shift parameter :math:`\\phi_{\\text{in}}`.
            phi_ex (float): External phase shift parameter :math:`\\phi_{\\text{ex}}`.
            cutoff (int): Fock space cutoff dimension.

        Returns:
            np.ndarray: 4D tensor of shape ``(cutoff, cutoff, cutoff, cutoff)`` representing
            the Mach-Zehnder gate operator.
        """
        ret = mzgate_tw(phi_in, phi_ex, cutoff)
        return ret.transpose((0, 2, 1, 3))
    
    def phase_uncached(theta: float, trunc: int) -> np.ndarray:
        """Computes the single-mode phase shift operator matrix in the Fock basis without caching.

        Args:
            theta (float): Phase rotation angle :math:`\\theta`.
            trunc (int): Fock space cutoff dimension.

        Returns:
            np.ndarray: 2D diagonal matrix of shape ``(trunc, trunc)`` representing the phase shift operator.
        """
        return np.array(np.diag([np.exp(1j * n * theta) for n in range(trunc)]), 
                       dtype=np.complex128)
    
    def kerr_uncached(kappa: float, trunc: int) -> np.ndarray:
        """Computes the single-mode Kerr non-linearity operator matrix in the Fock basis without caching.

        Args:
            kappa (float): Kerr non-linearity strength parameter :math:`\\kappa`.
            trunc (int): Fock space cutoff dimension.

        Returns:
            np.ndarray: 2D diagonal matrix of shape ``(trunc, trunc)`` representing the Kerr operator.
        """
        n = np.arange(trunc)
        return np.diag(np.exp(1j * kappa * n**2))
    
    def cross_kerr_uncached(kappa: float, trunc: int) -> np.ndarray:
        """Computes the two-mode cross-Kerr non-linearity operator tensor in the Fock basis without caching.

        Args:
            kappa (float): Cross-Kerr interaction strength parameter :math:`\\kappa`.
            trunc (int): Fock space cutoff dimension.

        Returns:
            np.ndarray: 4D tensor of shape ``(trunc, trunc, trunc, trunc)`` representing
            the cross-Kerr operator.
        """
        n1 = np.arange(trunc)[None, :]
        n2 = np.arange(trunc)[:, None]
        n1n2 = np.ravel(n1 * n2)
        return np.diag(np.exp(1j * kappa * n1n2)).reshape([trunc] * 4).swapaxes(1, 2)
    
    # Replace the cached functions with uncached versions
    ops.beamsplitter = beamsplitter_uncached
    ops.squeezing = squeezing_uncached
    ops.displacement = displacement_uncached
    ops.two_mode_squeeze = two_mode_squeeze_uncached
    ops.mzgate = mzgate_uncached
    ops.phase = phase_uncached
    ops.kerr = kerr_uncached
    ops.cross_kerr = cross_kerr_uncached
    
    print("Caching DISABLED for all Fock backend operations!")
    print("   Gates will be recomputed every time.")
    print("   Memory usage will stay constant - no buildup!")
