"""Target state generators implementing the TargetGenerator interface."""


# 1. Import the module we need to patch
import scipy.integrate

# 2. Check if the patch is needed to avoid errors
if not hasattr(scipy.integrate, 'simps'):
    print("Monkey patching scipy.integrate: 'simps' not found. Pointing to 'simpson'.")
    # 3. Create the 'simps' attribute and point it to the existing 'simpson' function.
    scipy.integrate.simps = scipy.integrate.simpson
else:
    print("'simps' already exists in scipy.integrate. No patch needed.")

import abc
import numpy as np
import strawberryfields as sf
from strawberryfields.ops import Sgate, Dgate, Vgate, Catstate, Ket
import qutip as qt
from scipy.special import factorial
import scipy.linalg
import pandas as pd

from quantum_modules import TargetGenerator


# --- QuTiP Target Generators (Helper Functions) ---

def sqrGKP_qutip(mu, d, delta, cutoff, nmax=25):  
    """Generates Square GKP target using QuTiP."""
    n1 = np.arange(-nmax, nmax+1)[:, None]
    n2 = np.arange(-nmax, nmax+1)[None, :]

    # Lattice spacing L = sqrt(4*pi) for square
    # arg1 handles the phase checkerboard pattern for logical states
    arg1 = 1j * np.pi * n2 * (d * n1 + mu) / d
    amplitude = (np.exp(arg1)).flatten()[:, None]

    alpha = np.sqrt(np.pi / d) * ((d * n1 + mu - 1j * n2))
    alpha = alpha.flatten()[:, None]
    n = np.arange(cutoff)[None, :]
    
    coherent = np.exp(-0.5 * np.abs(alpha)**2) * alpha**n / np.sqrt(factorial(n))
    state_vector = np.sum(amplitude * coherent * np.exp(-n * delta**2), axis=0).reshape(-1, 1)
    
    return qt.Qobj(state_vector).unit()


def hexGKP(mu, d, delta, cutoff, nmax=20):
    r"""Hexagonal GKP code state (QuTiP 5 compatible)."""
    n1 = np.arange(-nmax, nmax+1)[:, None]
    n2 = np.arange(-nmax, nmax+1)[None, :]

    n1sq = n1**2
    n2sq = n2**2

    sqrt3 = np.sqrt(3)

    # Complex phase and envelope arguments
    arg1 = -1j * np.pi * n2 * (d * n1 + mu) / d
    arg2 = -np.pi * (d**2 * n1sq + n2sq - d * n1 * (n2 - 2 * mu) - n2 * mu + mu**2) / (sqrt3 * d)
    arg2 *= 1 - np.exp(-2 * delta**2)

    amplitude = (np.exp(arg1)).flatten()[:, None]

    # Hexagonal lattice displacement amplitudes
    alpha = np.sqrt(np.pi / (2 * sqrt3 * d)) * (sqrt3 * (d * n1 + mu) - 1j * (d * n1 - 2 * n2 + mu))
    alpha = alpha.flatten()[:, None]

    n = np.arange(cutoff)[None, :]
    coherent = np.exp(-0.5 * np.abs(alpha)**2) * alpha**n / np.sqrt(factorial(n))
    
    # Sum and Reshape for QuTiP 5
    state_vector = np.sum(amplitude * coherent * np.exp(-n * delta**2), axis=0).reshape(-1, 1)
    
    return qt.Qobj(state_vector).unit()


# --- Target Implementations ---

class GKPTarget(TargetGenerator):
    """
    Generates GKP target states (Square or Hexagonal).
    
    Parameters:
    - gkp_type: 'square' or 'hex'
    - mu: Logical state (0 or 1)
    - delta: Finite energy envelope parameter
    """
    
    def __init__(self, gkp_type='square', mu=0, delta=0.4):
        self.gkp_type = gkp_type.lower()
        self.mu = mu
        self.delta = delta
    
    def get_target_ket(self, cutoff_dim: int) -> np.ndarray:
        """Generate GKP target state using QuTiP."""
        print(f"Generating {self.gkp_type.upper()} GKP Target (mu={self.mu}, delta={self.delta}, N={cutoff_dim})...")
        
        if 'hex' in self.gkp_type:
            qobj_tgt = hexGKP(self.mu, 2, self.delta, cutoff_dim)
        else:
            qobj_tgt = sqrGKP_qutip(self.mu, 2, self.delta, cutoff_dim)

        # Convert QuTiP Qobj to Numpy Array (flattened for SF)
        target_np = qobj_tgt.full().flatten()
        
        # Ensure it's normalized
        target_np /= np.linalg.norm(target_np)
        
        return target_np


class SqueezedCatTarget(TargetGenerator):
    """
    Generates squeezed cat states (even and odd parity superposition).
    
    Parameters:
    - alpha: Cat state amplitude
    - r: Squeezing parameter
    """
    
    def __init__(self, alpha=3.0, r=1.38, p=0):
        self.alpha = alpha
        self.r = r
        self.p=p
    
    def get_target_ket(self, cutoff_dim: int) -> np.ndarray:
        """Generate squeezed cat target state. Returns the even parity state."""
        print(f"Generating Squeezed Cat Target (alpha={self.alpha}, r={self.r}, p={self.p})...")
        
        prog = sf.Program(1)
        with prog.context as q:
            Catstate(a=self.alpha, p=self.p) | q[0]
            Sgate(r=self.r) | q[0]
        
        eng = sf.Engine("fock", backend_options={"cutoff_dim": cutoff_dim})
        state = eng.run(prog).state
        
        return state.ket()


class CubicPhaseTarget(TargetGenerator):
    """
    Generates Cubic Phase target state.
    
    Target: |γ, r, α⟩ = D(α) exp(iγQ^3) S(r) |0⟩
    
    Parameters:
    - gamma: Cubic phase strength
    - r: Squeezing parameter
    - alpha: Displacement (typically imaginary)
    """
    
    def __init__(self, gamma=-0.2, r=-0.7, alpha=1.25):
        self.gamma = gamma
        self.r = r
        self.alpha = alpha
    
    def get_target_ket(self, cutoff_dim: int) -> np.ndarray:
        """Generate cubic phase target state."""
        print(f"Generating Target Cubic Phase State (gamma={self.gamma}, r={self.r}, alpha={self.alpha})...")
        
        prog = sf.Program(1)
        with prog.context as q:
            # Order: vacuum -> squeezing -> cubic phase -> displacement
            Sgate(self.r) | q[0]
            
            # Cubic Phase: exp(i * gamma * Q^3)
            Vgate(self.gamma * 2) | q[0]
            
            # Displacement (typically imaginary)
            z = 1j * self.alpha
            r = np.abs(z)
            theta = np.angle(z)
            Dgate(r, theta) | q[0]

        eng = sf.Engine("fock", backend_options={"cutoff_dim": cutoff_dim})
        state = eng.run(prog).state
        
        return state.ket()


class QuarticPhaseTarget(TargetGenerator):
    """
    Generates Quartic Phase target state.
    
    Target: |δ, s⟩ = exp(i * δ * Q^4) S(s) |0⟩
    
    Requires higher cutoff_dim (>= 60) due to non-Gaussian state complexity.
    
    Parameters:
    - delta: Quartic phase strength
    - s_r: Squeezing parameter (default 1)
    """
    
    def __init__(self, delta=0.05, s_r=1):
        self.delta = delta
        self.s_r = s_r
    
    def get_target_ket(self, cutoff_dim: int) -> np.ndarray:
        """Generate quartic phase target state via matrix exponentiation."""
        print(f"Generating Target Quartic Phase State (delta={self.delta}, cutoff={cutoff_dim})...")
        
        # 1. Generate Squeezed State S(s)|0> using SF
        prog = sf.Program(1)
        with prog.context as q:
            Sgate(-self.s_r) | q[0]

        eng = sf.Engine("fock", backend_options={"cutoff_dim": cutoff_dim})
        state = eng.run(prog).state
        ket_init = state.ket()

        # 2. Precompute x^4 operator
        dim = cutoff_dim
        sqrt_n = np.sqrt(np.arange(1, dim))
        import scipy.sparse as sp
        
        a_op = sp.diags([sqrt_n], [1], shape=(dim, dim), format='csr')
        a_dag_op = a_op.T
        x_op = (a_op + a_dag_op) / np.sqrt(2)
        x2 = x_op.dot(x_op)
        x4 = x2.dot(x2)
        x4_dense = x4.toarray()
        
        # 3. Matrix Exponentiation U = exp(i * delta * x^4)
        U_quartic = scipy.linalg.expm(1j * self.delta * x4_dense)
        
        # 4. Apply U to ket
        target_ket = U_quartic @ ket_init
        
        # 5. Normalize
        norm = np.linalg.norm(target_ket)
        target_ket = target_ket / norm
        
        return target_ket


class CubicResourceTarget(TargetGenerator):
    """
    Generates Cubic Phase Resource State for the Gadget circuit.
    
    Used as a target for non-Gaussian state preparation in 2-mode circuits.
    
    Parameters:
    - a: Normalization constant (default 0.61)
    """
    
    def __init__(self, a=0.61):
        self.a = a
    
    def get_target_ket(self, cutoff_dim: int) -> np.ndarray:
        """Generate cubic resource state."""
        print("Initializing Cubic Phase Target State for Gadget...")
        base_ket = np.zeros(cutoff_dim, dtype=np.complex128)
        base_ket[0] = 1.0
        base_ket[1] = 1j * self.a * np.sqrt(1.5)
        base_ket[3] = 1j * self.a
        base_ket /= np.linalg.norm(base_ket)
        return base_ket
    


class CoreGKPTarget(TargetGenerator):
    """
    Generates an approximate GKP state using the "Core + Squeezing" (Stellar Representation)
    method described in the paper.
    
    This class requires the 'GKP_core_coefficients.csv' file containing the optimized
    squeezing parameters (r) and Fock coefficients (c_n).
    
    Parameters:
    - csv_path: Path to the GKP coefficients CSV.
    - n_max: The stellar rank / truncation (2, 4, 6, 8, 10, 12).
    - delta_db: Target Delta in dB (e.g., 10.0).
    - mu: Logical state (0 or 1).
    """
    
    def __init__(self, csv_path='GKP_core_coefficients.csv', n_max=4, delta_db=10.0, mu=0):
        self.csv_path = csv_path
        self.n_max = n_max
        self.delta_db = delta_db
        self.mu = mu
    
    def get_target_ket(self, cutoff_dim: int) -> np.ndarray:
        """Load coefficients from CSV and apply squeezing to the core state."""
        print(f"Generating Core GKP Target (|{self.mu}>_A, n_max={self.n_max}, Delta={self.delta_db}dB)...")

        # 1. Load CSV data
        try:
            df = pd.read_csv(self.csv_path)
        except FileNotFoundError:
            raise FileNotFoundError(f"GKP coefficients file not found at {self.csv_path}. Please ensure the CSV is present.")

        # 2. Find matching row
        # Use a small tolerance for float comparison of Delta (dB)
        row = df[(df['n_max'] == self.n_max) & (np.abs(df['Delta (dB)'] - self.delta_db) < 1e-5)]

        if row.empty:
            # Fallback: find closest delta_db if exact match not found
            closest_idx = (df[df['n_max'] == self.n_max]['Delta (dB)'] - self.delta_db).abs().idxmin()
            row = df.loc[[closest_idx]]
            print(f"Warning: Exact Delta={self.delta_db}dB not found for n_max={self.n_max}. Using closest match: {row['Delta (dB)'].values[0]}dB")

        # 3. Extract squeezing (r) and core coefficients (c)
        # Suffix '0' for logical 0, '1' for logical 1
        sfx = str(self.mu)
        r_db = row[f'r{sfx} (dB)'].values[0]

        # Convert r_dB to r (Strawberry Fields parameter)
        # Paper Eq A3: r = ln(10)/20 * r_db
        r_param = (np.log(10) / 20.0) * r_db

        # Collect even Fock coefficients up to n_max
        # CSV columns are named c0_0, c0_2, ... or c1_0, c1_2, ...
        base_ket = np.zeros(cutoff_dim, dtype=np.complex128)
        for n in range(0, self.n_max + 1, 2):
            col_name = f'c{sfx}_{n}'
            if col_name in row.columns:
                base_ket[n] = row[col_name].values[0]

        # Ensure core is normalized
        base_ket /= np.linalg.norm(base_ket)

        # Print the core state coefficients
        print(f"Core state (stellar representation) for n_max={self.n_max}:")
        for n, val in enumerate(base_ket):
            if np.abs(val) > 1e-6:
                # Print real part if imaginary is negligible, otherwise show complex
                out_val = val.real if np.abs(val.imag) < 1e-8 else val
                print(f"  |{n}>: {out_val:.6f}")

        # 4. Use Strawberry Fields to apply Squeezing to the core ket
        prog = sf.Program(1)
        with prog.context as q:
            Ket(base_ket) | q[0]
            # Sgate applies exp(0.5 * r * (exp(-i*phi)a^2 - exp(i*phi)a_dag^2))
            # The paper assumes real squeezing
            # Sgate(r_param) | q[0]

        eng = sf.Engine("fock", backend_options={"cutoff_dim": cutoff_dim})
        state = eng.run(prog).state

        return state.ket()
