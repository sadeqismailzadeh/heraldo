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
from scipy.special import factorial, comb
import pandas as pd

class TargetGenerator(abc.ABC):
    """Responsible for generating the target state ket."""
    @abc.abstractmethod
    def get_target_ket(self, cutoff_dim: int) -> np.ndarray:
        pass


# --- Target Implementations ---


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
    


class CatTarget(TargetGenerator):
    """
    Generates squeezed cat states (even and odd parity superposition).
    
    Parameters:
    - alpha: Cat state amplitude
    - r: Squeezing parameter
    """
    
    def __init__(self, alpha=3.0, p=0):
        self.alpha = alpha
        self.p=p
    
    def get_target_ket(self, cutoff_dim: int) -> np.ndarray:
        """Generate squeezed cat target state. Returns the even parity state."""
        print(f"Generating Squeezed Cat Target (alpha={self.alpha}, p={self.p})...")
        
        prog = sf.Program(1)
        with prog.context as q:
            Catstate(a=self.alpha, p=self.p) | q[0]
        
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
    
    def __init__(self, csv_path='GKP_core_coefficients.csv', n_max=4, delta_db=10.0, mu=0, apply_squeezing=False):
        self.csv_path = csv_path
        self.n_max = n_max
        self.delta_db = delta_db
        self.mu = mu
        self.apply_squeezing = apply_squeezing  # To store the squeezing parameter used
    
    def get_target_ket(self, cutoff_dim: int) -> np.ndarray:
        """Load coefficients from CSV and apply squeezing to the core state."""
        verbose = 0
        if (verbose > 0):
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

        if (verbose > 0):
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
            if self.apply_squeezing:
               Sgate(r_param) | q[0]

        eng = sf.Engine("fock", backend_options={"cutoff_dim": cutoff_dim})
        state = eng.run(prog).state

        return state.ket()


class BinomialCodeTarget(TargetGenerator):
    """
    Generates Binomial Code target states.

    Defined by (N, S) parameters:
    |W_mu> = (1/sqrt(2^N)) * sum_{p} sqrt(binom(N+1, p)) |p(S+1)>

    where p sums over even integers for mu=0 (logical 0/up)
    and odd integers for mu=1 (logical 1/down).

    Parameters:
    - N: Order of the code (max(L, G, 2D)).
    - S: Spacing parameter (L + G).
    - mu: Logical state (0 or 1).
    """

    def __init__(self, N=1, S=1, mu=0):
        self.N = N
        self.S = S
        self.mu = mu

    def get_target_ket(self, cutoff_dim: int) -> np.ndarray:
        print(f"Generating Binomial Code Target (N={self.N}, S={self.S}, mu={self.mu})...")

        target_ket = np.zeros(cutoff_dim, dtype=np.complex128)

        # Iterate p from 0 to N+1
        # mu=0 -> even p, mu=1 -> odd p
        start_p = 1 if self.mu == 1 else 0

        for p in range(start_p, self.N + 2, 2):
            fock_n = p * (self.S + 1)

            if fock_n < cutoff_dim:
                # coeff = sqrt(binom(N+1, p))
                c = np.sqrt(comb(self.N + 1, p))
                target_ket[fock_n] = c
            else:
                pass

        # Normalize
        norm = np.linalg.norm(target_ket)
        if norm > 1e-9:
            target_ket /= norm

        return target_ket


