
import scipy.integrate
# 2. Check if the patch is needed to avoid errors
if not hasattr(scipy.integrate, 'simps'):
    print("Monkey patching scipy.integrate: 'simps' not found. Pointing to 'simpson'.")
    # 3. Create the 'simps' attribute and point it to the existing 'simpson' function.
    scipy.integrate.simps = scipy.integrate.simpson
else:
    print("'simps' already exists in scipy.integrate. No patch needed.")

import time
import numpy as np
from scipy.optimize import basinhopping
import strawberryfields as sf
from thewalrus.quantum import state_vector, density_matrix_element, pure_state_amplitude

from quantum_agent.components.targets import TargetGenerator
from quantum_agent.optimization.interfaces import OptimizableCircuit
from quantum_agent.envs.modular_env import fidelity_pure_state

class GaussianOptimizationRunner:
    """
    Manages the optimization process using the Gaussian backend and TheWalrus.
    
    This is significantly faster than the Fock backend for Gaussian circuits
    (Squeezing, Displacement, Beamsplitters, Rotation) combined with 
    Photon Number Resolving (PNR) post-selection.
    """
    def __init__(self, 
                 circuit: OptimizableCircuit, 
                 target_gen: TargetGenerator, 
                 cutoff_dim: int,
                 post_select_dict: dict,
                 alpha_prob: float = 1.0):
        
        self.circuit = circuit
        self.cutoff_dim = cutoff_dim
        self.post_select_dict = post_select_dict
        self.alpha_prob = alpha_prob
        
        # Identify measured modes and their values for probability calculation
        self.measured_modes = list(post_select_dict.keys())
        self.measured_values = list(post_select_dict.values())
        
        # Generate target ket once
        self.target_ket = target_gen.get_target_ket(cutoff_dim)

        # OPTIMIZATION: Identify non-zero target indices to avoid computing unnecessary Hafnians
        # This speeds up fidelity calculation significantly for sparse targets (e.g. Cubic Resource)
        self.nonzero_indices = np.where(np.abs(self.target_ket) > 1e-6)[0]
        self.nonzero_coeffs = self.target_ket[self.nonzero_indices]
        print(f"[GaussianRunner] Sparse Optimization: Computing {len(self.nonzero_indices)}/{len(self.target_ket)} amplitudes.")
        
    def _loss_function(self, params):
        """
        Calculates loss: -Fidelity - alpha * Prob
        """
        # 1. Run Circuit on Gaussian Backend
        eng = sf.Engine("gaussian")
        
        try:
            # OptimizableCircuit.run_circuit usually expects an engine and returns a Result/State
            state = self.circuit.run_circuit(params, eng)
            mu = state.means()
            cov = state.cov()
        except Exception:
            # If simulation fails (e.g. unstable parameters), return high loss
            return 100.0

        # 2. Calculate Probability of Post-Selection
        # We need the reduced Gaussian parameters of the *measured* modes
        try:
            mu_meas, cov_meas = state.reduced_gaussian(self.measured_modes)
            # density_matrix_element calculates <n| rho |m>. We want diagonal <n| rho |n>
            prob = density_matrix_element(mu_meas, cov_meas, self.measured_values, self.measured_values).real
        except Exception:
            return 100.0

        # Avoid division by zero if probability is extremely low
        if prob < 1e-12:
            # Penalty for zero probability (failed post-selection)
            return 10.0

        # 3. FAST Fidelity Calculation (Sparse)
        # Instead of computing the full state_vector (which calculates 'cutoff' Hafnians),
        # we only calculate amplitudes for indices where target_ket is non-zero.
        try:
            overlap = 0.0 + 0.0j
            n_modes = state.num_modes
            
            # Determine unmeasured mode index
            # (Assumes single mode output for sparse optimization currently)
            all_idxs = set(range(n_modes))
            meas_idxs = set(self.measured_modes)
            rem_idxs = list(all_idxs - meas_idxs)
            
            if len(rem_idxs) == 1:
                target_mode_idx = rem_idxs[0]
                
                # Base pattern with measured values
                pattern = [0] * n_modes
                for m, v in self.post_select_dict.items():
                    pattern[m] = v
                
                # Loop only over relevant indices
                for i, idx in enumerate(self.nonzero_indices):
                    pattern[target_mode_idx] = int(idx)
                    
                    # pure_state_amplitude computes <n_out, n_meas | U | 0>
                    # This is much faster than computing the whole vector if sparse
                    amp = pure_state_amplitude(mu, cov, pattern)
                    
                    # Accumulate overlap: <target|psi>
                    overlap += np.conj(self.nonzero_coeffs[i]) * amp
                    
                # Normalize by sqrt(Prob)
                overlap /= np.sqrt(prob)
                fid = np.abs(overlap)**2

            else:
                # Fallback to slow full vector if multi-mode output (rare for gadgets)
                raw_ket = state_vector(mu, cov, post_select=self.post_select_dict, cutoff=self.cutoff_dim)
                ket = raw_ket / np.sqrt(prob)
                fid = fidelity_pure_state(self.target_ket, ket)

        except Exception:
            return 100.0
        
        # 4. Total Loss
        # Maximize F and P -> Minimize -F - alpha*P
        loss = -fid - (self.alpha_prob * prob)
        
        return loss

    def callback(self, x, f, accept):
        """Optional callback to print progress."""
        if accept:
            print(f"  [Accept] Loss: {f:.5f}")

    def run(self, n_iter=20, method="SLSQP"):
        """
        Runs the global optimization using Basin Hopping.
        """
        # Initial guess: Random within bounds
        bounds = self.circuit.parameter_bounds
        x0 = np.array([np.random.uniform(l, h) for l, h in bounds])
        
        minimizer_kwargs = {
            "method": method,
            "bounds": bounds,
            "tol": 1e-6
        }
        
        print(f"Starting Gaussian Basin Hopping (n_iter={n_iter})...")
        start_time = time.time()
        
        result = basinhopping(
            self._loss_function,
            x0,
            niter=n_iter,
            minimizer_kwargs=minimizer_kwargs,
            callback=self.callback,
            stepsize=0.5
        )
        
        duration = time.time() - start_time
        
        # --- Evaluate final result details ---
        eng = sf.Engine("gaussian")
        state = self.circuit.run_circuit(result.x, eng)
        mu, cov = state.means(), state.cov()
        
        # Recalculate final metrics
        mu_meas, cov_meas = state.reduced_gaussian(self.measured_modes)
        fin_prob = density_matrix_element(mu_meas, cov_meas, self.measured_values, self.measured_values).real
        
        if fin_prob > 1e-12:
            raw_ket = state_vector(mu, cov, post_select=self.post_select_dict, cutoff=self.cutoff_dim)
            fin_ket = raw_ket / np.sqrt(fin_prob)
            fin_fid = fidelity_pure_state(self.target_ket, fin_ket)
        else:
            fin_fid = 0.0
        
        return {
            "x": result.x,
            "loss": result.fun,
            "fidelity": fin_fid,
            "probability": fin_prob,
            "duration": duration,
            "message": result.message
        }
