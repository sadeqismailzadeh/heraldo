import time
import numpy as np
from scipy.optimize import basinhopping
import strawberryfields as sf


# disable caching to save memory for large cutoff dims
from quantum_agent.patches.sf_operations_no_cache import disable_fock_caching
disable_fock_caching()

# optimized loss channel
from quantum_agent.patches.monitored_loss_measure_fock_patch import MonitoredLossMeasureFock, patch_monitored_loss_measure_fock, decode_measurement_result
patch_monitored_loss_measure_fock()

from quantum_agent.patches.beamsplitter_patch import patch_beamsplitter
patch_beamsplitter()

from quantum_agent.patches.prepare_multimode_patch import patch_prepare_multimode
patch_prepare_multimode()


from quantum_agent.components.targets import TargetGenerator
from quantum_agent.optimization.interfaces import OptimizableCircuit
from quantum_agent.envs.modular_env import fidelity_max_rotation

class BatchOptimizationRunner:
    """
    Manages the optimization process considering ALL measurement outcomes.
    
    Objective: Maximize Expected Fidelity = Sum( P(outcome) * Max_i(F(state_outcome, target_i)) )
    """
    def __init__(self, 
                 circuit: OptimizableCircuit, 
                 target_gens: list[TargetGenerator], 
                 cutoff_dim: int,
                 measure_modes: list[int],
                 penalty_strength: float = 10.0,
                 success_threshold: float = 0.98,
                 success_weight: float = 10.0,
                 max_post_select: int = None):
        
        self.circuit = circuit
        self.cutoff_dim = cutoff_dim
        self.measure_modes = measure_modes
        self.penalty_strength = penalty_strength
        self.success_threshold = success_threshold
        self.success_weight = success_weight
        self.max_post_select = max_post_select
        
        # Precompute all target kets once
        self.target_kets = [gen.get_target_ket(cutoff_dim) for gen in target_gens]
        print(f"[BatchRunner] Loaded {len(self.target_kets)} target states.")
        self.eval_count = 0
        
    def _loss_function(self, params):
        """
        Calculates loss: -Expected_Fidelity + Penalties
        """
        self.eval_count += 1
        # 1. Run Circuit
        eng = sf.Engine("fock", backend_options={"cutoff_dim": self.cutoff_dim})
        
        try:
            state = self.circuit.run_circuit(params, eng)
            
            # Check for global truncation error (norm conservation)
            # The trace of the density matrix (or norm of ket) should be 1.0
            ket = state.ket().flatten()
            global_norm = np.real(np.vdot(ket, ket))
        except Exception:
            # If simulation fails (e.g. numerical instability), return high loss
            return 100.0

        # 2. Extract All Possible Outputs (Branches)
        # Returns arrays directly
        kets, probs, outcomes = self.circuit.extract_all_outputs(state, self.measure_modes)
        
        expected_fidelity = 0.0
        soft_success_prob = 0.0
        
        # Sigmoid steepness for soft indicator
        steepness = 100.0
        
        # 3. Vectorized Expected Fidelity and Soft Success Probability
        total_prob = np.sum(probs)
        
        # Filter branches: outcome sum > 0 AND prob > 1e-6
        mask = (np.sum(outcomes, axis=1) > 0) & (probs > 1e-6)
        if self.max_post_select is not None:
             mask &= (np.max(outcomes, axis=1) <= self.max_post_select)
             
        filtered_kets = kets[mask]
        filtered_probs = probs[mask]

        if len(filtered_kets) > 0:
            # filtered_kets is already an array
            # Targets shape: (N_targets, D)
            targets_arr = np.array(self.target_kets)

            # --- Batched Fidelity Max Rotation ---
            # We need max_phi |<target | e^{i n phi} | state>|^2
            # This is equivalent to: max |FFT(conj(state) * target)|^2
            
            # Broadcast element-wise multiplication: 
            # (N_branches, 1, D) * (1, N_targets, D) -> (N_branches, N_targets, D)
            product_matrix = np.conj(filtered_kets[:, None, :]) * targets_arr[None, :, :]
            
            # Perform Batched FFT along the Fock dimension (axis -1)
            # Using n=2048 to match standard resolution for phase optimization
            fft_vals = np.fft.fft(product_matrix, n=256, axis=-1)
            
            # Compute squared magnitude and find max over rotation angle (FFT axis)
            # Shape: (N_branches, N_targets)
            pairwise_fidelities = np.max(np.abs(fft_vals)**2, axis=-1)
            
            # Find best fidelity across all targets for each branch
            # Shape: (N_branches,)
            best_fids = np.max(pairwise_fidelities, axis=1)

            # --- Vectorized Aggregation ---
            
            # Soft Indicator: sigmoid(fidelity)
            sigmoids = 1.0 / (1.0 + np.exp(-steepness * (best_fids - self.success_threshold)))
            soft_success_prob = np.sum(filtered_probs * sigmoids)

            # Logarithmic Reward
            infidelities = np.maximum(1.0 - best_fids, 1e-3)
            log_vals = -np.log10(infidelities)
            expected_fidelity = np.sum((filtered_probs**0.1) * (best_fids*log_vals)**2)

        # 4. Calculate Penalties
        
        # Truncation error: 1.0 - global_norm
        truncation_penalty = self.penalty_strength * abs(1.0 - global_norm)
        
        # Sum of probabilities penalty: Should sum to ~1.0 (or global_norm)
        # If extraction missed valid states (unlikely given loop) or cutoff too small
        prob_sum_penalty = self.penalty_strength * abs(1.0 - total_prob)
        
        # 5. Total Loss
        # Maximize Expected Fidelity and Soft Success Probability
        loss = -expected_fidelity - (self.success_weight * soft_success_prob) + \
               truncation_penalty + prob_sum_penalty
        
        return loss

    def callback(self, x, f, accept):
        """Optional callback to print progress."""
        if accept:
            print(f"  [Accept] Loss: {f:.5f} (Evals: {self.eval_count})")
        self.eval_count = 0

    def run(self, n_iter=20, method="SLSQP"):
        """
        Runs the global optimization.
        """
        # Initial guess: Random within bounds
        bounds = self.circuit.parameter_bounds
        x0 = np.array([np.random.uniform(l, h) for l, h in bounds])
        
        minimizer_kwargs = {
            "method": method,
            "bounds": bounds,
            "tol": 1e-6
        }
        
        print(f"Starting Batch Basin Hopping (n_iter={n_iter})...")
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
        
        # --- Evaluate final statistics ---
        eng = sf.Engine("fock", backend_options={"cutoff_dim": self.cutoff_dim})
        final_state = self.circuit.run_circuit(result.x, eng)
        kets, probs, outcomes = self.circuit.extract_all_outputs(final_state, self.measure_modes)
        
        final_expected_fid = 0.0
        branch_details = []
        
        for i in range(len(probs)):
            branch_ket = kets[i]
            prob = probs[i]
            outcome = tuple(outcomes[i])

            best_fid = 0.0
            best_target_idx = -1
            
            for i, target_ket in enumerate(self.target_kets):
                fid = fidelity_max_rotation(target_ket, branch_ket)
                if fid > best_fid:
                    best_fid = fid
                    best_target_idx = i
            
            final_expected_fid += prob * best_fid
            
            # Store details for all branches
            branch_details.append({
                "outcome": outcome,
                "prob": prob,
                "fidelity": best_fid,
                "target_idx": best_target_idx
            })
        
        # Sort details by probability descending
        branch_details.sort(key=lambda x: x["prob"], reverse=True)

        return {
            "x": result.x,
            "loss": result.fun,
            "expected_fidelity": final_expected_fid,
            "duration": duration,
            "branches": branch_details,
            "message": result.message
        }
