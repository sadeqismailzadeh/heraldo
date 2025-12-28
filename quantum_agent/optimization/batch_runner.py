import time
import numpy as np
from scipy.optimize import basinhopping
import strawberryfields as sf

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
                 penalty_strength: float = 10.0):
        
        self.circuit = circuit
        self.cutoff_dim = cutoff_dim
        self.measure_modes = measure_modes
        self.penalty_strength = penalty_strength
        
        # Precompute all target kets once
        self.target_kets = [gen.get_target_ket(cutoff_dim) for gen in target_gens]
        print(f"[BatchRunner] Loaded {len(self.target_kets)} target states.")
        
    def _loss_function(self, params):
        """
        Calculates loss: -Expected_Fidelity + Penalties
        """
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
        # Returns list of (normalized_ket, prob, outcome_tuple)
        branches = self.circuit.extract_all_outputs(state, self.measure_modes)
        
        expected_fidelity = 0.0
        total_prob = 0.0
        
        # 3. Calculate Expected Fidelity
        for branch_ket, prob, _ in branches:
            total_prob += prob
            
            # Find best match among all targets for this specific branch
            # We use fidelity_max_rotation to be phase insensitive
            best_branch_fid = 0.0
            for target_ket in self.target_kets:
                fid = fidelity_max_rotation(target_ket, branch_ket)
                if fid > best_branch_fid:
                    best_branch_fid = fid
            
            # Accumulate expectation
            expected_fidelity += prob * best_branch_fid

        # 4. Calculate Penalties
        
        # Truncation error: 1.0 - global_norm
        truncation_penalty = self.penalty_strength * abs(1.0 - global_norm)
        
        # Sum of probabilities penalty: Should sum to ~1.0 (or global_norm)
        # If extraction missed valid states (unlikely given loop) or cutoff too small
        prob_sum_penalty = self.penalty_strength * abs(1.0 - total_prob)
        
        # 5. Total Loss
        # Maximize Expected Fidelity -> Minimize negative
        loss = -expected_fidelity + truncation_penalty + prob_sum_penalty
        
        return loss

    def callback(self, x, f, accept):
        """Optional callback to print progress."""
        if accept:
            print(f"  [Accept] Loss: {f:.5f}")

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
        branches = self.circuit.extract_all_outputs(final_state, self.measure_modes)
        
        final_expected_fid = 0.0
        branch_details = []
        
        for branch_ket, prob, outcome in branches:
            best_fid = 0.0
            best_target_idx = -1
            
            for i, target_ket in enumerate(self.target_kets):
                fid = fidelity_max_rotation(target_ket, branch_ket)
                if fid > best_fid:
                    best_fid = fid
                    best_target_idx = i
            
            final_expected_fid += prob * best_fid
            
            # Store details for top probabilities
            if prob > 0.01:
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