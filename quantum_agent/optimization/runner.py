import time
import numpy as np
from scipy.optimize import basinhopping
import strawberryfields as sf

from quantum_agent.components.targets import TargetGenerator
from quantum_agent.optimization.interfaces import OptimizableCircuit
from quantum_agent.envs.modular_env import fidelity_pure_state

class OptimizationRunner:
    """
    Manages the Basin Hopping optimization process.
    """
    def __init__(self, 
                 circuit: OptimizableCircuit, 
                 target_gen: TargetGenerator, 
                 cutoff_dim: int,
                 post_select_dict: dict,
                 alpha_prob: float = 1.0,
                 penalty_strength: float = 10.0):
        
        self.circuit = circuit
        self.cutoff_dim = cutoff_dim
        self.post_select_dict = post_select_dict
        self.alpha_prob = alpha_prob
        self.penalty_strength = penalty_strength
        
        # Generate target ket once
        self.target_ket = target_gen.get_target_ket(cutoff_dim)
        self.eval_count = 0
        
    def _loss_function(self, params):
        """
        Calculates loss: -Fidelity - alpha * Prob + Penalties
        """
        self.eval_count += 1
        # 1. Run Circuit
        eng = sf.Engine("fock", backend_options={"cutoff_dim": self.cutoff_dim})
        
        try:
            state = self.circuit.run_circuit(params, eng)
            # Trace of pure state ket^2 checks norm conservation (truncation error)
            ket = state.ket().flatten()
            norm_in_trace = np.real(np.vdot(ket, ket))
        except Exception as e:
            # If simulation fails (e.g. numerical instability), return high loss
            return 100.0

        # 2. Extract Output
        out_ket, prob = self.circuit.extract_output(state, self.post_select_dict)
        
        # 3. Calculate Fidelity
        fid = fidelity_pure_state(self.target_ket, out_ket)
        
        # 4. Calculate Penalties (Truncation error penalty)
        # Ideally pure state trace should be 1.0. 
        # Deviation indicates significant energy in higher Fock states (truncation).
        truncation_error = 1.0 - np.abs(norm_in_trace)
        
        # 5. Total Loss
        # Maximize F and P -> Minimize -F - alpha*P
        loss = -fid - (self.alpha_prob * prob) + (self.penalty_strength * truncation_error)
        
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
        
        print(f"Starting Basin Hopping (n_iter={n_iter})...")
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
        
        # Evaluate final result details
        eng = sf.Engine("fock", backend_options={"cutoff_dim": self.cutoff_dim})
        final_state = self.circuit.run_circuit(result.x, eng)
        fin_ket, fin_prob = self.circuit.extract_output(final_state, self.post_select_dict)
        fin_fid = fidelity_pure_state(self.target_ket, fin_ket)
        
        return {
            "x": result.x,
            "loss": result.fun,
            "fidelity": fin_fid,
            "probability": fin_prob,
            "duration": duration,
            "message": result.message
        }
