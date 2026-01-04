"""
Optimization runner for time-domain multiplexed circuits using Beam Search.
"""
import time
import numpy as np
from scipy.optimize import basinhopping
import strawberryfields as sf
from strawberryfields.ops import Ket

from quantum_agent.optimization.time_interfaces import TimeMultiplexedCircuit
from quantum_agent.components.targets import TargetGenerator
from quantum_agent.envs.modular_env import fidelity_max_rotation

class TimeDomainRunner:
    """
    Optimizes time-domain circuits using a Beam Search strategy.
    
    This runner executes the circuit step-by-step, maintaining a 'beam' of the most 
    probable trajectories (measurement outcomes).
    
    Phases:
    A. Evolution: Serial execution of the circuit step for each active branch.
    B. Virtual Branching: Vectorized calculation of all possible measurement outcome probabilities.
    C. Pruning: Vectorized selection of the top-K probable paths (Beam Search).
    D. Realization: Lazy projection and normalization of the selected states.
    """
    def __init__(self, 
                 circuit: TimeMultiplexedCircuit, 
                 target_gen: TargetGenerator, 
                 cutoff_dim: int,
                 beam_width: int = 5,
                 penalty_strength: float = 10.0):
        
        self.circuit = circuit
        self.target_ket = target_gen.get_target_ket(cutoff_dim)
        self.cutoff_dim = cutoff_dim
        self.beam_width = beam_width
        self.penalty_strength = penalty_strength
        self.eval_count = 0
        
    def _loss_function(self, flat_params):
        """
        Calculates loss: -Expected_Fidelity + Penalties
        """
        self.eval_count += 1
        
        # 0. Setup
        mapped_params = self.circuit.map_parameters(flat_params)
        meas_mode, meas_cutoff = self.circuit.get_measurement_spec()
        
        # Initialize Beam: Single vacuum state on Mode 0 (Loop)
        # We assume Mode 0 is the persistent loop mode.
        init_ket = np.zeros(self.cutoff_dim, dtype=np.complex128)
        init_ket[0] = 1.0
        
        # active_branches: List of (ket_mode_0, path_probability)
        active_branches = [(init_ket, 1.0)]
        
        # Reuse engine for performance
        eng = sf.Engine("fock", backend_options={"cutoff_dim": self.cutoff_dim})
        
        # 1. Time Loop (Step 0 to T-1)
        for step in range(self.circuit.steps):
            step_params = mapped_params[step]
            
            # Containers for vectorized processing
            branch_raw_probs = [] # List of 1D arrays (meas_cutoff,)
            branch_tensors = []   # List of 2D arrays (cutoff, meas_cutoff)
            
            # --- Phase A: Evolution (Loop unavoidable for SF) ---
            for parent_ket, _ in active_branches:
                eng.reset()
                
                # Load state into Mode 0 (Loop)
                prog_prep = sf.Program(2)
                with prog_prep.context as q:
                    Ket(parent_ket) | q[0]
                eng.run(prog_prep)
                
                # Run Step (Applies Unitary)
                result = self.circuit.run_step(None, step, step_params, eng)
                state = result.state
                
                # Extract State Tensor: Indices [q0, q1] (Loop, Ancilla)
                full_ket = state.ket()
                
                # Optimization: Only keep columns up to meas_cutoff for Mode 1
                if full_ket.shape[1] > meas_cutoff:
                    sliced_ket = full_ket[:, :meas_cutoff]
                else:
                    sliced_ket = full_ket
                    
                # Store tensor for potential projection later
                branch_tensors.append(sliced_ket)
                
                # Calculate Probabilities for Mode 1 (Sum over Mode 0)
                # P(n) = sum_m |<m, n| psi>|^2
                probs_n = np.sum(np.abs(sliced_ket)**2, axis=0)
                branch_raw_probs.append(probs_n)

            # --- Phase B: Vectorized Virtual Branching ---
            
            # P_parent: (N_branches,)
            P_parent = np.array([b[1] for b in active_branches])
            
            # P_raw: (N_branches, meas_cutoff)
            P_raw = np.stack(branch_raw_probs) 
            
            # Joint Probability: P(parent) * P(outcome|parent)
            # Broadcasting: (N, 1) * (N, M) -> (N, M)
            P_total = P_parent[:, None] * P_raw
            
            # --- Phase C: Vectorized Pruning ---
            
            flat_P = P_total.flatten()
            current_n_candidates = flat_P.size
            k = min(self.beam_width, current_n_candidates)
            
            # Find indices of the top K probabilities
            # np.argpartition puts the k largest elements at the end (unsorted)
            top_indices = np.argpartition(flat_P, -k)[-k:]
            
            # Resolve flattened indices back to (parent_index, outcome_index)
            parent_indices, outcome_indices = np.unravel_index(top_indices, P_total.shape)
            
            # --- Phase D: Lazy Realization ---
            
            new_branches = []
            
            # Iterate selected paths to realize states
            for p_idx, outcome in zip(parent_indices, outcome_indices):
                prob = P_total[p_idx, outcome]
                
                if prob < 1e-12:
                    continue
                
                # Project: Slice the tensor column corresponding to measurement outcome 'n'
                # |psi_next>_0 = <n|_1 |psi>_01
                # Numpy slice creates a copy or view, usually sufficient for next step
                raw_new_ket = branch_tensors[p_idx][:, outcome]
                
                # Normalize
                norm = np.linalg.norm(raw_new_ket)
                if norm > 1e-9:
                    normalized_ket = raw_new_ket / norm
                    new_branches.append((normalized_ket, prob))
            
            active_branches = new_branches
            
            if not active_branches:
                # All branches died (numerical issues)
                return 100.0

        # --- Final Objective ---
        expected_fidelity = 0.0
        total_prob = 0.0
        
        # Sum Expected Fidelity over surviving branches
        for ket, prob in active_branches:
            fid = fidelity_max_rotation(self.target_ket, ket)
            expected_fidelity += prob * fid
            total_prob += prob
            
        # Penalties
        # 1. Total Probability Loss (indicates truncation or dropped branches)
        loss_prob = np.abs(1.0 - total_prob)
        
        return -expected_fidelity + (self.penalty_strength * loss_prob)

    def callback(self, x, f, accept):
        if accept:
            print(f"  [Accept] Loss: {f:.5f} (Evals: {self.eval_count})")
        self.eval_count = 0

    def run(self, n_iter=20, method="SLSQP"):
        """
        Runs the global optimization.
        """
        # Construct full parameter bounds
        per_step_bounds = self.circuit.per_step_parameter_bounds
        if self.circuit.time_invariant:
            full_bounds = per_step_bounds
        else:
            full_bounds = per_step_bounds * self.circuit.steps
            
        # Initial guess
        x0 = np.array([np.random.uniform(l, h) for l, h in full_bounds])
        
        minimizer_kwargs = {
            "method": method,
            "bounds": full_bounds,
            "tol": 1e-4
        }
        
        print(f"Starting Time-Domain Beam Search (Width={self.beam_width}, Steps={self.circuit.steps})...")
        start_time = time.time()
        
        result = basinhopping(
            self._loss_function,
            x0,
            niter=n_iter,
            minimizer_kwargs=minimizer_kwargs,
            callback=self.callback,
            stepsize=0.5
        )
        
        return {
            "x": result.x,
            "loss": result.fun,
            "duration": time.time() - start_time,
            "message": result.message
        }