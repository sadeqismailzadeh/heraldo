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
        active_kets = np.zeros((1, self.cutoff_dim), dtype=np.complex128)
        active_kets[0, 0] = 1.0
        active_probs = np.array([1.0])
        
        # 1. Time Loop (Step 0 to T-1)
        for step in range(self.circuit.steps):
            step_params = mapped_params[step]
            
            # Containers for vectorized processing
            branch_raw_probs = [] # List of 1D arrays (meas_cutoff,)
            branch_tensors = []   # List of 2D arrays (cutoff, meas_cutoff)
            
            # --- Phase A: Evolution (Loop unavoidable for SF) ---
            for idx in range(len(active_kets)):
                parent_ket = active_kets[idx]
                
                # Instantiate engine per branch to ensure clean state (avoids _trunc errors)
                eng = sf.Engine("fock", backend_options={"cutoff_dim": self.cutoff_dim})
                
                # Load state into Mode 0 (Loop)
                prog_prep = sf.Program(2)
                with prog_prep.context as q:
                    Ket(parent_ket) | q[0]
                
                # Run Preparation
                eng.run(prog_prep)
                
                # Run Step (Applies Unitary)
                # Note: circuit.run_step calls eng.run(prog), which appends to the execution
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
            P_parent = active_probs
            
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
            
            # --- Phase D: Vectorized Realization ---
            
            # Stack tensors: (N_parents, cutoff, meas_cutoff)
            tensor_stack = np.stack(branch_tensors)
            
            # Extract kets: (K, cutoff) via fancy indexing
            # tensor_stack[p, :, o] selects the specific columns
            raw_new_kets = tensor_stack[parent_indices, :, outcome_indices]
            
            # Normalize
            norms = np.linalg.norm(raw_new_kets, axis=1)
            
            # Probabilities for selected paths
            selected_probs = P_total[parent_indices, outcome_indices]
            
            # Filter
            mask = (norms > 1e-9) & (selected_probs > 1e-12)
            
            if not np.any(mask):
                return 100.0
            
            # Update Active Beam
            active_kets = raw_new_kets[mask] / norms[mask][:, None]
            active_probs = selected_probs[mask]

        # --- Final Objective (Vectorized) ---
        
        # Batched Fidelity Max Rotation
        # max_phi |<target | e^{i n phi} | state>|^2
        # Equivalent to max |FFT(conj(target) * state)|^2 along Fock axis
        
        # Product: (N_branches, D)
        prod = np.conj(self.target_ket) * active_kets
        
        # FFT (n=256 for phase resolution)
        fft_vals = np.fft.fft(prod, n=256, axis=-1)
        fidelities = np.max(np.abs(fft_vals)**2, axis=-1)
        
        expected_fidelity = np.sum((active_probs**0.1) * fidelities)
        total_prob = np.sum(active_probs)
            
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
        
        # --- Evaluate final details with history ---
        mapped_params = self.circuit.map_parameters(result.x)
        _, meas_cutoff = self.circuit.get_measurement_spec()

        # Initialize Beam
        active_kets = np.zeros((1, self.cutoff_dim), dtype=np.complex128)
        active_kets[0, 0] = 1.0
        active_probs = np.array([1.0])
        active_outcomes = [()]  # List of tuples

        for step in range(self.circuit.steps):
            step_params = mapped_params[step]

            branch_raw_probs = []
            branch_tensors = []

            # Phase A: Evolution
            for idx in range(len(active_kets)):
                parent_ket = active_kets[idx]
                eng = sf.Engine("fock", backend_options={"cutoff_dim": self.cutoff_dim})

                prog_prep = sf.Program(2)
                with prog_prep.context as q:
                    Ket(parent_ket) | q[0]
                eng.run(prog_prep)

                result_step = self.circuit.run_step(None, step, step_params, eng)
                state = result_step.state
                full_ket = state.ket()

                if full_ket.shape[1] > meas_cutoff:
                    sliced_ket = full_ket[:, :meas_cutoff]
                else:
                    sliced_ket = full_ket

                branch_tensors.append(sliced_ket)
                probs_n = np.sum(np.abs(sliced_ket) ** 2, axis=0)
                branch_raw_probs.append(probs_n)

            # Phase B: Virtual Branching
            P_parent = active_probs
            P_raw = np.stack(branch_raw_probs)
            P_total = P_parent[:, None] * P_raw

            # Phase C: Pruning
            flat_P = P_total.flatten()
            current_n_candidates = flat_P.size
            k = min(self.beam_width, current_n_candidates)

            # Find indices of the top K probabilities
            top_indices = np.argpartition(flat_P, -k)[-k:]
            # Sort for stability
            top_indices = top_indices[np.argsort(flat_P[top_indices])[::-1]]

            parent_indices, outcome_indices = np.unravel_index(top_indices, P_total.shape)

            # Phase D: Realization
            tensor_stack = np.stack(branch_tensors)
            raw_new_kets = tensor_stack[parent_indices, :, outcome_indices]
            norms = np.linalg.norm(raw_new_kets, axis=1)
            selected_probs = P_total[parent_indices, outcome_indices]

            mask = (norms > 1e-9) & (selected_probs > 1e-12)

            if not np.any(mask):
                break

            active_kets = raw_new_kets[mask] / norms[mask][:, None]
            active_probs = selected_probs[mask]

            # Track history
            new_outcomes = []
            masked_parent_indices = parent_indices[mask]
            masked_outcome_indices = outcome_indices[mask]

            for p_idx, o_val in zip(masked_parent_indices, masked_outcome_indices):
                new_outcomes.append(active_outcomes[p_idx] + (int(o_val),))
            active_outcomes = new_outcomes

        branch_details = []
        if len(active_kets) > 0:
            prod = np.conj(self.target_ket) * active_kets
            fft_vals = np.fft.fft(prod, n=256, axis=-1)
            fidelities = np.max(np.abs(fft_vals) ** 2, axis=-1)

            for i in range(len(active_probs)):
                branch_details.append({
                    "outcome": active_outcomes[i],
                    "prob": float(active_probs[i]),
                    "fidelity": float(fidelities[i]),
                    "target_idx": 0
                })

        return {
            "x": result.x,
            "loss": result.fun,
            "duration": time.time() - start_time,
            "message": result.message,
            "branches": branch_details
        }
