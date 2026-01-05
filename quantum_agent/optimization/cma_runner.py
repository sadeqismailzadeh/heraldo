import time
import numpy as np
import multiprocessing
import strawberryfields as sf
from strawberryfields.ops import Ket
from functools import partial

try:
    import cma
except ImportError:
    raise ImportError("CMA-ES runner requires the 'cma' package. Please install it via 'pip install cma'.")

from quantum_agent.optimization.time_interfaces import TimeMultiplexedCircuit
from quantum_agent.optimization.time_runner import evaluate_time_domain_circuit
from quantum_agent.components.targets import TargetGenerator

class CMAESOptimizationRunner:
    """
    Optimizes time-domain circuits using CMA-ES (Covariance Matrix Adaptation Evolution Strategy).
    
    This runner parallelizes the evaluation of candidates using multiprocessing.
    """
    def __init__(self, 
                 circuit: TimeMultiplexedCircuit, 
                 target_gens: list[TargetGenerator], 
                 cutoff_dim: int,
                 beam_width: int = 5,
                 penalty_strength: float = 10.0,
                 success_threshold: float = 0.99,
                 success_weight: float = 5.0,
                 num_processes: int = 4,
                 sigma0: float = 0.5):
        
        self.circuit = circuit
        self.cutoff_dim = cutoff_dim
        self.beam_width = beam_width
        self.penalty_strength = penalty_strength
        self.success_threshold = success_threshold
        self.success_weight = success_weight
        self.num_processes = num_processes
        self.sigma0 = sigma0

        if not isinstance(target_gens, list):
            target_gens = [target_gens]
        
        # Precompute targets
        self.target_kets = [gen.get_target_ket(cutoff_dim) for gen in target_gens]

    def run(self, n_generations=50, population_size=None):
        """
        Runs the CMA-ES optimization.
        
        Args:
            n_generations (int): Maximum number of generations.
            population_size (int): Size of the population (lambda). If None, CMA defaults are used.
        """
        # 1. Setup Parameters and Bounds
        per_step_bounds = self.circuit.per_step_parameter_bounds
        if self.circuit.time_invariant:
            full_bounds = per_step_bounds
        else:
            full_bounds = per_step_bounds * self.circuit.steps
            
        # CMA expects bounds in format [[lower_1, lower_2...], [upper_1, upper_2...]]
        lower_bounds = [b[0] for b in full_bounds]
        upper_bounds = [b[1] for b in full_bounds]
        
        # Initial guess (random within bounds)
        x0 = np.array([np.random.uniform(l, h) for l, h in full_bounds])
        
        # 2. Initialize CMA-ES
        opts = {
            'bounds': [lower_bounds, upper_bounds],
            'maxiter': n_generations,
            'verbose': -1,  # Suppress internal printing
        }
        if population_size:
            opts['popsize'] = population_size
            
        es = cma.CMAEvolutionStrategy(x0, self.sigma0, opts)
        
        print(f"Starting CMA-ES (Generations={n_generations}, PopSize={es.popsize}, Processes={self.num_processes})...")
        start_time = time.time()
        
        # Create a partial function with fixed arguments for the worker
        # We bind the complex objects here so the pool only transmits the params
        worker_func = partial(
            evaluate_time_domain_circuit,
            circuit=self.circuit,
            target_kets=self.target_kets,
            cutoff_dim=self.cutoff_dim,
            beam_width=self.beam_width,
            penalty_strength=self.penalty_strength,
            success_threshold=self.success_threshold,
            success_weight=self.success_weight
        )
        
        best_loss = float('inf')
        best_x = None

        # 3. Evolution Loop
        with multiprocessing.Pool(processes=self.num_processes) as pool:
            for gen in range(n_generations):
                if es.stop():
                    break
                    
                X = es.ask()
                
                # Parallel Evaluation
                fitness_values = pool.map(worker_func, X)
                
                es.tell(X, fitness_values)
                es.logger.add()  # write to cma files
                # es.disp()      # print info
                
                current_best_loss = min(fitness_values)
                if current_best_loss < best_loss:
                    best_loss = current_best_loss
                    best_x = es.result.xbest
                    
                print(f"  Gen {gen+1}/{n_generations} | Min Loss: {current_best_loss:.5f} | Sigma: {es.sigma:.3f}")

        duration = time.time() - start_time
        final_x = es.result.xbest
        final_loss = best_loss # strictly tracked best
        
        # 4. Final Reconstruction (Re-run best to get details)
        # This logic mirrors TimeDomainRunner to produce the detailed 'branches' output
        
        mapped_params = self.circuit.map_parameters(final_x)
        meas_specs = self.circuit.get_measurement_specs()
        meas_modes = [m for m, c in meas_specs]
        meas_cutoffs = [c for m, c in meas_specs]
        perm = [0] + meas_modes

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

                prog_prep = sf.Program(len(meas_modes) + 1)
                with prog_prep.context as q:
                    Ket(parent_ket) | q[0]
                eng.run(prog_prep)

                result_step = self.circuit.run_step(None, step, step_params, eng)
                full_ket = result_step.state.ket()

                transposed_ket = np.transpose(full_ket, axes=perm)
                slices = [slice(None)] + [slice(0, c) for c in meas_cutoffs]
                sliced_ket = transposed_ket[tuple(slices)]

                branch_tensors.append(sliced_ket)
                probs_tensor = np.sum(np.abs(sliced_ket) ** 2, axis=0)
                branch_raw_probs.append(probs_tensor.flatten())

            # Phase B: Virtual Branching
            P_parent = active_probs
            P_raw = np.stack(branch_raw_probs)
            P_total = P_parent[:, None] * P_raw

            # Phase C: Pruning
            flat_P = P_total.flatten()
            k = min(self.beam_width, flat_P.size)
            top_indices = np.argpartition(flat_P, -k)[-k:]
            top_indices = top_indices[np.argsort(flat_P[top_indices])[::-1]]

            parent_indices, outcome_indices_flat = np.unravel_index(top_indices, P_total.shape)
            outcomes_unraveled = np.unravel_index(outcome_indices_flat, tuple(meas_cutoffs))

            # Phase D: Realization
            tensor_stack = np.stack(branch_tensors)
            indexer = (parent_indices, slice(None)) + outcomes_unraveled
            raw_new_kets = tensor_stack[indexer]
            
            norms = np.linalg.norm(raw_new_kets, axis=1)
            selected_probs = P_total[parent_indices, outcome_indices_flat]

            mask = (norms > 1e-9) & (selected_probs > 1e-12)

            if not np.any(mask):
                break

            active_kets = raw_new_kets[mask] / norms[mask][:, None]
            active_probs = selected_probs[mask]

            # Track history
            new_outcomes = []
            masked_parent_indices = parent_indices[mask]
            
            # Re-slice outcomes unraveled based on mask
            masked_outcomes_tuple = tuple(arr[mask] for arr in outcomes_unraveled)
            
            for i in range(len(masked_parent_indices)):
                p_idx = masked_parent_indices[i]
                step_outcome = tuple(int(arr[i]) for arr in masked_outcomes_tuple)
                new_outcomes.append(active_outcomes[p_idx] + step_outcome)
            active_outcomes = new_outcomes

        branch_details = []
        if len(active_kets) > 0:
            targets_arr = np.array(self.target_kets)
            prod = np.conj(active_kets[:, None, :]) * targets_arr[None, :, :]
            fft_vals = np.fft.fft(prod, n=256, axis=-1)
            pairwise_fidelities = np.max(np.abs(fft_vals) ** 2, axis=-1)

            best_fidelities = np.max(pairwise_fidelities, axis=1)
            best_target_idxs = np.argmax(pairwise_fidelities, axis=1)

            for i in range(len(active_probs)):
                branch_details.append({
                    "outcome": active_outcomes[i],
                    "prob": float(active_probs[i]),
                    "fidelity": float(best_fidelities[i]),
                    "target_idx": int(best_target_idxs[i])
                })

        return {
            "x": final_x,
            "loss": final_loss,
            "duration": duration,
            "branches": branch_details,
            "message": "CMA-ES Finished"
        }