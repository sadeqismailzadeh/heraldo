import numpy as np
import quantum_agent
import strawberryfields as sf
from strawberryfields.ops import Ket
from quantum_agent.optimization.time_circuits import *
from quantum_agent.optimization.time_runner import evaluate_time_domain_circuit
from quantum_agent.components.targets import *
from pathlib import Path

import numpy as np
import strawberryfields as sf
from quantum_agent.optimization.time_interfaces import TimeMultiplexedCircuit
from quantum_agent.optimization.time_runner import evaluate_time_domain_circuit
from quantum_agent.utils import *


def run_deterministic_path(circuit: TimeMultiplexedCircuit, flat_params: np.ndarray, measurement_outcomes: tuple, cutoff_dim):
    """
    Evaluates a time-domain circuit for a specific path of measurements.
    
    Args:
        circuit: The TimeMultiplexedCircuit instance
        flat_params: Flat parameter array for the circuit
        measurement_outcomes: Tuple of measurement outcomes (one per step)
        
    Returns:
        A dictionary with final probability and final state ket if the path is valid,
        or None if the path is not physically possible.
    """
    # Split initial parameters if needed
    n_init = circuit.num_initial_parameters
    if n_init > 0:
        init_params = flat_params[:n_init]
        step_params = flat_params[n_init:]
    else:
        init_params = np.array([])
        step_params = flat_params

    mapped_params = circuit.map_parameters(step_params)
    
    # Get measurement specs (modes and cutoffs)
    meas_specs = circuit.get_measurement_specs()
    meas_modes = [m for m, c in meas_specs]
    meas_cutoffs = [c for m, c in meas_specs]
    perm = [0] + meas_modes
    
    # Initialize the state
    initial_ket = circuit.get_initial_state_ket(init_params, cutoff_dim)  # Use a reasonable cutoff
    current_ket = initial_ket.copy()
    
    # Store probability at each step
    probabilities = []
    
    for step in range(circuit.steps):
        step_params = mapped_params[step]
        
        # Run the circuit step
        eng = sf.Engine("fock", backend_options={"cutoff_dim": cutoff_dim})
        
        prog_prep = sf.Program(len(meas_modes) + 1)
        with prog_prep.context as q:
            Ket(current_ket) | q[0]
        
        eng.run(prog_prep)
        
        result = circuit.run_step(None, step, step_params, eng)
        full_ket = result.state.ket()
    
        # Transpose to [loop, ancilla1, ancilla2,...]
        transposed_ket = np.transpose(full_ket, axes=perm)
        
        # Slice measured modes
        slices = [slice(None)] + [slice(0, c) for c in meas_cutoffs]
        sliced_ket = transposed_ket[tuple(slices)]
        
        # Calculate probabilities and check if the desired measurement is possible
        probs_tensor = np.sum(np.abs(sliced_ket)**2, axis=0)
        
        # Get the probability of the specified outcome at this step
        current_outcome_index = tuple([int(outcome) for outcome in measurement_outcomes[step]])
        
        if len(meas_specs) == 1:
            # Single ancilla case
            prob = probs_tensor[current_outcome_index[0]]
        else:
            # Multiple ancillas: index into the tensor
            prob = probs_tensor[current_outcome_index]
            
        probabilities.append(prob)
        
        # Project onto the measurement outcome (discard other possibilities)
        if len(meas_specs) == 1:
            # Single mode projection
            proj_ket = sliced_ket[:, current_outcome_index[0]]
        else:
            # Multi-mode projection: take the specific tensor element
            indices = [slice(None)] + [int(outcome) for outcome in measurement_outcomes[step]]
            proj_ket = sliced_ket[tuple(indices)]
            
        # Normalize and store as new state
        norm = np.linalg.norm(proj_ket)
        
        if abs(norm) < 1e-9:
            return None  # Path is not physically possible
            
        current_ket = proj_ket / norm
        
    final_probability = np.prod(probabilities)
    
    result = {
        "final_probability": final_probability,
        "final_state_ket": current_ket
    }
    
    return result

def main():
    # Example usage with ThreeModeTimeDomainSqueezeOnly circuit
    # You can change this to any TimeMultiplexedCircuit subclass
    CUTOFF_DIM=100
    # Initialize the circuit (modify parameters as needed)
    squeezing = db_to_r(12)
    circuit2 = ThreeModeTimeDomainSqueezeOnly(steps=1,
                                    time_invariant=False,
                                    clip_size=squeezing,
                                    measure_fock_cutoff=30,
                                    num_single_photon=0,
                                    train_initial_state=True, 
                                    initial_r=squeezing )

    # Define fixed parameters - for demonstration, we'll use dummy values
    # In a real usage, you would replace this with actual parameter values
    flat_params = np.array([
        1.3436, -2.5916 ,
        1.0420, 1.0716, 
        -1.8277, -0.3456, 
        0.8137, 3.9270, 3.1415,
        1.9529 , -0.0495 , 1.9843
    ])
    
    # Define the measurement outcomes we want to trace
    # Format: tuple of integers for each step
    # For three steps with two ancillas:
    # Each step has a tuple (ancilla1_outcome, ancilla2_outcome)
    measurement_outcomes = (
        (1, 3),   # Step 1 outcomes (ancilla 1 and 2)
        # (0, 1),   # Step 2 outcomes
        # (1, 0)    # Step 3 outcomes
    )
    
    result = run_deterministic_path(circuit2, flat_params, measurement_outcomes, CUTOFF_DIM)

    csv_path =  Path(__file__).resolve().parent.parent.parent / "data" / "GKP_core_coefficients.csv"
    target3=CoreGKPTarget(csv_path=csv_path, 
                        n_max=4, 
                        delta_db=10, 
                        mu=0)
    
    if result is None:
        print("The specified measurement path is not physically possible.")
    else:
        print(f"Final probability: {result['final_probability']:.4f}")
        print("Final state ket:")
        print(result['final_state_ket'])
    
    ket = result['final_state_ket']
    target_ket= target3.get_target_ket(CUTOFF_DIM)
    fidelity = fidelity_max_rotation(target_ket, ket)
    # fidelity = fidelity_pure_state(target_ket, ket)
    print(f"fidelity = {fidelity}")



    measurement_outcomes = (
        (3, 1),   # Step 1 outcomes (ancilla 1 and 2)
    )
    
    result = run_deterministic_path(circuit2, flat_params, measurement_outcomes, CUTOFF_DIM)
    ket1 = result['final_state_ket']


    measurement_outcomes = (
        (1, 3),   # Step 1 outcomes (ancilla 1 and 2)
    )
    
    result = run_deterministic_path(circuit2, flat_params, measurement_outcomes, CUTOFF_DIM)
    ket2 = result['final_state_ket']

    fidelity = fidelity_pure_state(ket1, ket2)
    print(f"fidelity 2 states= {fidelity}")
    
if __name__ == "__main__":
    main()
