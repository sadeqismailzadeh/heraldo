# 1. Patch Scipy for compatibility
import scipy.integrate
if not hasattr(scipy.integrate, 'simps'):
    scipy.integrate.simps = scipy.integrate.simpson

import numpy as np
import matplotlib.pyplot as plt
import strawberryfields as sf
from strawberryfields import ops
from scipy.optimize import minimize
import matplotlib.cm as cm
import json
import os

# ==========================================
# PART 1: FAST PURE STATE MATH
# ==========================================

def fidelity_pure_state(target_ket, state_ket):
    """
    Computes fidelity between two pure states (vectors) via overlap.
    F = |<psi|phi>|^2
    Computational Cost: O(N) -- Very Fast
    """
    # Flatten ensures they are 1D arrays
    target_ket = np.asarray(target_ket, dtype=np.complex128).flatten()
    state_ket = np.asarray(state_ket, dtype=np.complex128).flatten()
    
    # Calculate overlap (dot product)
    overlap = np.vdot(target_ket, state_ket)
    
    # Fidelity is magnitude squared
    return np.abs(overlap) ** 2

# ==========================================
# PART 2: EFFICIENT SIMULATION
# ==========================================

def get_joint_state(tau_sq, cutoff):
    """
    Runs the interference circuit ONCE.
    Returns the joint state vector |Psi>_01 of shape (cutoff, cutoff).
    """
    eng = sf.Engine("fock", backend_options={"cutoff_dim": cutoff})
    prog = sf.Program(2)
    
    theta = np.arccos(np.sqrt(tau_sq))
    
    with prog.context as q:
        # Orthogonal Squeezing Setup
        ops.Sgate(-1.38) | q[0] # Vertical
        ops.Sgate(1.38) | q[1]  # Horizontal
        
        # Beam Splitter Interaction
        ops.BSgate(theta, 0) | (q[0], q[1])
        
    result = eng.run(prog)
    return result.state.ket()

def generate_target_ket(alpha_mag, r_mag, parity, cutoff, rotation=0):
    """
    Generates a candidate Squeezed Cat ket vector for optimization.
    Orientation is fixed to Vertical (Imaginary Alpha).
    Optional rotation can be applied.
    """
    eng = sf.Engine("fock", backend_options={"cutoff_dim": cutoff})
    prog = sf.Program(1)

    # Vertical Cat orientation
    alpha_val = alpha_mag

    with prog.context as q:
        ops.Catstate(alpha_val, p=parity) | q[0]
        ops.Sgate(r_mag) | q[0]
        if rotation != 0:
            ops.Rgate(rotation) | q[0]

    result = eng.run(prog)
    assert result.state.is_pure
    return result.state.ket()


# ==========================================
# PART 3: OPTIMIZATION LOOP
# ==========================================

def run_fig6_fast():
    CUTOFF = 50
    
    # Full range of photons as in Fig 6
    N_VALUES = range(1, 13)
    
    # Increased resolution since calculation is now faster
    TAU_SQ_VALUES = np.linspace(1e-6, 0.5, 30) 

    # --- STRATEGY: WARM STARTING ---
    # We need to tell the optimizer where to look for the first step (Tau=0).
    # Looking at Fig 6 (Left side):
    #   1. Alpha starts low (bottom middle panel) -> approx 0.5
    #   2. r starts high (bottom panel) -> approx 1.4 to 1.6
    #
    # We store these guesses in a dictionary. As the loop progresses, 
    # we will update these guesses with the result from the previous step.
    current_guesses = {}
    for n in N_VALUES:
        current_guesses[n] = [0.5, 1.45] 
    
    # Data Structure
    data = {n: {'tau': [], 'fid': [], 'alpha': [], 'r': []} for n in N_VALUES}
    
    print(f"Starting Fast Optimization over {len(TAU_SQ_VALUES)} points...")
    
    for i, tau_sq in enumerate(TAU_SQ_VALUES):
        # 1. Run Circuit ONCE for this Tau to get joint state
        joint_ket = get_joint_state(tau_sq, CUTOFF)
        
        print(f"Processing Tau^2 = {tau_sq:.3f} ({i+1}/{len(TAU_SQ_VALUES)})")
        
        for n in N_VALUES:
            # 2. SLICE the joint tensor to get the conditional state for photon count n
            # joint_ket shape is (dim0, dim1). We want column n of dim1.
            slice_ket = joint_ket[:, n]
            
            # 3. Normalize
            prob = np.sum(np.abs(slice_ket)**2)
            
            if prob < 1e-9:
                # Impossible state (probability ~ 0)
                data[n]['tau'].append(tau_sq)
                data[n]['fid'].append(0)
                data[n]['alpha'].append(0)
                data[n]['r'].append(0)
                continue
                
            state_ket = slice_ket / np.sqrt(prob)
            
            # 4. Determine Parity (Input is Even+Even, measurement removes n)
            # If n is Even -> Output Even (0). If n is Odd -> Output Odd (1).
            target_parity = n % 2
            
            # 5. DEFINE OPTIMIZER OBJECTIVE
            # We want to find (alpha, r) that maximizes fidelity with state_ket
            def objective(params):
                a, r = params
                
                # --- PENALTY METHOD ---
                # Nelder-Mead doesn't support hard bounds (like L-BFGS-B).
                # If the optimizer tries negative alpha/r, we return a "bad" score (2.0).
                # This forces it back into the valid region.
                if a < 0.0 or r < 0.0: return 2.0
                if a > 10.0 or r > 3.0: return 2.0 # Sanity upper bounds
                
                # Generate candidate state
                target = generate_target_ket(a, r, target_parity, CUTOFF)
                
                # Calculate Fidelity
                fid = fidelity_pure_state(target, state_ket)
                
                # We want to MAXIMIZE Fidelity, but Scipy only MINIMIZES.
                # So we minimize (1.0 - Fidelity).
                return 1.0 - fid 

            # 5. Run Optimization
            # We use the "Warm Start" guess from the dictionary.
            guess = current_guesses[n]
            
            res = minimize(
                objective,
                guess,
                method='Nelder-Mead',
                # tol: when to stop. 1e-4 is precise enough for plotting.
                # maxfev: max iterations to prevent infinite loops.
                options={'xatol': 1e-4, 'fatol': 1e-4, 'maxfev': 400}
            )

            # 6. UPDATE WARM START
            # The result of this step becomes the starting guess for the NEXT tau step.
            # This ensures the optimizer follows the smooth lines in Fig 6.
            current_guesses[n] = res.x

            # 7. Store Results
            data[n]['tau'].append(tau_sq)
            data[n]['fid'].append(1.0 - res.fun) # Convert loss back to fidelity
            data[n]['alpha'].append(res.x[0])
            data[n]['r'].append(res.x[1])

            # ADD THIS PRINT STATEMENT:
            print(f"n={n}, Tau={tau_sq:.2f} | Steps: {res.nit} | Evaluations: {res.nfev} | "
                  f"alpha: {res.x[0]:.4f} |  r: {res.x[1]:.4f} |  Fid: {1.0-res.fun:.4f}")


    # Save to JSON
    print("Saving data to 'fig6_data.json'...")
    file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'fig6_data.json')
    with open(file, 'w') as f:
        json.dump(data, f, indent=4)
    print("Done!")
    return data


if __name__ == "__main__":
    run_fig6_fast()
    