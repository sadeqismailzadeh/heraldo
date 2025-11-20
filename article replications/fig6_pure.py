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



def precompute_target_sqrts(cutoff_dim):
    """
    Generates the 4 target Squeezed Cat states used by the agent
    and returns their matrix square roots.
    
    Targets:
    1. Even Parity (Standard)
    2. Odd Parity (Standard)
    3. Even Parity (Rotated 90 deg)
    4. Odd Parity (Rotated 90 deg)
    """
    alpha = 3.0
    r = 1.38
    
    target_sqrts = []
    
    # We need a temporary engine to generate these static states
    eng = sf.Engine("fock", backend_options={"cutoff_dim": cutoff_dim})
    # eng = sf.Engine("bosonic")

    # --- Define the 4 Programs ---
    programs = []
    
    # 1. Rho Plus (Even)
    p1 = sf.Program(1)
    with p1.context as q:
        ops.Catstate(alpha, p=0) | q[0]
        ops.Sgate(r) | q[0]
    programs.append(p1)

    # 2. Rho Minus (Odd)
    p2 = sf.Program(1)
    with p2.context as q:
        ops.Catstate(alpha, p=1) | q[0]
        ops.Sgate(r) | q[0]
    programs.append(p2)

    # 3. Rho Plus Rotated
    p3 = sf.Program(1)
    with p3.context as q:
        ops.Catstate(alpha, p=0) | q[0]
        ops.Sgate(r) | q[0]
        ops.Rgate(np.pi/2) | q[0]
    programs.append(p3)

    # 4. Rho Minus Rotated
    p4 = sf.Program(1)
    with p4.context as q:
        ops.Catstate(alpha, p=1) | q[0]
        ops.Sgate(r) | q[0]
        ops.Rgate(np.pi/2) | q[0]
    programs.append(p4)

    # # --- Run and Compute Sqrt ---
    print("Pre-computing target states...")
    for prog in programs:
        result = eng.run(prog)
        dm = result.state.dm(cutoff=cutoff_dim)
        target_sqrts.append(compute_matrix_sqrt(dm))
        
    return target_sqrts

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


def estimate_alpha_r(state_ket, cutoff):
    """
    Mathematically estimates alpha and r to give a perfect starting guess.
    """
    # Create number operator diagonal [0, 1, 2, ..., cutoff-1]
    n_op = np.arange(cutoff)
    
    # Calculate Mean Photon Number <n>
    # state_ket is a 1D vector of coefficients c_n
    # <n> = sum( |c_n|^2 * n )
    probs = np.abs(state_ket)**2
    mean_n = np.sum(probs * n_op)
    
    # HEURISTIC GUESS:
    # Assume squeezing r is roughly proportional to size (typical for these circuits)
    # A standard guess for this paper is r approx 1.0
    # <n> = alpha^2 + sinh^2(r)
    # alpha^2 = <n> - sinh^2(1.0)
    
    sinh_sq_r = np.sinh(1.0)**2
    if mean_n > sinh_sq_r:
        guess_alpha = np.sqrt(mean_n - sinh_sq_r)
        guess_r = 1.0
    else:
        # If mean photon count is tiny, it's mostly just squeezing
        guess_alpha = 0.1
        # sinh^2(r) = mean_n -> r = arcsinh(sqrt(mean_n))
        guess_r = np.arcsinh(np.sqrt(mean_n))
        
    return guess_alpha, guess_r



# ==========================================
# PART 3: OPTIMIZATION LOOP
# ==========================================

def run_fig6_fast():
    CUTOFF = 50
    
    # Full range of photons as in Fig 6
    N_VALUES = range(1, 13)
    
    # Increased resolution since calculation is now faster
    TAU_SQ_VALUES = np.linspace(0.2, 0.5, 30) 
    
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
                # Physics constraints
                if a < 0.1 or r < 0.0: return 1.0 
                
                # target = generate_target_ket(a, r, target_parity, CUTOFF)
                # fid = fidelity_pure_state(target, state_ket)


                targets = [
                    generate_target_ket(a, r, parity, CUTOFF, rotation=rot)
                    for parity in [0, 1] for rot in [0, np.pi/2]
                ]
                current_fidelities = [
                    fidelity_pure_state(target, state_ket)
                    for target in targets
                ]
                fid = np.max(current_fidelities)

                return 1.0 - fid # Minimize Loss
            
            # Initial Guess (Heuristic based on Fig 6)
            # High Tau -> Low Alpha, Low r
            # Low Tau -> High Alpha, High r
            # ---- INSIDE YOUR LOOP ----
            # Replace the previous guess logic with:
            # guess_alpha, guess_r = estimate_alpha_r(state_ket, CUTOFF)
            guess_alpha = 7.5
            guess_r = 1.8

            # guess_alpha = 3.0 * (1.0 - tau_sq*1.5) 
            # guess_alpha = max(guess_alpha, 1.0)
            # guess_r = 1.5 * (1.0 - tau_sq)

            # Run Optimizer (Nelder-Mead is robust here)
            # res = minimize(objective, [guess_alpha, guess_r], method='Nelder-Mead', tol=1e-2)


            # Define bounds: alpha in [0.1, 5.0], r in [0.0, 2.0]
            # This prevents the optimizer from looking at impossible negative values
            bounds = [(0.1, 8), (0.0, 2.0)]

            # Run Optimizer
            # 'eps': 1e-4 tells it how big a step to take to calculate the slope
            res = minimize(
                objective,
                [guess_alpha, guess_r],
                method='L-BFGS-B',
                bounds=bounds,
                options={'ftol': 1e-5, 'eps': 1e-4}
            )


            # res = minimize(
            #     objective,
            #     [guess_alpha, guess_r],
            #     method='BFGS',
            # )
            # ADD THIS PRINT STATEMENT:
            print(f"n={n}, Tau={tau_sq:.2f} | Steps: {res.nit} | Evaluations: {res.nfev} | "
                  f"alpha: {res.x[0]:.4f} |  r: {res.x[1]:.4f} |  Fid: {1.0-res.fun:.4f}")

            # Store Results
            data[n]['tau'].append(tau_sq)
            data[n]['fid'].append(1.0 - res.fun)
            data[n]['alpha'].append(res.x[0])
            data[n]['r'].append(res.x[1])

    # Save to JSON
    print("Saving data to 'fig6_data.json'...")
    with open('fig6_data.json', 'w') as f:
        json.dump(data, f, indent=4)
    print("Done!")
    return data



if __name__ == "__main__":
    run_fig6_fast()
    