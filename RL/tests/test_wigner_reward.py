import numpy as np
import qutip as qt
from RL.components.rewards import WignerWeightedReward

# --- Helper Functions for State Generation ---

def get_fock_state(n, dim):
    """Returns normalized ket for Fock state |n>."""
    ket = np.zeros(dim, dtype=np.complex128)
    ket[n] = 1.0
    return ket

def get_coherent_state(alpha, dim):
    """Returns normalized ket for Coherent state |alpha>."""
    return qt.coherent(dim, alpha).full().flatten()

def get_cat_state(alpha, dim, parity='odd'):
    """Returns normalized ket for Cat state (superposition of coherent states)."""
    coh_p = qt.coherent(dim, alpha)
    coh_m = qt.coherent(dim, -alpha)
    cat = (coh_p - coh_m) if parity == 'odd' else (coh_p + coh_m)
    return cat.unit().full().flatten()

def print_header(title):
    print("\n" + "#"*80)
    print(f" {title}")
    print("#"*80)

# --- Test Sections ---

def test_sanity_check(dim=20):
    print_header("TEST 1: SANITY CHECK (Hermiticity)")
    
    # Target: Fock |1>
    target = get_fock_state(1, dim)
    rewarder = WignerWeightedReward(target, dim, neg_weight=2.0, grid_points=40, grid_range=4.0)
    
    op = rewarder.reward_operator
    is_hermitian = np.allclose(op, op.conj().T, atol=1e-8)
    
    print(f"[-] Constructed Reward Operator ({dim}x{dim} matrix)")
    print(f"[-] Is Matrix Hermitian? : {'YES [PASS]' if is_hermitian else 'NO [FAIL]'}")


def test_selectivity(dim=20):
    print_header("TEST 2: SELECTIVITY (Does it prefer the correct state?)")
    print("Goal: The reward should be highest for the Target state.\n")
    
    # Setup: Target is an Odd Cat State (highly non-Gaussian)
    target = get_cat_state(alpha=2.0, dim=dim, parity='odd')
    
    # We use a high negative weight to strictly enforce the 'non-Gaussian' shape
    rewarder = WignerWeightedReward(target, dim, neg_weight=5.0, pos_weight=1.0, grid_points=60)
    
    # Define candidates
    candidates = [
        ("Target (Odd Cat)", target),
        ("Even Cat",        get_cat_state(2.0, dim, parity='even')),
        ("Coherent (a=2)",  get_coherent_state(2.0, dim)),
        ("Vacuum |0>",      get_fock_state(0, dim))
    ]
    
    # Calculate Scores
    scores = []
    for name, state in candidates:
        score, _, _ = rewarder.compute(state, [target], {}, 0.99)
        scores.append((name, score))
    
    # Find winner
    winner_name, winner_score = max(scores, key=lambda x: x[1])
    
    # Print Table
    print(f"{'Candidate State':<20} | {'Reward Score':<15} | {'Status'}")
    print("-" * 55)
    for name, score in scores:
        is_winner = (name == winner_name)
        marker = "WINNER <--" if is_winner else ""
        print(f"{name:<20} | {score:<15.4f} | {marker}")
        
    if winner_name == "Target (Odd Cat)":
        print("\n[PASS] Logic confirmed: Target state has the highest reward.")
    else:
        print(f"\n[FAIL] Logic error: {winner_name} beat the target!")


def test_negativity_boost(dim=20):
    print_header("TEST 3: THE 'NEGATIVITY BOOST' (Intuition Check)")
    print("This test explains why the reward values change based on 'neg_weight'.")
    print("We compare a GAUSSIAN target (Vacuum) vs a NON-GAUSSIAN target (Fock |1>).")
    print("We compute the 'Self-Reward' (Target scored against itself) as we increase weight.\n")

    weights = [1.0, 2.0, 5.0, 10.0]
    
    # --- Case A: Vacuum ---
    print("--- CASE A: Gaussian Target (Vacuum |0>) ---")
    print("Expectation: Score should STAY CONSTANT.")
    print("Reason: Vacuum Wigner function is ALL POSITIVE. Multiplying negative regions by 10x does nothing.\n")
    
    vac_target = get_fock_state(0, dim)
    
    print(f"{'Neg Weight':<12} | {'Self-Reward':<12} | {'Change?'}")
    print("-" * 45)
    
    base_score = None
    gaussian_passed = True
    
    for w in weights:
        # Re-build operator with new weight
        r = WignerWeightedReward(vac_target, dim, neg_weight=w, grid_points=40)
        score, _, _ = r.compute(vac_target, [vac_target], {}, 0.99)
        
        if base_score is None: base_score = score
        
        # Check stability
        is_stable = np.isclose(score, base_score, atol=1e-4)
        status = "Constant" if is_stable else "CHANGED (!)"
        if not is_stable: gaussian_passed = False
        
        print(f"{w:<12.1f} | {score:<12.4f} | {status}")

    if gaussian_passed:
        print("\n[PASS] Gaussian behavior confirmed (Weight irrelevant).\n")
    else:
        print("\n[FAIL] Gaussian score changed unexpectedly.\n")


    # --- Case B: Fock |1> ---
    print("--- CASE B: Non-Gaussian Target (Fock |1>) ---")
    print("Expectation: Score should INCREASE.")
    print("Reason: Fock |1> has a negative dip at the origin. Increasing weight makes this dip 'more valuable'.\n")
    
    fock_target = get_fock_state(1, dim)
    
    print(f"{'Neg Weight':<12} | {'Self-Reward':<12} | {'Boost Factor'}")
    print("-" * 45)
    
    base_score = None
    nongaussian_passed = True
    
    for w in weights:
        r = WignerWeightedReward(fock_target, dim, neg_weight=w, grid_points=40)
        score, _, _ = r.compute(fock_target, [fock_target], {}, 0.99)
        
        if base_score is None: base_score = score
        
        ratio = score / base_score
        if w > 1.0 and ratio <= 1.0: nongaussian_passed = False
        
        print(f"{w:<12.1f} | {score:<12.4f} | {ratio:.2f}x")
        
    if nongaussian_passed:
        print("\n[PASS] Non-Gaussian behavior confirmed (Score boosts with weight).")
    else:
        print("\n[FAIL] Score did not increase as expected.")


def test_optimality(dim=20):
    print_header("TEST 4: OPTIMALITY (Is Target the Global Max?)")
    print("Goal: Ensure that the Target state yields the MAXIMUM reward compared to neighbors.")
    print("      We check random perturbations and the operator's max eigenstate.\n")
    
    # Setup: Target Fock |1> (Simple non-Gaussian)
    target = get_fock_state(1, dim)
    
    # Use standard weights
    rewarder = WignerWeightedReward(target, dim, neg_weight=2.0, pos_weight=1.0, grid_points=50)
    
    # 1. Score of Target
    target_score, _, _ = rewarder.compute(target, [target], {}, 0.99)
    print(f"Target Score: {target_score:.6f}")
    
    # 2. Perturbation Test
    n_trials = 10
    epsilon = 0.2 # Perturbation magnitude
    
    success_count = 0
    
    print(f"\n[A] Comparing against {n_trials} random perturbations (epsilon={epsilon})...")
    print(f"{'Trial':<6} | {'Perturbed Score':<15} | {'Delta (Target - Pert)'}")
    print("-" * 55)
    
    for i in range(n_trials):
        # Create random state
        rand_ket = qt.rand_ket(dim).full().flatten()
        
        # Perturb: |psi'> = normalize( |target> + eps * |rand> )
        perturbed = target + epsilon * rand_ket
        perturbed /= np.linalg.norm(perturbed)
        
        p_score, _, _ = rewarder.compute(perturbed, [target], {}, 0.99)
        
        delta = target_score - p_score
        
        # We allow a tiny floating point tolerance
        is_lower = (delta > -1e-9) 
        
        if is_lower:
            success_count += 1
            
        print(f"{i+1:<6} | {p_score:<15.6f} | {delta:+.6f} {'[OK]' if is_lower else '[FAIL]'}")

    if success_count == n_trials:
        print("\n[PASS] Target state is a local maximum against perturbations.")
    else:
        print("\n[FAIL] Some perturbations scored higher than the target!")

    # 3. Eigenvector Check (Global Max Check)
    # The operator is Hermitian. The state with global max reward is the eigenvector with max eigenvalue.
    print("\n[B] Checking Eigenstructure of Reward Operator (Global Optimality)...")
    vals, vecs = np.linalg.eigh(rewarder.reward_operator)
    
    # Max eigenvalue and corresponding vector (sorted ascending, so last one)
    max_eval = vals[-1]
    max_evec = vecs[:, -1]
    
    # Fidelity between Target and Max Eigenvector (Overlap squared)
    overlap = np.abs(np.vdot(target, max_evec))**2
    
    print(f"Max Eigenvalue:             {max_eval:.6f}")
    print(f"Target Self-Reward:         {target_score:.6f}")
    print(f"Overlap (Target vs MaxEig): {overlap:.6f}")
    
    # Threshold: The target should be very close to the optimal eigenvector
    if overlap > 0.98:
        print("\n[PASS] Target is effectively the global maximum eigenstate (>98% overlap).")
    else:
        print("\n[WARN] Target is NOT the global maximum. The reward shaping might be distorting the optimum.")


def run_tests():
    test_sanity_check()
    test_selectivity()
    test_negativity_boost()
    test_optimality()
    print("\n" + "="*80)
    print(" ALL TESTS COMPLETE")
    print("="*80)

if __name__ == "__main__":
    run_tests()