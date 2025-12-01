
# 1. Import the module we need to patch
import scipy.integrate

# 2. Check if the patch is needed to avoid errors
if not hasattr(scipy.integrate, 'simps'):
    print("Monkey patching scipy.integrate: 'simps' not found. Pointing to 'simpson'.")
    # 3. Create the 'simps' attribute and point it to the existing 'simpson' function.
    scipy.integrate.simps = scipy.integrate.simpson
else:
    print("'simps' already exists in scipy.integrate. No patch needed.")


import numpy as np
import matplotlib.pyplot as plt
import strawberryfields as sf
from strawberryfields import ops
from strawberryfields.ops import *


# 1. The Database of Parameters (Extracted from Table 2)
optimal_circuit_params = [
    {"a": 0.3, "r1": -0.27, "r2": 0.65, "r3": 0.66, "phi_r1": -4.14, "phi_r2": 0.53, "phi_r3": -1.94, "d1": 0.38, "d2": -0.19, "d3": -0.47, "theta1": 2.29, "theta2": -4.19, "theta3": 3.5, "phi1": 2.77, "phi2": 0.59, "phi3": -0.2},
    {"a": 0.38, "r1": -0.53, "r2": -0.75, "r3": -0.55, "phi_r1": 7.19, "phi_r2": 10.16, "phi_r3": 10.73, "d1": 0.51, "d2": -0.03, "d3": 0.53, "theta1": 2.3, "theta2": -2.04, "theta3": -4.11, "phi1": -0.29, "phi2": -1.99, "phi3": -0.89},
    {"a": 0.46, "r1": -0.75, "r2": -0.54, "r3": 0.27, "phi_r1": 1.23, "phi_r2": -5.31, "phi_r3": 1.53, "d1": -0.04, "d2": 0.38, "d3": 0.63, "theta1": 0.7, "theta2": 1.97, "theta3": -0.88, "phi1": -1.62, "phi2": 0.72, "phi3": -1.79},
    {"a": 0.53, "r1": 0.71, "r2": 0.67, "r3": -0.42, "phi_r1": -2.07, "phi_r2": 0.06, "phi_r3": -3.79, "d1": -0.02, "d2": 0.34, "d3": 0.02, "theta1": -1.57, "theta2": 0.68, "theta3": 2.5, "phi1": 0.53, "phi2": -4.51, "phi3": 0.72},
    {"a": 0.61, "r1": 0.72, "r2": 0.65, "r3": -0.45, "phi_r1": 0.23, "phi_r2": 0.49, "phi_r3": -3.81, "d1": 0, "d2": -0.33, "d3": -0.01, "theta1": -1.57, "theta2": -2.46, "theta3": 0.63, "phi1": -1.81, "phi2": -0.22, "phi3": 6.42},
    {"a": 0.67, "r1": 0.52, "r2": 0.56, "r3": 0.74, "phi_r1": 0.09, "phi_r2": 3.35, "phi_r3": -3.26, "d1": 0.64, "d2": -0.33, "d3": 0.02, "theta1": -3.31, "theta2": 2.29, "theta3": 2.63, "phi1": -3.33, "phi2": 1.41, "phi3": -6.29},
    {"a": 0.77, "r1": -0.55, "r2": 0.73, "r3": 0.03, "phi_r1": 1.14, "phi_r2": 0.98, "phi_r3": -3.03, "d1": 0.34, "d2": 0.11, "d3": 0.51, "theta1": -2.3, "theta2": -1.96, "theta3": 0.69, "phi1": -3.22, "phi2": 4.32, "phi3": -4.07},
    {"a": 0.84, "r1": 0.74, "r2": 0.54, "r3": -0.48, "phi_r1": 0.5, "phi_r2": -5.29, "phi_r3": 2.73, "d1": -0.01, "d2": -0.36, "d3": 0.01, "theta1": -1.52, "theta2": 0.7, "theta3": -3.8, "phi1": -0.95, "phi2": -1.06, "phi3": -1.24},
    {"a": 0.92, "r1": -0.54, "r2": 0.49, "r3": 0.72, "phi_r1": -0.44, "phi_r2": 2.36, "phi_r3": 3.14, "d1": 0.15, "d2": -0.48, "d3": 0.25, "theta1": -1.3, "theta2": 2.27, "theta3": 0.67, "phi1": -2.13, "phi2": 3.94, "phi3": 1.99},
    {"a": 1, "r1": 0.66, "r2": -0.38, "r3": -0.76, "phi_r1": 3.36, "phi_r2": -0.73, "phi_r3": 0.4, "d1": -0.05, "d2": -0.82, "d3": 0, "theta1": 1.56, "theta2": -2.32, "theta3": 0.61, "phi1": -3.67, "phi2": -0.97, "phi3": -0.71}
]


# --- 2. CIRCUIT CONSTRUCTION (Top-to-Bottom Sorting) ---
def get_circuit_for_a(target_a):
    p = next((item for item in optimal_circuit_params if item["a"] == target_a), None)
    
    prog = sf.Program(3)
    with prog.context as q:
        # MAPPING:
        # q[0] = Top Mode    (Index 1 in Table)
        # q[1] = Middle Mode (Index 2 in Table)
        # q[2] = Bottom Mode (Index 3 in Table) --> OUTPUT

        # --- PREPARATION ---
        # Top (Index 1)
        Sgate(p["r1"], p["phi_r1"]) | q[0]
        Dgate(p["d1"], 0) | q[0]

        # Middle (Index 2)
        Sgate(p["r2"], p["phi_r2"]) | q[1]
        Dgate(p["d2"], 0) | q[1]
        
        # Bottom (Index 3)
        Sgate(p["r3"], p["phi_r3"]) | q[2]
        Dgate(p["d3"], 0) | q[2]
        
        # --- INTERFEROMETER ---
        # 1. Top (q0) mixes with Middle (q1)
        BSgate(p["theta1"]+0.07, p["phi1"]) | (q[0], q[1])
        
        # 2. Middle (q1) mixes with Bottom (q2)
        BSgate(p["theta2"], p["phi2"]) | (q[1], q[2])
        
        # 3. Top (q0) mixes with Middle (q1)
        BSgate(p["theta3"], p["phi3"]) | (q[0], q[1])
        
        # --- MEASUREMENTS ---
        # Paper says: PNR (m1=1, m2=2)
        # m1 is Top wire (q0), m2 is Middle wire (q1)
        MeasureFock(select=1) | q[0]
        MeasureFock(select=2) | q[1]
        
    return prog


def get_target_ket(a, cutoff_dim):
    """
    Constructs the theoretical target ket vector based on Eq. (1) in the paper.
    """
    # Initialize an empty complex vector
    target_ket = np.zeros(cutoff_dim, dtype=np.complex128)
    
    # Set the specific coefficients from Eq (1)
    # Coeff for |0> is 1
    target_ket[0] = 1.0 + 0j
    
    # Coeff for |1> is i * a * sqrt(3/2)
    target_ket[1] = 1j * a * np.sqrt(3/2)
    
    # Coeff for |2> is 0 (Explicitly left as 0)
    
    # Coeff for |3> is i * a
    target_ket[3] = 1j * a
    
    # Normalize the vector
    # The paper gives the norm factor as sqrt(1 + 5|a|^2 / 2), 
    # but using linear algebra norm is safer numerically.
    norm = np.linalg.norm(target_ket)
    target_ket = target_ket / norm
    
    return target_ket

# --- Example Usage ---

# Select 'a' from the table (e.g., 0.61)
target_a_value = 0.61


# Build program
my_program = get_circuit_for_a(target_a_value)

# Run simulation
# Note: Since MeasureFock is used, we need a backend that supports sampling or Fock representation.
cutoff_dim = 15
grid_size = 200
x_limit = 5

eng = sf.Engine(backend="fock", backend_options={"cutoff_dim": cutoff_dim})

result = eng.run(my_program)
state = result.state



rho_out = state.reduced_dm(2)

target_ket = get_target_ket(target_a_value, cutoff_dim)
fidelity = np.real(np.dot(target_ket.conj(), np.dot(rho_out, target_ket)))

print(f"Fidelity with Target (a={target_a_value}): {fidelity:.6f}")


# --- 4. CALCULATE WIGNER FUNCTION ---
print("Calculating Wigner function...")
xvec = np.linspace(-x_limit, x_limit, grid_size)
pvec = np.linspace(-x_limit, x_limit, grid_size)
W = state.wigner(mode=2, xvec=xvec, pvec=pvec)

# --- 5. CALCULATE FOCK PROBABILITIES ---
ket = state.ket()
probs = np.abs(ket[0,0, :])**2
probs = probs.flatten().real


# --- 6. PLOTTING ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Plot 1: Wigner Function
# We use RdBu_r colormap: Red = Positive, Blue = Negative (Standard in Quantum Optics)
# Negative regions indicate Non-Gaussianity.
X, P = np.meshgrid(xvec, pvec)
c = ax1.pcolormesh(X, P, W, cmap='RdBu', shading='auto', vmin=-np.max(np.abs(W)), vmax=np.max(np.abs(W)))
fig.colorbar(c, ax=ax1, label='W(x, p)')
ax1.set_title(f"Fidelity ={fidelity:.4f})")
ax1.set_xlabel("x (Position)")
ax1.set_ylabel("p (Momentum)")
ax1.set_aspect('equal')

# Add grid lines to see the center
ax1.axhline(0, color='black', linestyle='--', alpha=0.3)
ax1.axvline(0, color='black', linestyle='--', alpha=0.3)

# Plot 2: Fock Distribution
ax2.bar(range(cutoff_dim), probs, color='teal', alpha=0.7, edgecolor='black')
ax2.set_title("Fock State Probabilities")
ax2.set_xlabel("Fock Number |n>")
ax2.set_ylabel("Probability")
ax2.set_xticks(range(cutoff_dim))

# Highlight the "Hole" at |2>
ax2.annotate('The Hole\n(Must be 0)', xy=(2, probs[2]), xytext=(2, 0.3),
                arrowprops=dict(facecolor='red', shrink=0.05),
                ha='center', color='red', fontweight='bold')

# Highlight the cutoff at |4+>
ax2.axvline(3.5, color='red', linestyle='--', label='Truncation')
ax2.text(4, 0.2, "Forbidden\nRegion", color='red')

plt.tight_layout()
plt.show()
