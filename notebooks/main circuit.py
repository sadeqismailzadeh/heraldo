# %%
import strawberryfields as sf
from strawberryfields import ops
from strawberryfields.ops import Sgate, BSgate, MeasureFock

from strawberryfields.tdm import shift_by
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import cm

import strawberryfields as sf
from strawberryfields.ops import *
import numpy as np
from scipy.linalg import sqrtm

# %% [markdown]
# # fidelity

# %%

def uhlmann_jozsa_fidelity(rho, sigma):
    """Calculates the Uhlmann-Jozsa fidelity between two density matrices."""
    
    # Ensure inputs are numpy arrays
    rho = np.array(rho)
    sigma = np.array(sigma)
    
    # Calculate the square root of rho
    # sqrtm is the matrix square root, not element-wise
    rho_sqrt = sqrtm(rho)
    
    # Calculate the product inside the trace
    product = rho_sqrt @ sigma @ rho_sqrt
    
    # Calculate the square root of the product
    sqrt_product = sqrtm(product)
    
    # The trace of the result is the "trace distance" part
    # We take the real part to handle potential small imaginary numerical errors
    trace = np.trace(sqrt_product).real
    
    # Fidelity is the square of this trace
    fidelity = trace**2
    
    return fidelity

# %% [markdown]
# # circuit

# %%
# Parameters
r0 = 1.38  # Squeezing parameter
tau_1 = 0.367
tau_2 = 0  # Switchable mirror for VBS2

# # Convert transmission coefficient to beam splitter angle.
theta_2 = np.arccos(tau_2)

# Create a Strawberry Fields program
eng = sf.Engine("fock", backend_options={"cutoff_dim": 20})
results = []

is_finished = True
while True:
    # agent can control tau1
    theta_1 = np.arccos(tau_1)

    # Create a new program for each iteration. it must be set before each iteration
    # note that the variable q[1] and q[2] are not changed when setting the program
    prog = sf.Program(2)

    with prog.context as q:
        # Initialize mode 0 with a squeezed vacuum state
        # agent can control squeezed state rotation, but not the squeezing strength
        rot_theta=0
        Sgate(r0, rot_theta) | q[1]

        # Apply variable beam splitter (VBS1)
        BSgate(theta_1, 0) | (q[0], q[1])

        # Photon-number-resolving measurement (PNR)
        MeasureFock() | q[0]

        # agent can control when to finish
        if is_finished:
            theta_2 = np.arccos(1)

        BSgate(theta_2, 0) | (q[0], q[1])  # Fully reflective mirror

    # Run the simulation and store result
    result = eng.run(prog)
    results.append(result)
    if is_finished:
        break

# Use the last result for plotting
result = results[-1]

state_dm=result.state.dm()

print("Final state:", result.state)
print(result.samples)

xvec = np.linspace(-15, 15, 401)                        # Create vector of 401 points from -15 to 15 for phase-space coordinates
W = result.state.wigner(mode=1, xvec=xvec, pvec=xvec)   # Calculate Wigner function for quantum state (mode 1)
scale = np.max(W.real)                                  # Get max value for symmetric color scaling
nrm = mpl.colors.Normalize(-scale, scale)               # Create symmetric color normalization
plt.axes().set_aspect("equal")                          # Set square aspect ratio for phase-space plot
plt.contourf(xvec, xvec, W, 60, cmap=cm.RdBu, norm=nrm) # Plot Wigner function using filled contours (red: positive, blue: negative)
plt.show()

# %% [markdown]
# # reward states

# %%
cutoff = 50
eng = sf.Engine("fock", backend_options={"cutoff_dim": cutoff})

prog = sf.Program(1)

# Define the parameters for the squeezed cat state
alpha = 3
r = 1.38

# --- 2. Construct the Quantum Circuit ---
with prog.context as q:
    # First, prepare the initial cat state
    # A cat state is a superposition of two coherent states
    Catstate(alpha) | q[0]
    
    # Second, apply the squeezing operation
    Sgate(r) | q[0]

state_theoretical_dm_rho_plus = eng.run(prog).state.dm(cutoff=cutoff)

with prog.context as q:
    # First, prepare the initial cat state
    # A cat state is a superposition of two coherent states
    Catstate(alpha,p=1) | q[0]
    
    # Second, apply the squeezing operation
    Sgate(r) | q[0]

state_theoretical_dm_rho_minus = eng.run(prog).state.dm(cutoff=cutoff)

with prog.context as q:
    # First, prepare the initial cat state
    # A cat state is a superposition of two coherent states
    Catstate(alpha) | q[0]
    
    # Second, apply the squeezing operation
    Sgate(r) | q[0]

    Rgate(np.pi/2) | q[0]

state_theoretical_dm_rho_plus_rot = eng.run(prog).state.dm(cutoff=cutoff)

with prog.context as q:
    # First, prepare the initial cat state
    # A cat state is a superposition of two coherent states
    Catstate(alpha,p=1) | q[0]
    
    # Second, apply the squeezing operation
    Sgate(r) | q[0]

    Rgate(np.pi/2) | q[0]

state_theoretical_dm_rho_minus_rot = eng.run(prog).state.dm(cutoff=cutoff)



