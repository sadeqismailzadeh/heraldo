"""
Coefficient Tensor Structure Documentation for AI Understanding

This file demonstrates the structure of coefficient tensors in quantum pure states
represented by the Strawberry Fields quantum computing framework.
"""

# Import required modules
import scipy.integrate
import time
import numba

# Handle scipy version compatibility
if not hasattr(scipy.integrate, 'simps'):
    print("Monkey patching scipy.integrate: 'simps' not found. Pointing to 'simpson'.")
    scipy.integrate.simps = scipy.integrate.simpson
else:
    print("'simps' already exists in scipy.integrate. No patch needed.")
    
import strawberryfields as sf
import strawberryfields.ops as ops
import numpy as np

print("=" * 80)
print("EXAMPLE 1: Single Mode Displacement")
print("=" * 80)

# Configuration: 2 modes, cutoff dimension 4
# This means each mode can have photon numbers 0, 1, 2, 3
prog = sf.Program(2)
eng = sf.Engine("fock", backend_options={"cutoff_dim": 4})

# Apply displacement operation to mode 0
# Mode 1 remains in vacuum state |0⟩
with prog.context as q:
    ops.Dgate(3) | q[0]

# Execute the quantum program
state = eng.run(prog).state

# Coefficient tensor access
coeffs = state.data  # Alternative: state.ket()
print(f"Pure state: {state.is_pure}")
print(f"Tensor shape: {coeffs.shape}")
print(f"Full tensor:\n{coeffs}")


print("\n" + "=" * 80)
print("EXAMPLE 2: Single Mode Squeezing")
print("=" * 80)

# Reset program configuration
prog = sf.Program(2)
eng = sf.Engine("fock", backend_options={"cutoff_dim": 4})

# Apply squeezing operation to mode 1
# Mode 0 remains in vacuum state |0⟩
with prog.context as q:
    ops.Sgate(1) | q[1]

# Execute the quantum program
state = eng.run(prog).state

# Coefficient tensor access
coeffs = state.data
print(f"Pure state: {state.is_pure}")
print(f"Tensor shape: {coeffs.shape}")
print(f"Full tensor:\n{coeffs}")


print("\n" + "=" * 80)
print("EXAMPLE 3: Two-Mode State")
print("=" * 80)

# Reset program configuration
prog = sf.Program(2)
eng = sf.Engine("fock", backend_options={"cutoff_dim": 4})

# Apply operations to both modes
with prog.context as q:
    ops.Dgate(3) | q[0]
    ops.Sgate(1) | q[1]

# Execute the quantum program
state = eng.run(prog).state

# Coefficient tensor access
coeffs = state.data
print(f"Pure state: {state.is_pure}")
print(f"Tensor shape: {coeffs.shape}")
print(f"Full tensor:\n{coeffs}")
