import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import cm

import strawberryfields as sf
from strawberryfields.ops import Load

from stable_baselines3 import PPO

# Import our custom quantum environment and the fidelity function
from quantum_circuit_env import QuantumCircuitEnv, uhlmann_jozsa_fidelity

import os
# temporary fix. it may cause crashes or silently produce incorrect results
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE" 


# --- 1. Setup the Environment and Model ---

# IMPORTANT: The parameters here (especially cutoff_dim) MUST match
# the parameters used during training.
CUTOFF_DIM = 20 # The same cutoff_dim used in train_quantum.py
env = QuantumCircuitEnv(cutoff_dim=CUTOFF_DIM)

# Load the trained model
MODEL_PATH = "ppo_quantum_circuit.zip"
try:
    model = PPO.load(MODEL_PATH, env=env)
except FileNotFoundError:
    print(f"Error: Trained model not found at '{MODEL_PATH}'")
    print("Please run train_quantum.py first to train and save the model.")
    exit()

# --- 2. Run the Evaluation ---
num_episodes = 2 # Run for a few episodes to see different outcomes
for episode in range(num_episodes):
    print(f"\n{'='*20} Starting Evaluation Episode {episode + 1} {'='*20}")
    
    obs, info = env.reset()
    terminated, truncated = False, False
    total_reward = 0
    
    # Loop through the steps of a single episode
    while not (terminated or truncated):
        # Use the deterministic policy for evaluation
        action, _ = model.predict(obs, deterministic=True)
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Get the measured photon number from the info dictionary
        measured_n = info.get('measured_photons', 'N/A')
        
        # Log the details of this step
        print(
            f"Step {env.current_step:2d}: "
            f"Action=[r={action[0]:.4f}, T={action[1]:.4f}], "
            f"Measured_n={measured_n}, "
            f"Step Reward={reward:.6f}"
        )
        total_reward += reward

    # --- 3. Post-Episode Analysis ---
    print(f"\n--- Episode {episode + 1} Finished ---")
    print(f"Total Steps: {env.current_step}")
    print(f"Total Reward: {total_reward:.4f}")

    # Get the final density matrix from the environment
    final_dm = env.current_dm

    # Calculate the fidelity against the four target states
    final_fidelities = [uhlmann_jozsa_fidelity(final_dm, target) for target in env.target_dms]
    best_fidelity = np.max(final_fidelities)
    best_target_index = np.argmax(final_fidelities)

    print(f"Final State Fidelity vs Target States: {[f'{f:.4f}' for f in final_fidelities]}")
    print(f"BEST FIDELITY: {best_fidelity:.4f} (with Target #{best_target_index})")

    # --- 4. Visualize the Final State ---
    
    
    # Generate the Wigner function plot
    final_state = env.current_state
    xvec = np.linspace(-6, 6, 200)
    W = final_state.wigner(mode=0, xvec=xvec, pvec=xvec)
    
    # Plotting boilerplate
    scale = np.max(np.abs(W))
    nrm = mpl.colors.Normalize(-scale, scale)
    
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.set_aspect("equal")
    ax.contourf(xvec, xvec, W, 60, cmap=cm.RdBu, norm=nrm)
    
    fig.colorbar(cm.ScalarMappable(norm=nrm, cmap=cm.RdBu), ax=ax)
    plt.title(f"Episode {episode+1} - Final State Wigner Function\nBest Fidelity: {best_fidelity:.4f}", fontsize=14)
    plt.xlabel("q", fontsize=12)
    plt.ylabel("p", fontsize=12)
    plt.grid(True, linestyle='--')
    plt.show()

env.close()