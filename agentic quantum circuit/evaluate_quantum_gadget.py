"""Interactive evaluation script for a single trained agent.

This script loads a pre-trained PPO model and runs it in the QuantumCircuitEnv
for a few episodes. It provides a detailed, step-by-step log of the agent's
actions, the resulting measurements, and the rewards received.

After each episode, it calculates the final state's fidelity against the
target states and generates a Wigner function plot to visualize the
quantum state in phase space.

Usage:
    1. Edit the `MODEL_PATH` variable below to point to your trained `.zip` file.
    2. Run the script from the command line: `python evaluate_quantum.py`
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import cm

from stable_baselines3 import PPO
from quantum_circuit_env import QuantumCircuitEnv, fidelity_pure_state
from quantum_gadget_env import QuantumGadgetEnv
from quantum_gadget_env_3mode import ThreeModeGadgetEnv

import strawberryfields as sf


def main():
    """Runs the main evaluation loop."""
    # --- Configuration ---
    # IMPORTANT: Environment parameters MUST match those used during training.
    CUTOFF_DIM = 15
    MAX_STEPS = 10
    REWARD_POWER = 2
    NUM_EPISODES = 10 # Number of episodes to run

    # --- EDIT THIS: Path to the trained model ---
    MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ppo_quantum_circuit.zip")

    # --- Environment and Model Setup ---
    env = ThreeModeGadgetEnv(cutoff_dim=CUTOFF_DIM,
                           max_steps=MAX_STEPS,
                           reward_power=REWARD_POWER,
                           tunable_r=True,
                           is_loss_channel=True,
                           loss_channel=1)

    try:
        model = PPO.load(MODEL_PATH, env=env)
    except FileNotFoundError:
        print(f"Error: Trained model not found at '{MODEL_PATH}'")
        print("Please edit the MODEL_PATH variable in this script.")
        exit()

    # --- Run Evaluation Episodes ---
    for episode in range(NUM_EPISODES):
        print(f"\n{'='*20} Starting Evaluation Episode {episode + 1} {'='*20}")
        
        obs, info = env.reset()
        terminated, truncated = False, False
        total_reward = 0
        
        while not (terminated or truncated):
            # Use the deterministic policy for evaluation
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            
            measured_n = info.get('detected_photons', 'N/A')
            photon_loss = info.get('photon_loss', 'N/A')
            fidelity = info.get('fidelity', 'N/A')
            print(
                f"Step {env.current_step:2d}: "
                f"Action=[r={action[0]:.4f},  phi_sq={action[2]:.4f}, tau={np.cos(action[1]):.4f}, alpha_mag={action[3]:.4f}, alpha_phi={action[4]:.4f}], "
                f"Measured_n={measured_n}, "
                f"photon_loss={photon_loss}, "
                f"fidelity={fidelity:.4f}"
            )
            total_reward += reward

        # --- Post-Episode Analysis and Visualization ---
        print(f"\n--- Episode {episode + 1} Finished ---")
        print(f"Total Steps: {env.current_step}")
        print(f"Total Reward: {total_reward:.4f}")

        # Generate and display the Wigner function plot
        final_state = env.current_state
        state = final_state

        x_limit = 5
        grid_size = 200
        cutoff_dim = CUTOFF_DIM
        # --- 4. CALCULATE WIGNER FUNCTION ---
        print("Calculating Wigner function...")
        xvec = np.linspace(-x_limit, x_limit, grid_size)
        pvec = np.linspace(-x_limit, x_limit, grid_size)
        W = state.wigner(mode=0, xvec=xvec, pvec=pvec)

        # --- 5. CALCULATE FOCK PROBABILITIES ---

        ket = state.ket()
        probs = np.abs(ket[:,0, 0])**2
        probs = probs.flatten().real

        # --- 6. PLOTTING ---
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # Plot 1: Wigner Function
        # We use RdBu_r colormap: Red = Positive, Blue = Negative (Standard in Quantum Optics)
        # Negative regions indicate Non-Gaussianity.
        X, P = np.meshgrid(xvec, pvec)
        c = ax1.pcolormesh(X, P, W, cmap='RdBu', shading='auto', vmin=-np.max(np.abs(W)), vmax=np.max(np.abs(W)))
        fig.colorbar(c, ax=ax1, label='W(x, p)')
        ax1.set_title(f"Wigner Function")
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

    env.close()

if __name__ == '__main__':
    main()