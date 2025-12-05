"""Script to calculate the average final fidelity over a specified number of episodes.

This script loads a pre-trained PPO model and runs it in the QuantumCircuitEnv
for a specified number of episodes. It collects the final fidelity values from
each episode (the best fidelity against target states) and computes their average.

Usage:
    1. Edit the `N_EPISODES` variable below to set the number of episodes.
    2. Edit the `MODEL_PATH` variable below to point to your trained `.zip` file.
    3. Run the script from the command line: `python average_fidelity_quantum.py`
"""
import os
import numpy as np
from tqdm import tqdm

from stable_baselines3 import PPO
from quantum_circuit_env import QuantumCircuitEnv, fidelity_pure_state

import strawberryfields as sf



def main():
    """Runs the evaluation and computes average fidelity."""
    # --- Configuration ---
    # IMPORTANT: Environment parameters MUST match those used during training.
    CUTOFF_DIM = 25
    MAX_STEPS = 50
    REWARD_POWER = 2
    N_EPISODES = 1000  # Number of episodes to run

    # --- EDIT THIS: Path to the trained model ---
    MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ppo_quantum_circuit.zip")

    # --- Environment and Model Setup ---
    env = QuantumCircuitEnv(cutoff_dim=CUTOFF_DIM,
                            max_steps=MAX_STEPS,
                            reward_power=REWARD_POWER,
                            tunable_r=True,
                            is_loss_channel=True,
                            loss_channel=1)

    try:
        model = PPO.load(MODEL_PATH, env=env)
    except FileNotFoundError:
        print(f"Error: Trained model not found at '{MODEL_PATH}'")
        print("Please specify the correct model path using --model_path.")
        exit()

    # --- Run Evaluation Episodes and Collect Fidelities ---
    fidelities = []
    fid_per_steps = []
    total_steps_list = []
    pbar = tqdm(range(N_EPISODES))
    for episode in pbar:
        obs, info = env.reset()
        terminated, truncated = False, False
        total_steps = 0
        while not (terminated or truncated):
            # Use the deterministic policy for evaluation
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_steps += 1

        # --- Collect Final Fidelity ---
        final_fidelity = info.get('fidelity', 0.0)
        fidelities.append(final_fidelity)

        fid_per_step = final_fidelity / total_steps
        fid_per_steps.append(fid_per_step)

        total_steps_list.append(total_steps)
        # --- Update Progress Bar with Running Average ---
        running_avg = np.mean(total_steps_list)
        pbar.set_description(f"Avg: {running_avg:.4f}")

    # --- Compute and Output Average ---
    average_fidelity = np.mean(fidelities)
    print(f"Average final fidelity over {N_EPISODES} episodes: {average_fidelity:.4f}")

    env.close()


if __name__ == '__main__':
    main()