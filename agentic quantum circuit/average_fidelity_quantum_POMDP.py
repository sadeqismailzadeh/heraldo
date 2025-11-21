"""Script to calculate the average final fidelity over a specified number of episodes for POMDP.

This script loads a pre-trained RecurrentPPO model and runs it in the PartiallyObservableQuantumEnv
for a specified number of episodes. It collects the final fidelity values from
each episode (from info) and computes their average.

Usage:
    1. Edit the `N_EPISODES` variable below to set the number of episodes.
    2. Edit the `MODEL_PATH` variable below to point to your trained `.zip` file.
    3. Run the script from the command line: `python average_fidelity_quantum_POMDP.py`
"""
import os
import numpy as np
from tqdm import tqdm

from quantum_circuit_env import QuantumCircuitEnv, fidelity_pure_state
from partially_observable_env import PartiallyObservableQuantumEnv

from sb3_contrib import RecurrentPPO

import strawberryfields as sf
# temporary fix. it may cause crashes or silently produce incorrect results
# os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


def main():
    """Runs the evaluation and computes average fidelity."""
    # --- Configuration ---
    # IMPORTANT: Environment parameters MUST match those used during training.
    CUTOFF_DIM = 25
    MAX_STEPS = 10
    REWARD_POWER = 2
    N_EPISODES = 10  # Number of episodes to run

    # --- EDIT THIS: Path to the trained model ---
    MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ppo_quantum_circuit.zip")

    # --- Environment and Model Setup ---
    env = PartiallyObservableQuantumEnv(cutoff_dim=CUTOFF_DIM,
                            max_steps=MAX_STEPS,
                            reward_power=REWARD_POWER,
                            tunable_r=True,
                            is_loss_channel=True,
                            loss_channel=1)

    try:
        model = RecurrentPPO.load(MODEL_PATH, env=env)
    except FileNotFoundError:
        print(f"Error: Trained model not found at '{MODEL_PATH}'")
        print("Please edit the MODEL_PATH variable in this script.")
        exit()

    # --- Run Evaluation Episodes and Collect Fidelities ---
    fidelities = []
    pbar = tqdm(range(N_EPISODES))
    for episode in pbar:
        obs, info = env.reset()
        # Cell and hidden state of the LSTM
        lstm_states = None
        num_envs = 1
        # Episode start signals are used to reset the lstm states
        episode_starts = np.ones((num_envs,), dtype=bool)

        terminated, truncated = False, False
        while not (terminated or truncated):
            action, lstm_states = model.predict(obs, state=lstm_states, episode_start=episode_starts, deterministic=True)
            episode_starts = np.zeros((num_envs,), dtype=bool)
            obs, reward, terminated, truncated, info = env.step(action)

        # --- Collect Final Fidelity ---
        final_fidelity = info.get('fidelity', 0.0)
        fidelities.append(final_fidelity)

        # --- Update Progress Bar with Running Average ---
        running_avg = np.mean(fidelities)
        pbar.set_description(f"Avg: {running_avg:.4f}")

    # --- Compute and Output Average ---
    average_fidelity = np.mean(fidelities)
    print(f"Average final fidelity over {N_EPISODES} episodes: {average_fidelity:.4f}")

    env.close()


if __name__ == '__main__':
    main()