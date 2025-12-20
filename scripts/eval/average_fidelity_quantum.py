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
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.env_util import make_vec_env
from quantum_circuit_env import QuantumCircuitEnv, fidelity_pure_state
from quantum_cubic_env import CubicPhaseEnv

import strawberryfields as sf



def main():
    """Runs the evaluation and computes average fidelity."""
    # --- Configuration ---
    # IMPORTANT: Environment parameters MUST match those used during training.
    CUTOFF_DIM = 50
    MAX_STEPS = 50
    REWARD_POWER = 2
    N_EPISODES = 1000  # Number of episodes to run

    # --- EDIT THIS: Path to the trained model ---
    MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ppo_quantum_circuit.zip")

    # --- Environment and Model Setup ---
    env = make_vec_env(
        QuantumCircuitEnv,
        n_envs=1,
        env_kwargs=dict(
            cutoff_dim=CUTOFF_DIM,
            max_steps=MAX_STEPS,
            reward_power=REWARD_POWER,
            tunable_r=True,
            is_loss_channel=True,
            loss_channel=1,
            initial_target_fidelity=0.95,
        ),
    )

    try:
        stats_path = MODEL_PATH.replace('.zip', '_vecnormalize.pkl')
        env = VecNormalize.load(stats_path, env)
        env.training = False
        env.norm_reward = False
        model = PPO.load(MODEL_PATH, env=env)
    except FileNotFoundError as e:
        if 'vecnormalize' in str(e) or not os.path.exists(MODEL_PATH):
            print(f"Error: Trained model or VecNormalize stats not found at '{MODEL_PATH}' or '{stats_path}'")
            print("Please specify the correct model path.")
            exit()
        else:
            print(f"VecNormalize stats not found at '{stats_path}', loading model without VecNormalize.")
            model = PPO.load(MODEL_PATH, env=env)

    # --- Run Evaluation Episodes and Collect Fidelities ---
    fidelities = []
    fid_per_steps = []
    total_steps_list = []
    pbar = tqdm(range(N_EPISODES))
    for episode in pbar:
        obs = env.reset()
        done = False
        total_steps = 0
        while not done:
            # Use the deterministic policy for evaluation
            actions, _ = model.predict(obs, deterministic=True)
            obs, rewards, dones, infos = env.step(actions)
            done = dones[0]
            total_steps += 1

        # --- Collect Final Fidelity ---
        final_fidelity = infos[0].get('fidelity', 0.0)
        fidelities.append(final_fidelity)

        fid_per_step = final_fidelity / total_steps
        fid_per_steps.append(fid_per_step)

        total_steps_list.append(total_steps)
        # --- Update Progress Bar with Running Average ---
        running_avg = np.mean(fidelities)
        pbar.set_description(f"Avg: {running_avg:.4f}")

    # --- Compute and Output Average ---
    average_fidelity = np.mean(fidelities)
    print(f"Average final fidelity over {N_EPISODES} episodes: {average_fidelity:.4f}")

    env.close()


if __name__ == '__main__':
    main()