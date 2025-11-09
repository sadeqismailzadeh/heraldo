"""Batch evaluation script to reproduce Figure 3 from the research article.

This script runs a trained PPO model over a large number of episodes in
parallel to collect statistics about its performance. It measures metrics
such as output state fidelity, episode length, and detected photons.

The collected data is then used to generate a 2x2 plot that replicates the
layout and content of Figure 3 in the original paper, providing a clear
benchmark of the agent's learned strategy.

Usage:
    1. Edit the `MODEL_PATH` variable in the "Configuration" section below to
       point to your trained `.zip` file.
    2. Adjust other evaluation parameters as needed.
    3. Run the script from the command line: `python article_fig3.py`
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# --- CRITICAL: Set thread limits BEFORE importing numpy/sb3 ---
# This is essential for good performance with multiprocessing.
print("--- Configuring thread limits for NumPy/OpenBLAS/MKL ---")
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from quantum_circuit_env import QuantumCircuitEnv, fidelity_with_sqrt

# --- Configuration ---

# Simulation Parameters (MUST match the training environment)
CUTOFF_DIM = 25
MAX_STEPS = 50
REWARD_POWER = 2 

# Evaluation Parameters
NUM_EPISODES_TO_COLLECT = 1250  # Total episodes for statistics
N_ENVS = 4                      # Number of parallel environments

# --- EDIT THIS: Path to the trained model ---
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ppo_quantum_circuit.zip")

# Analysis Parameters
SUCCESS_FIDELITY_THRESHOLD = 0.90 

def plot_results(fidelities, photons, episode_lengths, steps_between_resets):
    """Replicate the paper's Figure 3 layout for collected episode statistics."""
    print("\n--- Generating Plots (Replicating Figure 3 Layout) ---")
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    fig.suptitle("Evaluation Results (Replication of Paper's Figure 3)", fontsize=16, y=0.98)
    
    # (a) Output State Fidelity
    ax = axes[0, 0]
    ax.hist(fidelities, bins=50, range=(0.0, 1.0), color='tab:blue', edgecolor='black')
    ax.set_title("(a) Output State Fidelity")
    ax.set_xlabel("Output state fidelity")
    ax.set_ylabel("Episode count")
    ax.grid(True, linestyle='--')
    
    # (c) Steps per Episode, with Resets
    ax = axes[0, 1]
    ax.hist(episode_lengths, bins=50, range=(0, MAX_STEPS), color='tab:green', edgecolor='black')
    ax.set_title("(c) Steps per Episode, with Resets")
    ax.set_xlabel("Number of steps")
    ax.set_ylabel("Episode count")
    ax.grid(True, linestyle='--')
    
    # (b) Total Detected Photons per Episode
    ax = axes[1, 0]
    ax.hist(photons, bins=60, range=(0, 120), color='tab:orange', edgecolor='black')
    ax.set_title("(b) Total Detected Photons per Episode")
    ax.set_xlabel("Detected photon number")
    ax.set_ylabel("Episode count")
    ax.grid(True, linestyle='--')
    
    # (d) Steps per Episode, Between Resets
    ax = axes[1, 1]
    ax.hist(steps_between_resets, bins=50, range=(0, MAX_STEPS), color='tab:red', edgecolor='black')
    ax.set_title("(d) Steps per Episode, Between Resets")
    ax.set_xlabel("Number of steps (between resets)")
    ax.set_ylabel("Episode count")
    ax.grid(True, linestyle='--')
    
    fig.tight_layout()
    plt.show()

def main():
    """Evaluate the policy in parallel and summarize outcomes for Figure 3."""
    print("--- Starting Parallel Evaluation ---")
    
    print(f"Creating {N_ENVS} parallel environments...")
    env = make_vec_env(
        QuantumCircuitEnv,
        n_envs=N_ENVS,
        env_kwargs=dict(
            cutoff_dim=CUTOFF_DIM,
            max_steps=MAX_STEPS,
            reward_power=REWARD_POWER,
            tunable_r=False  # Make sure this matches your trained model
        ),
        vec_env_cls=SubprocVecEnv,
        vec_env_kwargs=dict(start_method='spawn')
    )

    print(f"Loading trained model from '{MODEL_PATH}'...")
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Trained model not found at '{MODEL_PATH}'")
        print("Please edit the MODEL_PATH variable in this script.")
        return

    model = PPO.load(MODEL_PATH, env=env)

    # Data storage for completed episodes
    fidelities = []
    episode_lengths = []
    steps_between_resets = []
    photons_between_resets = []

    # Trackers for ongoing episodes in each parallel environment
    is_reset = np.zeros(N_ENVS, dtype=bool)
    current_episode_steps = np.zeros(N_ENVS, dtype=int)
    last_reset_step = np.zeros(N_ENVS, dtype=int)
    current_segment_photons = np.zeros(N_ENVS, dtype=int)
    
    obs = env.reset()
    
    with tqdm(total=NUM_EPISODES_TO_COLLECT, desc="Collecting Episodes") as pbar:
        while len(fidelities) < NUM_EPISODES_TO_COLLECT:
            actions, _ = model.predict(obs, deterministic=True)
            new_obs, rewards, dones, infos = env.step(actions)

            for i in range(N_ENVS):
                current_episode_steps[i] += 1
                
                measured_photons = infos[i].get('measured_photons', 0)
                current_segment_photons[i] += 0 if is_reset[i] else measured_photons
                
                # Check for the agent's "reset" action (high transmissivity)
                theta_1 = actions[i][0]
                vbs1_transmissivity = np.cos(theta_1)**2
                
                if vbs1_transmissivity <= 0.01 and not is_reset[i]:
                    is_reset[i] = True                
                    last_reset_step[i] = current_episode_steps[i]
                
                if dones[i]:
                    # Episode finished, collect final data
                    final_info = infos[i]
                    final_dm = final_info.get('final_dm')
                    
                    if final_dm is not None:
                        target_sqrts = env.get_attr('target_sqrts')[0]
                        final_fidelities = np.array([fidelity_with_sqrt(sqrt, final_dm) for sqrt in target_sqrts])
                        fidelities.append(np.max(final_fidelities))
                    else:
                        fidelities.append(0.0)
                    
                    episode_lengths.append(current_episode_steps[i])
                    steps_between_resets.append(last_reset_step[i])
                    photons_between_resets.append(current_segment_photons[i])
                
                    # Reset trackers for this environment
                    current_episode_steps[i] = 0
                    last_reset_step[i] = 0
                    current_segment_photons[i] = 0
                    is_reset[i] = False                  
                    pbar.update(1)
            
            obs = new_obs

    env.close()

    print("\n--- Evaluation Complete ---")
    fidelities = np.array(fidelities)
    successful_episodes = np.sum(fidelities >= SUCCESS_FIDELITY_THRESHOLD)
    success_rate = (successful_episodes / len(fidelities)) * 100
    
    print(f"Total Episodes Collected: {len(fidelities)}")
    print(f"Average Final Fidelity: {np.mean(fidelities):.4f}")
    print(f"Success Rate (Fidelity >= {SUCCESS_FIDELITY_THRESHOLD}): {success_rate:.2f}%")
    
    plot_results(fidelities, photons_between_resets, episode_lengths, steps_between_resets)

if __name__ == '__main__':
    main()