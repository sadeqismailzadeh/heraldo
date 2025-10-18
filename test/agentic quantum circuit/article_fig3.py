"""Batch evaluation script that reproduces Figure 3 statistics from the article."""

import numpy as np
import matplotlib.pyplot as plt
import os
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

# Import your custom quantum environment and helper functions
from quantum_circuit_env import QuantumCircuitEnv, fidelity_with_sqrt

# --- Configuration ---

# --- 1. Simulation Parameters (MUST match training) ---
CUTOFF_DIM = 25
MAX_STEPS = 50
REWARD_POWER = 2 

# --- 2. Evaluation Parameters ---
NUM_EPISODES_TO_COLLECT = 1250 # Total episodes to run for statistics
N_ENVS = 4 # Number of parallel environments
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ppo_quantum_circuit.zip")

# --- 3. Analysis Parameters ---
RESET_TRANSMISSIVITY_THRESHOLD = 0.99 
SUCCESS_FIDELITY_THRESHOLD = 0.90 

# --- Helper Function for Plotting (Unchanged from before) ---
def plot_results(fidelities, photons, episode_lengths, steps_between_resets):
    """Replicate the paper's Figure 3 layout for collected episode statistics.

    Args:
        fidelities (Iterable[float]): Maximum fidelity achieved per episode.
        photons (Iterable[int]): Photon counts accumulated between agent resets.
        episode_lengths (Iterable[int]): Number of interaction steps per episode.
        steps_between_resets (Iterable[int]): Steps elapsed before each
            transmissivity reset event.
    """
    print("\n--- Generating Plots (Replicating Figure 3 Layout) ---")
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    fig.suptitle("Evaluation Results (Replication of Paper's Figure 3)", fontsize=16, y=0.98)
    # (a)
    ax = axes[0, 0]
    ax.hist(fidelities, bins=50, range=(0.0, 1.0), color='tab:blue', edgecolor='black')
    ax.set_title("(a) Output State Fidelity")
    ax.set_xlabel("Output state fidelity")
    ax.set_ylabel("Episode count")
    ax.grid(True, linestyle='--')
    # (c)
    ax = axes[0, 1]
    ax.hist(episode_lengths, bins=50, range=(0, MAX_STEPS), color='tab:green', edgecolor='black')
    ax.set_title("(c) Steps per Episode, with Resets")
    ax.set_xlabel("Number of steps")
    ax.set_ylabel("Episode count")
    ax.grid(True, linestyle='--')
    # (b)
    ax = axes[1, 0]
    ax.hist(photons, bins=60, range=(0, 120), color='tab:orange', edgecolor='black')
    ax.set_title("(b) Total Detected Photons per Episode")
    ax.set_xlabel("Detected photon number")
    ax.set_ylabel("Episode count")
    ax.grid(True, linestyle='--')
    # (d)
    ax = axes[1, 1]
    ax.hist(steps_between_resets, bins=50, range=(0, MAX_STEPS), color='tab:red', edgecolor='black')
    ax.set_title("(d) Steps per Episode, Between Resets")
    ax.set_xlabel("Number of steps (between resets)")
    ax.set_ylabel("Episode count")
    ax.grid(True, linestyle='--')
    fig.tight_layout()
    plt.show()

# --- Main Evaluation Script ---
def main():
    """Evaluate the policy in parallel and summarize outcomes for Figure 3."""
    print("--- Starting Parallel Evaluation ---")
    
    # --- 1. Setup Parallel Environment and Load Model ---
    
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

    # Note: Use your final trained model path here
    MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ppo_quantum_circuit.zip")
    print(f"Loading trained model from '{MODEL_PATH}'...")
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Trained model not found at '{MODEL_PATH}'")
        return

    model = PPO.load(MODEL_PATH, env=env)

    # --- 2. Run Parallel Evaluation and Collect Data ---
    
    # Data storage lists for completed episodes/segments
    fidelities = []
    episode_lengths = []
    steps_between_resets = []
    # --- NEW: List to store photon counts per segment ---
    photons_between_resets = []


    is_reset=np.zeros(N_ENVS, dtype=bool)
    # Trackers for ongoing episodes in each parallel environment
    current_episode_steps = np.zeros(N_ENVS, dtype=int)
    last_reset_step = np.zeros(N_ENVS, dtype=int)
    # --- NEW: Tracker for photons in the current segment ---
    current_segment_photons = np.zeros(N_ENVS, dtype=int)
    
    obs = env.reset()
    
    # Use tqdm for a progress bar
    with tqdm(total=NUM_EPISODES_TO_COLLECT, desc="Collecting Episodes") as pbar:
        while len(fidelities) < NUM_EPISODES_TO_COLLECT:
            actions, _ = model.predict(obs, deterministic=True)
            new_obs, rewards, dones, infos = env.step(actions)

            # Process results for each environment
            for i in range(N_ENVS):
                current_episode_steps[i] += 1
                
                # --- MODIFIED: Add measured photons to the current SEGMENT's count ---
                measured_photons = infos[i].get('measured_photons', 0)
                current_segment_photons[i] += 0 if is_reset[i] else measured_photons
                
                # Check for agent's "reset" action
                # Assumes action[0] is theta_1 for the BSgate
                # and you are using the non-tunable_r version of the env.
                # If your action space is different, adjust this line.
                theta_1 = actions[i][0]
                vbs1_transmissivity = np.cos(theta_1)**2
                
                if vbs1_transmissivity <= 0.01 and is_reset[i] == False:
                    is_reset[i] = True                
                    last_reset_step[i] = current_episode_steps[i]
                
                # Check if the episode in this environment is done
                if dones[i]:
                    # --- An episode has finished, collect its final data ---
                    final_info = infos[i]
                    final_dm = final_info.get('final_dm')
                    
                    if final_dm is not None:
                        target_sqrts_list = env.get_attr('target_sqrts')
                        target_sqrts = target_sqrts_list[0]
                        final_fidelities = np.array([fidelity_with_sqrt(sqrt, final_dm) for sqrt in target_sqrts])
                        fidelities.append(np.max(final_fidelities))
                    else:
                        fidelities.append(0.0)
                    
                    episode_lengths.append(current_episode_steps[i])
                    steps_between_resets.append(last_reset_step[i])
                    photons_between_resets.append(current_segment_photons[i])
                
                    # --- Reset ALL trackers for this environment ---
                    current_episode_steps[i] = 0
                    last_reset_step[i] = 0
                    current_segment_photons[i] = 0
                    is_reset[i] = False                  
                    pbar.update(1)
            
            obs = new_obs

    env.close()

    # --- 3. Analyze and Display Results ---
    print("\n--- Evaluation Complete ---")
    fidelities = np.array(fidelities)
    
    successful_episodes = np.sum(fidelities >= SUCCESS_FIDELITY_THRESHOLD)
    success_rate = (successful_episodes / len(fidelities)) * 100
    
    print(f"Total Episodes Collected: {len(fidelities)}")
    print(f"Total Segments Recorded: {len(steps_between_resets)}")
    print(f"Average Final Fidelity: {np.mean(fidelities):.4f}")
    print(f"Success Rate (Fidelity >= {SUCCESS_FIDELITY_THRESHOLD}): {success_rate:.2f}%")
    
    # --- MODIFIED: Pass the new photon data to the plotting function ---
    plot_results(fidelities, photons_between_resets, episode_lengths, steps_between_resets)

if __name__ == '__main__':
    main()