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
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.env_util import make_vec_env
from quantum_circuit_env import QuantumCircuitEnv, fidelity_pure_state
from quantum_cubic_env import CubicPhaseEnv

# from quantum_gadget_env import QuantumGadgetEnv, fidelity_pure_state
# from quantum_circuit_env_mixed import QuantumCircuitEnv

# --- Configuration ---

# Simulation Parameters
CUTOFF_DIM = 50
MAX_STEPS = 50
REWARD_POWER = 2 

# Evaluation Parameters
NUM_EPISODES_TO_COLLECT = 1250
N_ENVS = 4

# --- EDIT THIS: Path to the trained model ---
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ppo_quantum_circuit.zip")

def plot_results(fidelities, photons, episode_lengths, steps_between_resets):
    """Replicate the paper's Figure 3 layout."""
    print("\n--- Generating Plots (Replicating Figure 3 Layout) ---")
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    fig.suptitle(f"Evaluation Results (N={len(fidelities)})", fontsize=16, y=0.98)
    
    # (a) Output State Fidelity
    ax = axes[0, 0]
    ax.hist(fidelities, bins=50, range=(0.0, 1.0), color='#1f77b4', edgecolor='black', alpha=0.7)
    ax.set_title("(a) Output State Fidelity")
    ax.set_xlabel("Fidelity")
    ax.set_ylabel("Episode count")
    ax.grid(True, linestyle='--', alpha=0.5)
    
    # (b) Total Detected Photons per Episode
    ax = axes[1, 0]
    # Log scale helps visualize the spread if there are many resets
    ax.hist(photons, bins=50, range=(0, 100), color="#ff7f0e", edgecolor='black', alpha=0.7)
    ax.set_title("(b) Total Detected Photons (per Episode)")
    ax.set_xlabel("Detected photon number")
    ax.set_ylabel("Episode count")
    ax.grid(True, linestyle='--', alpha=0.5)

    # (c) Steps per Episode, with Resets
    # This is the raw length of the episode until termination or max_steps
    ax = axes[0, 1]
    ax.hist(episode_lengths, bins=MAX_STEPS, range=(0, MAX_STEPS), color="#2ca02c", edgecolor='black', alpha=0.7)
    ax.set_title("(c) Total Steps per Episode (With Resets)")
    ax.set_xlabel("Number of steps")
    ax.set_ylabel("Episode count")
    ax.grid(True, linestyle='--', alpha=0.5)
    
    # (d) Steps per Episode, Between Resets
    # This shows the length of the *successful* sequence
    ax = axes[1, 1]
    ax.hist(steps_between_resets, bins=MAX_STEPS, range=(0, MAX_STEPS), color='#d62728', edgecolor='black', alpha=0.7)
    ax.set_title("(d) Steps in Final Sequence (Since Last Reset)")
    ax.set_xlabel("Number of steps")
    ax.set_ylabel("Episode count")
    ax.grid(True, linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    # save the figure
    file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fig3.png")
    plt.savefig(file_path, dpi=300, bbox_inches='tight')
    plt.show()

def main():
    print("--- Starting Parallel Evaluation ---")

    env = make_vec_env(
        QuantumCircuitEnv,
        n_envs=N_ENVS,
        env_kwargs=dict(
            cutoff_dim=CUTOFF_DIM,
            max_steps=MAX_STEPS,
            reward_power=REWARD_POWER,
            tunable_r=True, # Article 1 usually assumes fixed r=1.38, agent controls theta
            is_loss_channel=False,
            loss_channel=1,
            initial_target_fidelity=0.95,
        ),
        vec_env_cls=SubprocVecEnv
    )
    
    # Load VecNormalize stats if available
    stats_path = MODEL_PATH.replace('.zip', '_vecnormalize.pkl')
    if os.path.exists(stats_path):
        env = VecNormalize.load(stats_path, env)
        env.training = False
        env.norm_reward = False

    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model not found at {MODEL_PATH}")
        return

    model = PPO.load(MODEL_PATH, env=env)

    # --- STORAGE ---
    # Global results
    final_fidelities = []
    total_steps_data = []       # For Fig 3(c)
    segment_steps_data = []     # For Fig 3(d)
    total_photons_data = []     # For Fig 3(b)

    # Per-environment trackers
    # Tracks total steps in the current episode (Fig 3c)
    current_total_steps = np.zeros(N_ENVS, dtype=int)
    # Tracks steps since the last "Reset" action (Fig 3d)
    current_segment_steps = np.zeros(N_ENVS, dtype=int)
    # Tracks total photons accumulated in the episode
    current_total_photons = np.zeros(N_ENVS, dtype=int)
    
    obs = env.reset()
    
    with tqdm(total=NUM_EPISODES_TO_COLLECT, desc="Evaluating") as pbar:
        while len(final_fidelities) < NUM_EPISODES_TO_COLLECT:
            actions, _ = model.predict(obs, deterministic=True)
            new_obs, rewards, dones, infos = env.step(actions)
            
            for i in range(N_ENVS):
                # 1. Update Photon Counts
                n_photons = infos[i].get('detected_photons', 0)
                
                # 2. Analyze Action for "Reset" vs "Hold" vs "Build"
                # Action[0] is Theta. Transmissivity tau = cos(theta)^2
                denorm_action = env.env_method('_denormalize_action', actions[i], indices=[i])[0]
                theta = denorm_action[1]

                transmissivity = np.cos(theta)**2
                
                # Thresholds
                RESET_THRESHOLD = 0.95  # tau approx 1 (Flush loop)
                HOLD_THRESHOLD = 0.05   # tau approx 0 (Trap light / Stop)

                if transmissivity > RESET_THRESHOLD:
                    # --- RESET ACTION ---
                    # Agent discards state. Reset segment counter.
                    current_segment_steps[i] = 0
                    
                elif transmissivity < HOLD_THRESHOLD:
                    # --- HOLD ACTION ---
                    # Agent is happy and freezing the state.
                    # Do NOT increment the segment counter. 
                    # (It effectively "stopped" building the state here)
                    pass
                    
                else:
                    # --- BUILD ACTION ---
                    # Agent is actively mixing/measuring.
                    # Increment the counter for the current attempt.
                    current_segment_steps[i] += 1
                    current_total_steps[i] += 1
                    current_total_photons[i] += n_photons
                

                # 3. Handle Episode Termination
                if dones[i]:
                    # Extract Final Fidelity
                    fid = infos[i].get('fidelity')
                    # Access target kets from the env object (wrapped in VecEnv)
                    # Note: get_attr returns a list of attributes from all envs
                    if fid is not None:
                        final_fidelities.append(fid)
                    else:
                        final_fidelities.append(0.0)
                    
                    # Store Statistics
                    total_steps_data.append(current_total_steps[i])
                    segment_steps_data.append(current_segment_steps[i])
                    total_photons_data.append(current_total_photons[i])
                    
                    # Reset Trackers for this environment
                    current_total_steps[i] = 0
                    current_segment_steps[i] = 0
                    current_total_photons[i] = 0
                    
                    pbar.update(1)
            
            obs = new_obs

    env.close()
    
    # --- RESULTS ---
    final_fidelities = np.array(final_fidelities)
    print(f"Average Fidelity: {np.mean(final_fidelities):.4f}")
    
    plot_results(final_fidelities, total_photons_data, total_steps_data, segment_steps_data)

if __name__ == '__main__':
    main()