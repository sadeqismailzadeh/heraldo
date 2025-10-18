"""Training entry point for reinforcement-learning control of a quantum circuit.

This script configures multiprocessing-safe threading limits, constructs a
vectorized :class:`QuantumCircuitEnv`, and trains a Stable-Baselines3 PPO agent
with checkpoint-based auto-resume support. The resulting policy checkpoints and
TensorBoard logs are stored under the local ``Train`` directory.
"""

# dependencies
# pip install strawberryfields gymnasium stable-baselines3[extra]

import os
import platform
import multiprocessing as mp

# ==============================================================================
# === CRITICAL: CONTROL NUMPY THREADING FOR MULTIPROCESSING ====================
# ==============================================================================
# Set these environment variables BEFORE importing numpy, sf, or sb3.
# This prevents NumPy's backend from creating a thread storm when using
# multiple environments in parallel. We want each process to use only ONE core.
print("--- Configuring thread limits for NumPy/OpenBLAS/MKL ---")
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

import glob
import re
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList 
from stable_baselines3.common.vec_env import SubprocVecEnv
# NEW: Import for creating parallel environments
from stable_baselines3.common.env_util import make_vec_env

# Import our custom quantum environment
from quantum_circuit_env import QuantumCircuitEnv
from thread_manager_callback import ThreadManagerCallback
import torch

import warnings
from scipy.linalg import LinAlgWarning
warnings.simplefilter('always', LinAlgWarning)  # show every occurrence

# TODO SUPERVISOR: document the code. flowchart

# TODO validity of no cache functions 
# TODO train with tunable r, vacuum inital state
# TODO termination with set full reflective action ask ai
# TODO loss channel
# TODO Map a full noise-phase diagram. Find noise thresholds where RL still performs like baselines?
# TODO stable baseline zoo for hyperparameter tuning
# TODO venv on ssd no cuda

# It's good practice to wrap the main execution logic in a function
def main():
    """Configure the environment, resume if possible, and launch PPO training."""
    # ==============================================================================
    # === 1. CONFIGURATION =========================================================
    # ==============================================================================

    # The reward_power for the environment
    REWARD_POWER = 2

    # ==============================================================================
    # === 2. MULTIPROCESSING CONFIGURATION =========================================
    # ==============================================================================
    
    # Determine the number of parallel environments
    # Leave at least one core free for the main process
    cpu_count = mp.cpu_count()
    # N_ENVS = max(1, cpu_count - 1)  # At least 1, at most (cpu_count - 1)
    N_ENVS = 4  # At least 1, at most (cpu_count - 1)
    print(f"Using {N_ENVS} parallel environments (detected {cpu_count} CPU cores)")
    

    # Create the vectorized environment
    env = make_vec_env(
        QuantumCircuitEnv,
        n_envs=N_ENVS,
        env_kwargs=dict(
            cutoff_dim=25,
            max_steps=10,
            reward_power=REWARD_POWER,
            tunable_r=False,
            is_agent_able_to_terminate=False
        ),
        vec_env_cls=SubprocVecEnv,
        # Use the platform-appropriate start method determined above
        # 'spawn': Works on all platforms, creates fresh Python interpreter for each process
        # 'fork': Linux-only, faster but can have issues with certain libraries
        vec_env_kwargs=dict(start_method='spawn')
    )

    # You can add this check to be 100% sure
    print(f"Vectorized environment type: {type(env.unwrapped)}")
    assert isinstance(env.unwrapped, SubprocVecEnv), "FATAL: Not using SubprocVecEnv for multiprocessing!"
    
    # ==============================================================================
    # === 3. SETUP PATHS ===========================================================
    # ==============================================================================

    # Define the base directory where everything will be saved.
    # on script directory   
    log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Train")
    os.makedirs(log_dir, exist_ok=True)

    # Define a prefix for your saved model files
    model_prefix = "ppo_quantum_circuit"


    # ==============================================================================
    # === 4. AUTO-RESUME LOGIC =====================================================
    # ==============================================================================
    latest_checkpoint = None
    current_steps = 0
    
    print("--- Checking for existing checkpoints... ---")

    # Find all checkpoint files in the log directory that match the prefix
    checkpoint_files = glob.glob(os.path.join(log_dir, f"{model_prefix}_*.zip"))

    if checkpoint_files:
        # If checkpoints exist, find the one with the highest step number
        # We extract the number from the filename (e.g., "ppo_..._120000_steps.zip")
        try:
            # Exclude the final model from checkpoint resume
            checkpoint_files = [f for f in checkpoint_files if "_final.zip" not in f]
            if checkpoint_files:
                latest_checkpoint = max(
                    checkpoint_files,
                    key=lambda f: int(re.search(r'_(\d+)_steps.zip', f).group(1))
                )
                # --- NEW: PARSE THE STEP COUNT FROM THE FILENAME ---
                current_steps = int(re.search(r'_(\d+)_steps.zip', latest_checkpoint).group(1))
                print(f"✅ Found latest checkpoint: {os.path.basename(latest_checkpoint)}")
        except (ValueError, AttributeError):
            print("⚠️ Could not determine the latest checkpoint. Starting fresh.")
            # This can happen if filenames are not in the expected format


    # ==============================================================================
    # === 5. CREATE OR LOAD MODEL ==================================================
    # ==============================================================================
    
    # Create or load the model
    if latest_checkpoint:
        print("\n--- RESUMING TRAINING ---")
        # Load the model from the latest checkpoint
        model = PPO.load(latest_checkpoint, env=env)
        # model = SAC.load(latest_checkpoint, env=env)
        print("Model loaded. Continuing from where it left off.")

        # --- SET a new, much smaller learning rate ---
        # One order of magnitude smaller is a great starting point.
        # new_learning_rate = 3e-5 
        # # Update lr_schedule, which is called to determine current learning rate
        # # here a constant learning rate
        # model.lr_schedule = lambda _: new_learning_rate
        # # Update `learning_rate` too in case we want to save/load the model
        # # (cf. remark below)
        # model.learning_rate = lambda _: new_learning_rate
        # print(f"New learning rate set to: {new_learning_rate}")
        
    else:
        print("\n--- STARTING NEW TRAINING ---")
        policy_kwargs = dict(
            net_arch=dict(pi=[256, 256], vf=[256, 256]) # pi=policy network, vf=value network
        )

        model = PPO(
            "MlpPolicy",
            env,
            policy_kwargs=policy_kwargs,
            learning_rate=3e-4,      # Default is good. Can try 1e-4 if unstable.
            n_steps=2*1024,            # Crucial: Number of steps per env before an update.
            batch_size=64,           # Mini-batch size for the update.
            n_epochs=10,             # How many times to iterate over the collected data.
            gamma=0.98,              # Discount factor. Slightly lower for short episodes.
            gae_lambda=0.95,         # Factor for trade-off of bias vs variance for GAE.
            ent_coef=0.01,           # Entropy coefficient to encourage exploration.
            verbose=1,
            tensorboard_log=log_dir
        )

        # policy_kwargs = dict(
        #     net_arch=dict(pi=[256, 256], qf=[256, 256]) # pi=policy network, qf=Q-function network
        # )

        # model = SAC(
        #     "MlpPolicy",
        #     env,
        #     policy_kwargs=policy_kwargs,
        #     learning_rate=3e-4,        # Good default. Can be tuned with a scheduler.
        #     buffer_size=200_000,       # How many transitions to store in the replay buffer.
        #     batch_size=256,            # How many samples to use for each gradient update.
        #     gamma=0.98,                # Discount factor.
        #     tau=0.005,                 # The soft update coefficient for target networks.
        #     ent_coef='auto',           # Crucial: Automatically tunes the entropy bonus.
        #     train_freq = (1, "step"),   # Update the model after every step.
        #     gradient_steps=-1,          # Perform one gradient step per update.
        #     learning_starts=8000,      # Collect 1000 random steps before starting to train.
        #     verbose=0,
        #     device='cuda' if torch.cuda.is_available() else 'cpu',
        #     tensorboard_log=log_dir
        # )


        print("New model created.")


    # ==============================================================================
    # === 6. DEFINE CALLBACKS ======================================================
    # ==============================================================================
    # This callback will save the model every `save_freq` steps.
    # A frequency of 10,000 to 20,000 steps is a good starting point.
    checkpoint_callback = CheckpointCallback(
    save_freq=20000,
    save_path=log_dir,
    name_prefix=model_prefix,
    save_replay_buffer=True,
    save_vecnormalize=True
    )


    # NEW: Add the thread management callback
    # It's good practice to get the cpu_count once and reuse it
    # num_cpus = mp.cpu_count()
    num_cpus = 4
    print(f"--- Configuring dynamic threading ---")
    print(f"Threads during rollout: 1")
    print(f"Threads during model update: {num_cpus}")

    thread_manager_callback = ThreadManagerCallback(rollout_threads=1, update_threads=num_cpus, verbose=1)

    # Combine checkpoint and thread callbacks
    callback_list = CallbackList([checkpoint_callback, thread_manager_callback])
    print("Using checkpoint-based training with dynamic threading.")

    # ==============================================================================
    # === 7. TRAIN THE AGENT =======================================================
    # ==============================================================================
    # Set the total number of timesteps for the entire training run
    TARGET_TIMESTEPS = 7_000_000
    # --- NEW: CALCULATE THE REMAINING STEPS TO TRAIN ---
    remaining_timesteps = TARGET_TIMESTEPS - current_steps

    print(f"\n--- Starting/Resuming training ---")
    print(f"Total timesteps: {TARGET_TIMESTEPS}")
    print(f"Current timesteps: {current_steps}")
    print(f"Remaining timesteps to learn: {remaining_timesteps}")


    # The `learn` call
    # `reset_num_timesteps=False` is CRUCIAL for resuming. It ensures the step
    # counter continues from the loaded model's progress.


    model.learn(
        total_timesteps=remaining_timesteps,
        callback=callback_list,
        reset_num_timesteps=(False if latest_checkpoint else True), # IMPORTANT FOR RESUMING
        progress_bar = True,
    )

    print("\n--- Training Finished! ---")


    # ==============================================================================
    # === 8. SAVE THE FINAL MODEL ==================================================
    # ==============================================================================
    final_model_path = os.path.join(log_dir, f"{model_prefix}_final.zip")
    model.save(final_model_path)
    print(f"\n✅ Final model saved to: {final_model_path}")

# ==============================================================================
# === SCRIPT ENTRY POINT =======================================================
# ==============================================================================
if __name__ == "__main__":
    # This is the crucial part. The main() function will only be called
    # when the script is executed directly.
    
    main()
