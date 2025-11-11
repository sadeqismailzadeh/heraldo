"""Main training script for the quantum circuit reinforcement learning agent
using an Asymmetric Actor-Critic (AAC) architecture.

This script sets up and runs the training process for a PPO agent from the
Stable Baselines3 library, adapted for the AAC setup.
"""
import os
import re
import glob
import platform
import multiprocessing as mp

# --- CRITICAL: Set thread limits BEFORE importing other libraries ---
print("--- Configuring thread limits for NumPy/OpenBLAS/MKL ---")
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

from stable_baselines3.ppo import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env

from aac_quantum_env import AACQuantumCircuitEnv
from custom_policy import AsymmetricRecurrentCriticPolicy
from thread_manager_callback import ThreadManagerCallback
from metrics_callback import MetricsCallback
from typing import Callable

def linear_schedule(initial_value: float, end_value: float) -> Callable[[float], float]:
    """
    Linear learning rate schedule.
    """
    def func(progress_remaining: float) -> float:
        return end_value + progress_remaining * (initial_value - end_value)
    return func

def main():
    """Configures the AAC environment, resumes if possible, and launches PPO training."""
    # --- Configuration ---
    # Environment Parameters
    CUTOFF_DIM = 25
    MAX_STEPS = 10
    REWARD_POWER = 5
    TUNABLE_R = True

    # Training Parameters
    N_ENVS = 4
    TARGET_TIMESTEPS = 10_000_000
    CHECKPOINT_FREQ = 20_000

    # PPO Hyperparameters for AAC
    POLICY_KWARGS = dict(net_arch=dict(pi=[128, 128], vf=[256, 256]))
    LEARNING_RATE = 3e-4
    N_STEPS_PER_UPDATE = 2048 * 2
    BATCH_SIZE = 64 * 2
    N_EPOCHS = 10
    GAMMA = 0.99
    GAE_LAMBDA = 0.95
    ENT_COEF = 0.01

    # --- Setup Paths ---
    log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Train_AAC")
    os.makedirs(log_dir, exist_ok=True)
    model_prefix = "ppo_aac_quantum_circuit"

    # --- Setup Parallel Environments ---
    print(f"Using {N_ENVS} parallel environments for AAC training.")
    env = make_vec_env(
        AACQuantumCircuitEnv,
        n_envs=N_ENVS,
        env_kwargs=dict(
            cutoff_dim=CUTOFF_DIM,
            max_steps=MAX_STEPS,
            reward_power=REWARD_POWER,
            tunable_r=TUNABLE_R,
            is_loss_channel=False,
            loss_channel=1
        ),
        vec_env_cls=SubprocVecEnv,
        vec_env_kwargs=dict(start_method='spawn')
    )

    # --- Auto-Resume Logic ---
    latest_checkpoint = None
    current_steps = 0
    print("--- Checking for existing checkpoints... ---")
    checkpoint_files = glob.glob(os.path.join(log_dir, f"{model_prefix}_*.zip"))
    
    checkpoint_files = [f for f in checkpoint_files if "_final.zip" not in f]

    if checkpoint_files:
        try:
            latest_checkpoint = max(
                checkpoint_files,
                key=lambda f: int(re.search(r'_(\d+)_steps.zip', f).group(1))
            )
            current_steps = int(re.search(r'_(\d+)_steps.zip', latest_checkpoint).group(1))
            print(f"✅ Found latest checkpoint: {os.path.basename(latest_checkpoint)}")
        except (ValueError, AttributeError):
            print("⚠️ Could not parse step count from checkpoint names. Starting fresh.")

    # --- Create or Load Model ---
    if latest_checkpoint:
        print("\n--- RESUMING AAC TRAINING ---")
        model = PPO.load(latest_checkpoint, env=env, policy=AsymmetricRecurrentCriticPolicy)
        print(f"Model loaded. Resuming from {current_steps} timesteps.")
    else:
        print("\n--- STARTING NEW AAC TRAINING ---")
        model = PPO(
            AsymmetricRecurrentCriticPolicy,
            env,
            learning_rate=LEARNING_RATE,
            n_steps=N_STEPS_PER_UPDATE,
            batch_size=BATCH_SIZE,
            n_epochs=N_EPOCHS,
            gamma=GAMMA,
            gae_lambda=GAE_LAMBDA,
            ent_coef=ENT_COEF,
            verbose=1,
            device='cpu',
            tensorboard_log=log_dir
        )
        print("New PPO model with AsymmetricCriticPolicy created.")

    # --- Define Callbacks ---
    checkpoint_callback = CheckpointCallback(
        save_freq=CHECKPOINT_FREQ,
        save_path=log_dir,
        name_prefix=model_prefix,
        save_replay_buffer=True,
        save_vecnormalize=True
    )
    
    num_cpus = 4
    thread_manager_callback = ThreadManagerCallback(rollout_threads=1, update_threads=num_cpus, verbose=1)
    metrics_callback = MetricsCallback(verbose=1)

    callback_list = CallbackList([checkpoint_callback, thread_manager_callback, metrics_callback])

    # --- Train the Agent ---
    remaining_timesteps = TARGET_TIMESTEPS - current_steps
    print(f"\n--- Starting/Resuming AAC training ---")
    print(f"Total target timesteps: {TARGET_TIMESTEPS}")
    print(f"Remaining timesteps to learn: {remaining_timesteps}")

    model.learn(
        total_timesteps=remaining_timesteps,
        callback=callback_list,
        reset_num_timesteps=(latest_checkpoint is None),
        progress_bar=True,
    )

    print("\n--- AAC Training Finished! ---")

    # --- Save the Final Model ---
    final_model_path = os.path.join(log_dir, f"{model_prefix}_final.zip")
    model.save(final_model_path)
    print(f"\n✅ Final AAC model saved to: {final_model_path}")

if __name__ == "__main__":
    main()
