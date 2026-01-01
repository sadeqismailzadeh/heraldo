import os
import re
import sys
import glob
import torch
import multiprocessing as mp

# --- CRITICAL: Set thread limits BEFORE importing other libraries ---
print("--- Configuring thread limits for NumPy/OpenBLAS/MKL ---")
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.env_util import make_vec_env

# Ensure we can import the environment file if running this script directly
current_dir = os.path.dirname(os.path.abspath(__file__))
# sys.path.append(current_dir)
# # Add project root to path for quantum_agent imports
# project_root = os.path.abspath(os.path.join(current_dir, "../../../"))
# if project_root not in sys.path:
#     sys.path.append(project_root)

from hypersphere_env import HypersphereNavigationEnv

# Callback imports
from quantum_agent.callbacks.thread_manager_callback import ThreadManagerCallback
from quantum_agent.callbacks.metrics_callback import MetricsCallback
from quantum_agent.callbacks.curriculum_callback import CurriculumCallback
from quantum_agent.callbacks.adaptive_kl_callback import AdaptiveKLCallback

def main():
    # --- Configuration ---
    # Environment Parameters
    N_DIMS = 25
    MAX_STEPS = 500
    INITIAL_FIDELITY = 0.85

    # Training Parameters
    N_ENVS = 4
    TARGET_TIMESTEPS = 10_000_000
    CHECKPOINT_FREQ = 20_000
    USE_VEC_NORMALIZE = False

    # PPO Hyperparameters (Similar to modular circuit training)
    POLICY_KWARGS = dict(
        net_arch=dict(pi=[256, 256, 256], vf=[256, 256, 256]),
        optimizer_class=torch.optim.Adam,
        activation_fn=torch.nn.Tanh
    )
    LEARNING_RATE = 3e-4
    N_STEPS_PER_UPDATE = 2048 // N_ENVS * 4*2*2
    BATCH_SIZE = 1024*2*2
    N_EPOCHS = 3
    GAMMA = 0.99
    
    # --- Setup Paths ---
    log_dir = os.path.join(current_dir, "logs/hypersphere_tuning")
    os.makedirs(log_dir, exist_ok=True)
    model_prefix = "ppo_hypersphere_modular"

    # --- Setup Parallel Environments ---
    print(f"Using {N_ENVS} parallel environments with HypersphereNavigationEnv.")
    
    env_kwargs = dict(
        n_dims=N_DIMS,
        max_steps=MAX_STEPS,
        terminate_on_success=True, # Maintenance mode learning
        target_fidelity=INITIAL_FIDELITY
    )

    vec_env = make_vec_env(
        HypersphereNavigationEnv,
        n_envs=N_ENVS,
        env_kwargs=env_kwargs,
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
            print("⚠️ Could not parse step count. Starting fresh.")
            latest_checkpoint = None

    # --- Create or Load Model ---
    if latest_checkpoint:
        print("\n--- RESUMING TRAINING ---")
        if USE_VEC_NORMALIZE:
            stats_filename = f"{model_prefix}_vecnormalize_{current_steps}_steps.pkl"
            stats_path = os.path.join(log_dir, stats_filename)
            if os.path.exists(stats_path):
                env = VecNormalize.load(stats_path, vec_env)
                print(f"Loaded VecNormalize stats from {stats_path}")
            else:
                env = VecNormalize(vec_env, norm_obs=False, gamma=GAMMA, norm_reward=True, clip_reward=10.0)
        else:
            env = vec_env
            
        model = PPO.load(latest_checkpoint, 
                        learning_rate=LEARNING_RATE,
                        n_steps=N_STEPS_PER_UPDATE,
                        batch_size=BATCH_SIZE,
                        n_epochs=N_EPOCHS,
                        gamma=GAMMA,
                        env=env, 
                        ent_coef=0.01)
        print(f"Model loaded. Resuming from {current_steps} timesteps.")
    else:
        if USE_VEC_NORMALIZE:
            env = VecNormalize(vec_env, norm_obs=False, gamma=GAMMA, norm_reward=True, clip_reward=10.0)
        else:
            env = vec_env

        print("\n--- STARTING NEW TRAINING ---")
        model = PPO(
            "MlpPolicy",
            env,
            policy_kwargs=POLICY_KWARGS,
            learning_rate=LEARNING_RATE,
            n_steps=N_STEPS_PER_UPDATE,
            batch_size=BATCH_SIZE,
            n_epochs=N_EPOCHS,
            gamma=GAMMA,
            ent_coef=0.01,
            verbose=1,
            device='auto',
            tensorboard_log=log_dir
        )

    # --- Define Callbacks ---
    checkpoint_callback = CheckpointCallback(
        save_freq=CHECKPOINT_FREQ,
        save_path=log_dir,
        name_prefix=model_prefix,
        save_replay_buffer=True,
        save_vecnormalize=USE_VEC_NORMALIZE
    )
    
    num_cpus = mp.cpu_count()
    thread_manager_callback = ThreadManagerCallback(rollout_threads=1, update_threads=num_cpus, verbose=1)
    metrics_callback = MetricsCallback(verbose=1)

    curriculum_callback = CurriculumCallback(
        log_dir=log_dir,
        success_threshold=0.98, 
        max_difficulty=0.999,
        initial_difficulty=INITIAL_FIDELITY,
        verbose=1
    )

    kl_callback = AdaptiveKLCallback(
        kl_threshold=0.03, 
        window_size=15, 
        decay_factor=0.5,
        min_lr=1e-6
    )

    callback_list = CallbackList([
        checkpoint_callback, 
        thread_manager_callback, 
        metrics_callback, 
        curriculum_callback, 
        kl_callback
    ])

    # --- Train the Agent ---
    remaining_timesteps = TARGET_TIMESTEPS - current_steps
    print(f"\n--- Starting/Resuming training ---")
    
    model.learn(
        total_timesteps=remaining_timesteps,
        callback=callback_list,
        reset_num_timesteps=(latest_checkpoint is None),
        progress_bar=True,
    )

    # --- Save Final ---
    final_model_path = os.path.join(log_dir, f"{model_prefix}_final.zip")
    model.save(final_model_path)
    if USE_VEC_NORMALIZE:
        stats_path = os.path.join(log_dir, "vec_normalize.pkl")
        env.save(stats_path)
    print(f"\n✅ Training finished. Saved to {final_model_path}")

if __name__ == "__main__":
    main()
