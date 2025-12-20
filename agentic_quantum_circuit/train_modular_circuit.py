"""
Main training script for the MODULAR quantum environment.

This script duplicates the training procedure of 'train_quantum_circuit.py'
but utilizes 'ModularQuantumEnv' by injecting the specific Circuit, Target, 
and Reward modules required to reproduce the Squeezed Cat experiment.
"""
import os
import re
import glob
import platform
import multiprocessing as mp

from sympy import false

# --- CRITICAL: Set thread limits BEFORE importing other libraries ---
print("--- Configuring thread limits for NumPy/OpenBLAS/MKL ---")
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList, BaseCallback
from stable_baselines3.common.vec_env import VecNormalize, VecMonitor
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env

import torch

# --- Custom Imports ---
# 1. The Modular Environment
from modular_quantum_env import ModularQuantumEnv

# 2. The Specific Modules to assemble the environment
from circuits import *
from targets import *
from rewards import *

# 3. Callbacks (Assumed to exist based on your file list)
from thread_manager_callback import ThreadManagerCallback
from metrics_callback import MetricsCallback
from curriculum_callback import CurriculumCallback 
from adaptive_kl_callback import AdaptiveKLCallback

def main():
    """Configures the modular environment and launches PPO training."""
    
    # --- Configuration ---
    # Environment Parameters
    CUTOFF_DIM = 50
    MAX_STEPS = 50
    INITIAL_DIFFICULTY = 0.67 # Start with lower fidelity target


    # Training Parameters
    N_ENVS = 4  # Number of parallel environments
    TARGET_TIMESTEPS = 100_000_000
    CHECKPOINT_FREQ = 20_000

    # PPO Hyperparameters
    POLICY_KWARGS = dict(
        net_arch=dict(pi=[256, 256], vf=[256, 256]),
        optimizer_class=torch.optim.Adam,
        activation_fn=torch.nn.Tanh
    )
    LEARNING_RATE = 3e-4
    N_STEPS_PER_UPDATE = 2048 // N_ENVS 
    BATCH_SIZE = 64  
    N_EPOCHS = 10
    GAMMA = 0.99
    
    # --- Setup Paths ---
    log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Train_Modular")
    os.makedirs(log_dir, exist_ok=True)
    model_prefix = "ppo_modular_cat"

    # --- Instantiate Modules ---
    # We create instances of the components to define what the environment does.
    
    # 1. Circuit Context: The physical loop setup
    circuit_context = GeneralLoopCircuit(
        tunable_r=True, 
        max_squeezing=1.38
    )

    circuit_context2 = ThreeModeGadgetCircuit(
        max_sq_r=1, 
        max_disp=1,
        tunable_bs_phase=false
    )

    circuit_context3 = CubicSpecificCircuit(max_sq_r=1.38)
    
    # 2. Target Generator: The state we want to reach (Squeezed Cat)
    target1 = SqueezedCatTarget(
        alpha=3, 
        r=1.38,
        p=0
    )

    target2 = SqueezedCatTarget(
        alpha=3, 
        r=1.38,
        p=1
    )
    
    csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "GKP_core_coefficients.csv")
    target3=CoreGKPTarget(csv_path=csv_path, 
                          n_max=4, 
                          delta_db=10.4, 
                          mu=0)
    
    # 3. Reward Mechanism: How we calculate success
    reward_mech = LogFidelityReward()

    # --- Setup Parallel Environments ---
    print(f"Using {N_ENVS} parallel environments with ModularQuantumEnv.")
    
    # We pass the instances via env_kwargs. SubprocVecEnv will pickle them 
    # and send them to the worker processes.
    env_kwargs = dict(
        cutoff_dim=CUTOFF_DIM,
        max_steps=MAX_STEPS,
        loss_channel=1.0, # Assumes no loss for now, consistent with basic setup
        initial_target_fidelity=INITIAL_DIFFICULTY,
        # INJECT MODULES HERE:
        circuit_context=circuit_context,
        target_gens=[target3],
        reward_mech=reward_mech
    )

    vec_env = make_vec_env(
        ModularQuantumEnv,
        n_envs=N_ENVS,
        env_kwargs=env_kwargs,
        vec_env_cls=SubprocVecEnv,
        vec_env_kwargs=dict(start_method='spawn') 
    )

    # --- Auto-Resume Logic (Identical to original script) ---
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
        stats_filename = f"{model_prefix}_vecnormalize_{current_steps}_steps.pkl"
        stats_path = os.path.join(log_dir, stats_filename)
        if os.path.exists(stats_path):
            env = VecNormalize.load(stats_path, vec_env)
            print(f"Loaded VecNormalize stats from {stats_path}")
        else:
            # Fallback if stats missing
            env = VecNormalize(vec_env, norm_obs=False, gamma=GAMMA, norm_reward=True, clip_reward=10.0)
            
        model = PPO.load(latest_checkpoint, env=env, ent_coef=0)
        print(f"Model loaded. Resuming from {current_steps} timesteps.")
    else:
        env = VecNormalize(vec_env, norm_obs=False, gamma=GAMMA, norm_reward=True, clip_reward=10.0)

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
            verbose=1,
            device='cpu',
            tensorboard_log=log_dir
        )

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

    # Note: Ensure your CurriculumCallback calls env.set_difficulty().
    # Ideally, ModularQuantumEnv should propagate this to the reward mechanism 
    # or the reward mechanism should read it from the env state.
    curriculum_callback = CurriculumCallback(
        log_dir=log_dir,
        success_threshold=0.9, 
        max_difficulty=0.98,
        initial_difficulty=INITIAL_DIFFICULTY,
        verbose=1
    )

    kl_callback = AdaptiveKLCallback(
        kl_threshold=0.05, 
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
    stats_path = os.path.join(log_dir, "vec_normalize.pkl")
    env.save(stats_path)
    print(f"\n✅ Training finished. Saved to {final_model_path}")

if __name__ == "__main__":
    main()