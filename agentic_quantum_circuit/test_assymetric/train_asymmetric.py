"""Main training script for the asymmetric quantum circuit reinforcement learning agent.

This script sets up and runs the training process for a RecurrentPPO agent from the
Stable Baselines3 library with a custom asymmetric policy. It is designed for efficient,
parallel training and includes robust features like automatic resumption from the latest
checkpoint.

Key Features:
   - **Parallel Training:** Utilizes `SubprocVecEnv` to run multiple environments
     in parallel, significantly speeding up data collection.
   - **Auto-Resume:** Automatically detects and loads the latest model checkpoint
     from the `Train/` directory, allowing training to be stopped and started
     without losing progress.
   - **TensorBoard Logging:** Logs key training metrics (reward, loss, etc.)
     to a TensorBoard instance for real-time monitoring.
   - **Dynamic Threading:** Manages CPU threads to optimize for both data
     collection (rollout) and model updates, preventing performance bottlenecks.
   - **Asymmetric Policy:** Uses a custom LSTM policy tailored for asymmetric
     training environments with blind dimensions.

Usage:
    1. Adjust the parameters in the "Configuration" section of the `main()`
       function below.
    2. Run the script from the command line: `python train_asymmetric.py`
    3. To monitor training, run: `python run_tensorboard.py`
"""
import os
import re
import glob
import platform
import multiprocessing as mp

# --- CRITICAL: Set thread limits BEFORE importing other libraries ---
# This prevents NumPy's backend from creating a thread storm when using
# multiple environments in parallel. Each process should use only one core.
print("--- Configuring thread limits for NumPy/OpenBLAS/MKL ---")
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

from sb3_contrib import RecurrentPPO
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env

from asymmetric_env import AsymmetricTrainingEnv
from asymmetric_policy import AsymmetricLstmPolicy
from thread_manager_callback import ThreadManagerCallback
from metrics_callback import MetricsCallback
from typing import Callable


def linear_schedule(initial_value: float, end_value: float) -> Callable[[float], float]:
    """
    Linear learning rate schedule.

    :param initial_value: The initial learning rate.
    :param end_value: The final learning rate.
    :return: schedule that computes current learning rate depending on progress
    """
    def func(progress_remaining: float) -> float:
        """
        Progress will decrease from 1 (beginning) to 0 (end).
        """
        return end_value + progress_remaining * (initial_value - end_value)

    return func


def main():
    """Configures the environment, resumes if possible, and launches RecurrentPPO training."""
    # --- Configuration ---
    # Environment Parameters
    CUTOFF_DIM = 25
    MAX_STEPS = 10
    REWARD_POWER = 2
    TUNABLE_R = True
    LOSS_CHANNEL = 1  # Specific to asymmetric training: The Hard Problem

    # Training Parameters
    N_ENVS = 4  # Number of parallel environments
    TARGET_TIMESTEPS = 40_000_000  # Total steps for the entire training run
    CHECKPOINT_FREQ = 20_000  # Save a checkpoint every N steps

    # PPO Hyperparameters


    LEARNING_RATE = 3e-4
    N_STEPS_PER_UPDATE = 4096
    BATCH_SIZE = 128
    N_EPOCHS = 10
    GAMMA = 0.99
    GAE_LAMBDA = 0.95
    ENT_COEF = 0.01

    # --- Setup Paths ---
    log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Train")
    os.makedirs(log_dir, exist_ok=True)
    model_prefix = "ppo_quantum_circuit"

    # --- Calculate Blind Dimension ---
    # We need to know 'blind_dim' to pass it to the Policy.
    # Let's create a dummy env to check.
    temp_env = AsymmetricTrainingEnv(
        cutoff_dim=CUTOFF_DIM,
        max_steps=MAX_STEPS,
        is_loss_channel=True,
        loss_channel=LOSS_CHANNEL,
    )
    BLIND_DIM = temp_env.blind_dim
    print(f"Detected Blind Dimension: {BLIND_DIM}")
    temp_env.close()

    # We pass the BLIND_DIM to the policy so it knows where to slice.
    EXTRACTOR_ARCH = dict(
        pi=[256, 256],  # Actor: Input -> 256 -> 256 -> LSTM
        vf=[256, 256]   # Critic: Input -> 256 -> 256 -> Linear (No LSTM)
    )

    POLICY_KWARGS = dict(
        blind_dim=BLIND_DIM,
        lstm_hidden_size=256,
        enable_critic_lstm=False,
        shared_lstm=False,  # these two flags combined make the critic memory less
        extractor_arch=EXTRACTOR_ARCH
    )

    # --- Setup Parallel Environments ---
    print(f"Using {N_ENVS} parallel environments.")
    env = make_vec_env(
        AsymmetricTrainingEnv,
        n_envs=N_ENVS,
        env_kwargs=dict(
            cutoff_dim=CUTOFF_DIM,
            max_steps=MAX_STEPS,
            reward_power=REWARD_POWER,
            tunable_r=TUNABLE_R,
            is_loss_channel=True,
            loss_channel=LOSS_CHANNEL
        ),
        vec_env_cls=SubprocVecEnv,
        vec_env_kwargs=dict(start_method='spawn')  # 'spawn' is safer for cross-platform
    )

    # --- Setup Asymmetric Policy ---
    

    # --- Auto-Resume Logic ---
    latest_checkpoint = None
    current_steps = 0
    print("--- Checking for existing checkpoints... ---")
    checkpoint_files = glob.glob(os.path.join(log_dir, f"{model_prefix}_*.zip"))

    # Exclude the final model from the list of checkpoints to resume from
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
        print("\n--- RESUMING TRAINING ---")
        model = RecurrentPPO.load(latest_checkpoint, env=env)

        # --- SET a new, much smaller learning rate ---
        # One order of magnitude smaller is a great starting point.
        # new_learning_rate = 3e-5
        # Update lr_schedule, which is called to determine current learning rate
        # here a constant learning rate

        # model.lr_schedule = lambda _: 1e-4
        # # Update `learning_rate` too in case we want to save/load the model
        # # (cf. remark below)
        # model.learning_rate = lambda _: initial_lr
        # print(f"New learning rate set to: {initial_lr}")

        print(f"Model loaded. Resuming from {current_steps} timesteps.")
    else:
        print("\n--- STARTING NEW TRAINING ---")
        model = RecurrentPPO(
            AsymmetricLstmPolicy,  # Use our custom class
            env,
            policy_kwargs=POLICY_KWARGS,
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
        print("New RecurrentPPO model created.")

    # --- Define Callbacks ---
    checkpoint_callback = CheckpointCallback(
        save_freq=CHECKPOINT_FREQ,
        save_path=log_dir,
        name_prefix=model_prefix,
        save_replay_buffer=True,
        save_vecnormalize=True
    )

    # Optimize thread usage for different parts of the training loop
    num_cpus = 4
    thread_manager_callback = ThreadManagerCallback(rollout_threads=1, update_threads=num_cpus, verbose=1)
    metrics_callback = MetricsCallback(verbose=1)

    callback_list = CallbackList([checkpoint_callback, thread_manager_callback, metrics_callback])

    # --- Train the Agent ---
    remaining_timesteps = TARGET_TIMESTEPS - current_steps
    print(f"\n--- Starting/Resuming training ---")
    print(f"Total target timesteps: {TARGET_TIMESTEPS}")
    print(f"Remaining timesteps to learn: {remaining_timesteps}")

    model.learn(
        total_timesteps=remaining_timesteps,
        callback=callback_list,
        reset_num_timesteps=(latest_checkpoint is None),  # Crucial for resuming
        progress_bar=True,
    )

    print("\n--- Training Finished! ---")

    # --- Save the Final Model ---
    final_model_path = os.path.join(log_dir, f"{model_prefix}_final.zip")
    model.save(final_model_path)
    print(f"\n✅ Final model saved to: {final_model_path}")


if __name__ == "__main__":
    main()