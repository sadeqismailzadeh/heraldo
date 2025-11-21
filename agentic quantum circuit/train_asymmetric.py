import os
print("--- Configuring thread limits for NumPy/OpenBLAS/MKL ---")
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

import gymnasium as gym
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CheckpointCallback
from thread_manager_callback import ThreadManagerCallback
from metrics_callback import MetricsCallback
import glob
import re


# Import our custom modules
from asymmetric_env import AsymmetricTrainingEnv
from asymmetric_policy import AsymmetricLstmPolicy

def main():

    # --- Hyperparameters for High-Loss Regime (0.9) ---
    CUTOFF_DIM = 25
    MAX_STEPS = 10
    LOSS_CHANNEL = 1 # The Hard Problem

    TARGET_TIMESTEPS = 40_000_000  # Total steps for the entire training run
    CHECKPOINT_FREQ = 20_000  # Save a checkpoint every N steps
    
    # Robust PPO Params
    LEARNING_RATE = 3e-4       # Slow and steady
    N_STEPS_PER_UPDATE = 4096  # Large buffer
    BATCH_SIZE = 128          # Large batch to average out loss noise
    N_EPOCHS = 10
    ENT_COEF = 0.02            # Force exploration
    
    # Parallelism
    N_ENVS = 4 # Increase this if your CPU allows!
    
    # --- 1. Calculate Dimensions ---
    # We need to know 'blind_dim' to pass it to the Policy.
    # Let's create a dummy env to check.
    temp_env = AsymmetricTrainingEnv(
        cutoff_dim=CUTOFF_DIM, 
        max_steps=MAX_STEPS, 
        loss_channel=LOSS_CHANNEL,
        is_loss_channel=True
    )
    BLIND_DIM = temp_env.blind_dim
    print(f"Detected Blind Dimension: {BLIND_DIM}")
    temp_env.close()

    # --- 2. Setup Environment ---
    env = make_vec_env(
        AsymmetricTrainingEnv,
        n_envs=N_ENVS,
        env_kwargs=dict(
            cutoff_dim=CUTOFF_DIM,
            max_steps=MAX_STEPS,
            reward_power=2,
            tunable_r=True,
            is_loss_channel=True, 
            loss_channel=LOSS_CHANNEL
        ),
        vec_env_cls=SubprocVecEnv,
        vec_env_kwargs=dict(start_method='spawn')
    )

    # --- 3. Setup Asymmetric Policy ---
    # We pass the BLIND_DIM to the policy so it knows where to slice.
    policy_kwargs = dict(
        blind_dim=BLIND_DIM,
        lstm_hidden_size=256,
        enable_critic_lstm=False,
        shared_lstm= False, # these two flags combined ake the critic memory less
        # net_arch=[], # We handled the architecture in the custom policy class
    )

    # --- Setup Paths ---
    log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Train")
    os.makedirs(log_dir, exist_ok=True)
    model_prefix = "ppo_quantum_circuit"


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
            AsymmetricLstmPolicy, # Use our custom class
            env,
            learning_rate=LEARNING_RATE,
            n_steps=N_STEPS_PER_UPDATE,
            batch_size=BATCH_SIZE,
            n_epochs=N_EPOCHS,
            ent_coef=ENT_COEF,
            gamma=0.99,
            policy_kwargs=policy_kwargs,
            verbose=1,
            tensorboard_log=log_dir,
            device='cpu' 
        )   
        print("New PPO model created.")

    # --- Define Callbacks ---
    checkpoint_callback = CheckpointCallback(
        save_freq=20000,
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
        reset_num_timesteps=(latest_checkpoint is None), # Crucial for resuming
        progress_bar=True,
    )

    print("\n--- Training Finished! ---")

    
    
    model.save(f"{log_dir}/final_model")
    print("Training Complete.")

if __name__ == "__main__":
    main()