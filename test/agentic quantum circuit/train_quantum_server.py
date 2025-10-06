import os
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
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList 
# NEW: Import for creating parallel environments
from stable_baselines3.common.env_util import make_vec_env

# Import our custom quantum environment
from quantum_circuit_env import QuantumCircuitEnv
from curriculum_callback import CurriculumCallback

import warnings
from scipy.linalg import LinAlgWarning
warnings.simplefilter('always', LinAlgWarning)  # show every occurrence


# It's good practice to wrap the main execution logic in a function
def main():
    # ==============================================================================
    # === 1. CONFIGURATION =========================================================
    # ==============================================================================
    # Set this to True to use curriculum learning, False to use checkpoint resume
    USE_CURRICULUM = False  # Toggle this flag to switch between modes
    
    # ==============================================================================
    # === 2. DEFINE CURRICULUM & SETUP ENVIRONMENT =================================
    # ==============================================================================
    # Define the stages: {mean_reward_threshold: new_reward_power}
    # This requires tuning! Start with thresholds you think are achievable at each stage.
    # The reward is calculated as: max_fidelity ** reward_power
    # Higher reward_power makes it harder to get high rewards (sharper curve)
    # Lower reward_power makes it easier to get rewards (flatter curve)
    # Start with low power (easy) and gradually increase to make learning harder
    CURRICULUM_STAGES = {
        0.5: 10,   # When mean reward >= 0.5, increase power to 10
        0.7: 20,   # When mean reward >= 0.7, increase power to 20
        0.85: 30,  # When mean reward >= 0.85, increase power to 30
        0.95: 50   # When mean reward >= 0.95, increase power to final 50
    }

    # The starting reward_power for the environment. Make it easy!
    # Use easier starting power when using curriculum, harder when not
    STARTING_REWARD_POWER = 5 if USE_CURRICULUM else 50

    N_ENVS = 4 # Or os.cpu_count() - 1


    # Pass the starting difficulty to the environment constructor
    env = make_vec_env(
        QuantumCircuitEnv,
        n_envs=N_ENVS,
        env_kwargs=dict(
            cutoff_dim=25,
            max_steps=10,
            reward_power=STARTING_REWARD_POWER  # <-- Start easy with low power
        ),
        vec_env_cls=SubprocVecEnv,
        # On Windows and macOS, 'spawn' is the only safe start method.
        # This is the default but we make it explicit for clarity.
        vec_env_kwargs=dict(start_method='spawn')
    )

    # You can add this check to be 100% sure
    print(f"Vectorized environment type: {type(env.unwrapped)}")
    assert isinstance(env.unwrapped, SubprocVecEnv), "FATAL: Not using SubprocVecEnv for multiprocessing!"



    # env = QuantumCircuitEnv(cutoff_dim=25, max_steps=10)
    # env = make_vec_env(QuantumCircuitEnv, n_envs=4, env_kwargs=dict(cutoff_dim=25, max_steps=10))


    # ==============================================================================
    # === 3. SETUP PATHS ===========================================================
    # ==============================================================================

    # Define the base directory where everything will be saved.
    log_dir = "./Train/"
    os.makedirs(log_dir, exist_ok=True)

    # Define a prefix for your saved model files
    model_prefix = "ppo_quantum_circuit"


    # ==============================================================================
    # === 4. AUTO-RESUME LOGIC (when not using curriculum) ========================
    # ==============================================================================
    latest_checkpoint = None
    
    if not USE_CURRICULUM:
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
                    print(f"✅ Found latest checkpoint: {os.path.basename(latest_checkpoint)}")
            except (ValueError, AttributeError):
                print("⚠️ Could not determine the latest checkpoint. Starting fresh.")
                # This can happen if filenames are not in the expected format


    # ==============================================================================
    # === 5. CREATE OR LOAD MODEL ==================================================
    # ==============================================================================
    
    # Create or load the model
    if latest_checkpoint and not USE_CURRICULUM:
        print("\n--- RESUMING TRAINING ---")
        # Load the model from the latest checkpoint
        model = PPO.load(latest_checkpoint, env=env)
        print("Model loaded. Continuing from where it left off.")
    else:
        if USE_CURRICULUM:
            print("\n--- STARTING NEW TRAINING WITH CURRICULUM ---")
        else:
            print("\n--- STARTING NEW TRAINING ---")
        # If no checkpoint was found, create a new PPO model
        model = PPO(
        # "MlpPolicy": This tells SB3 to use a standard neural network (Multi-Layer Perceptron)
        # as the agent's "brain". This is the right choice for vector-based states like ours.
        # If we had image-based states, we would use "CnnPolicy".
        "MlpPolicy",

        # The environment the agent will interact with and learn from.
        env,

        # verbose=1 prints out training progress (rewards, episode lengths, etc.) to the console.
        verbose=1,

        # ======================================================================
        # === KEY HYPERPARAMETERS  =============================================
        # ======================================================================
        # These values control the learning process. Tuning them can improve performance.

        # gamma: The discount factor. A value close to 1 (like 0.99) makes the agent "patient",
        # caring about long-term rewards. A value close to 0 would make it "short-sighted".
        gamma=0.999,

        # n_steps: The number of steps the agent takes in the environment before it updates
        # its policy network. A larger value provides more data for each update, which
        # can lead to more stable training.
        n_steps=12500,

        # batch_size: During the policy update, the collected data is split into
        # mini-batches of this size.
        batch_size=5000,

        # n_epochs: The number of times the agent will iterate over the collected data
        # during each policy update.
        n_epochs=14,

        # learning_rate: Controls how much the neural network's weights are adjusted
        # during each update. A smaller value leads to slower but often more stable learning.
        learning_rate=0.001,

        # ======================================================================
        # === OTHER CONFIGURATIONS =============================================
        # ======================================================================

        # policy_kwargs: A dictionary for passing extra arguments to the policy
        # network, such as network architecture and the optimizer.
        # net_arch: Defines the size of the neural networks for the policy (pi)
        # and the value function (vf).
        policy_kwargs = dict(net_arch=dict(pi=[256, 128, 64], vf=[256, 128, 64])),

        # tensorboard_log: Specifies a directory to save training logs. These can be
        # viewed with a tool called TensorBoard for detailed graphs of the training process.
        tensorboard_log=log_dir,

        device="cpu",

        # for debug. remove in actual training
        # seed=42 
        )
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

    # Conditionally set up callbacks based on USE_CURRICULUM flag
    if USE_CURRICULUM:
        # Callback for curriculum learning (verbose=2 for detailed debugging)
        curriculum_callback = CurriculumCallback(curriculum_stages=CURRICULUM_STAGES, verbose=2)
        # Combine both callbacks into a list
        callback_list = CallbackList([checkpoint_callback, curriculum_callback])
        print("Using curriculum learning with checkpoint saving.")
    else:
        # Use only checkpoint callback for regular training with resume capability
        callback_list = checkpoint_callback
        print("Using checkpoint-based training (no curriculum learning).")


    # ==============================================================================
    # === 7. TRAIN THE AGENT =======================================================
    # ==============================================================================
    # Set the total number of timesteps for the entire training run
    TOTAL_TIMESTEPS = 7_000_000

    print(f"\n--- Starting/Resuming training for {TOTAL_TIMESTEPS} total timesteps ---")

    # The `learn` call
    # `reset_num_timesteps=False` is CRUCIAL for resuming. It ensures the step
    # counter continues from the loaded model's progress.


    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
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
    from stable_baselines3.common.vec_env import SubprocVecEnv # Need this for the vec_env_cls argument
    main()