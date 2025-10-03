import os
import glob
import re
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback

# Import our custom quantum environment
from quantum_circuit_env import QuantumCircuitEnv

env = QuantumCircuitEnv(cutoff_dim=25, max_steps=10)

# ==============================================================================
# === 1. SETUP GOOGLE DRIVE AND PATHS ==========================================
# ==============================================================================

# Define the base directory where everything will be saved.
log_dir = "/content/drive/My Drive/Colab_RL_Training/QuantumCircuit/"
os.makedirs(log_dir, exist_ok=True)

# Define a prefix for your saved model files
model_prefix = "ppo_quantum_circuit"


# ==============================================================================
# === 2. AUTO-RESUME LOGIC =====================================================
# ==============================================================================
print("--- Checking for existing checkpoints... ---")

# Find all checkpoint files in the log directory that match the prefix
checkpoint_files = glob.glob(os.path.join(log_dir, f"{model_prefix}_*.zip"))
latest_checkpoint = None

if checkpoint_files:
    # If checkpoints exist, find the one with the highest step number
    # We extract the number from the filename (e.g., "ppo_..._120000_steps.zip")
    try:
        latest_checkpoint = max(
            checkpoint_files,
            key=lambda f: int(re.search(r'_(\d+)_steps.zip', f).group(1))
        )
        print(f"✅ Found latest checkpoint: {os.path.basename(latest_checkpoint)}")
    except (ValueError, AttributeError):
        print("⚠️ Could not determine the latest checkpoint. Starting fresh.")
        # This can happen if filenames are not in the expected format

# Create or load the model
if latest_checkpoint:
    print("\n--- RESUMING TRAINING ---")
    # Load the model from the latest checkpoint
    # The environment (`env`) must be defined before this cell is run
    model = PPO.load(latest_checkpoint, env=env)
    print("Model loaded. Continuing from where it left off.")
else:
    print("\n--- STARTING NEW TRAINING ---")
    # If no checkpoint was found, create a new PPO model
    # The environment (`env`) must be defined before this cell is run
    model = PPO(
    # "MlpPolicy": This tells SB3 to use a standard neural network (Multi-Layer Perceptron)
    # as the agent's "brain". This is the right choice for vector-based states like ours.
    # If we had image-based states, we would use "CnnPolicy".
    "MlpPolicy",

    # The environment the agent will interact with and learn from.
    env,

    # verbose=1 prints out training progress (rewards, episode lengths, etc.) to the console.
    verbose=1,

    # --- Key Hyperparameters ---
    # These values control the learning process. Tuning them can improve performance.

    # gamma: The discount factor. A value close to 1 (like 0.99) makes the agent "patient",
    # caring about long-term rewards. A value close to 0 would make it "short-sighted".
    gamma=0.999,

    # n_steps: The number of steps the agent takes in the environment before it updates
    # its policy network. A larger value provides more data for each update, which
    # can lead to more stable training.
    n_steps=50000,

    # batch_size: During the policy update, the collected data is split into
    # mini-batches of this size.
    batch_size=5000,

    # n_epochs: The number of times the agent will iterate over the collected data
    # during each policy update.
    n_epochs=14,

    # learning_rate: Controls how much the neural network's weights are adjusted
    # during each update. A smaller value leads to slower but often more stable learning.
    learning_rate=0.001,

    # This is a complex problem, so we need a larger network and more experience.
    policy_kwargs = dict(net_arch=dict(pi=[256, 128, 64], vf=[256, 128, 64])),
    # pi = policy network, vf = value network

    # tensorboard_log: Specifies a directory to save training logs. These can be
    # viewed with a tool called TensorBoard for detailed graphs of the training process.
    tensorboard_log=log_dir
    )
    print("New model created.")


# ==============================================================================
# === 3. DEFINE THE CHECKPOINT CALLBACK ========================================
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


# ==============================================================================
# === 4. TRAIN THE AGENT =======================================================
# ==============================================================================
# Set the total number of timesteps for the entire training run
TOTAL_TIMESTEPS = 200_000

print(f"\n--- Starting/Resuming training for {TOTAL_TIMESTEPS} total timesteps ---")

# The `learn` call
# `reset_num_timesteps=False` is CRUCIAL for resuming. It ensures the step
# counter continues from the loaded model's progress.
model.learn(
    total_timesteps=TOTAL_TIMESTEPS,
    callback=checkpoint_callback,
    reset_num_timesteps=False # IMPORTANT FOR RESUMING
)

print("\n--- Training Finished! ---")


# ==============================================================================
# === 5. SAVE THE FINAL MODEL ==================================================
# ==============================================================================
final_model_path = os.path.join(log_dir, f"{model_prefix}_final.zip")
model.save(final_model_path)
print(f"\n✅ Final model saved to: {final_model_path}")