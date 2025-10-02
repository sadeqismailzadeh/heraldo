from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
import os
# temporary fix. it may cause crashes or silently produce incorrect results
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE" 


# Import our custom quantum environment
from quantum_circuit_env import QuantumCircuitEnv

# --- 1. Create the Environment ---
# Using a smaller cutoff_dim for faster training. Increase for higher accuracy.
# The paper mentions 10-step episodes.
env = QuantumCircuitEnv(cutoff_dim=20, max_steps=20)

# Optional but recommended: Check if the custom environment follows the gymnasium API
# check_env(env) 

# --- 2. Define the Agent and Hyperparameters ---
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
    gamma=0.99,
    
    # n_steps: The number of steps the agent takes in the environment before it updates
    # its policy network. A larger value provides more data for each update, which
    # can lead to more stable training.
    n_steps=2048,
    
    # batch_size: During the policy update, the collected data is split into
    # mini-batches of this size.
    batch_size=64,
    
    # n_epochs: The number of times the agent will iterate over the collected data
    # during each policy update.
    n_epochs=10,
    
    # learning_rate: Controls how much the neural network's weights are adjusted
    # during each update. A smaller value leads to slower but often more stable learning.
    learning_rate=3e-4,

    # This is a complex problem, so we need a larger network and more experience.
    policy_kwargs = dict(net_arch=dict(pi=[256, 128], vf=[256, 128])),
    # pi = policy network, vf = value network
    
    # tensorboard_log: Specifies a directory to save training logs. These can be
    # viewed with a tool called TensorBoard for detailed graphs of the training process.
    tensorboard_log="./ppo_navigation_tensorboard/"
)

# --- 3. Train the Agent ---
# This will take a significant amount of time.
# Start with a smaller number to test, then increase for full training.
print("Starting training on Quantum Circuit Environment...")
model.learn(total_timesteps=200_000) # Start with 200k, aim for 1M+
print("Training finished!")

# --- 4. Save the Model ---
model.save("ppo_quantum_circuit")
print("Model saved to ppo_quantum_circuit.zip")

env.close()