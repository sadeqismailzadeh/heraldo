# Import the PPO algorithm from stable-baselines3 and our custom environment
from stable_baselines3 import PPO
from simple_navigation_env import SimpleNavigationEnv

# --- 1. Create the Environment ---
# Instantiate the environment we just defined. The agent will be trained in this "world".
env = SimpleNavigationEnv()

# --- 2. Instantiate the PPO Agent ---
# We create an instance of the PPO algorithm.
# PPO (Proximal Policy Optimization) is a powerful and popular RL algorithm
# that works well with continuous action and state spaces, just like in our problem.
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
    
    # tensorboard_log: Specifies a directory to save training logs. These can be
    # viewed with a tool called TensorBoard for detailed graphs of the training process.
    tensorboard_log="./ppo_navigation_tensorboard/"
)

# --- 3. Train the Agent ---
# The learn() method starts the training process. The agent will now interact
# with the environment for a specified number of steps, collecting experience
# and updating its policy to maximize the total reward.
print("Starting training...")
model.learn(total_timesteps=100000)
print("Training finished!")

# --- 4. Save the Trained Model ---
# After training, we save the agent's learned policy (the neural network weights)
# to a file. This allows us to load it later for evaluation without retraining.
model.save("ppo_simple_navigation")
print("Model saved to ppo_simple_navigation.zip")

# Clean up the environment
env.close()