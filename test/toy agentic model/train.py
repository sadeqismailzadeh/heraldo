from stable_baselines3 import PPO
from simple_navigation_env import SimpleNavigationEnv

# 1. Create the environment
env = SimpleNavigationEnv()

# 2. Instantiate the PPO agent
# "MlpPolicy" is a standard feedforward neural network, suitable for our vector-based state.
# Hyperparameters are chosen to be simple, but you can tune them.
# The `gamma` and `n_steps` are taken from the paper's structure.
model = PPO(
    "MlpPolicy",
    env,
    verbose=1,  # Print training progress
    gamma=0.99, # Discount factor, similar to paper
    n_steps=2048, # Number of steps to run for each environment per update
    batch_size=64,
    n_epochs=10,
    learning_rate=3e-4,
    tensorboard_log="./ppo_navigation_tensorboard/"
)

# 3. Train the agent
print("Starting training...")
model.learn(total_timesteps=100000)
print("Training finished!")

# 4. Save the trained model
model.save("ppo_simple_navigation")
print("Model saved to ppo_simple_navigation.zip")

env.close()