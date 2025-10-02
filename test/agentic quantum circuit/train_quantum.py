from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

# Import our custom quantum environment
from quantum_circuit_env import QuantumCircuitEnv

# --- 1. Create the Environment ---
# Using a smaller cutoff_dim for faster training. Increase for higher accuracy.
# The paper mentions 10-step episodes.
env = QuantumCircuitEnv(cutoff_dim=15, max_steps=10)

# Optional but recommended: Check if the custom environment follows the gymnasium API
# check_env(env) 

# --- 2. Define the Agent and Hyperparameters ---
# This is a complex problem, so we need a larger network and more experience.
policy_kwargs = dict(net_arch=dict(pi=[256, 128], vf=[256, 128]))

model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    policy_kwargs=policy_kwargs,
    n_steps=2048,  # Collect a good amount of data before each update
    batch_size=64,
    n_epochs=10,
    gamma=0.99,    # The paper uses 0.999, but 0.99 is also a good start
    learning_rate=1e-4, # Use a smaller learning rate for stability
    tensorboard_log="./ppo_quantum_tensorboard/"
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