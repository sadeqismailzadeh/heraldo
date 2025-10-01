# Import the PPO algorithm and our custom environment
from stable_baselines3 import PPO
from simple_navigation_env import SimpleNavigationEnv

# --- 1. Create the Environment ---
# We create the same environment again, this time to test the agent.
env = SimpleNavigationEnv()

# --- 2. Load the Trained Model ---
# Instead of creating a new model, we load the one we saved after training.
try:
    model = PPO.load("ppo_simple_navigation", env=env)
except FileNotFoundError:
    print("Error: Trained model 'ppo_simple_navigation.zip' not found.")
    print("Please run train.py first to train and save the model.")
    exit()

# --- 3. Run the Evaluation Loop ---
# We will run the agent for a few episodes to see how it performs.
num_episodes = 5
for episode in range(num_episodes):
    # Reset the environment to get the starting observation
    obs, info = env.reset()
    
    # Flags to track if the episode has ended
    terminated, truncated = False, False
    total_reward = 0
    
    print(f"\n--- Starting Episode {episode + 1} ---")
    
    # Loop until the episode ends (either by reaching the goal or timing out)
    while not (terminated or truncated):
        # --- Get the Agent's Action ---
        # The model.predict() method uses the learned policy to choose an action
        # based on the current observation 'obs'.
        # deterministic=True: During evaluation, we want the agent to choose its
        # best-known action, not a random one for exploration.
        action, _states = model.predict(obs, deterministic=True)
        
        # --- Perform the Action in the Environment ---
        # The step function returns the results of taking that action.
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Accumulate the reward for this episode
        total_reward += reward
        
        # --- Render the Environment ---
        # Call the render() method to show the agent's movement on the screen.
        env.render()

    print(f"Episode finished. Total Reward: {total_reward:.2f}")

# Clean up the environment and close the visualization window
env.close()