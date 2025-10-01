from stable_baselines3 import PPO
from simple_navigation_env import SimpleNavigationEnv

# Create the environment with rendering enabled
env = SimpleNavigationEnv()

# Load the trained model
try:
    model = PPO.load("ppo_simple_navigation", env=env)
except FileNotFoundError:
    print("Error: Trained model not found. Please run train.py first.")
    exit()

# Evaluate the agent for 5 episodes
num_episodes = 5
for episode in range(num_episodes):
    obs, info = env.reset()
    terminated, truncated = False, False
    total_reward = 0
    
    print(f"\n--- Starting Episode {episode + 1} ---")
    
    while not (terminated or truncated):
        # Use the model to predict the best action (deterministic=True)
        action, _states = model.predict(obs, deterministic=True)
        
        # Perform the action in the environment
        obs, reward, terminated, truncated, info = env.step(action)
        
        total_reward += reward
        
        # Render the environment to visualize
        env.render()

    print(f"Episode finished. Total Reward: {total_reward:.2f}")

env.close()