import os
import sys
import numpy as np
from stable_baselines3 import PPO

# Ensure we can import the environment file
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from hypersphere_env import HypersphereNavigationEnv, plot_hypersphere_trajectory

# --- CONFIGURATION ---
MODEL_PATH = "logs/hypersphere_tuning/ppo_hypersphere_surrogate_final.zip"
N_DIMS = 50
EVAL_SEED = 12345  # Fixed seed for reproducibility

def main():
    # 1. Check if model exists
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model not found at {MODEL_PATH}")
        print("Please run train_hypersphere.py first.")
        return

    # 2. Setup Environment for Evaluation
    # We set terminate_on_success=True to measure how quickly it reaches the target.
    env = HypersphereNavigationEnv(
        n_dims=N_DIMS,
        terminate_on_success=True, 
        target_fidelity=0.99
    )

    # 3. Load Agent
    print(f"Loading model from {MODEL_PATH}...")
    model = PPO.load(MODEL_PATH)

    # 4. Run Evaluation Episode
    print(f"Running evaluation episode with seed {EVAL_SEED}...")
    
    # Reset with specific seed
    obs, _ = env.reset(seed=EVAL_SEED)
    
    done = False
    truncated = False
    step = 0
    total_reward = 0
    fidelity_history = []

    # Get initial fidelity (optional, but good for the plot starting point)
    # The env doesn't return info on reset, so we calculate it manually or just wait for step 1
    # For plotting consistency, let's just log steps 1..T
    
    while not (done or truncated):
        # Deterministic=True ensures we see the best behavior of the policy, no noise
        action, _ = model.predict(obs, deterministic=True)
        
        obs, reward, done, truncated, info = env.step(action)
        
        fidelity = info['fidelity']
        fidelity_history.append(fidelity)
        total_reward += reward
        step += 1
        
        print(f"Step {step:02d}: Fidelity={fidelity:.4f} | Reward={reward:.4f}")

    # 5. Results
    status = "SUCCESS" if info['is_success'] else "TIMEOUT"
    print(f"\nEpisode Finished. Status: {status}")
    print(f"Total Steps: {step}")
    print(f"Total Reward: {total_reward:.4f}")
    print(f"Final Fidelity: {fidelity_history[-1]:.4f}")

    # 6. Visualization
    print("\nPlotting trajectory...")
    plot_hypersphere_trajectory(
        fidelity_history, 
        title=f"Agent Evaluation (Seed {EVAL_SEED}): {status}"
    )

if __name__ == "__main__":
    main()