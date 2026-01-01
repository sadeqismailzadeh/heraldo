import os
import sys
import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList

# Ensure we can import the environment file if running this script directly
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from hypersphere_env import HypersphereNavigationEnv

# --- HYPERPARAMETERS FOR TUNING ---
LEARNING_RATE = 3e-4        # Step size for optimization
ENT_COEF = 0.01             # Entropy coefficient (encourage exploration)
CLIP_RANGE = 0.2            # PPO Clipping parameter
GAMMA = 0.99                # Discount factor
GAE_LAMBDA = 0.95           # Generalized Advantage Estimation smoothing
BATCH_SIZE = 64             # Minibatch size
N_STEPS = 2048              # Steps per rollout

# --- CONFIGURATION ---
TOTAL_TIMESTEPS = 200_000
N_DIMS = 50                 # State dimension (Hypersphere)
LOG_DIR = "logs/hypersphere_tuning/"
MODEL_NAME = "ppo_hypersphere_surrogate"

def main():
    # 1. Ensure log directory exists
    os.makedirs(LOG_DIR, exist_ok=True)

    # 2. Environment Setup
    # We set terminate_on_success=False to force the agent to LEARN MAINTENANCE.
    # In Quantum Control, reaching the state is half the battle; keeping it there 
    # against drift/decoherence is the other half.
    env = HypersphereNavigationEnv(
        n_dims=N_DIMS,
        terminate_on_success=False, 
        target_fidelity=0.99
    )
    
    # Wrap in Monitor for logging episode rewards/lengths to disk
    env = Monitor(env, filename=os.path.join(LOG_DIR, "monitor.csv"))
    
    # Wrap in DummyVecEnv (SB3 standard wrapper)
    env = DummyVecEnv([lambda: env])

    # 3. Callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=20_000,
        save_path=LOG_DIR,
        name_prefix=MODEL_NAME
    )

    # --- PLACEHOLDER: Custom Callbacks ---
    # If you want to use your existing callbacks, import them and add them to the list here.
    # e.g.:
    # from quantum_agent.callbacks.curriculum_callback import CurriculumCallback
    # curriculum_cb = CurriculumCallback(...)
    # 
    # from quantum_agent.callbacks.adaptive_kl_callback import AdaptiveKLCallback
    # kl_cb = AdaptiveKLCallback(...)
    
    callbacks = CallbackList([
        checkpoint_callback,
        # curriculum_cb,
        # kl_cb
    ])

    # 4. Initialize Agent
    print(f"Initializing PPO Agent with LR={LEARNING_RATE}, Ent={ENT_COEF}...")
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=LEARNING_RATE,
        ent_coef=ENT_COEF,
        clip_range=CLIP_RANGE,
        gamma=GAMMA,
        gae_lambda=GAE_LAMBDA,
        batch_size=BATCH_SIZE,
        n_steps=N_STEPS,
        verbose=1,
        tensorboard_log=LOG_DIR,
        device="auto" # Use GPU if available, else CPU
    )

    # 5. Train
    print(f"Starting training for {TOTAL_TIMESTEPS} timesteps...")
    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=callbacks,
        progress_bar=True
    )

    # 6. Save Final Model
    save_path = os.path.join(LOG_DIR, f"{MODEL_NAME}_final")
    model.save(save_path)
    print(f"Training complete. Model saved to {save_path}.zip")

    # Optional: Basic Evaluation
    obs = env.reset()
    print("\n--- Running Quick Evaluation (Maintenance Mode) ---")
    total_reward = 0
    for i in range(50):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        total_reward += reward[0]
        # Access info from the first env in the VecEnv
        fid = info[0]['fidelity']
        print(f"Step {i+1}: Fidelity={fid:.4f} | Reward={reward[0]:.4f}")
    
    print(f"Total Reward (50 steps): {total_reward:.4f}")

if __name__ == "__main__":
    main()