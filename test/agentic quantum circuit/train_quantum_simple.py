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

from stable_baselines3 import PPO
# NEW: Import for creating parallel environments
from stable_baselines3.common.env_util import make_vec_env

# Import our custom quantum environment
from quantum_circuit_env import QuantumCircuitEnv

import warnings
from scipy.linalg import LinAlgWarning
warnings.simplefilter('always', LinAlgWarning)  # show every occurrence


# It's good practice to wrap the main execution logic in a function
def main():
    # ==============================================================================
    # === 1. CONFIGURATION =========================================================
    # ==============================================================================
    
    # Number of parallel environments
    N_ENVS = 3  # Or os.cpu_count() - 1
    
    # Reward power: controls difficulty (higher = harder)
    # Lower values (e.g., 5-10) make it easier to get high rewards
    # Higher values (e.g., 50-100) make it much harder
    REWARD_POWER = 50
    
    # Training duration
    TOTAL_TIMESTEPS = 800_000

    # ==============================================================================
    # === 2. SETUP ENVIRONMENT =====================================================
    # ==============================================================================
    
    print(f"\n--- Setting up {N_ENVS} parallel environments ---")
    
    # Create vectorized parallel environments
    env = make_vec_env(
        QuantumCircuitEnv,
        n_envs=N_ENVS,
        env_kwargs=dict(
            cutoff_dim=25,
            max_steps=10,
            reward_power=REWARD_POWER
        ),
        vec_env_cls=SubprocVecEnv,
        # On Windows and macOS, 'spawn' is the only safe start method.
        vec_env_kwargs=dict(start_method='spawn')
    )

    # Verify we're using SubprocVecEnv for multiprocessing
    print(f"Vectorized environment type: {type(env.unwrapped)}")
    assert isinstance(env.unwrapped, SubprocVecEnv), "FATAL: Not using SubprocVecEnv for multiprocessing!"

    # ==============================================================================
    # === 3. SETUP DIRECTORIES =====================================================
    # ==============================================================================
    
    log_dir = "./Train/"
    os.makedirs(log_dir, exist_ok=True)
    print(f"Logs and models will be saved to: {log_dir}")

    # ==============================================================================
    # === 4. CREATE MODEL ==========================================================
    # ==============================================================================
    
    print("\n--- Creating new PPO model ---")
    
    model = PPO(
        # "MlpPolicy": Use a standard neural network (Multi-Layer Perceptron)
        # as the agent's "brain". This is the right choice for vector-based states.
        "MlpPolicy",
        
        # The environment the agent will interact with and learn from
        env,
        
        # verbose=1 prints training progress to the console
        verbose=1,
        
        # ======================================================================
        # === KEY HYPERPARAMETERS ==============================================
        # ======================================================================
        
        # gamma: Discount factor for future rewards (0.999 = very patient agent)
        gamma=0.999,
        
        # n_steps: Steps taken before each policy update
        n_steps=8192,
        
        # batch_size: Mini-batch size during policy update
        batch_size=64,
        
        # n_epochs: Number of epochs for each policy update
        n_epochs=10,
        
        # learning_rate: How much to adjust network weights during updates
        learning_rate=3e-4,
        
        # ======================================================================
        # === NETWORK ARCHITECTURE =============================================
        # ======================================================================
        
        # Define the neural network architecture
        # Both policy (pi) and value function (vf) use the same structure
        policy_kwargs=dict(net_arch=dict(pi=[256, 128, 64], vf=[256, 128, 64])),
        
        # tensorboard_log: Directory for TensorBoard logging
        tensorboard_log=log_dir,
        
        # device: Use "cpu" or "cuda" for GPU training
        device="cpu",
    )
    
    print("Model created successfully!")

    # ==============================================================================
    # === 5. TRAIN THE AGENT =======================================================
    # ==============================================================================
    
    print(f"\n{'='*70}")
    print(f"Starting training for {TOTAL_TIMESTEPS:,} timesteps")
    print(f"Reward Power: {REWARD_POWER} (higher = harder)")
    print(f"Parallel Environments: {N_ENVS}")
    print(f"{'='*70}\n")
    
    # Start training
    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        progress_bar=True,
    )
    
    print("\n--- Training Finished! ---")

    # ==============================================================================
    # === 6. SAVE THE FINAL MODEL ==================================================
    # ==============================================================================
    
    final_model_path = os.path.join(log_dir, "ppo_quantum_circuit_simple.zip")
    model.save(final_model_path)
    print(f"\n✅ Final model saved to: {final_model_path}")
    
    # Close environments
    env.close()
    print("\n✅ Training complete!")


# ==============================================================================
# === SCRIPT ENTRY POINT =======================================================
# ==============================================================================
if __name__ == "__main__":
    # This is crucial for multiprocessing on Windows
    from stable_baselines3.common.vec_env import SubprocVecEnv
    main()
