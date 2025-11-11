"""
Short training test script for the Asymmetric Actor-Critic (AAC) architecture.
This script performs a minimal training run to verify end-to-end functionality.
"""
import os

# --- CRITICAL: Set thread limits BEFORE importing other libraries ---
print("--- Configuring thread limits for NumPy/OpenBLAS/MKL ---")
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

from sb3_contrib.ppo_recurrent import RecurrentPPO as PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env

from aac_quantum_env import AACQuantumCircuitEnv
from custom_policy import AsymmetricRecurrentCriticPolicy

def main():
    print("\n--- Starting short AAC training test ---")

    # --- Configuration ---
    CUTOFF_DIM = 10  # Smaller for faster testing
    MAX_STEPS = 5    # Shorter episodes
    REWARD_POWER = 2
    TUNABLE_R = False

    N_ENVS = 2  # Fewer parallel environments
    TOTAL_TIMESTEPS = 1000  # Very short training run

    # --- Setup Environment ---
    print(f"Using {N_ENVS} parallel environments for test training.")
    env = make_vec_env(
        AACQuantumCircuitEnv,
        n_envs=N_ENVS,
        env_kwargs=dict(
            cutoff_dim=CUTOFF_DIM,
            max_steps=MAX_STEPS,
            reward_power=REWARD_POWER,
            tunable_r=TUNABLE_R,
            is_loss_channel=False,
            loss_channel=1
        ),
        vec_env_cls=SubprocVecEnv,
        vec_env_kwargs=dict(start_method='spawn')
    )

    # --- Create Model ---
    print("Creating PPO model with AsymmetricRecurrentCriticPolicy...")
    model = PPO(
        AsymmetricRecurrentCriticPolicy,
        env,
        verbose=0, # Suppress verbose output during test
        device='cpu',
        n_steps=TOTAL_TIMESTEPS // N_ENVS, # Ensure n_steps is at least 1
        batch_size=64,
        n_epochs=4,
    )
    print("Model created.")

    # --- Train the Agent ---
    print(f"Training for {TOTAL_TIMESTEPS} timesteps...")
    try:
        model.learn(
            total_timesteps=TOTAL_TIMESTEPS,
            progress_bar=True,
        )
        print("\n✅ Short training test completed successfully!")
    except Exception as e:
        print(f"\n❌ Short training test failed: {e}")
        raise

    # --- Save the Model (optional, for verification) ---
    test_log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Test_AAC_Train")
    os.makedirs(test_log_dir, exist_ok=True)
    model_path = os.path.join(test_log_dir, "ppo_aac_short_test.zip")
    model.save(model_path)
    print(f"Test model saved to: {model_path}")

    env.close()

if __name__ == "__main__":
    main()
