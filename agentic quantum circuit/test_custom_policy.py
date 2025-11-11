"""
Unit tests for the custom_policy.py module.
"""

import torch
import pytest
from stable_baselines3 import PPO
from aac_quantum_env import AACQuantumCircuitEnv
from custom_policy import AsymmetricCriticPolicy

@pytest.fixture
def aac_env():
    """Create a dummy AACQuantumCircuitEnv for testing."""
    return AACQuantumCircuitEnv(cutoff_dim=10, max_steps=5)

def test_policy_instantiation(aac_env):
    """Test if the AsymmetricCriticPolicy can be instantiated correctly."""
    try:
        policy = AsymmetricCriticPolicy(
            observation_space=aac_env.observation_space,
            action_space=aac_env.action_space,
            lr_schedule=lambda _: 3e-4,
        )
        assert policy is not None
        print("\n✅ Policy instantiation successful.")
    except Exception as e:
        pytest.fail(f"Policy instantiation failed: {e}")

def test_ppo_model_with_custom_policy(aac_env):
    """Test if a PPO model can be created with the custom AAC policy."""
    try:
        model = PPO(
            AsymmetricCriticPolicy,
            aac_env,
            verbose=0
        )
        assert model is not None
        print("\n✅ PPO model creation with custom policy successful.")
    except Exception as e:
        pytest.fail(f"PPO model creation failed: {e}")

def test_forward_pass(aac_env):
    """Test the forward pass of the policy."""
    model = PPO(AsymmetricCriticPolicy, aac_env)
    obs, _ = aac_env.reset()
    
    # Convert observation to tensor
    obs_tensor = {k: torch.as_tensor([v]).to(model.device) for k, v in obs.items()}

    try:
        actions, values, log_probs = model.policy.forward(obs_tensor)
        assert actions is not None
        assert values is not None
        assert log_probs is not None
        print("\n✅ Forward pass successful.")
        print(f"  - Actions shape: {actions.shape}")
        print(f"  - Values shape: {values.shape}")
        print(f"  - Log probs shape: {log_probs.shape}")
    except Exception as e:
        pytest.fail(f"Forward pass failed: {e}")

def test_evaluate_actions(aac_env):
    """Test the evaluate_actions method of the policy."""
    model = PPO(AsymmetricCriticPolicy, aac_env)
    obs, _ = aac_env.reset()
    
    obs_tensor = {k: torch.as_tensor([v]).to(model.device) for k, v in obs.items()}
    
    # Get some actions
    actions, _, _ = model.policy.forward(obs_tensor)

    try:
        values, log_probs, entropy = model.policy.evaluate_actions(obs_tensor, actions)
        assert values is not None
        assert log_probs is not None
        assert entropy is not None
        print("\n✅ evaluate_actions successful.")
        print(f"  - Values shape: {values.shape}")
        print(f"  - Log probs shape: {log_probs.shape}")
        print(f"  - Entropy: {entropy}")
    except Exception as e:
        pytest.fail(f"evaluate_actions failed: {e}")

def test_predict_values(aac_env):
    """Test the predict_values method of the policy."""
    model = PPO(AsymmetricCriticPolicy, aac_env)
    obs, _ = aac_env.reset()

    obs_tensor = {k: torch.as_tensor([v]).to(model.device) for k, v in obs.items()}

    try:
        values = model.policy.predict_values(obs_tensor)
        assert values is not None
        print("\n✅ predict_values successful.")
        print(f"  - Values shape: {values.shape}")
    except Exception as e:
        pytest.fail(f"predict_values failed: {e}")

if __name__ == "__main__":
    pytest.main([__file__])
