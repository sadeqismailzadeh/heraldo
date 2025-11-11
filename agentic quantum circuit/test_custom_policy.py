"""
Unit tests for the custom_policy.py module, specifically for the recurrent policy.
"""

import torch
import pytest
from sb3_contrib.ppo_recurrent import RecurrentPPO as PPO
from aac_quantum_env import AACQuantumCircuitEnv
from custom_policy import AsymmetricRecurrentCriticPolicy

@pytest.fixture
def aac_env():
    """Create a dummy AACQuantumCircuitEnv for testing."""
    return AACQuantumCircuitEnv(cutoff_dim=10, max_steps=5)

def test_policy_instantiation(aac_env):
    """Test if the AsymmetricRecurrentCriticPolicy can be instantiated correctly."""
    try:
        policy = AsymmetricRecurrentCriticPolicy(
            observation_space=aac_env.observation_space,
            action_space=aac_env.action_space,
            lr_schedule=lambda _: 3e-4,
        )
        assert policy is not None
        print("\n✅ Recurrent Policy instantiation successful.")
    except Exception as e:
        pytest.fail(f"Recurrent Policy instantiation failed: {e}")

def test_ppo_model_with_custom_recurrent_policy(aac_env):
    """Test if a PPO model can be created with the custom recurrent AAC policy."""
    try:
        model = PPO(
            AsymmetricRecurrentCriticPolicy,
            aac_env,
            verbose=0
        )
        assert model is not None
        print("\n✅ PPO model creation with custom recurrent policy successful.")
    except Exception as e:
        pytest.fail(f"PPO model creation failed: {e}")

def test_forward_pass_recurrent(aac_env):
    """Test the forward pass of the recurrent policy, including LSTM states."""
    model = PPO(AsymmetricRecurrentCriticPolicy, aac_env)
    obs, _ = aac_env.reset()
    
    # Convert observation to tensor and add batch dimension
    obs_tensor = {k: torch.as_tensor([v]).to(model.device) for k, v in obs.items()}

    # Initialize LSTM states (hidden and cell states)
    num_envs = 1 # Assuming single environment for testing
    n_lstm_layers = 1 # Default in RecurrentActorCriticPolicy
    lstm_hidden_size = 64 # Default in RecurrentActorCriticPolicy
    
    # LSTM states are (hidden_state, cell_state)
    # Each state has shape (n_lstm_layers, num_envs, lstm_hidden_size)
    lstm_states = (
        torch.zeros(n_lstm_layers, num_envs, lstm_hidden_size).to(model.device),
        torch.zeros(n_lstm_layers, num_envs, lstm_hidden_size).to(model.device),
    )
    episode_starts = torch.tensor([True]).to(model.device) # Start of a new episode

    try:
        actions, values, log_probs, new_lstm_states = model.policy.forward(obs_tensor, lstm_states, episode_starts)
        assert actions is not None
        assert values is not None
        assert log_probs is not None
        assert new_lstm_states is not None
        assert len(new_lstm_states) == 2 # (hidden_state, cell_state)
        assert new_lstm_states[0].shape == lstm_states[0].shape
        assert new_lstm_states[1].shape == lstm_states[1].shape
        print("\n✅ Recurrent Forward pass successful.")
        print(f"  - Actions shape: {actions.shape}")
        print(f"  - Values shape: {values.shape}")
        print(f"  - Log probs shape: {log_probs.shape}")
    except Exception as e:
        pytest.fail(f"Recurrent Forward pass failed: {e}")

def test_evaluate_actions_recurrent(aac_env):
    """Test the evaluate_actions method of the recurrent policy."""
    model = PPO(AsymmetricRecurrentCriticPolicy, aac_env)
    obs, _ = aac_env.reset()
    
    obs_tensor = {k: torch.as_tensor([v]).to(model.device) for k, v in obs.items()}
    
    num_envs = 1
    n_lstm_layers = 1
    lstm_hidden_size = 64
    lstm_states = (
        torch.zeros(n_lstm_layers, num_envs, lstm_hidden_size).to(model.device),
        torch.zeros(n_lstm_layers, num_envs, lstm_hidden_size).to(model.device),
    )
    episode_starts = torch.tensor([True]).to(model.device)

    # Get some actions
    actions, _, _, _ = model.policy.forward(obs_tensor, lstm_states, episode_starts)

    try:
        values, log_probs, entropy = model.policy.evaluate_actions(obs_tensor, actions, lstm_states, episode_starts)
        assert values is not None
        assert log_probs is not None
        assert entropy is not None
        print("\n✅ Recurrent evaluate_actions successful.")
        print(f"  - Values shape: {values.shape}")
        print(f"  - Log probs shape: {log_probs.shape}")
        print(f"  - Entropy: {entropy}")
    except Exception as e:
        pytest.fail(f"Recurrent evaluate_actions failed: {e}")

    # predict_values does not take LSTM states, so it can remain similar
def test_predict_values(aac_env):
    """Test the predict_values method of the policy."""
    model = PPO(AsymmetricRecurrentCriticPolicy, aac_env)
    obs, _ = aac_env.reset()

    obs_tensor = {k: torch.as_tensor([v]).to(model.device) for k, v in obs.items()}

    num_envs = 1
    n_lstm_layers = 1
    lstm_hidden_size = 64
    lstm_states = (
        torch.zeros(n_lstm_layers, num_envs, lstm_hidden_size).to(model.device),
        torch.zeros(n_lstm_layers, num_envs, lstm_hidden_size).to(model.device),
    )
    episode_starts = torch.tensor([True]).to(model.device)

    try:
        values = model.policy.predict_values(obs_tensor, lstm_states, episode_starts)
        assert values is not None
        print("\n✅ predict_values successful.")
        print(f"  - Values shape: {values.shape}")
    except Exception as e:
        pytest.fail(f"predict_values failed: {e}")
if __name__ == "__main__":
    pytest.main([__file__])