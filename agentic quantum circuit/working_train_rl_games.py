"""
Working training script for Quantum Circuit Environment with rl-games.

This script properly registers the custom gym environment in rl-games configuration
system and handles the missing synchronous vectorized environment by creating one.
"""
import os
import numpy as np

# --- CRITICAL: Set thread limits BEFORE importing other libraries ---
print("--- Configuring thread limits for NumPy/OpenBLAS/MKL ---")
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

def create_sync_vectorized_environment():
    """Create a synchronous vectorized environment for gym environments without Ray."""
    from rl_games.common.ivecenv import IVecEnv
    import gym
    import numpy as np
    import random
    from rl_games.common.tr_helpers import dicts_to_dict_with_arrays

    class SyncVecEnv(IVecEnv):
        """
        Synchronous vectorized environment that runs multiple gym environments sequentially.
        This is an alternative to RayVecEnv when Ray is not available.
        """
        def __init__(self, config_name, num_actors, **kwargs):
            from rl_games.common.env_configurations import configurations
            self.config_name = config_name
            self.num_actors = num_actors
            self.use_torch = False
            
            # Create multiple environments using the registered creator
            env_config = configurations[config_name]
            self.envs = []
            for i in range(num_actors):
                # Create a new instance of the environment
                env = env_config['env_creator'](**kwargs)
                self.envs.append(env)

            # Get environment info from the first environment
            env_info = self._get_env_info(self.envs[0])
            self.use_global_obs = env_info['use_global_observations']
            self.obs_type_dict = type(env_info['observation_space']) is gym.spaces.Dict
            self.state_type_dict = type(env_info.get('state_space')) is gym.spaces.Dict
            self.env_info = env_info

        def _obs_to_fp32(self, obs):
            if isinstance(obs, dict):
                for k, v in obs.items():
                    if isinstance(v, dict):
                        for dk, dv in v.items():
                            if hasattr(dv, 'dtype') and dv.dtype == np.float64:
                                v[dk] = dv.astype(np.float32)
                    else:
                        if hasattr(v, 'dtype') and v.dtype == np.float64:
                            obs[k] = v.astype(np.float32)
            else:
                if hasattr(obs, 'dtype') and obs.dtype == np.float64:
                    obs = obs.astype(np.float32)
            return obs

        def _get_env_info(self, env):
            info = {}
            observation_space = env.observation_space
            info['action_space'] = env.action_space
            info['observation_space'] = observation_space
            info['state_space'] = None
            info['use_global_observations'] = False
            info['agents'] = 1  # Assuming single agent environments
            info['value_size'] = 1
            if hasattr(env, 'use_central_value'):
                info['use_global_observations'] = env.use_central_value
            if hasattr(env, 'value_size'):
                info['value_size'] = env.value_size
            if hasattr(env, 'state_space'):
                info['state_space'] = env.state_space
            return info

        def step(self, actions):
            """Step all environments and return stacked results."""
            new_obs, new_rewards, new_dones, new_infos = [], [], [], []

            for i, env in enumerate(self.envs):
                action = actions[i]
                obs, reward, done, info = env.step(action)
                
                # Handle environment reset if done
                if done:
                    obs = env.reset()
                
                obs = self._obs_to_fp32(obs)
                new_obs.append(obs)
                new_rewards.append(reward)
                new_dones.append(done)
                new_infos.append(info)

            # Stack or concatenate results
            if self.obs_type_dict:
                ret_obs = dicts_to_dict_with_arrays(new_obs, True)
            else:
                ret_obs = np.stack(new_obs)

            return ret_obs, np.stack(new_rewards), np.stack(new_dones), new_infos

        def reset(self):
            """Reset all environments."""
            obs_list = []
            for env in self.envs:
                obs = env.reset()
                obs = self._obs_to_fp32(obs)
                obs_list.append(obs)

            if self.obs_type_dict:
                ret_obs = dicts_to_dict_with_arrays(obs_list, True)
            else:
                ret_obs = np.stack(obs_list)

            return ret_obs

        def get_env_info(self):
            """Get environment information."""
            return self.env_info

    # Register this sync environment type in vecenv_config
    from rl_games.common import vecenv
    vecenv.register('SYNC', lambda config_name, num_actors, **kwargs: 
                    SyncVecEnv(config_name, num_actors, **kwargs))
    
    print("Registered synchronous vectorized environment (SYNC)")

def register_quantum_env():
    """Register the quantum circuit environment in the rl-games configuration."""
    from rl_games.common import env_configurations
    from quantum_circuit_env_gym import QuantumCircuitEnvGym
    
    # Register the environment with a synchronous vectorized environment
    def create_env_fn(cutoff_dim=8, max_steps=3, reward_power=1, tunable_r=False, 
                      is_loss_channel=False, loss_channel=1.0, **kwargs):
        return QuantumCircuitEnvGym(
            cutoff_dim=cutoff_dim,
            max_steps=max_steps,
            reward_power=reward_power,
            tunable_r=tunable_r,
            is_loss_channel=is_loss_channel,
            loss_channel=loss_channel
        )
    
    env_configurations.register('QuantumCircuitEnvGym-v0', {
        'env_creator': create_env_fn,
        'vecenv_type': 'SYNC'  # Use our synchronous environment
    })
    
    print("Registered QuantumCircuitEnvGym-v0 with SYNC vectorized environment")

def main():
    """Main training function."""
    print("Starting RL-Games Quantum Circuit Training with Sync Environment\n")
    
    try:
        # Create the synchronous vectorized environment
        create_sync_vectorized_environment()
        
        # Register the quantum environment
        register_quantum_env()
        
        from rl_games.torch_runner import Runner
        
        # Create the configuration with the registered environment name
        config = {
            'params': {
                'seed': 42,
                'algo': {'name': 'a2c_continuous'},
                'model': {'name': 'continuous_a2c_logstd'},
                'network': {
                    'name': 'actor_critic',
                    'separate': True,
                    'space': {
                        'continuous': {
                            'mu_activation': 'None',
                            'sigma_activation': 'None',
                            'mu_init': {'name': 'default'},
                            'sigma_init': {'name': 'const_initializer', 'val': 0},
                            'fixed_sigma': True
                        }
                    },
                    'mlp': {
                        'units': [64, 64],  # Smaller network for testing
                        'activation': 'elu',
                        'd2rl': False,
                        'initializer': {'name': 'default'},
                        'regularizer': {'name': 'None'}
                    }
                },
                'config': {
                    'name': 'QuantumCircuit_SyncTest',
                    'env_name': 'QuantumCircuitEnvGym-v0',  # Registered name
                    'ppo': True,
                    'mixed_precision': False,
                    'normalize_input': True,
                    'normalize_value': True,
                    'value_bootstrap': True,
                    'num_actors': 2,
                    'reward_shaper': {'scale_value': 1},
                    'normalize_advantage': True,
                    'gamma': 0.99,
                    'tau': 0.95,
                    'learning_rate': 3e-4,
                    'kl_threshold': 0.016,
                    'score_to_win': 10000,
                    
                    # Very short training for debugging
                    'max_epochs': 2,  # Just 2 epochs
                    'save_best_after': 1,
                    'save_frequency': 1,
                    
                    'grad_norm': 1.0,
                    'entropy_coef': 0.01,
                    'truncate_grads': True,
                    'e_clip': 0.2,
                    'horizon_length': 64,  # Very small for quick test
                    'minibatch_size': 16,   # Very small for quick test
                    'mini_epochs': 2,       # Few mini-epochs
                    'critic_coef': 1.0,
                    'clip_value': True,
                    'seq_len': 4,
                    'bounds_loss_coef': 0.0001,
                    'lr_schedule': 'linear',  # Add the missing lr_schedule
                    
                    # Environment specific - minimal setup (these will be passed as kwargs to env_creator)
                    'cutoff_dim': 8,      # Very small cutoff for fast computation
                    'max_steps': 3,       # Very few steps for quick episodes
                    'reward_power': 1,    # Simple reward
                    'tunable_r': False,
                    'is_loss_channel': False,
                    'loss_channel': 1.0
                }
            }
        }
        
        # Set up logging directory
        log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "SyncTrain")
        os.makedirs(log_dir, exist_ok=True)
        config['params']['config']['train_dir'] = log_dir
        
        print("Loading configuration...")
        runner = Runner()
        runner.load(config)
        
        print("Starting training run...")
        runner.run({
            'train': True,
            'play': False,
            'load_path': None,
        })
        
        print("Training completed successfully!")
        return True
        
    except Exception as e:
        print(f"Error during training: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    if success:
        print("\nTraining completed successfully!")
    else:
        print("\nTraining failed!")
        exit(1)