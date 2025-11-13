"""
Working training script for Quantum Circuit Environment with rl-games and Ray.

This script uses the custom RayVecEnv approach from rl-games documentation
to handle custom environments properly with Ray workers.
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

def create_custom_ray_environments():
    """Create custom Ray environments that properly handle configuration passing."""
    from rl_games.common.ivecenv import IVecEnv
    import gym
    import numpy as np
    import random
    from time import sleep
    import torch
    from rl_games.common.tr_helpers import dicts_to_dict_with_arrays
    
    class CustomRayWorker:
        """Custom Ray worker that receives the configuration dictionary directly."""
        def __init__(self, config_dict, config_name, config):
            """Initialize the worker with the full configuration dictionary."""
            # Import the quantum environment in the worker process
            from quantum_circuit_env_gym import QuantumCircuitEnvGym
            
            # Now use the config_dict to get the env_creator function
            # The config_dict contains the environment configurations that were registered
            self.env = config_dict[config_name]['env_creator'](**config)

        def _obs_to_fp32(self, obs):
            if isinstance(obs, dict):
                for k, v in obs.items():
                    if isinstance(v, dict):
                        for dk, dv in v.items():
                            if dv.dtype == np.float64:
                                v[dk] = dv.astype(np.float32)
                    else:
                        if dv.dtype == np.float64:
                            obs[k] = v.astype(np.float32)
            else:
                if obs.dtype == np.float64:
                    obs = obs.astype(np.float32)
            return obs

        def step(self, action):
            """Step the environment and reset if done"""
            next_state, reward, is_done, info = self.env.step(action)

            if np.isscalar(is_done):
                episode_done = is_done
            else:
                episode_done = is_done.all()
            if episode_done:
                next_state = self.reset()
            next_state = self._obs_to_fp32(next_state)
            return next_state, reward, is_done, info

        def seed(self, seed):
            if hasattr(self.env, 'seed'):
                torch.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
                np.random.seed(seed)
                random.seed(seed)
                self.env.seed(seed)

        def render(self):
            self.env.render()

        def reset(self):
            obs = self.env.reset()
            obs = self._obs_to_fp32(obs)
            return obs

        def get_action_mask(self):
            return self.env.get_action_mask()

        def get_number_of_agents(self):
            if hasattr(self.env, 'get_number_of_agents'):
                return self.env.get_number_of_agents()
            else:
                return 1

        def set_weights(self, weights):
            self.env.update_weights(weights)

        def can_concat_infos(self):
            if hasattr(self.env, 'concat_infos'):
                return self.env.concat_infos
            else:
                return False

        def get_env_info(self):
            info = {}
            observation_space = self.env.observation_space

            info['action_space'] = self.env.action_space
            info['observation_space'] = observation_space
            info['state_space'] = None
            info['use_global_observations'] = False
            info['agents'] = self.get_number_of_agents()
            info['value_size'] = 1
            if hasattr(self.env, 'use_central_value'):
                info['use_global_observations'] = self.env.use_central_value
            if hasattr(self.env, 'value_size'):
                info['value_size'] = self.env.value_size
            if hasattr(self.env, 'state_space'):
                info['state_space'] = self.env.state_space
            return info


    class CustomRayVecEnv(IVecEnv):
        """Custom Ray vectorized environment that passes configuration dict to workers."""
        try:
            import ray
        except ImportError:
            pass

        def __init__(self, config_dict, config_name, num_actors, **kwargs):
            """Initialize with the configuration dictionary passed explicitly."""
            self.config_dict = config_dict  # Store the configuration dict
            self.config_name = config_name
            self.num_actors = num_actors
            self.use_torch = False
            self.seed = kwargs.pop('seed', None)

            self.remote_worker = self.ray.remote(CustomRayWorker)
            # Pass config_dict to each worker
            self.workers = [self.remote_worker.remote(self.config_dict, self.config_name, kwargs) for i in range(self.num_actors)]

            if self.seed is not None:
                seeds = range(self.seed, self.seed + self.num_actors)
                seed_set = []
                for (seed, worker) in zip(seeds, self.workers):
                    seed_set.append(worker.seed.remote(seed))
                self.ray.get(seed_set)

            res = self.workers[0].get_number_of_agents.remote()
            self.num_agents = self.ray.get(res)

            res = self.workers[0].get_env_info.remote()
            env_info = self.ray.get(res)
            res = self.workers[0].can_concat_infos.remote()
            can_concat_infos = self.ray.get(res)
            self.use_global_obs = env_info['use_global_observations']
            self.concat_infos = can_concat_infos
            self.obs_type_dict = type(env_info.get('observation_space')) is gym.spaces.Dict
            self.state_type_dict = type(env_info.get('state_space')) is gym.spaces.Dict
            if self.num_agents == 1:
                self.concat_func = np.stack
            else:
                self.concat_func = np.concatenate

        def step(self, actions):
            """Step all individual environments (using the created workers)."""
            newobs, newstates, newrewards, newdones, newinfos = [], [], [], [], []
            res_obs = []
            if self.num_agents == 1:
                for (action, worker) in zip(actions, self.workers):
                    res_obs.append(worker.step.remote(action))
            else:
                for num, worker in enumerate(self.workers):
                    res_obs.append(worker.step.remote(actions[self.num_agents * num: self.num_agents * num + self.num_agents]))

            all_res = self.ray.get(res_obs)
            for res in all_res:
                cobs, crewards, cdones, cinfos = res
                if self.use_global_obs:
                    newobs.append(cobs["obs"])
                    newstates.append(cobs["state"])
                else:
                    newobs.append(cobs)
                newrewards.append(crewards)
                newdones.append(cdones)
                newinfos.append(cinfos)

            if self.obs_type_dict:
                ret_obs = dicts_to_dict_with_arrays(newobs, self.num_agents == 1)
            else:
                ret_obs = self.concat_func(newobs)

            if self.use_global_obs:
                newobsdict = {}
                newobsdict["obs"] = ret_obs

                if self.state_type_dict:
                    newobsdict["states"] = dicts_to_dict_with_arrays(newstates, True)
                else:
                    newobsdict["states"] = np.stack(newstates)
                ret_obs = newobsdict
            if self.concat_infos:
                newinfos = dicts_to_dict_with_arrays(newinfos, False)
            return ret_obs, self.concat_func(newrewards), self.concat_func(newdones), newinfos

        def get_env_info(self):
            res = self.workers[0].get_env_info.remote()
            return self.ray.get(res)

        def set_weights(self, indices, weights):
            res = []
            for ind in indices:
                res.append(self.workers[ind].set_weights.remote(weights))
            self.ray.get(res)

        def has_action_masks(self):
            return True

        def get_action_masks(self):
            mask = [worker.get_action_mask.remote() for worker in self.workers]
            masks = self.ray.get(mask)
            return np.concatenate(masks, axis=0)

        def reset(self):
            res_obs = [worker.reset.remote() for worker in self.workers]
            newobs, newstates = [],[]
            for res in res_obs:
                cobs = self.ray.get(res)
                if self.use_global_obs:
                    newobs.append(cobs["obs"])
                    newstates.append(cobs["state"])
                else:
                    newobs.append(cobs)

            if self.obs_type_dict:
                ret_obs = dicts_to_dict_with_arrays(newobs, self.num_agents == 1)
            else:
                ret_obs = self.concat_func(newobs)

            if self.use_global_obs:
                newobsdict = {}
                newobsdict["obs"] = ret_obs

                if self.state_type_dict:
                    newobsdict["states"] = dicts_to_dict_with_arrays(newstates, True)
                else:
                    newobsdict["states"] = np.stack(newstates)
                ret_obs = newobsdict
            return ret_obs

    # Register the custom ray environment type
    from rl_games.common import env_configurations
    from rl_games.common import vecenv
    vecenv.register('CUSTOMRAY', lambda config_name, num_actors, **kwargs: 
                    CustomRayVecEnv(env_configurations.configurations, config_name, num_actors, **kwargs))
    
    print("Registered custom Ray vectorized environment (CUSTOMRAY)")


def register_quantum_env():
    """Register the quantum circuit environment in the rl-games configuration."""
    from rl_games.common import env_configurations
    from quantum_circuit_env_gym import QuantumCircuitEnvGym

    # Register the environment with a custom Ray vectorized environment
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
        'vecenv_type': 'CUSTOMRAY'  # Use our custom Ray environment
    })

    print("Registered QuantumCircuitEnvGym-v0 with CUSTOMRAY vectorized environment")


def main():
    """Main training function."""
    print("Starting RL-Games Quantum Circuit Training with Custom Ray Parallelization\n")

    try:
        # Create the custom Ray environments that properly pass configuration
        create_custom_ray_environments()

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
                    'name': 'QuantumCircuit_CustomRayTest',
                    'env_name': 'QuantumCircuitEnvGym-v0',  # Registered name
                    'ppo': True,
                    'mixed_precision': False,
                    'normalize_input': True,
                    'normalize_value': True,
                    'value_bootstrap': True,
                    'num_actors': 4,  # Increased for better parallelization with Ray
                    'reward_shaper': {'scale_value': 1},
                    'normalize_advantage': True,
                    'gamma': 0.99,
                    'tau': 0.95,
                    'learning_rate': 3e-4,
                    'kl_threshold': 0.016,
                    'score_to_win': 10000,

                    # Short training for testing Ray parallelization
                    'max_epochs': 3,  # Slightly more epochs for better testing
                    'save_best_after': 1,
                    'save_frequency': 1,

                    'grad_norm': 1.0,
                    'entropy_coef': 0.01,
                    'truncate_grads': True,
                    'e_clip': 0.2,
                    'horizon_length': 128,  # Slightly larger for better training
                    'minibatch_size': 32,   # Slightly larger for better training
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
        log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "CustomRayTrain")
        os.makedirs(log_dir, exist_ok=True)
        config['params']['config']['train_dir'] = log_dir

        print("Loading configuration...")
        runner = Runner()
        runner.load(config)

        print("Starting Custom Ray-parallelized training run...")
        print(f"Using {config['params']['config']['num_actors']} parallel environments with Custom Ray")
        runner.run({
            'train': True,
            'play': False,
            'load_path': None,
        })

        print("Custom Ray-parallelized training completed successfully!")
        return True

    except Exception as e:
        print(f"Error during Custom Ray-parallelized training: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    if success:
        print("\nCustom Ray-parallelized training completed successfully!")
    else:
        print("\nCustom Ray-parallelized training failed!")
        exit(1)