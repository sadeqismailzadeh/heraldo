import multiprocessing as mp
import numpy as np  # Import numpy for calculating the mean
import torch
from stable_baselines3.common.callbacks import BaseCallback


class ThreadManagerCallback(BaseCallback):
    """
    Dynamically adjust PyTorch threading and log custom episode metrics.

    - Manages PyTorch thread counts for rollout and update phases.
    - Records the mean photon loss for each completed episode.
    - Records the final fidelity at the end of each completed episode.
    - Records the minimum inner product (state normalization) for each episode.
    """

    def __init__(self, rollout_threads: int = 1, update_threads: int = -1, verbose: int = 0):
        """Store rollout/update thread targets and resolve CPU availability."""
        super().__init__(verbose)
        self.rollout_threads = rollout_threads
        self.update_threads = mp.cpu_count() if update_threads == -1 else update_threads
        
        # This will be initialized in _on_training_start
        self.episode_photon_losses = []

    def _on_training_start(self) -> None:
        """
        This method is called before the first rollout starts.
        Perfect for initializing per-environment trackers.
        """
        # Get the number of parallel environments
        n_envs = self.training_env.num_envs
        # Create a list of empty lists to store losses for each env
        self.episode_photon_losses = [[] for _ in range(n_envs)]

    def _on_rollout_start(self) -> None:
        """Limit PyTorch to ``rollout_threads`` before the environment step loop."""
        if self.verbose > 0:
            print(f"Callback: Setting PyTorch threads to {self.rollout_threads} for rollout phase.")
        torch.set_num_threads(self.rollout_threads)

    def _on_rollout_end(self) -> None:
        """Restore ``update_threads`` before the policy update begins."""
        if self.verbose > 0:
            print(f"Callback: Setting PyTorch threads to {self.update_threads} for model update phase.")
        torch.set_num_threads(self.update_threads)

    def _on_step(self) -> bool:
        """
        Record photon loss at each step and log summary metrics on episode completion.
        
        This method is called after each step in the environment.
        For vectorized environments, it is called once per step for all envs.
        """
        # self.locals['infos'] is a list of info dicts, one for each env
        for i, info in enumerate(self.locals["infos"]):
            # --- 1. Accumulate metrics at every step ---
            # IMPORTANT: Assumes your env's info dict contains 'photon_loss'
            if "photon_loss" in info:
                self.episode_photon_losses[i].append(info["photon_loss"])

            # --- 2. Check for episode completion ---
            # SB3 automatically adds an "episode" key to the info dict when an episode ends
            if "episode" in info.keys():
                # --- 3. Log the final metrics for the completed episode ---
                
                # A. Log final fidelity
                # IMPORTANT: Assumes your env's info dict contains 'fidelity' on the final step
                if "fidelity" in info:
                    final_fidelity = info["fidelity"]
                    self.logger.record("quantum/final_fidelity", final_fidelity)
                
                # A2. Log minimum inner product (normalization check)
                if "min_inner_product" in info:
                    min_inner_product = info["min_inner_product"]
                    self.logger.record("quantum/min_inner_product", min_inner_product)

                # B. Calculate and log mean photon loss for the episode
                if self.episode_photon_losses[i]: # Ensure the list is not empty
                    mean_loss = np.mean(self.episode_photon_losses[i])
                    self.logger.record("quantum/mean_photon_loss", mean_loss)

                # --- 4. Reset the accumulator for this environment ---
                self.episode_photon_losses[i] = []

        return True