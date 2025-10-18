"""Callback utilities for managing PyTorch thread usage during PPO training."""

import multiprocessing as mp

import torch
from stable_baselines3.common.callbacks import BaseCallback


class ThreadManagerCallback(BaseCallback):
    """Dynamically adjust PyTorch threading across rollout and update phases.

    During rollouts each subprocess should use a single thread to avoid
    oversubscribing cores; during model updates the callback restores higher
    thread counts to speed up gradient computation.
    """

    def __init__(self, rollout_threads: int = 1, update_threads: int = -1, verbose: int = 0):
        """Store rollout/update thread targets and resolve CPU availability."""
        super().__init__(verbose)
        self.rollout_threads = rollout_threads

        # If update_threads is -1 (default), use all available CPU cores
        self.update_threads = mp.cpu_count() if update_threads == -1 else update_threads

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
        """Record fidelity metrics once episodes complete across subprocesses."""
        for done, info in zip(self.locals['dones'], self.locals['infos']):
            if done and 'max_fidelity' in info:
                self.logger.record('custom/final_fidelity', info['max_fidelity'])
        return True