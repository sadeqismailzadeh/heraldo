import torch
import multiprocessing as mp
from stable_baselines3.common.callbacks import BaseCallback

class ThreadManagerCallback(BaseCallback):
    """
    A custom callback to dynamically manage the number of threads used by PyTorch.
    - Sets threads to 1 for the data collection phase (rollout) to avoid contention
      between parallel environments.
    - Sets threads to the maximum available for the model update phase (learning)
      to speed up the neural network computations.
    """
    def __init__(self, rollout_threads: int = 1, update_threads: int = -1, verbose: int = 0):
        super(ThreadManagerCallback, self).__init__(verbose)
        self.rollout_threads = rollout_threads
        
        # If update_threads is -1 (default), use all available CPU cores
        self.update_threads = mp.cpu_count() if update_threads == -1 else update_threads

    def _on_rollout_start(self) -> None:
        """
        This method is called before a new rollout starts (data collection).
        """
        if self.verbose > 0:
            print(f"Callback: Setting PyTorch threads to {self.rollout_threads} for rollout phase.")
        torch.set_num_threads(self.rollout_threads)

    def _on_rollout_end(self) -> None:
        """
        This method is called after a rollout ends and before the model update.
        """
        if self.verbose > 0:
            print(f"Callback: Setting PyTorch threads to {self.update_threads} for model update phase.")
        torch.set_num_threads(self.update_threads)

    def _on_step(self) -> bool:
        # Check for "done" signals from all parallel environments
        for done, info in zip(self.locals['dones'], self.locals['infos']):
            if done and 'max_fidelity' in info:
                self.logger.record('custom/final_fidelity', info['max_fidelity'])
        return True