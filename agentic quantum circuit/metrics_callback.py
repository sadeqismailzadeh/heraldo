import numpy as np
from stable_baselines3.common.callbacks import BaseCallback, CallbackList

class MetricsCallback(BaseCallback):
    """Logs mean photon loss per step and mean final fidelity per rollout."""
    
    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
    
    def _on_step(self) -> bool:
        """
        Called after each environment step.
        - Accumulates photon loss from every step
        - Records fidelity only when an episode terminates
        """
        # Get info dicts and done flags from the training loop
        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", [])
        
        
        for info, done in zip(infos, dones):
            # Log photon loss for every step (automatically averaged across steps)
            photon_loss = info.get("photon_loss")
            if photon_loss is not None:
                self.logger.record_mean("quantum/photon_loss_per_step", float(photon_loss))

            # Log fidelity only at episode end (automatically averaged across episodes)
            done_flag = bool(done)
            if done_flag:
                fidelity = info.get("fidelity")
                if fidelity is not None:
                    self.logger.record_mean("quantum/final_fidelity", float(fidelity))
                                   
                min_inner_product = info.get("min_inner_product")
                if min_inner_product is not None:
                    self.logger.record_mean("quantum/min_inner_product", float(min_inner_product))
        
        return True  # Continue training