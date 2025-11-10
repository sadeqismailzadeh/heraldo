import numpy as np
from stable_baselines3.common.callbacks import BaseCallback, CallbackList

class PhotonLossFidelityCallback(BaseCallback):
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
        
        # Handle both vec and single env cases
        if not isinstance(infos, (list, tuple)):
            infos = [infos]
        if not isinstance(dones, (list, tuple)):
            dones = [dones]
        
        for info, done in zip(infos, dones):
            # Log photon loss for every step (automatically averaged across steps)
            photon_loss = info.get("photon_loss")
            if photon_loss is not None:
                self.logger.record_mean("rollout/photon_loss_per_step", float(photon_loss))
            
            # Log fidelity only at episode end (automatically averaged across episodes)
            if done:
                fidelity = info.get("fidelity")
                if fidelity is not None:
                    self.logger.record_mean("rollout/final_fidelity", float(fidelity))
        
        return True  # Continue training