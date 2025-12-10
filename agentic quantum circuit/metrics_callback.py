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
                    fidelity = float(fidelity)
                    self.logger.record_mean("quantum/final_fidelity", fidelity)

                    # --- NEW LOGIC HERE ---
                    # We log 1.0 if it hit the threshold, 0.0 if not.
                    # SB3 averages this, so 0.60 means it hit the target 60% of the time.
                    self.logger.record_mean("success_rate/hit_0.80", float(fidelity >= 0.80))
                    self.logger.record_mean("success_rate/hit_0.85", float(fidelity >= 0.85))
                    self.logger.record_mean("success_rate/hit_0.90", float(fidelity >= 0.90))
                    self.logger.record_mean("success_rate/hit_0.95", float(fidelity >= 0.95))
                    self.logger.record_mean("success_rate/hit_0.98", float(fidelity >= 0.98))

                    # 2. Log the "Cluttered" detailed breakdown silently
                    # This loops from 0.80 to 0.99
                    for i in range(80, 100):
                        threshold = i / 100.0
                        is_hit = float(fidelity >= threshold)
                        
                        # exclude="stdout" saves it to the TensorBoard file
                        # but keeps your terminal window clean!
                        self.logger.record_mean(
                            f"fidelity_bins/hit_{threshold:.2f}", 
                            is_hit, 
                            exclude="stdout" 
                        )
                
                                   
                min_inner_product = info.get("min_inner_product")
                if min_inner_product is not None:
                    self.logger.record_mean("quantum/min_inner_product", float(min_inner_product))

                ng_score = info.get("ng_score", 0)
                if min_inner_product is not None:
                    self.logger.record_mean("quantum/non_gaussianity", float(ng_score))

                target_fidelity = info.get("target_fidelity", 0)
                if min_inner_product is not None:
                    self.logger.record_mean("quantum/target_fidelity", float(target_fidelity))

                is_success = info.get("is_success", 0)
                if min_inner_product is not None:
                    self.logger.record_mean("rollout/success_rate_rollout", float(is_success))
        
        return True  # Continue training