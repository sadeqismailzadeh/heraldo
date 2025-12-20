import numpy as np
from collections import deque
from stable_baselines3.common.callbacks import BaseCallback

class AdaptiveKLCallback(BaseCallback):
    """
    Adjusts the learning rate based on the KL divergence (approx_kl) 
    from the previous updates.
    """
    def __init__(self, 
                 kl_threshold: float = 0.03, 
                 window_size: int = 10, 
                 decay_factor: float = 0.5, 
                 min_lr: float = 1e-6,
                 verbose: int = 1):
        super(AdaptiveKLCallback, self).__init__(verbose)
        self.kl_threshold = kl_threshold
        self.window_size = window_size
        self.decay_factor = decay_factor
        self.min_lr = min_lr
        
        # storage for the last N KL values
        self.kl_history = deque(maxlen=window_size)
        
        # To handle cooldown (don't drop LR twice in a row immediately)
        self.cooldown_counter = 0

    def _on_rollout_end(self) -> None:
        """
        This is called before the training phase starts.
        We check the metrics from the *previous* training phase here.
        """
        # Retrieve the last logged approx_kl
        # Note: 'train/approx_kl' is logged by PPO after training.
        # It might not exist on the very first rollout.
        logs = self.logger.name_to_value
        if 'train/approx_kl' in logs:
            last_kl = logs['train/approx_kl']
            self.kl_history.append(last_kl)

        # Check if we have enough data and if cooldown is over
        if len(self.kl_history) >= self.window_size and self.cooldown_counter == 0:
            avg_kl = np.mean(self.kl_history)
            
            if avg_kl > self.kl_threshold:
                self._update_learning_rate(avg_kl)
                # Reset history and set cooldown to allow stabilization
                self.kl_history.clear()
                self.cooldown_counter = 5 # wait 5 rollouts before checking again
        
        if self.cooldown_counter > 0:
            self.cooldown_counter -= 1

    def _update_learning_rate(self, current_avg_kl):
            # 1. Get the current LR from the optimizer
            #    (We take the first param group, usually there's only one)
            optimizer = self.model.policy.optimizer
            current_lr = optimizer.param_groups[0]['lr']
            
            # 2. Calculate the new LR
            new_lr = max(current_lr * self.decay_factor, self.min_lr)
            
            if new_lr < current_lr:
                # 3. Update the internal tracker for the callback (optional, but good for debugging)
                self.current_lr = new_lr

                # 4. OVERRIDE SB3'S SCHEDULE
                #    We use a lambda that ignores the progress input (_) 
                #    and always returns our new fixed 'new_lr'.
                #    We must capture the value immediately (using default arg trick) 
                #    or use a class attribute to avoid closure scope issues.
                self.model.lr_schedule = lambda _: new_lr

                # This ensures that if you save model.save("ppo_low_lr"), 
                # and load it back, it knows the LR is 1e-5 (float), not a lambda.
                self.model.learning_rate = new_lr
                
                # 5. Force an immediate update to the optimizer
                #    (Otherwise it waits until the start of the next rollout)
                self.model._update_learning_rate(optimizer)
                
                if self.verbose > 0:
                    print(f"\n[AdaptiveKL] High KL ({current_avg_kl:.4f}). "
                        f"Lowering LR: {current_lr:.2e} -> {new_lr:.2e}")

    def _on_step(self) -> bool:
        return True