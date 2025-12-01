from stable_baselines3.common.callbacks import BaseCallback
import numpy as np

class CurriculumCallback(BaseCallback):
    """
    Updates the curriculum ONLY at the end of a PPO rollout.
    This ensures the PPO batch data remains consistent (Stationarity).
    """
    def __init__(self, success_threshold: float = 0.8, max_difficulty: float = 0.9999, verbose: int = 1):
        super(CurriculumCallback, self).__init__(verbose)
        self.success_threshold = success_threshold
        self.max_difficulty = max_difficulty
        self.current_difficulty = 0.90 # Must match Env init
        
        # Temporary storage for the current rollout
        self.rollout_successes = []

    def _on_step(self) -> bool:
        """
        Collect success flags ONLY from finished episodes.
        """
        # Access the local variables from the PPO runner
        dones = self.locals['dones'] # Array of booleans [True, False, ...]
        infos = self.locals['infos'] # Array of dicts
        
        for idx, done in enumerate(dones):
            if done:
                # In SB3 VecEnv, when done=True, the 'info' dict usually 
                # corresponds to the terminal step.
                info = infos[idx]
                
                # Safety check: ensure the key exists (it should based on Env code)
                if 'is_success' in info:
                    self.rollout_successes.append(info['is_success'])
                    
        return True

    def _on_rollout_end(self) -> None:
        """
        Called by PPO after collecting 'n_steps' but BEFORE the gradient update.
        This is the perfect time to evaluate and upgrade.
        """
        if len(self.rollout_successes) == 0:
            return

        # 1. Calculate Success Rate for this ENTIRE batch
        success_rate = np.mean(self.rollout_successes)
        
        if self.verbose > 0:
            print(f"   [Curriculum] Batch Success Rate: {success_rate:.2%} "
                  f"(based on {len(self.rollout_successes)} episodes)")

        # 2. Check for Upgrade
        if success_rate >= self.success_threshold and self.current_difficulty < self.max_difficulty:
            self._upgrade_difficulty()
        
        # 3. Clear the buffer so the next rollout starts fresh
        self.rollout_successes = []

    def _upgrade_difficulty(self):
        # Increase difficulty
        if self.current_difficulty < 0.99:
            self.current_difficulty += 0.01
        else:
            self.current_difficulty += 0.001
            
        self.current_difficulty = min(self.current_difficulty, self.max_difficulty)

        # Update all parallel environments
        self.training_env.set_attr("target_fidelity", self.current_difficulty)
        
        print(f"\n🚀 CURRICULUM UPGRADE! New Target: {self.current_difficulty:.4f}\n")