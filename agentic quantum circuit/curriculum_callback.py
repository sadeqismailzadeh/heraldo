import os
import json
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

class CurriculumCallback(BaseCallback):
    """
    Manages the curriculum difficulty.
    Saves the current difficulty to a JSON file so it persists across restarts.
    """
    def __init__(self, 
                 log_dir: str, 
                 success_threshold: float = 0.3, 
                 max_difficulty: float = 0.999, 
                 initial_difficulty: float = 0.80, # Default if no save file found
                 verbose: int = 1):
        super(CurriculumCallback, self).__init__(verbose)
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, "curriculum_state.json")
        
        self.success_threshold = success_threshold
        self.max_difficulty = max_difficulty
        self.current_difficulty = initial_difficulty
        
        # Temporary storage for the current rollout
        self.rollout_successes = []

    def _on_training_start(self) -> None:
        """
        Runs once when model.learn() is called.
        We use this to load the saved difficulty and sync the environment.
        """
        # 1. Try to load previous state
        if os.path.exists(self.save_path):
            try:
                with open(self.save_path, "r") as f:
                    data = json.load(f)
                    self.current_difficulty = float(data.get("current_difficulty", self.current_difficulty))
                if self.verbose > 0:
                    print(f"Curriculum loaded! Resuming at target_fidelity: {self.current_difficulty:.4f}")
            except Exception as e:
                print(f"Error loading curriculum state: {e}. Using default: {self.current_difficulty}")
        else:
            if self.verbose > 0:
                print(f"No curriculum save found. Starting at: {self.current_difficulty:.4f}")

        # 2. FORCE the environment to update immediately
        # This overrides whatever default the environment was created with in main.py
        self.training_env.set_attr("target_fidelity", self.current_difficulty)

    def _on_step(self) -> bool:
        """Collect success flags from finished episodes."""
        dones = self.locals['dones']
        infos = self.locals['infos']
        
        for idx, done in enumerate(dones):
            if done:
                info = infos[idx]
                if 'is_success' in info:
                    self.rollout_successes.append(info['is_success'])
        return True

    def _on_rollout_end(self) -> None:
        """Evaluate performance and update difficulty if needed."""
        if len(self.rollout_successes) == 0:
            return

        success_rate = np.mean(self.rollout_successes)
        
        if self.verbose > 0:
            print(f"   [Curriculum] Target: {self.current_difficulty:.4f} | "
                  f"Batch Success: {success_rate:.2%} ({len(self.rollout_successes)} eps)")

        # Check for Upgrade
        if success_rate >= self.success_threshold and self.current_difficulty < self.max_difficulty:
            self._upgrade_difficulty()
        
        self.rollout_successes = []

    def _upgrade_difficulty(self):
        # 1. Math for new difficulty
        if self.current_difficulty < 0.99:
            self.current_difficulty += 0.01
        else:
            self.current_difficulty += 0.001
            
        self.current_difficulty = min(self.current_difficulty, self.max_difficulty)

        # 2. Update Environments
        self.training_env.set_attr("target_fidelity", self.current_difficulty)
        
        # 3. SAVE the new state to JSON
        with open(self.save_path, "w") as f:
            json.dump({"current_difficulty": self.current_difficulty}, f)
        
        print(f"CURRICULUM UPGRADE! New Target: {self.current_difficulty:.4f} (Saved to {self.save_path})\n")