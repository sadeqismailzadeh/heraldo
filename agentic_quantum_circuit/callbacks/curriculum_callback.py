import os
import json
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

class CurriculumCallback(BaseCallback):
    """
    Manages the curriculum difficulty.
    Accumulates episodes over up to 2 rollouts if episode count < 100.
    """
    def __init__(self,
                 log_dir: str,
                 success_threshold: float = 0.7,
                 max_difficulty: float = 0.999,
                 initial_difficulty: float = 0.80,
                 verbose: int = 1):
        super(CurriculumCallback, self).__init__(verbose)
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, "curriculum_state.json")

        self.success_threshold = success_threshold
        self.max_difficulty = max_difficulty
        self.current_difficulty = initial_difficulty

        # Temporary storage for the current rollout(s)
        self.rollout_successes = []
        self.rollout_fidelities = []
        
        # New: Track how many rollouts we have accumulated
        self.rollouts_accumulated = 0

    def _update_env_difficulty(self):
        """Helper to push changes to workers and verify they happened."""
        try:
            # env_method calls the function on every parallel environment
            updated_values = self.training_env.env_method("set_difficulty", self.current_difficulty)

            fmt_values = [f"{v:.3f}" for v in updated_values]
            if self.verbose > 0:
                # updated_values is a list containing the return value from each worker
                print(f"   [Curriculum] Synced Env Difficulties: {fmt_values}")
        except Exception as e:
            print(f"   [Curriculum] Error updating environments: {e}")

    def _calculate_next_difficulty(self):
        """Calculate the next difficulty level."""
        if self.current_difficulty < 0.7:
            increment = 0.04
        if self.current_difficulty < 0.92:
            increment = 0.02
        # elif self.current_difficulty < 0.97:
        #     increment = 0.0025 * 2
        # elif self.current_difficulty < 0.98:
        #     increment = 0.001 * 2
        elif self.current_difficulty < 0.98:
            increment = 0.01
        elif self.current_difficulty < 0.99:
            increment = 0.0025
        elif self.current_difficulty < 0.998:
            increment = 0.001
        elif self.current_difficulty < 0.999:
            increment = 0.00025
        elif self.current_difficulty < 0.9998:
            increment = 0.0001
        elif self.current_difficulty < 0.9999:
            increment = 0.000025
        elif self.current_difficulty < 0.99999:
            increment = 0.0001
        else:
            increment = 0
        return min(self.current_difficulty + increment, self.max_difficulty)

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

        # USE env_method HERE
        self._update_env_difficulty()


    def _on_step(self) -> bool:
        """Collect success flags from finished episodes."""
        dones = self.locals['dones']
        infos = self.locals['infos']
        
        for idx, done in enumerate(dones):
            if done:
                info = infos[idx]
                if 'is_success' in info:
                    self.rollout_successes.append(info['is_success'])
                if 'fidelity' in info:
                    self.rollout_fidelities.append(info['fidelity'])
        return True

    def _on_rollout_end(self) -> None:
        """Evaluate performance and update difficulty if needed."""
        self.rollouts_accumulated += 1
        num_episodes = len(self.rollout_successes)

        # --- LOGIC CHANGE STARTS HERE ---
        # 1. If we have 100+ episodes, evaluate immediately.
        # 2. If we have accumulated 2 rollouts, evaluate immediately (regardless of count).
        # 3. Otherwise, return and keep accumulating data in the next rollout.
        if num_episodes < 400:
            if self.verbose > 0:
                print(f"   [Curriculum] Accumulating... (Eps: {num_episodes}, Rollouts: {self.rollouts_accumulated})")
            return 
        # -------------------------------

        success_rate = np.mean(self.rollout_successes)

        if self.verbose > 0:
            print(f"   [Curriculum] Target: {self.current_difficulty:.4f} | "
                  f"Batch Success: {success_rate:.2%} ({num_episodes} eps over {self.rollouts_accumulated} rollouts)")

        # Log curriculum metrics
        self.logger.record("curriculum/current_target", self.current_difficulty)
        self.logger.record("curriculum/current_target_success", success_rate)

        # Next target metrics
        next_diff = self._calculate_next_difficulty()
        self.logger.record("curriculum/next_target", next_diff)
        next_success_rate = np.mean([f >= next_diff for f in self.rollout_fidelities]) if len(self.rollout_fidelities) > 0 else 0
        self.logger.record("curriculum/next_target_success", next_success_rate)

        # Check for Upgrade
        if success_rate >= self.success_threshold and self.current_difficulty < self.max_difficulty:
            self._upgrade_difficulty()

        self.rollout_successes = []
        self.rollout_fidelities = []
        self.rollouts_accumulated = 0

    def _upgrade_difficulty(self):
        # 1. Calculate new difficulty
        self.current_difficulty = self._calculate_next_difficulty()

        # 2. Update Environments using the Helper Method
        self._update_env_difficulty()

        # 3. SAVE the new state to JSON
        with open(self.save_path, "w") as f:
            json.dump({"current_difficulty": self.current_difficulty}, f)

        print(f"CURRICULUM UPGRADE! New Target: {self.current_difficulty:.4f} (Saved to {self.save_path})\n")