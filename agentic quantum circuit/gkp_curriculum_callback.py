import os
import json
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

class GKPCurriculumCallback(BaseCallback):
    def __init__(self,
                 log_dir: str,
                 # Fidelity Config
                 initial_fidelity: float = 0.75,
                 max_fidelity: float = 0.96,
                 reset_fidelity: float = 0.75, # Where to drop to after sharpening
                 
                 # Envelope (Delta) Config
                 initial_delta: float = 0.50,  # Starting "easy" wide peaks
                 min_delta: float = 0.20,      # Target "hard" sharp peaks
                 delta_step: float = 0.01,     # How much to reduce delta by
                 
                 success_threshold: float = 0.6, # When to trigger upgrade
                 verbose: int = 1):
        
        super().__init__(verbose)
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, "gkp_curriculum_state.json")

        # Config
        self.max_fidelity_cap = max_fidelity
        self.reset_fidelity = reset_fidelity
        self.min_delta = min_delta
        self.delta_step = delta_step
        self.success_threshold = success_threshold

        # State
        self.current_fidelity = initial_fidelity
        self.current_delta = initial_delta

        # Metrics
        self.rollout_successes = []

    def _update_env_params(self):
        """Push current state to all parallel workers."""
        try:
            # We pass both params to the new env method
            ret = self.training_env.env_method(
                "set_curriculum_params", 
                self.current_fidelity, 
                self.current_delta
            )
            if self.verbose > 0:
                # Just print the first worker's status to avoid spam
                print(f"   [Curriculum] Synced | Fid: {ret[0][0]:.3f} | Delta: {ret[0][1]:.3f}")
        except Exception as e:
            print(f"   [Curriculum] Error syncing envs: {e}")

    def _on_training_start(self) -> None:
        """Load state if exists."""
        if os.path.exists(self.save_path):
            try:
                with open(self.save_path, "r") as f:
                    data = json.load(f)
                    self.current_fidelity = data.get("fidelity", self.current_fidelity)
                    self.current_delta = data.get("delta", self.current_delta)
                print(f"Loaded Curriculum: Fid={self.current_fidelity}, Delta={self.current_delta}")
            except Exception as e:
                print(f"Error loading curriculum: {e}")
        
        self._update_env_params()

    def _on_step(self) -> bool:
        """Collect success info."""
        dones = self.locals['dones']
        infos = self.locals['infos']
        for idx, done in enumerate(dones):
            if done:
                # Assuming 'is_success' is based on hitting self.target_fidelity
                # which is dynamically updated by this callback
                if 'is_success' in infos[idx]:
                    self.rollout_successes.append(infos[idx]['is_success'])
        return True

    def _on_rollout_end(self) -> None:
        """Decide whether to climb fidelity or sharpen delta."""
        if not self.rollout_successes:
            return

        success_rate = np.mean(self.rollout_successes)
        
        self.logger.record("curriculum/target_fidelity", self.current_fidelity)
        self.logger.record("curriculum/target_delta", self.current_delta)
        self.logger.record("curriculum/success_rate", success_rate)

        if self.verbose > 0:
            print(f"   [Curriculum] Fid: {self.current_fidelity:.3f} | Delta: {self.current_delta:.3f} | Success: {success_rate:.2%}")

        # --- LOGIC CORE ---
        if success_rate >= self.success_threshold:
            self._trigger_upgrade()
        
        self.rollout_successes = []

    def _calculate_next_difficulty(self):
        """Calculate the next difficulty level."""
        if self.current_fidelity < 0.92:
            increment = 0.005 * 2
        elif self.current_fidelity < 0.97:
            increment = 0.0025 * 2
        elif self.current_fidelity < 0.98:
            increment = 0.001 * 2
        elif self.current_fidelity < 0.99:
            increment = 0.0005 * 2
        else:
            increment = 0.0001
        return self.current_fidelity + increment


    def _trigger_upgrade(self):
        """
        Logic: 
        1. If Fidelity < Max, increase Fidelity.
        2. If Fidelity >= Max, sharpen Delta (reduce) AND drop Fidelity.
        """
        
        # Are we at the peak fidelity for the current difficulty level?
        if self.current_fidelity < self.max_fidelity_cap:
            # Standard Climb
            self.current_fidelity = self._calculate_next_difficulty()
            print(f"   >>> UPGRADE: Increasing Fidelity Target to {self.current_fidelity:.3f}")
            
        else:
            # We hit 0.96 success rate. Time to make physics harder.
            if self.current_delta > self.min_delta:
                # Sharpen the state (Reduce Delta)
                old_delta = self.current_delta
                self.current_delta = max(self.current_delta - self.delta_step, self.min_delta)
                
                # SAWTOOTH: Drop fidelity back down
                self.current_fidelity = self.reset_fidelity
                
                print(f"   >>> PHASE CHANGE: Sharpening Delta ({old_delta:.3f}->{self.current_delta:.3f}) and Resetting Fidelity to {self.current_fidelity:.3f}")
            else:
                print("   >>> CURRICULUM COMPLETE: Max Fidelity and Min Delta reached.")

        # Sync and Save
        self._update_env_params()
        with open(self.save_path, "w") as f:
            json.dump({
                "fidelity": self.current_fidelity,
                "delta": self.current_delta
            }, f)