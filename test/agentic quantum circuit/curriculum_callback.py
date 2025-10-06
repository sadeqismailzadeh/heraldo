from stable_baselines3.common.callbacks import BaseCallback
import numpy as np

class CurriculumCallback(BaseCallback):
    """
    A stateful callback to implement curriculum learning one stage at a time.
    It increases the environment's difficulty (fidelity_threshold) only when the
    agent's performance (mean reward) exceeds the threshold for the *current* stage.
    """
    def __init__(self, curriculum_stages, verbose=1):
        """
        :param curriculum_stages: A dictionary where keys are the mean reward
                                  to achieve and values are the new fidelity_threshold.
                                  Example: {5.0: 0.5, 8.0: 0.7, 10.0: 0.9}
        """
        super(CurriculumCallback, self).__init__(verbose)
        
        # --- NEW LOGIC ---
        # 1. Sort stages by the reward threshold required for a clear progression.
        # This creates a list of tuples: [(2.0, 0.5), (5.0, 0.7), ...]
        self.stages = sorted(curriculum_stages.items())
        
        # 2. Add state to track our current position in the curriculum.
        self.current_stage_idx = 0
        
        # 3. Track when we last advanced to prevent multiple rapid advancements
        self.last_advancement_step = 0
        # Require at least this many steps at new difficulty before advancing again
        self.min_steps_between_advancements = 2  # At least 2 rollouts at new difficulty

    def _on_step(self) -> bool:
        # Check if we should evaluate for a curriculum update.
        # Checking once per rollout is a good frequency.
        if self.n_calls % self.model.n_steps == 0:
            self._update_curriculum()
        return True

    def _update_curriculum(self):
        # --- NEW LOGIC ---
        
        # 1. Check if the curriculum is already complete.
        if self.current_stage_idx >= len(self.stages):
            if self.verbose > 0:
                print(f"[Curriculum] All stages complete! ({self.current_stage_idx}/{len(self.stages)})")
            return
        
        # 1.5. Calculate how many rollouts have passed since last advancement
        current_rollout = self.num_timesteps // self.model.n_steps
        rollouts_since_advancement = current_rollout - self.last_advancement_step
        
        # Don't advance too quickly - require performance at the new difficulty level
        if rollouts_since_advancement < self.min_steps_between_advancements:
            if self.verbose > 1:
                print(f"[Curriculum] Waiting for stability: {rollouts_since_advancement}/{self.min_steps_between_advancements} rollouts at current difficulty")
            return

        # 2. Get the requirements for the VERY NEXT stage only.
        reward_threshold, new_difficulty = self.stages[self.current_stage_idx]

        # 3. Get the agent's current performance.
        # The logger stores the mean reward of the last 100 episodes.
        if 'rollout/ep_rew_mean' in self.model.logger.name_to_value:
            current_reward = self.model.logger.name_to_value['rollout/ep_rew_mean']
            
            # Add debug logging to see what's happening
            if self.verbose > 1:
                current_fidelity = self.training_env.get_attr('fidelity_threshold')[0]
                print(f"[Curriculum] Stage {self.current_stage_idx + 1}/{len(self.stages)}: "
                      f"Current reward: {current_reward:.3f}, Threshold: {reward_threshold:.3f}, "
                      f"Current fidelity goal: {current_fidelity:.3f}")

            # 4. Check if the agent has mastered the current stage.
            if current_reward >= reward_threshold:
                if self.verbose > 0:
                    print("\n" + "="*60)
                    print(f"✅ CURRICULUM ADVANCEMENT: Mean reward {current_reward:.2f} > threshold {reward_threshold:.2f}.")
                    print(f"   Stage {self.current_stage_idx + 1}/{len(self.stages)} complete. Promoting to next difficulty.")
                    print(f"   Fidelity Threshold: {self.training_env.get_attr('fidelity_threshold')[0]:.2f} -> {new_difficulty:.2f}")
                    print("="*60)

                # 5. Apply the new difficulty to all parallel environments.
                self.training_env.set_attr('fidelity_threshold', new_difficulty)
                
                # Log the new difficulty to TensorBoard for tracking.
                self.logger.record("curriculum/fidelity_threshold", new_difficulty)
                self.logger.record("curriculum/stage_index", self.current_stage_idx + 1)

                # 6. CRUCIAL: Increment the stage index so we look for the next goal.
                self.current_stage_idx += 1
                
                # 7. Record when we advanced to prevent rapid successive advancements
                self.last_advancement_step = self.num_timesteps // self.model.n_steps
