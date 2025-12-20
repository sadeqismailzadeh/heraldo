"""Reward mechanisms implementing the RewardMechanism interface."""

import numpy as np
from quantum_modules import RewardMechanism
from modular_quantum_env import fidelity_max_rotation, decode_measurement_result


class LogFidelityReward(RewardMechanism):
    """
    Logarithmic fidelity reward used for circuit environment (cat state).
    
    Combines fidelity reward with penalties for:
    - Time cost per step
    - Self-similarity (penalizes stagnation)
    
    Uses bonus for success.
    """
    
    def __init__(self):
        pass
    
    def compute(self, current_ket: np.ndarray, target_kets: list, step_info: dict, target_fidelity: float) -> tuple:
        """
        Compute reward with penalties and bonuses.
        
        Returns:
            (reward, terminated, info)
        """
        self.target_fidelity = target_fidelity

        fidelity = max([fidelity_max_rotation(target_ket, current_ket) for target_ket in target_kets])
        result = step_info.get('result', None)
        past_ket = step_info.get('past_ket', None)
        
        reward = 0
        terminated = False
        hit_target = (fidelity > self.target_fidelity)
        
        max_reward = self._calculate_reward(1)
        reward += self._calculate_reward(fidelity)
        reward -= max_reward
        
        # Time penalty
        # reward -= 1 * max_reward
        
        # Stagnation penalty: penalize if state hasn't changed much
        self_fidelity = fidelity_max_rotation(past_ket, current_ket)
        if self_fidelity > 0.95:
            reward -=  0.5*max_reward

        # Success bonus
        if hit_target:
            reward += 10 * max_reward
            terminated = True

        # Normalize reward
        reward /= (11*max_reward)
        
        # Extract measurement info
        info = {
            'is_success': hit_target,
            'fidelity': fidelity,
            'target_fidelity': self.target_fidelity
        }
        
        if result is not None:
            encoded_result = result.samples[0][0]
            lost_photons, detected_photons = decode_measurement_result(encoded_result)
            info.update({
                'photon_loss': lost_photons,
                'detected_photons': detected_photons,
                'total_photons': lost_photons + detected_photons
            })
        
        if past_ket is not None:
            info['self_fidelity'] = fidelity_max_rotation(past_ket, current_ket)

        return reward, terminated, info
    
    def _calculate_reward(self, fidelity):
        """Calculates logarithmic reward based on infidelity."""
        infidelity = max(1.0 - fidelity, 1e-3)
        log_val = -np.log10(infidelity)
        return ((fidelity**2) * log_val)**2
