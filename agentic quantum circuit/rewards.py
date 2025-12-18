"""Reward mechanisms implementing the RewardMechanism interface."""

import numpy as np
from quantum_modules import RewardMechanism
from base_quantum_env import fidelity_max_rotation, decode_measurement_result


class StandardFidelityReward(RewardMechanism):
    """
    Generic fidelity-based reward mechanism.
    
    Uses logarithmic reward based on infidelity:
    reward = ((fidelity^2) * log10(1/(1-fidelity)))^2
    
    Provides bonus for reaching target fidelity.
    """
    
    def __init__(self, target_fidelity=0.9, bonus_multiplier=10.0):
        self.target_fidelity = target_fidelity
        self.bonus_multiplier = bonus_multiplier
    
    def compute(self, current_ket: np.ndarray, target_kets: list, step_info: dict) -> tuple:
        """
        Compute reward and termination.
        
        Returns:
            (reward, terminated, info)
        """
        fidelity = max([fidelity_max_rotation(target_ket, current_ket) for target_ket in target_kets])
        result = step_info.get('result', None)
        past_ket = step_info.get('past_ket', None)
        
        terminated = False
        hit_target = (fidelity > self.target_fidelity)
        
        # Logarithmic reward
        max_reward = self._calculate_reward(self.target_fidelity)
        reward = self._calculate_reward(fidelity)
        reward -= max_reward

        if hit_target:
            reward += self.bonus_multiplier * max_reward
            terminated = True

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

        return reward, terminated, info
    
    def _calculate_reward(self, fidelity):
        """Calculates a logarithmic reward based on infidelity."""
        infidelity = max(1.0 - fidelity, 1e-3)
        log_val = -np.log10(infidelity)
        return ((fidelity**2) * log_val)**2


class LogFidelityReward(RewardMechanism):
    """
    Logarithmic fidelity reward used for circuit environment (cat state).
    
    Combines fidelity reward with penalties for:
    - Time cost per step
    - Self-similarity (penalizes stagnation)
    
    Uses bonus for success.
    """
    
    def __init__(self, target_fidelity=0.9, reward_power=2, time_penalty=0.5, 
                 stagnation_penalty=1.0, bonus_multiplier=3.0):
        self.target_fidelity = target_fidelity
        self.reward_power = reward_power
        self.time_penalty = time_penalty
        self.stagnation_penalty = stagnation_penalty
        self.bonus_multiplier = bonus_multiplier
    
    def compute(self, current_ket: np.ndarray, target_kets: list, step_info: dict) -> tuple:
        """
        Compute reward with penalties and bonuses.
        
        Returns:
            (reward, terminated, info)
        """
        fidelity = max([fidelity_max_rotation(target_ket, current_ket) for target_ket in target_kets])
        result = step_info.get('result', None)
        past_ket = step_info.get('past_ket', None)
        
        reward = 0
        terminated = False
        hit_target = (fidelity > self.target_fidelity)
        
        max_reward = self._calculate_reward(self.target_fidelity)
        reward += self._calculate_reward(fidelity)
        reward -= max_reward
        
        # Time penalty
        reward -= self.time_penalty * max_reward
        
        # Stagnation penalty: penalize if state hasn't changed much
        if past_ket is not None:
            self_fidelity = fidelity_max_rotation(past_ket, current_ket)
            if self_fidelity > 0.95:
                reward -= self.stagnation_penalty * max_reward

        # Success bonus
        if hit_target:
            reward += self.bonus_multiplier * max_reward
            terminated = True

        # Normalize reward
        reward /= (4 * max_reward)
        
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


class GadgetReward(RewardMechanism):
    """
    Gadget circuit reward with non-Gaussianity score.
    
    Combines fidelity with non-Gaussianity metric to encourage
    preparation of non-Gaussian quantum states.
    
    Parameters:
    - target_fidelity: Fidelity threshold for success
    - target_ng_score: Target non-Gaussianity score
    - ng_threshold_fraction: Success requires ng_score > target_ng_score * this fraction
    """
    
    def __init__(self, target_fidelity=0.9, target_ng_score=1.0, 
                 ng_threshold_fraction=0.33, bonus_multiplier=10.0,
                 stagnation_penalty=1.0):
        self.target_fidelity = target_fidelity
        self.target_ng_score = target_ng_score
        self.ng_threshold_fraction = ng_threshold_fraction
        self.bonus_multiplier = bonus_multiplier
        self.stagnation_penalty = stagnation_penalty
    
    def compute(self, current_ket: np.ndarray, target_kets: list, step_info: dict) -> tuple:
        """
        Compute gadget reward combining fidelity and non-Gaussianity.
        
        Returns:
            (reward, terminated, info)
        """
        fidelity = max([fidelity_max_rotation(target_ket, current_ket) for target_ket in target_kets])
        ng_score = step_info.get('ng_score', 0.0)
        result = step_info.get('result', None)
        past_ket = step_info.get('past_ket', None)
        
        terminated = False
        ng_threshold = self.target_ng_score * self.ng_threshold_fraction
        hit_target = (fidelity > self.target_fidelity) and (ng_score > ng_threshold)
        
        max_reward = self._calculate_reward(self.target_fidelity)
        reward = self._calculate_reward(fidelity)
        reward -= max_reward
        
        # Stagnation penalty
        if past_ket is not None:
            self_fidelity = fidelity_max_rotation(past_ket, current_ket)
            if self_fidelity > 0.95:
                reward -= self.stagnation_penalty * max_reward

        if hit_target:
            reward += self.bonus_multiplier * max_reward
            terminated = True
        
        info = {
            'is_success': hit_target,
            'fidelity': fidelity,
            'ng_score': ng_score,
            'target_fidelity': self.target_fidelity,
            'target_ng_score': self.target_ng_score
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
