"""Reward mechanisms implementing the RewardMechanism interface."""

import numpy as np
from scipy.special import eval_hermite
from quantum_agent.core.interfaces import RewardMechanism
from quantum_agent.envs.modular_env import fidelity_max_rotation, decode_measurement_result


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
            reward -=  0.1*max_reward

        # Success bonus
        if hit_target:
            reward = 10*self._calculate_reward(fidelity)
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


class PowerLawReward(RewardMechanism):
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
        # hit_target = (fidelity > self.target_fidelity)
        
        # max_reward = self._calculate_reward(1)
        reward += self._calculate_reward(fidelity)
        # reward -= max_reward
        
        # Time penalty
        # reward -= 1 * max_reward
        
        # Stagnation penalty: penalize if state hasn't changed much
        # self_fidelity = fidelity_max_rotation(past_ket, current_ket)
        # if self_fidelity > 0.95:
        #     reward -=  0.5*max_reward

        # Success bonus
        # if hit_target:
        #     reward = max_reward + self._calculate_reward(fidelity)
        #     terminated = True

        # Normalize reward
        # reward /= (2*max_reward)
        
        # Extract measurement info
        info = {
            # 'is_success': hit_target,
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
        return fidelity**50


class PotentialBasedReward(RewardMechanism):
    """
    Potential-based reward mechanism.
    
    Reward = gamma * Phi(s') - Phi(s) - step_penalty
    
    where Phi(s) = -log10(1 - fidelity(s))
    This rewards the agent for reducing the 'nines' of infidelity.
    """
    
    def __init__(self, step_penalty=0.01, gamma=0.99):
        self.step_penalty = step_penalty
        self.gamma = gamma
    
    def _calculate_potential(self, fidelity):
        # Clip fidelity to avoid log(0) and potential infinities
        # 1e-9 allows up to 99.9999999% fidelity
        safe_fidelity = min(fidelity, 1.0 - 1e-3)
        infidelity = 1.0 - safe_fidelity
        return -np.log10(infidelity)
    
    def compute(self, current_ket: np.ndarray, target_kets: list, step_info: dict, target_fidelity: float) -> tuple[float, bool, dict]:
        
        # 1. Calculate current fidelity (s')
        # We calculate max fidelity against all potential targets (e.g. if symmetric targets exist)
        current_fidelity = max([fidelity_max_rotation(t, current_ket) for t in target_kets])
        phi_prime = self._calculate_potential(current_fidelity)
        
        # 2. Calculate past fidelity (s)
        past_ket = step_info.get('past_ket')
        if past_ket is None:
            # If no past state (e.g. very first step or logic gap), assume no change
            past_fidelity = current_fidelity
        else:
            past_fidelity = max([fidelity_max_rotation(t, past_ket) for t in target_kets])
            
        phi = self._calculate_potential(past_fidelity)
        
        # 3. Compute Shaped Reward
        # R = gamma * Phi(s') - Phi(s)
        reward = (self.gamma * phi_prime) - phi
        
        # Subtract step cost
        reward -= self.step_penalty
        
        # 4. Check Termination
        terminated = False
        if current_fidelity >= target_fidelity:
            terminated = True
            # Success bonus to encourage termination
            reward += 1.0

        # 5. Info
        info = {
            'fidelity': current_fidelity,
            'target_fidelity': target_fidelity,
            'is_success': terminated,
            'phi': phi,
            'phi_prime': phi_prime
        }

        # Add measurement info if available
        result = step_info.get('result', None)
        if result is not None and hasattr(result, 'samples') and len(result.samples) > 0:
            try:
                encoded_result = result.samples[0][0]
                lost_photons, detected_photons = decode_measurement_result(encoded_result)
                info.update({
                    'photon_loss': lost_photons,
                    'detected_photons': detected_photons,
                    'total_photons': lost_photons + detected_photons
                })
            except (IndexError, TypeError, ValueError):
                pass
                
        if past_ket is not None:
             info['self_fidelity'] = fidelity_max_rotation(past_ket, current_ket)
        
        return reward, terminated, info


