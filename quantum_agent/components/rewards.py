"""Reward mechanisms implementing the RewardMechanism interface."""

import numpy as np
import scipy.integrate
from scipy.special import eval_hermite
from quantum_agent.core.interfaces import RewardMechanism
from quantum_agent.envs.modular_env import fidelity_max_rotation, decode_measurement_result
import thewalrus
import qutip as qt


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




class WignerWeightedReward(RewardMechanism):
    """
    Reward based on the weighted overlap of Wigner functions using QuTiP.
    
    It emphasizes the negative regions of the target state's Wigner function.
    Because the Wigner transform is linear, we can precompute a single 
    Hermitian operator 'O' such that:
       Reward(psi) = <psi| O |psi>
    
    This reduces the heavy Wigner integration (slow) to a simple 
    vector-matrix multiplication (fast, O(N^2)) during training.
    
    Args:
        target_ket (np.ndarray): The target state vector (numpy).
        cutoff_dim (int): Fock truncation dimension.
        neg_weight (float): Multiplier for the negative regions of the target (e.g. 2.0).
        pos_weight (float): Multiplier for positive regions (usually 1.0).
        grid_range (float): Max x/p extent for integration (e.g. 6.0).
        grid_points (int): Resolution of the integration grid. 
                           MUST be Odd to capture origin (e.g. 101 or 201).
    """
    
    def __init__(self, target_ket, cutoff_dim, neg_weight=2.0, pos_weight=1.0, grid_range=12.0, grid_points=201):
        self.neg_weight = neg_weight
        self.pos_weight = pos_weight
        self.cutoff_dim = cutoff_dim
        
        # Ensure grid points is odd to capture origin
        if grid_points % 2 == 0:
            grid_points += 1
            
        print(f"Initializing WignerWeightedReward (NegWeight={neg_weight}x, Range={grid_range}, Points={grid_points})...")
        self.reward_operator = self._build_weighted_wigner_operator(
            target_ket, cutoff_dim, grid_range, grid_points
        )
        print("Wigner Operator built successfully.")

    def _build_weighted_wigner_operator(self, target_ket, dim, g_range, g_points):
        """
        Constructs the Hermitian operator O such that <psi|O|psi> 
        approximates the weighted Wigner overlap integral.
        """
        # 1. Setup Phase Space Grid
        xvec = np.linspace(-g_range, g_range, g_points)
        pvec = np.linspace(-g_range, g_range, g_points)
        
        # 2. Compute Target Wigner Function using QuTiP
        tgt_qobj = qt.Qobj(target_ket.flatten())
        
        # QuTiP wigner returns W(x, p)
        W_target = qt.wigner(tgt_qobj, xvec, pvec)
        
        # 3. Calculate Difference Map
        # Instead of integrating the full W_weighted, we define:
        # W_weighted = W_target + W_diff
        # Where W_diff accounts for the extra weights (neg_weight - 1, etc.)
        # This allows us to set Op = |tgt><tgt| + Op_diff
        # Ensuring that for weight=1, we get the EXACT target state projector.
        
        W_diff = np.zeros_like(W_target)
        mask_neg = W_target < 0
        mask_pos = W_target >= 0
        
        # Delta weights (relative to 1.0)
        W_diff[mask_neg] = W_target[mask_neg] * (self.neg_weight - 1.0)
        W_diff[mask_pos] = W_target[mask_pos] * (self.pos_weight - 1.0)
        
        # 4. Construct Base Operator (Perfect fidelity for weight=1)
        # Note: dim might differ from len(target_ket), so we project.
        # But usually they match. We construct |psi><psi|.
        # Ensure we work in the truncated dim space.
        t_vec = target_ket.flatten()[:dim] 
        t_vec /= np.linalg.norm(t_vec)
        Op_base = np.outer(t_vec, np.conj(t_vec))
        
        # 5. Construct Correction Operator via Integration
        Op_diff = np.zeros((dim, dim), dtype=np.complex128)
        
        # Iterate over basis elements (Upper triangle)
        for m in range(dim):
            for n in range(m, dim):
                # Basis operator A = |n><m|
                basis_op = qt.basis(dim, n) * qt.basis(dim, m).dag()
                
                # Compute Wigner of this basis operator
                W_nm = qt.wigner(basis_op, xvec, pvec)
                
                # Integrate Difference Map
                integrand = W_nm * W_diff
                
                # Simpson's Rule Integration
                val = scipy.integrate.simps(scipy.integrate.simps(integrand, pvec), xvec)
                
                Op_diff[m, n] = val
                
                if m != n:
                    Op_diff[n, m] = np.conj(val)
        
        # 6. Combine
        # Scale factor: Tr(A B) = 2*pi * Integral(W_A W_B)
        # So Integral = 1/(2*pi) * Tr.
        # We computed Integral. We want Operator element (Tr).
        # So we multiply by 2*pi.
        Op_diff *= (2 * np.pi)
        
        return Op_base + Op_diff

    def compute(self, current_ket: np.ndarray, target_kets: list, step_info: dict, target_fidelity: float) -> tuple:
        
        # 1. Compute Expectation Value: <psi | O | psi>
        psi = current_ket.flatten()
        
        # Fast Vector-Matrix-Vector multiply: O(N^2)
        # reward = Re( psi.dag * O * psi )
        reward_val = np.real(np.vdot(psi, self.reward_operator.dot(psi)))
        
        # 2. Check termination via standard fidelity
        # (We can stick to fidelity for "success" metrics even if reward is Wigner-based)
        current_fidelity = max([fidelity_max_rotation(t, current_ket) for t in target_kets])
        
        terminated = False
        if current_fidelity >= target_fidelity:
            terminated = True
            # Optional: Bonus for actual success
            reward_val += 1.0

        info = {
            'wigner_reward': reward_val,
            'fidelity': current_fidelity,
            'is_success': terminated
        }
        
        # Log measurement details if available
        result = step_info.get('result', None)
        if result is not None and hasattr(result, 'samples') and len(result.samples) > 0:
            try:
                encoded_result = result.samples[0][0]
                lost_photons, detected_photons = decode_measurement_result(encoded_result)
                info.update({
                    'photon_loss': lost_photons,
                    'detected_photons': detected_photons
                })
            except:
                pass

        return reward_val, terminated, info
