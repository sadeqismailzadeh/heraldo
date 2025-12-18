
# 1. Import the module we need to patch
import scipy.integrate

# 2. Check if the patch is needed to avoid errors
if not hasattr(scipy.integrate, 'simps'):
    print("Monkey patching scipy.integrate: 'simps' not found. Pointing to 'simpson'.")
    # 3. Create the 'simps' attribute and point it to the existing 'simpson' function.
    scipy.integrate.simps = scipy.integrate.simpson
else:
    print("'simps' already exists in scipy.integrate. No patch needed.")

import abc
import numpy as np
import gymnasium as gym
import strawberryfields as sf


class TargetGenerator(abc.ABC):
    """Responsible for generating the target state ket."""
    @abc.abstractmethod
    def get_target_ket(self, cutoff_dim: int) -> np.ndarray:
        pass

class CircuitContext(abc.ABC):
    """Responsible for the physical settings and action space."""

    @property
    @abc.abstractmethod
    def action_keys(self) -> list:
        """List of strings naming the action parameters."""
        pass

    @property
    @abc.abstractmethod
    def action_ranges(self) -> dict:
        """Dictionary mapping keys to (min, max) physical values."""
        pass

    @abc.abstractmethod
    def get_action_space(self) -> gym.spaces.Box:
        pass

    @abc.abstractmethod
    def build_step_program(self, action: np.ndarray) -> sf.Program:
        """Returns the SF program for a specific step."""
        pass

    @abc.abstractmethod
    def build_reset_program(self) -> sf.Program:
        """Returns the SF program for initialization."""
        pass

    @abc.abstractmethod
    def _get_current_ket(self, state):
        """Extract the relevant ket from the SF state object."""
        pass


class RewardMechanism(abc.ABC):
    """Responsible for calculating reward and termination."""
    @abc.abstractmethod
    def compute(self, current_ket: np.ndarray, target_kets: list, step_info: dict, target_fidelity: float) -> tuple[float, bool, dict]:
        """
        Compute reward and termination signal.
        
        Args:
            current_ket: Current state vector
            target_kets: List of target state vectors
            step_info: Dictionary with step information
            
        Returns:
            Tuple of (reward, terminated, info)
        """
        pass