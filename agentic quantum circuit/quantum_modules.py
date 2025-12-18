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
    def _ket_to_observation(self, ket):
        # ... (Same implementation as your base class) ...
        # Can be moved to a utility function
        pass

class RewardMechanism(abc.ABC):
    """Responsible for calculating reward and termination."""
    @abc.abstractmethod
    def compute(self, current_ket: np.ndarray, target_ket: np.ndarray, step_info: dict) -> tuple[float, bool, dict]:
        """Returns (reward, terminated, info)."""
        pass