import gymnasium as gym
from gymnasium import spaces
import numpy as np
import matplotlib.pyplot as plt

class SimpleNavigationEnv(gym.Env):
    """
    A simple 2D navigation environment.
    The agent starts at (0,0) and needs to reach a target at a given coordinate.
    This environment mimics the structure of the quantum problem in the paper:
    - Continuous state space: Agent's (x, y) position.
    - Continuous action space: Change in position (dx, dy).
    - Reward based on proximity to a target.
    """
    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self, target_position=np.array([5.0, 5.0]), max_steps=100):
        super(SimpleNavigationEnv, self).__init__()

        self.max_steps = max_steps
        self.target_position = target_position
        
        # Define the boundaries of our 2D world
        self.world_bounds = 10.0

        # STATE SPACE: The agent's position [x, y].
        # Analogous to the density matrix vector in the paper.
        self.observation_space = spaces.Box(
            low=-self.world_bounds, high=self.world_bounds, shape=(2,), dtype=np.float32
        )

        # ACTION SPACE: The change in position [dx, dy].
        # Analogous to the [squeezing, transmissivity] parameters in the paper.
        # We limit the max change per step to 1 unit.
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(2,), dtype=np.float32
        )
        
        # For rendering
        self.fig, self.ax = None, None
        self.agent_path = []

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Reset agent to the origin at the start of each episode
        self.agent_position = np.array([0.0, 0.0], dtype=np.float32)
        self.current_step = 0
        
        # For rendering
        self.agent_path = [self.agent_position.copy()]
        
        # Return initial state and info
        return self.agent_position, {}

    def step(self, action):
        # 1. UPDATE STATE based on action
        self.agent_position += action
        
        # Enforce world boundaries
        self.agent_position = np.clip(
            self.agent_position, -self.world_bounds, self.world_bounds
        )
        
        self.current_step += 1
        self.agent_path.append(self.agent_position.copy())

        # 2. CALCULATE REWARD
        # Reward is based on how much closer the agent got to the target.
        # This is similar to maximizing fidelity in the paper.
        distance_to_target = np.linalg.norm(self.agent_position - self.target_position)
        
        # Using an exponential function to make the reward high only when very close,
        # similar to how fidelity works (reward is between 0 and 1).
        reward = np.exp(-0.5 * distance_to_target**2)

        # 3. CHECK FOR TERMINATION
        terminated = False
        if distance_to_target < 0.5: # Agent reached the target
            terminated = True
            reward += 10.0 # Bonus reward for reaching the goal
            
        # 4. CHECK FOR TRUNCATION (Episode ends due to time limit)
        truncated = False
        if self.current_step >= self.max_steps:
            truncated = True

        return self.agent_position, reward, terminated, truncated, {}
        
    def render(self):
        if self.fig is None:
            plt.ion() # Turn on interactive mode
            self.fig, self.ax = plt.subplots(figsize=(6, 6))

        self.ax.clear()
        
        # Draw the agent's path
        path = np.array(self.agent_path)
        self.ax.plot(path[:, 0], path[:, 1], 'b-', alpha=0.5, label='Agent Path')
        
        # Draw the agent's current position
        self.ax.plot(self.agent_position[0], self.agent_position[1], 'ro', markersize=10, label='Agent')
        
        # Draw the target
        self.ax.plot(self.target_position[0], self.target_position[1], 'gx', markersize=15, markeredgewidth=3, label='Target')

        # Set plot limits and labels
        self.ax.set_xlim(-self.world_bounds - 1, self.world_bounds + 1)
        self.ax.set_ylim(-self.world_bounds - 1, self.world_bounds + 1)
        self.ax.set_xlabel("X coordinate")
        self.ax.set_ylabel("Y coordinate")
        self.ax.set_title(f"Step: {self.current_step}")
        self.ax.grid(True)
        self.ax.legend()
        
        plt.draw()
        plt.pause(1/self.metadata["render_fps"])

    def close(self):
        if self.fig is not None:
            plt.ioff()
            plt.close(self.fig)
            self.fig, self.ax = None, None