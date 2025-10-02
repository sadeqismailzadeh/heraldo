import gymnasium as gym
from gymnasium import spaces
import numpy as np
import matplotlib.pyplot as plt

class RandomNavigationEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self, max_steps=100):
        super(RandomNavigationEnv, self).__init__()
        self.max_steps = max_steps
        self.world_bounds = 10.0

        # --- CHANGE 1: Update the Observation Space ---
        # The observation is now the vector from the agent to the target.
        # If the agent is at (-10,-10) and the target at (10,10), the
        # relative vector is (20,20). So the bounds must be larger.
        max_relative_pos = 2 * self.world_bounds
        self.observation_space = spaces.Box(
            low=-max_relative_pos, high=max_relative_pos, shape=(2,), dtype=np.float32
        )

        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(2,), dtype=np.float32
        )
        
        # We need to store both the agent's and target's absolute position
        # for physics and rendering, even though the agent only "sees" the relative vector.
        self.agent_position = None
        self.target_position = None
        
        self.fig, self.ax = None, None
        self.agent_path = []

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # --- CHANGE 2: Randomize Initial and Target Positions ---
        # Use the environment's random number generator for reproducibility.
        self.agent_position = self.np_random.uniform(
            low=-self.world_bounds, high=self.world_bounds, size=(2,)
        ).astype(np.float32)
        
        self.target_position = self.np_random.uniform(
            low=-self.world_bounds, high=self.world_bounds, size=(2,)
        ).astype(np.float32)

        # Make sure the agent and target don't start on top of each other
        while np.linalg.norm(self.agent_position - self.target_position) < 1.0:
            self.target_position = self.np_random.uniform(
                low=-self.world_bounds, high=self.world_bounds, size=(2,)
            ).astype(np.float32)

        self.current_step = 0
        self.agent_path = [self.agent_position.copy()]
        
        # --- CHANGE 3: Return the RELATIVE position as the observation ---
        observation = self.target_position - self.agent_position
        
        return observation, {}

    def step(self, action):
        self.agent_position += action
        self.agent_position = np.clip(
            self.agent_position, -self.world_bounds, self.world_bounds
        )
        
        self.current_step += 1
        self.agent_path.append(self.agent_position.copy())

        distance_to_target = np.linalg.norm(self.agent_position - self.target_position)
        
        # The reward can remain the same
        reward = np.exp(-0.5 * distance_to_target**2)

        terminated = False
        if distance_to_target < 0.5:
            terminated = True
            reward += 10.0
            
        truncated = False
        if self.current_step >= self.max_steps:
            truncated = True

        # --- CHANGE 4: Return the NEW RELATIVE position as the observation ---
        observation = self.target_position - self.agent_position

        return observation, reward, terminated, truncated, {}

    # The render() and close() methods do not need any changes,
    # as they use self.agent_position and self.target_position directly.
    def render(self):
        # (No changes needed here)
        if self.fig is None:
            plt.ion()
            self.fig, self.ax = plt.subplots(figsize=(6, 6))
        self.ax.clear()
        path = np.array(self.agent_path)
        self.ax.plot(path[:, 0], path[:, 1], 'b-', alpha=0.5, label='Agent Path')
        self.ax.plot(self.agent_position[0], self.agent_position[1], 'ro', markersize=10, label='Agent')
        self.ax.plot(self.target_position[0], self.target_position[1], 'gx', markersize=15, markeredgewidth=3, label='Target')
        self.ax.set_xlim(-self.world_bounds - 1, self.world_bounds + 1)
        self.ax.set_ylim(-self.world_bounds - 1, self.world_bounds + 1)
        self.ax.set_title(f"Step: {self.current_step}")
        self.ax.grid(True)
        self.ax.legend()
        plt.draw()
        plt.pause(1/self.metadata["render_fps"])

    def close(self):
        # (No changes needed here)
        if self.fig is not None:
            plt.ioff()
            plt.close(self.fig)
            self.fig, self.ax = None, None