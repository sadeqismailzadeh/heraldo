# Import the necessary libraries
import gymnasium as gym  # The core library for creating RL environments
from gymnasium import spaces # Used to define the structure of states and actions
import numpy as np # For numerical operations (vectors, matrices, etc.)
import matplotlib.pyplot as plt # For visualizing the environment

# All custom environments must inherit from the gymnasium.Env class.
# This ensures they have the required methods (reset, step, etc.).
class SimpleNavigationEnv(gym.Env):
    """
    A simple 2D navigation environment that follows the gymnasium interface.

    The agent's goal is to navigate from a starting point (0,0) to a target position.
    This environment is a stand-in for the complex quantum circuit from the paper,
    but it maintains the same structural properties:
    - Continuous state space: The agent's (x, y) position.
    - Continuous action space: The desired change in position (dx, dy).
    - Step-by-step interaction loop.
    - Reward function based on proximity to a target.
    """
    # metadata is used by gymnasium to handle things like rendering modes and frame rates.
    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self, target_position=np.array([5.0, 5.0]), max_steps=100):
        """
        The constructor for the environment. This is where you define
        the fundamental properties like the action and observation spaces.
        """
        super(SimpleNavigationEnv, self).__init__()

        # --- Environment Parameters ---
        self.max_steps = max_steps  # The maximum number of steps in one episode
        self.target_position = target_position # The coordinates of the goal
        self.world_bounds = 10.0 # The size of our 2D world (-10 to +10)

        # --- Define Observation and Action Spaces ---
        # These definitions are mandatory for a gymnasium environment. They tell the
        # RL agent what kind of data to expect as input (state) and what kind
        # of data it needs to output (action).

        # OBSERVATION SPACE (State): What the agent "sees".
        # In our case, it's the agent's [x, y] coordinates.
        # `spaces.Box` is used for continuous values within a certain range.
        # low/high: The minimum/maximum possible values for x and y.
        # shape: The shape of the state vector. (2,) means a 1D array of length 2.
        # This is analogous to the flattened density matrix vector in the paper.
        self.observation_space = spaces.Box(
            low=-self.world_bounds, high=self.world_bounds, shape=(2,), dtype=np.float32
        )

        # ACTION SPACE: What the agent can "do".
        # In our case, it's the change in position [dx, dy].
        # We limit the maximum movement in one step to 1.0 in any direction.
        # This is analogous to the [squeezing, transmissivity] parameters in the paper.
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(2,), dtype=np.float32
        )
        
        # --- For Visualization (Rendering) ---
        self.fig, self.ax = None, None # Matplotlib figure and axis
        self.agent_path = [] # To store the agent's trajectory for drawing

    def reset(self, seed=None, options=None):
        """
        This method is called at the beginning of each new episode.
        It resets the environment to its initial state.
        It MUST return the initial observation and an info dictionary.
        """
        super().reset(seed=seed)
        
        # Reset agent's position to the origin
        self.agent_position = np.array([0.0, 0.0], dtype=np.float32)
        self.current_step = 0 # Reset the step counter
        
        # Reset the path for the new episode's visualization
        self.agent_path = [self.agent_position.copy()]
        
        # The info dictionary can be used to pass auxiliary diagnostic information.
        # It's often empty for simple environments.
        info = {}
        
        # Return the initial state of the environment and the info dict
        return self.agent_position, info

    def step(self, action):
        """
        This method executes one time step in the environment.
        It takes an 'action' from the agent and computes the result.
        It MUST return: observation, reward, terminated, truncated, info
        """
        # 1. UPDATE STATE: Apply the agent's action to update the environment.
        # The action is a [dx, dy] vector from the agent's policy network.
        self.agent_position += action
        
        # Keep the agent within the defined world boundaries.
        self.agent_position = np.clip(
            self.agent_position, -self.world_bounds, self.world_bounds
        )
        
        self.current_step += 1
        self.agent_path.append(self.agent_position.copy()) # Record for rendering

        # 2. CALCULATE REWARD: Determine the feedback for the agent's action.
        # Good actions should receive positive rewards.
        distance_to_target = np.linalg.norm(self.agent_position - self.target_position)
        
        # The reward is designed to be high when the agent is very close to the target.
        # An exponential function gives a smooth reward between 0 and 1, which is
        # analogous to the 'fidelity' metric used in the quantum paper.
        reward = np.exp(-0.5 * distance_to_target**2)

        # 3. CHECK FOR 'terminated': Has the episode ended because the goal was reached?
        terminated = False
        if distance_to_target < 0.1:
            terminated = True
            reward += 10.0 # Give a large bonus for successfully reaching the goal.

        # 4. CHECK FOR 'truncated': Has the episode ended for another reason (e.g., time limit)?
        truncated = False
        if self.current_step >= self.max_steps:
            truncated = True

        # 5. INFO DICTIONARY: Can contain extra info, but we don't need it here.
        info = {}

        # Return the 5-tuple required by gymnasium
        return self.agent_position, reward, terminated, truncated, info
        
    def render(self):
        """
        This method visualizes the current state of the environment.
        """
        if self.fig is None:
            plt.ion() # Enable interactive plotting
            self.fig, self.ax = plt.subplots(figsize=(6, 6))

        self.ax.clear() # Clear the plot for the new frame
        
        # Draw the agent's full path for this episode
        path = np.array(self.agent_path)
        self.ax.plot(path[:, 0], path[:, 1], 'b-', alpha=0.5, label='Agent Path')
        
        # Draw the agent's current position (red circle)
        self.ax.plot(self.agent_position[0], self.agent_position[1], 'ro', markersize=10, label='Agent')
        
        # Draw the target position (green 'x')
        self.ax.plot(self.target_position[0], self.target_position[1], 'gx', markersize=15, markeredgewidth=3, label='Target')

        # Formatting the plot
        self.ax.set_xlim(-self.world_bounds - 1, self.world_bounds + 1)
        self.ax.set_ylim(-self.world_bounds - 1, self.world_bounds + 1)
        self.ax.set_title(f"Step: {self.current_step}")
        self.ax.grid(True)
        self.ax.legend()
        
        # Update the display
        plt.draw()
        plt.pause(1/self.metadata["render_fps"])

    def close(self):
        """
        This method is called when the environment is no longer needed.
        It's used for cleanup, like closing visualization windows.
        """
        if self.fig is not None:
            plt.ioff() # Turn off interactive mode
            plt.close(self.fig)
            self.fig, self.ax = None, None