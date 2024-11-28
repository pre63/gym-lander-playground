import gymnasium as gym
from gymnasium import spaces
import numpy as np
import matplotlib.pyplot as plt


class RandomWalk(gym.Env):
    def __init__(self, grid_size=10):
        super().__init__()
        self.grid_size = grid_size
        self.state = grid_size // 2  # Start in the middle
        self.action_space = spaces.Discrete(2)  # 0: Left, 1: Right
        self.observation_space = spaces.Discrete(grid_size)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.state = self.grid_size // 2
        return self.state, {}

    def step(self, action):
        if action == 0:  # Move left
            self.state = max(0, self.state - 1)
        elif action == 1:  # Move right
            self.state = min(self.grid_size - 1, self.state + 1)

        done = self.state == 0 or self.state == self.grid_size - 1
        reward = 1 if done else 0
        return self.state, reward, done, False, {}

    def render(self):
        grid = ['.'] * self.grid_size
        grid[self.state] = 'A'
        print("".join(grid))