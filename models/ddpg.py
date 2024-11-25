import numpy as np
import gymnasium as gym
from stable_baselines3 import DDPG

from success import check_success


class Model:
  def __init__(self, env: gym.Env, gamma=0.99, tau=0.005, learning_rate=1e-3, buffer_size=1000000, batch_size=64, train_freq=1, gradient_steps=1):
    """
    Initialize the Deep Deterministic Policy Gradient (DDPG) model using Stable-Baselines3.
    Args:
        env (gym.Env): The environment to train on.
    """
    self.parameters = {
        "gamma": gamma,
        "tau": tau,
        "learning_rate": learning_rate,
        "buffer_size": buffer_size,
        "batch_size": batch_size,
        "train_freq": train_freq,
        "gradient_steps": gradient_steps,
    }

    self.env = env

    self.model = DDPG(
        "MlpPolicy",
        env,
        gamma=gamma,
        tau=tau,
        learning_rate=learning_rate,
        buffer_size=buffer_size,
        batch_size=batch_size,
        train_freq=train_freq,
        gradient_steps=gradient_steps,
        verbose=0,
        device="auto"
    )

  def train(self, total_timesteps=10000):
    """
    Train the model.
    Args:
        total_timesteps (int): Total timesteps to train the model.
    Returns:
        None
    """
    self.model.learn(total_timesteps=total_timesteps)

  def save(self, filename):
    """
    Save the model to a file.
    Args:
        filename (str): The name of the file to save.
    """
    self.model.save(filename)

  def load(self, filename):
    """
    Load the model from a file.
    Args:
        filename (str): The name of the file to load.
    """
    self.model = DDPG.load(filename, env=self.env)

  def evaluate(self, render=False):
    """
    Evaluate the model without training and return success, rewards, and frames.
    Args:
        render (bool): Whether to render the environment during evaluation.
    Returns:
        success (bool): Whether the evaluation was successful based on the defined criteria.
        episode_reward (float): Total reward obtained in the episode.
        frames (list): List of frames if rendering is enabled.
    """
    state, _ = self.env.reset()
    done = False
    episode_reward = 0
    frames = []

    while not done:
      action, _ = self.model.predict(state, deterministic=True)
      next_state, reward, terminated, truncated, info = self.env.step(action)
      done = terminated or truncated

      episode_reward += reward

      if render:
        frame = self.env.render()
        frames.append(frame)

      state = next_state

    # Define success condition
    success = check_success(next_state, terminated)

    return success, episode_reward, frames
