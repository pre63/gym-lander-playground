import numpy as np
import torch
import gymnasium as gym
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from stable_baselines3 import PPO
from success import check_success
from telemetry import add_telemetry_overlay, add_success_failure_to_frames
from gymnasium.wrappers import TimeLimit
from reward import RewardWrapper


def make_env(env_name, max_episode_steps=2000, reward_strategy="default"):
  """
  Factory function to create an environment.
  Args:
      env_name (str): The name of the environment.
      max_episode_steps (int): Maximum steps per episode.
      reward_strategy (str): Strategy for reward wrapping.
  Returns:
      Callable: A callable environment creation function.
  """
  def _init():
    env = gym.make(env_name, render_mode="rgb_array")
    env = TimeLimit(env, max_episode_steps=max_episode_steps)
    env = RewardWrapper(env, reward_strategy)
    return env
  return _init


class SBase:
  def __init__(self, env_name, num_envs=512, max_episode_steps=5000, reward_strategy="default", model_type="vec", **kwargs):
    """
    Initialize the SBase class with a SubprocVecEnv.
    Args:
        env_name (str): The name of the environment to train on.
        num_envs (int): Number of parallel environments to use.
        kwargs: Additional arguments to pass to the algorithm's initialization.
    """
    self.env_name = env_name
    self.num_envs = num_envs

    if model_type == "vec":
      self.env = SubprocVecEnv([make_env(env_name, max_episode_steps, reward_strategy) for _ in range(num_envs)])
      self.env = VecMonitor(self.env)
    elif model_type == "gym":
      self.env = make_env(env_name, max_episode_steps, reward_strategy)()

    self.parameters = kwargs
    self.model = None  # To be defined in the subclass

  def train(self, total_timesteps=1000):
    """
    Train the model for the specified number of timesteps.
    Args:
        total_timesteps (int): Number of timesteps to train for.
    """
    self.model.learn(total_timesteps=total_timesteps, progress_bar=True)

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
    self.model = self.model.load(filename)
