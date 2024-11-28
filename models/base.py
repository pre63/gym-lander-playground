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


class BaseModel:
  def __init__(self, env_name, num_envs=16, max_episode_steps=5000, reward_strategy="default", env_type="vec", **kwargs):
    """
    Initialize the BaseModel class with a SubprocVecEnv.
    Args:
        env_name (str): The name of the environment to train on.
        num_envs (int): Number of parallel environments to use.
        kwargs: Additional arguments to pass to the algorithm's initialization.
    """
    self.env_name = env_name
    self.num_envs = num_envs
    self.env_type = env_type
    if env_type == "vec":
      self.env = SubprocVecEnv([make_env(env_name, max_episode_steps, reward_strategy) for _ in range(num_envs)])
      self.env = VecMonitor(self.env)
    elif env_type == "gym":
      self.env = make_env(env_name, max_episode_steps, reward_strategy)()

    self.parameters = kwargs
    self.model = None  # To be defined in the subclass

  def learn(self, total_timesteps=1000, progress_bar=False):
    """
    Train the model for the specified number of timesteps.
    Args:
        total_timesteps (int): Number of timesteps to train for.
    """
    self.model.learn(total_timesteps=total_timesteps, progress_bar=progress_bar)

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
    self.model.load(filename)

  def predict(self, state):
    """
    Predict the value of a state.
    Args:
        state (np.ndarray): The state to predict the value of.
    Returns:
        float: The value of the state.
    """
    return self.model.predict(state)


class BaseConvertStableBaselinesModel:
  def __init__(self, env, env_type="gym", num_envs=16):
    self.env = env
    self.env_type = env_type
    self.num_envs = self.env.num_envs if hasattr(self.env, "num_envs") else num_envs

    if hasattr(env.observation_space, "shape"):
      self.observation_dim = env.observation_space.shape[0]
    else:
      raise ValueError("Unsupported observation space type.")

    if hasattr(env.action_space, "shape"):
      self.action_dim = env.action_space.shape[0]
    else:
      raise ValueError("Unsupported action space type.")

  def train_step(self, state, action, reward, next_state, next_action, done, info):
    """
    Perform a single training step. Should be overridden by subclasses.
    """
    raise NotImplementedError("train_step must be implemented by subclasses.")

  def predict(self, state):
    """
    Predict the value of a state or the best action. Should be overridden by subclasses.
    """
    raise NotImplementedError("predict must be implemented by subclasses.")

  def learn_episode_setup(self):
    """
    Perform any necessary setup before learning an episode.
    """
    pass

  def _process_single_env(self, total_timesteps, progress_bar):
    discounted_timesteps = 0
    done = True

    print("Training on a single environment...")

    while discounted_timesteps < total_timesteps:
      if done:
        state, _ = self.env.reset()
        self.learn_episode_setup()
        action = self.env.action_space.sample()
        done = False

      next_state, reward, terminated, truncated, info = self.env.step(action)
      next_action, _ = self.predict(next_state)

      self.train_step(state, action, reward, next_state, next_action, terminated or truncated, info)

      state, action = next_state, next_action
      discounted_timesteps += 1
      done = terminated

      if progress_bar:
        print(f"\rProgress: {discounted_timesteps}/{total_timesteps}", end="")

  def _process_multi_env(self, total_timesteps, progress_bar):
    progress_bar = True
    discounted_timesteps = 0
    dones = [True] * self.num_envs

    print("Training on multiple environments...")

    while discounted_timesteps < total_timesteps:
      if all(dones):
        states = self.env.reset()
        self.learn_episode_setup()
        actions = np.array([self.env.action_space.sample() for _ in range(self.num_envs)])
      
      next_states, rewards, dones, infos = self.env.step(actions)

      next_actions = np.array([self.predict(next_states[i])[0] for i in range(self.num_envs)])

      for i in range(self.num_envs):
        self.train_step(states[i], actions[i], rewards[i], next_states[i], next_actions[i], dones[i], infos[i])

      states, actions = next_states, next_actions
      discounted_timesteps += self.num_envs

      if progress_bar:
        print(f"\rProgress: {discounted_timesteps}/{total_timesteps}", end="")

  def learn(self, total_timesteps=1000, progress_bar=False):
    if self.env_type == "gym":
      self._process_single_env(total_timesteps, progress_bar)
    else:
      self._process_multi_env(total_timesteps, progress_bar)
