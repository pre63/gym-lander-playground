import numpy as np
import torch
import gymnasium as gym
from stable_baselines3 import PPO

from success import check_success
from telemetry import add_telemetry_overlay, add_success_failure_to_frames

class Model:
  def __init__(self, env: gym.Env, gamma=0.99, gae_lambda=0.95, clip_range=0.2, ent_coef=0.01):
    """
    Initialize the PPO model.
    Args:
        env (gym.Env): The environment to train on.
    """
    self.parameters = {
        "gamma": gamma,
        "gae_lambda": gae_lambda,
        "clip_range": clip_range,
        "ent_coef": ent_coef
    }
    self.env = env

    self.model = PPO("MlpPolicy", env,
                     gamma=gamma,
                     gae_lambda=gae_lambda,
                     clip_range=clip_range,
                     ent_coef=ent_coef,

                     verbose=0, device="cpu")

  def train(self):
    """
    Train the model for one episode and return the episode reward.
    Returns:
        episode_reward (float): Total reward obtained in the episode.
        history (list): The history of the agent..
    """
    self.model.learn(total_timesteps=1000)  # Adjust timesteps as needed

    # Evaluate the model to get episode reward
    state, _ = self.env.reset()
    episode_reward = 0
    history = []

    done = False

    while not done:
      action, _states = self.model.predict(state, deterministic=True)
      next_state, reward, terminated, truncated, _ = self.env.step(action)
      done = terminated or truncated

      history.append({
          "state": state.tolist(),
          "action": action.tolist(),
          "reward": reward,
          "episode_reward": episode_reward,
          "next_state": next_state.tolist(),
          "done": done
      })
      episode_reward += reward
      state = next_state

      episode_reward += reward

    success = check_success(next_state, terminated)
    return success, episode_reward, history

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

  def evaluate(self, render=False):
    """
    Evaluate the model without training and return success, rewards, and frames.
    Args:
        render (bool): Whether to render the environment during evaluation.
    Returns:
        success (bool): Whether the evaluation was successful based on the defined criteria.
        episode_reward (float): Total reward obtained in the episode.
        frames (list): List of frames (always returned, even if empty).
    """
    state, _ = self.env.reset()
    episode_reward = 0
    frames = []
    done = False

    while not done:
      action, _states = self.model.predict(state, deterministic=True)
      next_state, reward, terminated, truncated, info = self.env.step(action)
      done = terminated or truncated

      if render:
        frame = add_telemetry_overlay(self.env.render(), next_state)
        frames.append(frame)

      episode_reward += reward
      state = next_state

    # Define success condition
    success = check_success(next_state, terminated)
    if render:
      frames = add_success_failure_to_frames(frames, success)

    # Always return frames, even if empty
    return success, episode_reward, frames
