import numpy as np
import gymnasium as gym
from stable_baselines3 import DDPG

from success import check_success
from telemetry import add_telemetry_overlay, add_success_failure_to_frames


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
    Train the model for a single episode.
    Args:
        total_timesteps (int): Maximum timesteps to train the model in one episode.
    Returns:
        success (bool): Whether the training episode was successful based on the defined criteria.
        episode_reward (float): Total reward obtained in the episode.
        history (list): List of dictionaries capturing per-step metrics like state, reward, and action.
    """
    state, _ = self.env.reset()
    done = False
    episode_reward = 0
    history = []

    for _ in range(total_timesteps):
      if done:
        break

      # Select action using the model's actor policy
      action, _ = self.model.predict(state, deterministic=False)

      # Interact with the environment
      next_state, reward, terminated, truncated, info = self.env.step(action)
      done = terminated or truncated

      # Accumulate reward
      episode_reward += reward

      # Store the step information for logging
      history.append({"state": state, "action": action, "reward": reward, "next_state": next_state})

      # Update the model using the replay buffer and sampled minibatch
      # This happens internally in Stable-Baselines3 during `learn`
      self.model.replay_buffer.add(state, next_state, action, reward, done)
      self.model.train_step()

      # Move to the next state
      state = next_state

    # Check success at the end of the episode
    success = check_success(state, terminated)
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
        frame = add_telemetry_overlay(self.env.render(), next_state)
        frames.append(frame)

      state = next_state

    # Define success condition
    success = check_success(next_state, terminated)
    if render:
      frames = add_success_failure_to_frames(frames, success)

    return success, episode_reward, frames
