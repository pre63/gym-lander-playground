import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO


class Model:
  def __init__(self, env: gym.Env):
    """
    Initialize the PPO model.
    Args:
        env (gym.Env): The environment to train on.
    """
    self.env = env
    self.model = PPO("MlpPolicy", env, verbose=0)

  def train(self):
    """
    Train the model for one episode and return the episode reward.
    Returns:
        episode_reward (float): Total reward obtained in the episode.
        trajectory (list): The trajectory of the agent..
    """
    self.model.learn(total_timesteps=1000)  # Adjust timesteps as needed

    # Evaluate the model to get episode reward
    state, _ = self.env.reset()
    episode_reward = 0
    trajectory = []
    done = False

    while not done:
      action, _states = self.model.predict(state, deterministic=True)
      next_state, reward, terminated, truncated, _ = self.env.step(action)
      done = terminated or truncated

      trajectory.append({
          "state": state.tolist(),
          "action": action.tolist(),
          "reward": reward,
          "next_state": next_state.tolist(),
          "done": done
      })
      episode_reward += reward
      state = next_state

      episode_reward += reward

    return episode_reward, trajectory
