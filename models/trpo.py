import gymnasium as gym
from sb3_contrib import TRPO

from success import check_success


class Model:
  def __init__(self, env: gym.Env, gamma=0.99, gae_lambda=0.95, target_kl=0.01, net_arch=[64, 64]):
    """
    Initialize the TRPO model with the given environment and hyperparameters.
    Args:
        env (gym.Env): The environment to train on.
        gamma (float): Discount factor for rewards.
        gae_lambda (float): lambda for Generalized Advantage Estimation (GAE).
        target_kl (float): Target Kullback-Leibler divergence for trust region update.
    """
    self.parameters = {
        "gamma": gamma,
        "gae_lambda": gae_lambda,
        "target_kl": target_kl,
        "net_arch": net_arch
    }

    self.env = env
    self.model = TRPO(
        policy="MlpPolicy",
        env=env,
        verbose=0,
        gamma=gamma,
        gae_lambda=gae_lambda,
        target_kl=target_kl,  # Correctly using target_kl here
        policy_kwargs={
            "net_arch": net_arch
        },
        device="cpu"
    )

  def train(self):
    """
    Train the TRPO model for one episode and return the episode reward and rewards per step.

    Returns:
        episode_reward (float): Total reward obtained in the episode.
        history (list): The history of the agent..
    """
    state, _ = self.env.reset()
    done = False
    episode_reward = 0
    history = []

    while not done:
      action, _ = self.model.predict(state, deterministic=False)
      next_state, reward, terminated, truncated, _ = self.env.step(action)
      done = terminated or truncated

      episode_reward += reward
      history.append({
          "state": state.tolist(),
          "action": action.tolist(),
          "reward": reward,
          "episode_reward": episode_reward,
          "next_state": next_state.tolist(),
          "done": done
      })
      state = next_state

    # Perform a learning step with a fixed number of timesteps
    self.model.learn(total_timesteps=1000, reset_num_timesteps=False)

    return episode_reward, history

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
    Evaluate the model for one episode and return success status and reward.
    Args:
        render (bool): Whether to render frames during evaluation.
    Returns:
        success (bool): Whether the episode was successful.
        episode_reward (float): Total reward obtained in the episode.
        frames (list): List of frames if rendering is enabled.
    """
    state, _ = self.env.reset()
    done = False
    episode_reward = 0
    frames = []

    while not done:
      action = self.env.action_space.sample()
      next_state, reward, terminated, truncated, info = self.env.step(action)
      done = terminated or truncated

      episode_reward += reward

      if render:
        frame = self.env.render()
        frames.append(frame)

      state = next_state

    success = check_success(next_state, terminated)

    return success, episode_reward, frames if render else None
