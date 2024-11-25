import numpy as np
import gymnasium as gym
from scipy import stats

from success import check_success


class TrueOnlineTDlambda_aReplay:
  def __init__(self, alpha, gamma, lambda_, theta_init):
    """
    Initialize the True Online TD(λ)-Replay algorithm parameters.

    Parameters:
    - alpha: Learning rate.
    - gamma: Discount factor.
    - lambda_: Trace decay parameter.
    - theta_init: Initial weights (numpy array).
    """
    self.alpha = alpha
    self.gamma = gamma
    self.lambda_ = lambda_
    self.theta = theta_init.copy()
    self.n = len(theta_init)
    self.reset_traces()

  def reset_traces(self):
    """
    Reset the eligibility traces and other temporary variables.
    """
    self.e = np.zeros(self.n)
    self.e_bar = np.zeros(self.n)
    self.A_bar = np.eye(self.n)
    self.V_old = 0.0

  def train_step(self, phi_t, R_t, phi_t1):
    """
    Update the weights based on a single transition.

    Parameters:
    - phi_t: Feature vector of the current state.
    - R_t: Reward received after taking the action.
    - phi_t1: Feature vector of the next state.
    """
    V_t = np.dot(self.theta, phi_t)
    V_t1 = np.dot(self.theta, phi_t1)
    delta_t = R_t + self.gamma * V_t1 - V_t

    # Update eligibility trace e
    e_phi_t = np.dot(self.e, phi_t)
    self.e = self.gamma * self.lambda_ * self.e - self.alpha * phi_t * (self.gamma * self.lambda_ * e_phi_t - 1)

    # Update adjusted eligibility trace e_bar
    e_bar_phi_t = np.dot(self.e_bar, phi_t)
    self.e_bar = self.e_bar - self.alpha * e_bar_phi_t * (phi_t - self.V_old) + self.e * (delta_t + V_t - self.V_old)

    # Update A_bar matrix
    A_bar_phi_t = self.A_bar.dot(phi_t)
    self.A_bar = self.A_bar - self.alpha * np.outer(phi_t, A_bar_phi_t)

    # Update weights theta
    self.theta = self.A_bar.dot(self.theta) + self.e_bar

    # Update V_old for next iteration
    self.V_old = V_t1


class Model:
  def __init__(self, env: gym.Env, alpha=0.1, gamma=0.95, lambda_=0.9):
    """
    Initialize the model as a wrapper for the environment and algorithm.
    Args:
        env (gym.Env): The environment to train on.
        alpha (float): Learning rate.
        gamma (float): Discount factor.
        lambda_ (float): Trace decay parameter.
    """
    self.env = env
    state_dim = env.observation_space.shape[0]
    theta_init = np.zeros(state_dim)
    self.algorithm = TrueOnlineTDlambda_aReplay(
        alpha=alpha,
        gamma=gamma,
        lambda_=lambda_,
        theta_init=theta_init
    )
    self.parameters = {
        "alpha": alpha,
        "gamma": gamma,
        "lambda_a": lambda_
    }
    self.best_cumulative_rewards = []

  def train(self):
    """
    Train the model for one episode and return the episode reward and history.
    Returns:
        episode_reward (float): Total reward obtained in the episode.
        history (list): The history of the agent.
    """
    state, _ = self.env.reset()
    done = False
    episode_reward = 0
    history = []

    cumulative_rewards = []
    t = 0  # Time step

    self.algorithm.reset_traces()

    while not done:
      action = self.env.action_space.sample()  # Random action; replace with a policy if available
      next_state, reward, terminated, truncated, info = self.env.step(action)
      done = terminated or truncated

      # Convert states to feature vectors (phi_t and phi_t1)
      phi_t = np.array(state)
      phi_t1 = np.array(next_state)
      self.algorithm.train_step(phi_t, reward, phi_t1)

      episode_reward += reward
      cumulative_rewards.append(episode_reward)
      t += 1

      # Append history
      history.append({
          "state": state.tolist(),
          "action": action.tolist() if isinstance(action, np.ndarray) else action,
          "reward": reward,
          "episode_reward": episode_reward,
          "next_state": next_state.tolist(),
          "done": done
      })

      state = next_state

    return episode_reward, history

  def save(self, filename):
    """
    Save the model to a file.
    Args:
        filename (str): The name of the file to save the model to.
    """
    np.save(filename, self.algorithm.theta)

  def load(self, filename):
    """
    Load the model from a file.
    Args:
        filename (str): The name of the file to load the model from.
    """
    self.algorithm.theta = np.load(filename + ".npy")

  def evaluate(self, render=False):
    """
    Evaluate the model without training and return success and rewards.
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
      action = self.env.action_space.sample()  # Random action; replace with a policy if available
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
