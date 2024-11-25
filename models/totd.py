import numpy as np
import gymnasium as gym
from scipy import stats

from success import check_success


class TrueOnlineTDLambdaReplay:
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
    self.algorithm = TrueOnlineTDLambdaReplay(
        alpha=alpha,
        gamma=gamma,
        lambda_=lambda_,
        theta_init=theta_init
    )
    
    self.parameters = {
        "alpha": alpha,
        "gamma": gamma,
        "lambda": lambda_
    }

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
      action = self.select_action(state)
      next_state, reward, terminated, truncated, info = self.env.step(action)
      done = terminated or truncated

      phi_t = np.array(state)
      phi_t1 = np.array(next_state)
      self.algorithm.train_step(phi_t, reward, phi_t1)

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

    return episode_reward, history

  def select_action(self, state):
    """
    Select an action for a continuous action space using a Gaussian policy.

    Args:
        state (np.ndarray): The current state of the environment.
    Returns:
        np.ndarray: The continuous action sampled from the policy.
    """
    mean_action = self.compute_policy(state)  # Mean action
    action = np.clip(
        mean_action + np.random.normal(0, 0.1, size=mean_action.shape),
        self.env.action_space.low,
        self.env.action_space.high
    )
    return action

  def compute_policy(self, state):
    """
    Compute the mean action for the current policy.

    Args:
        state (np.ndarray): The current state of the environment.
    Returns:
        np.ndarray: The mean action based on the policy.
    """
    phi_state = np.array(state)
    return np.dot(self.algorithm.theta, phi_state)

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
    state, _ = self.env.reset()
    done = False
    episode_reward = 0
    frames = []

    while not done:
      action = self.select_action(state)
      next_state, reward, terminated, truncated, info = self.env.step(action)
      done = terminated or truncated

      episode_reward += reward

      if render:
        frames.append(self.env.render())

      state = next_state

    success = check_success(next_state, terminated)
    return success, episode_reward, frames
