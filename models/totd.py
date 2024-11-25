import numpy as np
import gymnasium as gym
from scipy import stats


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
  def __init__(self, env: gym.Env, alpha=0.1, gamma=0.95, lambd=0.9):
    """
    Initialize the model as a wrapper for the environment and algorithm.
    Args:
        env (gym.Env): The environment to train on.
        alpha (float): Learning rate.
        gamma (float): Discount factor.
        lambd (float): Trace decay parameter.
    """
    self.env = env
    state_dim = env.observation_space.shape[0]
    theta_init = np.zeros(state_dim)
    self.algorithm = TrueOnlineTDLambdaReplay(
        alpha=alpha,
        gamma=gamma,
        lambda_=lambd,
        theta_init=theta_init
    )
    self.best_episode_reward = -np.inf
    self.best_cumulative_rewards = []
    self.previous_rewards_per_step = []

  def train(self):
    """
    Train the model for one episode and return the episode reward, trajectory, and frames.
    Returns:
        episode_reward (float): Total reward obtained in the episode.
        trajectory (list): The trajectory of the agent.
        frames (list): Rendered frames of the episode.
    """
    state, _ = self.env.reset()
    done = False
    episode_reward = 0
    trajectory = []
    frames = []
    cumulative_rewards = []
    t = 0  # Time step

    self.algorithm.reset_traces()

    while not done:
      action = self.env.action_space.sample()  # Random action; replace with a policy if needed
      next_state, reward, terminated, truncated, _ = self.env.step(action)
      done = terminated or truncated

      # Convert states to feature vectors (phi_t and phi_t1)
      phi_t = np.array(state)
      phi_t1 = np.array(next_state)
      self.algorithm.train_step(phi_t, reward, phi_t1)

      episode_reward += reward
      cumulative_rewards.append(episode_reward)
      t += 1

      # Append trajectory
      trajectory.append({
          "state": state.tolist() if isinstance(state, np.ndarray) else state,
          "action": action.tolist() if isinstance(action, np.ndarray) else action,
          "reward": reward,
          "next_state": next_state.tolist() if isinstance(next_state, np.ndarray) else next_state,
          "done": done
      })

      # **Inferential Statistics Logic**
      render_frame = True
      if self.best_episode_reward > -np.inf and t < len(self.best_cumulative_rewards):
        # Calculate the difference in cumulative rewards
        current_cumulative = episode_reward
        best_cumulative = self.best_cumulative_rewards[t - 1]  # Adjust index for zero-based

        # Estimate the anticipated total reward
        steps_remaining = self.env.spec.max_episode_steps - t if self.env.spec.max_episode_steps else 1000 - t
        avg_reward_per_step = episode_reward / t
        anticipated_total_reward = episode_reward + avg_reward_per_step * steps_remaining

        # Calculate mean and standard deviation of rewards per step from previous episodes
        if len(self.previous_rewards_per_step) > 1:
          mean_best = np.mean(self.previous_rewards_per_step)
          std_best = np.std(self.previous_rewards_per_step, ddof=1)

          # Perform one-sample t-test
          t_stat, p_value = stats.ttest_1samp(
              [avg_reward_per_step],
              popmean=mean_best
          )

          # If p-value is greater than 0.05 (not significantly better), stop rendering
          if p_value > 0.05 and anticipated_total_reward < self.best_episode_reward:
            render_frame = False
        else:
          # If not enough data, use a simple threshold
          if anticipated_total_reward < 0.9 * self.best_episode_reward:
            render_frame = False

      # Render frame if needed
      if render_frame:
        frames.append(self.env.render())

      state = next_state

    # Update best episode reward and cumulative rewards if current is better
    if episode_reward > self.best_episode_reward:
      self.best_episode_reward = episode_reward
      self.best_cumulative_rewards = cumulative_rewards

    # Store the average reward per step for inferential statistics
    avg_reward_per_step = episode_reward / t
    self.previous_rewards_per_step.append(avg_reward_per_step)

    return episode_reward, trajectory, frames
