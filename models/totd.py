import numpy as np
import gymnasium as gym
from scipy import stats

from models.base import BaseModel

from success import check_success
from telemetry import add_telemetry_overlay, add_success_failure_to_frames


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
    - phi_t: Feature vector of the current state-action pair.
    - R_t: Reward received after taking the action.
    - phi_t1: Feature vector of the next state-action pair.
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


# class TOTD:
#   def __init__(self, env, alpha=0.01, gamma=0.99, lambda_=0.9, theta_init=None):
#     self.env = env
#     self.alpha = alpha
#     self.gamma = gamma
#     self.lambda_ = lambda_

#     self.observation_dim = self.env.observation_space.shape[0]
#     self.action_dim = self.env.action_space.shape[0]

#     # Feature vector dimension: state_dim + action_dim
#     self.feature_dim = self.observation_dim + self.action_dim

#     # Initialize theta as a 1D weight vector for features
#     self.theta_init = theta_init or np.zeros(self.feature_dim)
#     self.model = TrueOnlineTDLambdaReplay(alpha, gamma, lambda_, self.theta_init)

#   def state_action_features(self, state, action):
#     state = np.asarray(state, dtype=np.float32).flatten()
#     action = np.asarray(action, dtype=np.float32).flatten()
#     assert len(action) == self.action_dim, f"Expected action dimension {self.action_dim}, got {len(action)}"
#     return np.concatenate([state, action])


#   def predict_action(self, state):
#     sampled_actions = [self.env.action_space.sample() for _ in range(10)]
#     q_values = [self.q_value(state, action) for action in sampled_actions]
#     best_action = sampled_actions[np.argmax(q_values)]
#     # Ensure the action matches the expected shape
#     return np.clip(np.asarray(best_action, dtype=np.float32), self.env.action_space.low, self.env.action_space.high)


#   def q_value(self, state, action):
#     # Compute Q(s, a) as the dot product of theta and the feature vector
#     phi = self.state_action_features(state, action)
#     return np.dot(self.model.theta, phi)

#   def learn(self, total_timesteps=1000, progress_bar=False):
#     discount_timesteps = 0
#     done = True

#     while discount_timesteps < total_timesteps:
#       if done:
#         state, _ = self.env.reset()
#         action = self.env.action_space.sample()  # Initial random action
#         phi = self.state_action_features(state, action)
#         done = False

#       next_state, reward, terminated,_, _ = self.env.step(action)
#       next_action = self.predict_action(next_state)
#       next_phi = self.state_action_features(next_state, next_action)

#       # Train step
#       self.model.train_step(phi, reward, next_phi)

#       # Update for the next timestep
#       state, action, phi = next_state, next_action, next_phi
#       discount_timesteps += 1
#       done = terminated

#       if progress_bar:
#         print(f"\rProgress: {discount_timesteps}/{total_timesteps}", end="")

#   def predict(self, state):
#     """
#     Predict the value of a state as the maximum Q(s, a) over possible actions.
#     """
#     sampled_actions = [self.env.action_space.sample() for _ in range(10)]
#     q_values = [self.q_value(state, action) for action in sampled_actions]
#     return max(q_values)

#   def save(self, path):
#     np.save(path, self.model.theta)

#   def load(self, path):
#     self.model.theta = np.load(path + ".npy")
class GenericRLBase:
  def __init__(self, env, env_type="gym", num_envs=1):
    self.env = env
    self.env_type = env_type
    self.num_envs = num_envs

    if hasattr(env.observation_space, "shape"):
      self.observation_dim = env.observation_space.shape[0]
    else:
      raise ValueError("Unsupported observation space type.")

    if hasattr(env.action_space, "shape"):
      self.action_dim = env.action_space.shape[0]
    else:
      raise ValueError("Unsupported action space type.")

  def state_action_features(self, state, action):
    """
    Combine state and action into a feature vector. Should be overridden by subclasses.
    """
    raise NotImplementedError("state_action_features must be implemented by subclasses.")

  def predict_action(self, state):
    """
    Predict the best action for a given state. Should be overridden by subclasses.
    """
    raise NotImplementedError("predict_action must be implemented by subclasses.")

  def train_step(self, state, action, reward, next_state, next_action):
    """
    Perform a single training step. Should be overridden by subclasses.
    """
    raise NotImplementedError("train_step must be implemented by subclasses.")

  def _process_single_env(self, total_timesteps, progress_bar):
    discount_timesteps = 0
    done = True

    print("Training on a single environment...")

    while discount_timesteps < total_timesteps:
      if done:
        state, _ = self.env.reset()
        action = self.env.action_space.sample()
        done = False

      next_state, reward, terminated, _, _ = self.env.step(action)
      next_action = self.predict_action(next_state)

      self.train_step(state, action, reward, next_state, next_action)

      state, action = next_state, next_action
      discount_timesteps += 1
      done = terminated

      if progress_bar:
        print(f"\rProgress: {discount_timesteps}/{total_timesteps}", end="")

  def _process_multi_env(self, total_timesteps, progress_bar):
    progress_bar = True
    discount_timesteps = 0
    dones = [True] * self.num_envs

    print("Training on multiple environments...")

    while discount_timesteps < total_timesteps:
      if all(dones):
        states = self.env.reset()
        actions = np.array([self.env.action_space.sample() for _ in range(self.num_envs)])

      next_states, rewards, dones, infos = self.env.step(actions)
      next_actions = np.array([self.predict_action(next_states[i]) for i in range(self.num_envs)])

      for i in range(self.num_envs):
        print(states[i], actions[i], rewards[i], next_states[i], next_actions[i])
        self.train_step(states[i], actions[i], rewards[i], next_states[i], next_actions[i])

      states, actions = next_states, next_actions
      discount_timesteps += self.num_envs

      if progress_bar:
        print(f"\rProgress: {discount_timesteps}/{total_timesteps}", end="")

  def learn(self, total_timesteps=1000, progress_bar=False):
    if self.env_type == "gym":
      self._process_single_env(total_timesteps, progress_bar)
    else:
      self._process_multi_env(total_timesteps, progress_bar)

  def predict(self, state):
    """
    Predict the value of a state or the best action. Should be overridden by subclasses.
    """
    raise NotImplementedError("predict must be implemented by subclasses.")


class TOTD(GenericRLBase):
  def __init__(self, env, alpha=0.01, gamma=0.99, lambda_=0.9, theta_init=None, env_type="gym", num_envs=1):
    super().__init__(env, env_type, num_envs)
    self.alpha = alpha
    self.gamma = gamma
    self.lambda_ = lambda_

    self.feature_dim = self.observation_dim + self.action_dim
    self.theta = theta_init or np.zeros(self.feature_dim)
    self.model = TrueOnlineTDLambdaReplay(alpha, gamma, lambda_, self.theta)

  def state_action_features(self, state, action):
    state = np.asarray(state, dtype=np.float32).flatten()
    action = np.asarray(action, dtype=np.float32).flatten()
    return np.concatenate([state, action])

  def predict_action(self, state):
    print("Predicting action...")
    sampled_actions = [self.env.action_space.sample() for _ in range(10)]
    print(sampled_actions)
    q_values = [self.q_value(state, action) for action in sampled_actions]
    best_action = sampled_actions[np.argmax(q_values)]
    return np.clip(np.asarray(best_action, dtype=np.float32), self.env.action_space.low, self.env.action_space.high)

  def q_value(self, state, action):
    phi = self.state_action_features(state, action)
    return np.dot(self.theta, phi)

  def train_step(self, state, action, reward, next_state, next_action):
    phi = self.state_action_features(state, action)
    next_phi = self.state_action_features(next_state, next_action)
    self.model.train_step(phi, reward, next_phi)

    


class Model(BaseModel):
  def __init__(self, env_name, num_envs=16, max_episode_steps=5000, reward_strategy="default", **kwargs):
    """
    Initialize the Model class specifically for PPO.
    Args:
        env_name (str): The name of the environment to train on.
        num_envs (int): Number of parallel environments to use.
        kwargs: Additional arguments to pass to the PPO initialization.
    """
    super().__init__(env_name, num_envs, max_episode_steps, reward_strategy, **kwargs)
    self.model = TOTD(self.env, env_type=self.env_type, ** self.parameters)


if __name__ == "__main__":
  # Single environment with TOTD
  env = gym.make("LunarLanderContinuous-v3")
  totd_model = TOTD(env, env_type="gym")
  totd_model.learn(total_timesteps=1000)

  class ZTOTD(TOTD):
    def __init__(self, env, alpha=0.005, gamma=0.95, lambda_=0.8, theta_init=None, env_type="gym", num_envs=1):
      super().__init__(env, alpha, gamma, lambda_, theta_init, env_type, num_envs)
      # Additional ZTOTD-specific initialization here

  # Multi-environment with ZTOTD
  from stable_baselines3.common.vec_env import SubprocVecEnv
  vec_env = SubprocVecEnv([lambda: gym.make("LunarLanderContinuous-v3") for _ in range(4)])
  ztotd_model = ZTOTD(vec_env, env_type="vec", num_envs=4)
  ztotd_model.learn(total_timesteps=1000)
