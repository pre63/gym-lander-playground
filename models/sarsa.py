import numpy as np
import gymnasium as gym
from gymnasium import spaces
from itertools import product

from models.iht import IHT, tiles
from models.base import BaseModel, BaseConvertStableBaselinesModel


class Sarsa:
  def __init__(self, action_space, alpha=0.01, epsilon=0.1, gamma=1, tilings=20, max_episode_steps=10000, action_resolution=3):
    self.alpha = alpha
    self.epsilon = epsilon
    self.gamma = gamma
    self.tilings = tilings
    self.max_episode_steps = max_episode_steps
    self.action_space_low, self.action_space_high = action_space

    self.action_values = np.array(
        list(product(
            *[np.linspace(low, high, action_resolution) for low, high in zip(self.action_space_low, self.action_space_high)]
        ))
    )
    self.w = np.random.uniform(low=-0.05, high=0.05, size=(tilings**4,))
    iht_size = int(tilings**4)
    print(f"Size of IHT: {iht_size}")
    self.tile_coding = IHT(iht_size)

  def q_(self, feature):
    return np.dot(self.w, feature)

  def update(self, reward, current_q, future_q, feature, terminal):
    if terminal:
      w_update = self.alpha * (reward - current_q)
    else:
      w_update = self.alpha * (reward + self.gamma * future_q - current_q)
    self.w += np.multiply(w_update, feature)

  def one_hot_encode(self, indices):
    size = len(self.w)
    one_hot_vec = np.zeros(size)
    for i in indices:
      one_hot_vec[i] = 1
    return one_hot_vec

  def hash(self, state, action):
    feature_ind = np.array(tiles(self.tile_coding, self.tilings, state.tolist(), action.tolist()))
    feature = self.one_hot_encode(feature_ind)
    return feature

  def choose_action(self, state):
    action_val_dict = {}
    for action in self.action_values:
      feature = self.hash(state, action)
      q = self.q_(feature)
      action_val_dict[tuple(action)] = q  # Store actions as tuples (hashable for dict keys)

    greedy_action = max(action_val_dict, key=action_val_dict.get)
    non_greedy_actions = [np.array(a) for a in set(action_val_dict.keys()) - {greedy_action}]

    prob_explorative_action = self.epsilon / len(self.action_values)
    prob_greedy_action = 1 - self.epsilon + prob_explorative_action

    # Prepare options and probabilities
    actions = [np.array(greedy_action)] + non_greedy_actions
    probabilities = [prob_greedy_action] + [prob_explorative_action] * len(non_greedy_actions)

    chosen_action_index = np.random.choice(range(len(actions)), p=probabilities)
    chosen_action = actions[chosen_action_index]

    return chosen_action, action_val_dict[tuple(chosen_action)]

  def decay_epsilon(self):
    rate = max(self.epsilon - (1.0 / (self.max_episode_steps)), 0)
    self.epsilon = self.epsilon * rate

  def step(self, state, _, reward, next_state, done):
    action, q = self.choose_action(state)
    feature = self.hash(state, action)
    if done:
      self.update(reward, q, None, feature, True)
    else:
      next_action, next_q = self.choose_action(next_state)
      self.update(reward, q, next_q, feature, False)

  def get_action(self, state):
    action, q = self.choose_action(state)
    action = np.clip(action, self.action_space_low, self.action_space_high)
    action = action.flatten()
    return np.array(action), q

  def save(self, filename):
    # Save parameters, weights, and tile coding
    np.savez(
        filename,
        params={
            "alpha": self.alpha,
            "epsilon": self.epsilon,
            "gamma": self.gamma,
            "tilings": self.tilings,
            "max_episode_steps": self.max_episode_steps,
            "action_space_low": self.action_space_low.tolist(),
            "action_space_high": self.action_space_high.tolist()
        },
        weights=self.w,
        tile_coding=self.tile_coding.dictionary
    )

  def load(self, filename):
      # Load parameters, weights, and tile coding
    data = np.load(filename + ".npz", allow_pickle=True)
    params = data["params"].item()

    # Assign parameters
    self.alpha = params["alpha"]
    self.epsilon = params["epsilon"]
    self.gamma = params["gamma"]
    self.tilings = params["tilings"]
    self.max_episode_steps = params["max_episode_steps"]
    self.action_space_low = np.array(params["action_space_low"])
    self.action_space_high = np.array(params["action_space_high"])

    # Reinitialize dependent values
    self.action_values = np.array(
        list(product(
            *[np.linspace(low, high, len(self.action_space_low))
              for low, high in zip(self.action_space_low, self.action_space_high)]
        ))
    )
    self.w = data["weights"]
    self.tile_coding = IHT(self.tilings**4)
    self.tile_coding.dictionary = data["tile_coding"].item()


class SARSA(BaseConvertStableBaselinesModel):
  def __init__(self, env, alpha=0.01, gamma=0.99, lambda_=0.9, env_type="gym", num_envs=1, max_episode_steps=10000):
    super().__init__(env, env_type, num_envs)
    self.alpha = alpha
    self.gamma = gamma
    self.lambda_ = lambda_

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    self.model = Sarsa(
        (env.action_space.low, env.action_space.high),
        alpha=alpha,
        gamma=gamma,
        max_episode_steps=max_episode_steps,
        action_resolution=3
    )

  def train_step(self, state, action, reward, next_state, next_action, done, info):
    if isinstance(action, (float, int)):
      action = np.array([action])

    if isinstance(next_action, (float, int)):
      next_action = np.array([next_action])

    self.model.step(state, action, reward, next_state, done)

  def predict(self, state):
    action, q = self.model.get_action(state)
    action = np.clip(action, self.env.action_space.low, self.env.action_space.high)
    action = np.squeeze(action)  # Ensure action has the correct dimensions
    return action, q

  def learn_episode_setup(self):
    pass

  def save(self, filename):
    self.model.save(filename)

  def load(self, filename):
    self.model.load(filename)


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
    self.model = SARSA(self.env, env_type=self.env_type, max_episode_steps=max_episode_steps, ** self.parameters)


if __name__ == "__main__":
  env = gym.make("MountainCarContinuous-v0", render_mode="human")
  # Initialize critic and actor-critic models
  totdlr = SARSA(env, alpha=0.01, gamma=0.99, lambda_=0.9)
  totdlr.learn(1000, max_episode_steps=1000, render_frequency=100)
