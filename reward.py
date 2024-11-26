import numpy as np
import gymnasium as gym
from success import check_success


class RewardWrapper(gym.Wrapper):
  def __init__(self, env, reward_strategy):
    super().__init__(env)

    self.reward_strategy = reward_strategy

    if reward_strategy not in REWARD_STRATEGIES:
      raise ValueError(f"Unknown reward strategy: {reward_strategy}")

    self.reward_fn = REWARD_STRATEGIES[reward_strategy]

  def step(self, action):
    state, reward, terminated, truncated, info = self.env.step(action)
    custom_reward = self.reward_fn(state, reward, action, terminated or truncated, info)
    return state, custom_reward, terminated, truncated, info


def default_reward(state, reward, action, done, info):
  """
  Default reward strategy from the environment.
  """
  if done:
    success = check_success(state, done)
    if success:
      reward = max(reward + 200.0, 200.0)

  return reward


def proximity_reward(state, reward, action, done, info):
  """
  Reward strategy prioritizing proximity to the target and low velocity.
  """
  x_position = observation[0]

  if done:
    success = check_success(state, done)
    if success:
      award = 200.0 + -x_position**2
      reward = max(reward + award, 200.0)
    else:
      reward = reward - 100.0

  return reward


def energy_efficient_reward(state, reward, action, done, info):
  """
  Reward strategy prioritizing energy efficiency during landing.
  """
  fuel_usage = np.linalg.norm(action)  # Action magnitude as a proxy for fuel usage
  x_position = state[0]

  if fuel_usage > 1.0:  # Penalize only excessive fuel usage
    reward -= 0.1 * (fuel_usage - 1.0)

  if done:
    success = check_success(state, done)
    if success:
      award = 200.0 + -x_position**2 * 0.1 * (fuel_usage - 1.0)

      reward = max(reward + award, 200.0)
    else:
      reward = reward - 100.0

  return reward


def combined_reward(state, reward, action, done, info):
  """
  Combines proximity and energy efficiency strategies.
  """
  proximity = proximity_reward(state, reward, action, done, info)
  efficiency = energy_efficient_reward(state, reward, action, done, info)

  # Adjust weights dynamically based on proximity
  distance_to_target = np.linalg.norm(state[:2])
  proximity_weight = 0.7 if distance_to_target > 1.0 else 0.9
  efficiency_weight = 1 - proximity_weight

  return proximity_weight * proximity + efficiency_weight * efficiency


# Registry for reward strategies
REWARD_STRATEGIES = {
    "default": default_reward,
    "proximity": proximity_reward,
    "energy": energy_efficient_reward,
    "combined": combined_reward,
}
