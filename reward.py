# reward.py
import numpy as np


def default_reward(state, reward, action, done, info):
  """
  Default reward strategy from the environment.
  """
  return reward


def proximity_reward(state, reward, action, done, info):
  """
  Reward strategy prioritizing proximity to the target and low velocity.
  """
  distance_to_target = np.linalg.norm(state[:2])
  vertical_velocity = abs(state[2])
  horizontal_velocity = abs(state[3])

  reward = -distance_to_target**2  # Quadratic penalty for distance
  reward -= 10 * vertical_velocity * (1 / (distance_to_target + 1))  # Scale penalty with proximity
  reward -= 5 * horizontal_velocity * (1 / (distance_to_target + 1))

  if done:
    if info.get("landed_successfully", False):
      reward += 100 - 2 * (vertical_velocity + horizontal_velocity)  # Scaled success bonus
    else:
      reward -= 100

  return reward


def energy_efficient_reward(state, reward, action, done, info):
  """
  Reward strategy prioritizing energy efficiency during landing.
  """
  fuel_usage = np.linalg.norm(action)  # Action magnitude as a proxy for fuel usage
  distance_to_target = np.linalg.norm(state[:2])

  reward = -distance_to_target**2  # Quadratic penalty for distance
  if fuel_usage > 1.0:  # Penalize only excessive fuel usage
    reward -= 0.1 * (fuel_usage - 1.0)

  if done:
    if info.get("landed_successfully", False):
      reward += 100 - 0.1 * fuel_usage  # Success bonus scaled by fuel efficiency
    else:
      reward -= 100

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
    "energy_efficient": energy_efficient_reward,
    "combined": combined_reward,
}
