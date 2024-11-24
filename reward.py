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
  distance_to_target = np.linalg.norm(state[:2])  # Assume first two state dimensions are position
  vertical_velocity = abs(state[2])  # Assume state[2] is vertical velocity
  horizontal_velocity = abs(state[3])  # Assume state[3] is horizontal velocity

  reward = -distance_to_target  # Penalize distance from the target
  reward -= 10 * vertical_velocity  # Penalize high vertical speed
  reward -= 5 * horizontal_velocity  # Penalize high horizontal speed

  if done:
    if info.get("landed_successfully", False):  # Check for landing success
      reward += 100  # Bonus for successful landing
    else:
      reward -= 100  # Penalty for crashing

  return reward


def energy_efficient_reward(state, reward, action, done, info):
  """
  Reward strategy prioritizing energy efficiency during landing.
  """
  fuel_usage = np.linalg.norm(action)  # Action magnitude as a proxy for fuel usage
  distance_to_target = np.linalg.norm(state[:2])
  reward = -distance_to_target
  reward -= 0.1 * fuel_usage  # Penalize fuel usage

  if done:
    if info.get("landed_successfully", False):
      reward += 100
    else:
      reward -= 100

  return reward


def combined_reward(state, reward, action, done, info):
  """
  Combines proximity and energy efficiency strategies.
  """
  proximity = proximity_reward(state, reward, action, done, info)
  efficiency = energy_efficient_reward(state, reward, action, done, info)
  return 0.7 * proximity + 0.3 * efficiency


# Registry for reward strategies
REWARD_STRATEGIES = {
    "default": default_reward,
    "proximity": proximity_reward,
    "energy_efficient": energy_efficient_reward,
    "combined": combined_reward,
}
