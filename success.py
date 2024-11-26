import numpy as np


def check_success(observation,
                  terminated,
                  angle_threshold=0.1,
                  landing_pad_threshold=0.2,
                  x_velocity_threshold=0.01,
                  y_velocity_threshold=0.01):
  """
  Determine if the agent has successfully landed in the Continuous Lunar Lander environment.

  Parameters:
  - observation: list or array containing the state values from the environment.
  - terminated: boolean indicating if the episode has ended in a terminal state.
  - angle_threshold: maximum allowable angle (in radians) for success.
  - landing_pad_threshold: maximum allowable horizontal distance from the landing pad center.
  - x_velocity_threshold: maximum allowable horizontal velocity.
  - y_velocity_threshold: maximum allowable vertical velocity.

  Returns:
  - success: boolean indicating if the agent has successfully landed.
  """
  # Extract values from the observation
  x_position = observation[0]
  y_position = observation[1]
  x_velocity = observation[2]
  y_velocity = observation[3]
  angle = observation[4]
  angular_velocity = observation[5]
  leg_contact_left = observation[6]
  leg_contact_right = observation[7]

  # Define success criteria
  lander_is_upright = abs(angle) < angle_threshold
  lander_within_landing_pad = abs(x_position) < landing_pad_threshold
  velocities_within_limits = abs(x_velocity) < x_velocity_threshold and abs(y_velocity) < y_velocity_threshold
  legs_in_contact = leg_contact_left and leg_contact_right

  # Success condition
  success = terminated and lander_is_upright and lander_within_landing_pad and velocities_within_limits and legs_in_contact and y_position >= 0

  return success
