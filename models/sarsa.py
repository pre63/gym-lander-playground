import numpy as np
import gymnasium as gym


class Model:
  def __init__(self, env: gym.Env):
    """
    Initialize the SARSA model.
    Args:
        env (gym.Env): The environment to train on.
    """
    self.env = env
    self.q_table = np.zeros((self.env.observation_space.n, self.env.action_space.n))
    self.learning_rate = 0.1
    self.discount_factor = 0.99
    self.epsilon = 0.1  # For ε-greedy policy

  def epsilon_greedy_action(self, state):
    """
    Select an action using the ε-greedy policy.
    """
    if np.random.rand() < self.epsilon:
      return self.env.action_space.sample()
    else:
      return np.argmax(self.q_table[state])

  def train(self):
    """
    Train the model for one episode and return the episode reward.
    Returns:
        episode_reward (float): Total reward obtained in the episode.
        trajectory (list): The trajectory of the agent..
    """
    state, _ = self.env.reset()
    action = self.epsilon_greedy_action(state)
    done = False
    episode_reward = 0
    trajectory = []

    while not done:
      # Take the action and observe the next state and reward
      next_state, reward, terminated, truncated, _ = self.env.step(action)
      done = terminated or truncated

      # Choose the next action using the ε-greedy policy
      next_action = self.epsilon_greedy_action(next_state)

      # Update Q-value using the SARSA formula
      td_target = reward + self.discount_factor * self.q_table[next_state, next_action]
      self.q_table[state, action] += self.learning_rate * (td_target - self.q_table[state, action])

      # Update state and action
      state, action = next_state, next_action

      # Accumulate rewards
      trajectory.append({
          "state": state.tolist(),
          "action": action.tolist(),
          "reward": reward,
          "next_state": next_state.tolist(),
          "done": done
      })
      episode_reward += reward

    return episode_reward, trajectory
