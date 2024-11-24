import numpy as np
import gymnasium as gym


class Model:
  def __init__(self, env: gym.Env, lambda_=0.9, learning_rate=0.1, discount_factor=0.99, epsilon=0.1):
    """
    Initialize the TD(λ) model.
    Args:
        env (gym.Env): The environment to train on.
        lambda_ (float): The eligibility trace decay parameter.
    """
    self.parameters = {
        "lambda_": lambda_,
        "learning_rate": learning_rate,
        "discount_factor": discount_factor,
        "epsilon": epsilon
    }
    self.env = env
    self.q_table = np.zeros((self.env.observation_space.n, self.env.action_space.n))
    self.eligibility_traces = np.zeros_like(self.q_table)
    self.learning_rate = learning_rate
    self.discount_factor = discount_factor
    self.lambda_ = lambda_
    self.epsilon = epsilon

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
    done = False
    episode_reward = 0
    trajectory = []
    frames = []

    # Reset eligibility traces
    self.eligibility_traces.fill(0)

    while not done:
      # Choose an action
      action = self.epsilon_greedy_action(state)

      # Take the action, observe the next state and reward
      next_state, reward, terminated, truncated, _ = self.env.step(action)
      done = terminated or truncated

      # Compute TD error
      best_next_action = np.argmax(self.q_table[next_state])
      td_target = reward + self.discount_factor * self.q_table[next_state, best_next_action]
      td_error = td_target - self.q_table[state, action]

      # Update eligibility trace
      self.eligibility_traces[state, action] += 1

      # Update Q-values and eligibility traces for all state-action pairs
      self.q_table += self.learning_rate * td_error * self.eligibility_traces
      self.eligibility_traces *= self.discount_factor * self.lambda_

      # Update state and accumulate reward
      state = next_state
      trajectory.append({
          "state": state.tolist(),
          "action": action.tolist(),
          "reward": reward,
          "next_state": next_state.tolist(),
          "done": done
      })
      episode_reward += reward

      frames.append(self.env.render())

    return episode_reward, trajectory, frames
