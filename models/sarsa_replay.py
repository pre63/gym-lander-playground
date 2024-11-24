import numpy as np
import gymnasium as gym
from collections import deque
import random


class Model:
  def __init__(self, env: gym.Env, buffer_size=1000, batch_size=32):
    """
    Initialize the SARSA model with experience replay.
    Args:
        env (gym.Env): The environment to train on.
        buffer_size (int): Maximum size of the replay buffer.
        batch_size (int): Number of samples to train on from the replay buffer.
    """
    self.env = env
    self.q_table = np.zeros((self.env.observation_space.n, self.env.action_space.n))
    self.learning_rate = 0.1
    self.discount_factor = 0.99
    self.epsilon = 0.1  # For ε-greedy policy
    self.buffer = deque(maxlen=buffer_size)
    self.batch_size = batch_size

  def epsilon_greedy_action(self, state):
    """
    Select an action using the ε-greedy policy.
    """
    if np.random.rand() < self.epsilon:
      return self.env.action_space.sample()
    else:
      return np.argmax(self.q_table[state])

  def store_transition(self, transition):
    """
    Store a transition in the replay buffer.
    Args:
        transition (tuple): (state, action, reward, next_state, next_action, done)
    """
    self.buffer.append(transition)

  def sample_replay_buffer(self):
    """
    Sample a batch of transitions from the replay buffer.
    Returns:
        list: A batch of transitions.
    """
    return random.sample(self.buffer, min(len(self.buffer), self.batch_size))

  def replay(self):
    """
    Perform experience replay to update Q-values.
    """
    batch = self.sample_replay_buffer()
    for state, action, reward, next_state, next_action, done in batch:
      td_target = reward
      if not done:
        td_target += self.discount_factor * self.q_table[next_state, next_action]
      self.q_table[state, action] += self.learning_rate * (td_target - self.q_table[state, action])

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
    frames = []

    while not done:
      # Take the action and observe the next state and reward
      next_state, reward, terminated, truncated, _ = self.env.step(action)
      done = terminated or truncated

      # Choose the next action using the ε-greedy policy
      next_action = self.epsilon_greedy_action(next_state)

      # Store the transition in the replay buffer
      self.store_transition((state, action, reward, next_state, next_action, done))

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

      # Perform replay to update Q-values
      self.replay()

      frames.append(self.env.render())

    return episode_reward, trajectory, frames
