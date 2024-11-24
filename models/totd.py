import torch
from collections import deque


class Model:
  def __init__(self, env, alpha=0.01, gamma=0.99, lambda_=0.9, buffer_size=10000):
    """
    True Online TD(lambda) with Replay.

    Args:
        env (gym.Env): The Gymnasium environment.
        alpha (float): Learning rate.
        gamma (float): Discount factor.
        lambda_ (float): Trace decay parameter.
        buffer_size (int): Maximum size of the replay buffer.
    """
    self.parameters = {
        "alpha": alpha,
        "gamma": gamma,
        "lambda_": lambda_,
        "buffer_size": buffer_size
    }

    self.env = env
    self.state_dim = env.observation_space.shape[0]
    self.action_dim = env.action_space.shape[0]

    self.alpha = alpha
    self.gamma = gamma
    self.lambda_ = lambda_
    self.buffer_size = buffer_size

    # Initialize weights and eligibility trace
    self.theta = torch.zeros(self.state_dim + self.action_dim, requires_grad=False)
    self.e_trace = torch.zeros(self.state_dim + self.action_dim, requires_grad=False)

    # Replay buffer
    self.replay_buffer = deque(maxlen=buffer_size)

  def store_transition(self, state, action, reward, next_state):
    """
    Store a transition in the replay buffer.
    """
    self.replay_buffer.append((state, action, reward, next_state))

  def compute_td_error(self, state, action, reward, next_state):
    """
    Compute the TD error.
    """
    sa_current = torch.cat((state, action))
    sa_next = torch.cat((next_state, torch.zeros(self.action_dim)))
    v_current = torch.dot(self.theta, sa_current)
    v_next = torch.dot(self.theta, sa_next)
    td_error = reward + self.gamma * v_next - v_current
    return td_error

  def update(self, state, action, reward, next_state):
    """
    Update weights using True Online TD(lambda).
    """
    sa_current = torch.cat((state, action))
    td_error = self.compute_td_error(state, action, reward, next_state)

    # Update eligibility trace
    self.e_trace = self.gamma * self.lambda_ * self.e_trace + sa_current

    # Update weights
    self.theta += self.alpha * td_error * self.e_trace

  def replay(self):
    """
    Perform updates using the replay buffer.
    """
    for state, action, reward, next_state in self.replay_buffer:
      self.update(state, action, reward, next_state)

  def train(self):
    """
    Train the model for one episode.

    Returns:
        episode_reward (float): Total reward obtained in the episode.
        trajectory (list): The trajectory of the agent..
    """
    state, _ = self.env.reset()
    done = False
    episode_reward = 0
    trajectory = []
    frames = []

    while not done:
      # Select a random action
      action = torch.FloatTensor(self.env.action_space.sample())

      # Take a step in the environment
      next_state, reward, terminated, truncated, _ = self.env.step(action.numpy())
      done = terminated or truncated

      # Convert to tensors
      state_tensor = torch.FloatTensor(state)
      action_tensor = torch.FloatTensor(action)
      next_state_tensor = torch.FloatTensor(next_state)

      # Store the transition and update
      self.store_transition(state_tensor, action_tensor, reward, next_state_tensor)
      self.update(state_tensor, action_tensor, reward, next_state_tensor)

      # Perform replay
      self.replay()

      # Update state and reward
      trajectory.append({
          "state": state.tolist(),
          "action": action.tolist(),
          "reward": reward,
          "next_state": next_state.tolist(),
          "done": done
      })
      episode_reward += reward
      state = next_state

      frames.append(self.env.render())

    return episode_reward, trajectory, frames
