import numpy as np
import random

import torch
import torch.nn as nn
import torch.optim as optim

from collections import deque

from success import check_success
from telemetry import add_telemetry_overlay, add_success_failure_to_frames


class ReplayBuffer:
  def __init__(self, buffer_size=100000, batch_size=256, large_batch_multiplier=4):
    self.buffer = deque(maxlen=buffer_size)
    self.batch_size = batch_size
    self.large_batch_size = batch_size * large_batch_multiplier

  def store(self, state, action, reward, next_state, done):
    self.buffer.append((state, action, reward, next_state, done))

  def sample_large_batch(self):
    return random.sample(self.buffer, min(len(self.buffer), self.large_batch_size))

  def sample_prioritized_batch(self, priorities):
    probabilities = priorities / np.sum(priorities)
    indices = np.random.choice(len(priorities), size=self.batch_size, p=probabilities)
    return indices

  def size(self):
    return len(self.buffer)

  def get_samples_by_indices(self, indices):
    return [self.buffer[i] for i in indices]


class Actor(nn.Module):
  def __init__(self, state_dim, action_dim, max_action):
    super(Actor, self).__init__()
    self.net = nn.Sequential(
        nn.Linear(state_dim, 256),
        nn.ReLU(),
        nn.Linear(256, 256),
        nn.ReLU(),
        nn.Linear(256, action_dim),
        nn.Tanh()
    )
    self.max_action = max_action

  def forward(self, state):
    return self.net(state) * self.max_action


class Critic(nn.Module):
  def __init__(self, state_dim, action_dim):
    super(Critic, self).__init__()
    self.net = nn.Sequential(
        nn.Linear(state_dim + action_dim, 256),
        nn.ReLU(),
        nn.Linear(256, 256),
        nn.ReLU(),
        nn.Linear(256, 1)
    )

  def forward(self, state, action):
    return self.net(torch.cat([state, action], dim=1))


class Model:
  def __init__(self, env, buffer_size=100000, batch_size=256, gamma=0.99, tau=0.005, actor_lr=3e-4, critic_lr=3e-4, alpha=0.2):

    self.parameters = {
        "buffer_size": buffer_size,
        "batch_size": batch_size,
        "gamma": gamma,
        "tau": tau,
        "actor_lr": actor_lr,
        "critic_lr": critic_lr,
        "alpha": alpha
    }

    self.env = env
    self.state_dim = env.observation_space.shape[0]
    self.action_dim = env.action_space.shape[0]
    self.max_action = env.action_space.high[0]

    # Device setup
    self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Networks
    self.actor = Actor(self.state_dim, self.action_dim, self.max_action).to(self.device)
    self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)

    self.critic = Critic(self.state_dim, self.action_dim).to(self.device)
    self.target_critic = Critic(self.state_dim, self.action_dim).to(self.device)
    self.target_critic.load_state_dict(self.critic.state_dict())
    self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)

    # Replay Buffer
    self.replay_buffer = ReplayBuffer(buffer_size, batch_size)

    # Hyperparameters
    self.gamma = gamma
    self.tau = tau
    self.alpha = alpha

  def store_transition(self, state, action, reward, next_state, done):
    self.replay_buffer.store(state, action, reward, next_state, done)

  def sample_large_batch(self):
    return self.replay_buffer.sample_large_batch()

  def train(self):
    """
    Train the model for one episode and return the episode reward and history.

    Returns:
        episode_reward (float): Total reward obtained in the episode.
        history (list): The history of the agent.
    """
    state, _ = self.env.reset()
    state = torch.FloatTensor(state).to(self.device)
    done = False
    episode_reward = 0
    history = []

    while not done:
      # Select an action
      state_tensor = state.unsqueeze(0)
      action = self.actor(state_tensor).detach().cpu().numpy()[0]
      next_state, reward, terminated, truncated, _ = self.env.step(action)
      next_state = torch.FloatTensor(next_state).to(self.device)
      done = terminated or truncated

      # Store transition in the replay buffer
      self.store_transition(state.cpu().numpy(), action, reward, next_state.cpu().numpy(), done)

      # Perform training step
      self.train_step()

      # Update state and rewards
      history.append({
          "state": state.cpu().numpy().tolist(),
          "action": action.tolist(),
          "reward": reward,
          "episode_reward": episode_reward,
          "next_state": next_state.cpu().numpy().tolist(),
          "done": done
      })
      state = next_state
      episode_reward += reward

    success = check_success(next_state, terminated)
    return success, episode_reward, history

  def train_step(self):
    """
    Perform a single training step using a prioritized replay buffer.
    """
    if self.replay_buffer.size() < self.replay_buffer.batch_size:
      return

    # Sample large batch
    large_batch = self.sample_large_batch()
    states, actions, rewards, next_states, dones = zip(*large_batch)

    # Convert to tensors and move to device
    states = torch.FloatTensor(np.array(states)).to(self.device)
    actions = torch.FloatTensor(np.array(actions)).to(self.device)
    rewards = torch.FloatTensor(np.array(rewards)).unsqueeze(1).to(self.device)
    next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
    dones = torch.FloatTensor(np.array(dones)).unsqueeze(1).to(self.device)

    # Compute priorities
    with torch.no_grad():
      target_q_values = rewards + self.gamma * (1 - dones) * self.target_critic(next_states, self.actor(next_states))
    current_q_values = self.critic(states, actions)
    td_errors = torch.abs(current_q_values - target_q_values).detach().cpu().numpy()
    priorities = td_errors.flatten()

    # Down-sample to prioritized batch
    prioritized_indices = self.replay_buffer.sample_prioritized_batch(priorities)
    prioritized_samples = self.replay_buffer.get_samples_by_indices(prioritized_indices)
    states, actions, rewards, next_states, dones = zip(*prioritized_samples)

    # Convert to tensors and move to device again
    states = torch.FloatTensor(np.array(states)).to(self.device)
    actions = torch.FloatTensor(np.array(actions)).to(self.device)
    rewards = torch.FloatTensor(np.array(rewards)).unsqueeze(1).to(self.device)
    next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
    dones = torch.FloatTensor(np.array(dones)).unsqueeze(1).to(self.device)

    # Critic update
    with torch.no_grad():
      target_q_values = rewards + self.gamma * (1 - dones) * self.target_critic(next_states, self.actor(next_states))
    current_q_values = self.critic(states, actions)
    critic_loss = nn.MSELoss()(current_q_values, target_q_values)

    self.critic_optimizer.zero_grad()
    critic_loss.backward()
    self.critic_optimizer.step()

    # Actor update
    actor_loss = -self.critic(states, self.actor(states)).mean()
    self.actor_optimizer.zero_grad()
    actor_loss.backward()
    self.actor_optimizer.step()

    # Soft update of target critic
    for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
      target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

  def save(self, filename):
    """
    Save the model parameters to a file.

    Args:
        filename (str): Path to the file.
    """
    torch.save({
        "actor": self.actor.state_dict(),
        "critic": self.critic.state_dict(),
        "actor_optimizer": self.actor_optimizer.state_dict(),
        "critic_optimizer": self.critic_optimizer.state_dict(),
        "parameters": self.parameters
    }, filename)

  def load(self, filename):
    """
    Load the model parameters from a file.

    Args:
        filename (str): Path to the file.
    """
    checkpoint = torch.load(filename)
    self.actor.load_state_dict(checkpoint["actor"])
    self.critic.load_state_dict(checkpoint["critic"])
    self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])
    self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer"])
    self.parameters = checkpoint["parameters"]

  def evaluate(self, render=False):
    """
    Run the algorithm without training and return success, rewards, and frames.

    Args:
        render (bool): Whether to render the environment during evaluation.

    Returns:
        success (bool): Whether the evaluation was successful based on the defined criteria.
        episode_reward (float): Total reward obtained in the episode.
        frames (list): List of frames (empty if render is False).
    """
    state, _ = self.env.reset()
    state = torch.FloatTensor(state).to(self.device)
    done = False
    episode_reward = 0
    frames = []

    while not done:
      # Select action using the current policy
      state_tensor = state.unsqueeze(0)
      action = self.actor(state_tensor).detach().cpu().numpy()[0]
      action = np.clip(action, -self.max_action, self.max_action)

      next_state, reward, terminated, truncated, info = self.env.step(action)
      next_state = torch.FloatTensor(next_state).to(self.device)
      done = terminated or truncated

      episode_reward += reward

      if render:
        frame = add_telemetry_overlay(self.env.render(), next_state)
        frames.append(frame)

      state = next_state

    # Define success condition
    success = check_success(next_state, terminated)
    if render:
      frames = add_success_failure_to_frames(frames, success)

    return success, episode_reward, frames
