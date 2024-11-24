import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
from collections import deque
import random


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
    return self.max_action * self.net(state)


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
  def __init__(self, env: gym.Env, buffer_size=100000, batch_size=64, gamma=0.99, tau=0.005, actor_lr=1e-3, critic_lr=1e-3):
    """
    Initialize the Deep Deterministic Policy Gradient (DDPG) model.
    Args:
        env (gym.Env): The environment to train on.
        buffer_size (int): Maximum size of the replay buffer.
        batch_size (int): Number of samples per training batch.
        gamma (float): Discount factor.
        tau (float): Soft update factor for target networks.
        actor_lr (float): Learning rate for the actor network.
        critic_lr (float): Learning rate for the critic network.
    """
    self.env = env
    self.state_dim = env.observation_space.shape[0]
    self.action_dim = env.action_space.shape[0]
    self.max_action = env.action_space.high[0]

    # Actor and Critic networks
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    self.actor = Actor(self.state_dim, self.action_dim, self.max_action).to(device)
    self.actor_target = Actor(self.state_dim, self.action_dim, self.max_action).to(device)
    self.actor_target.load_state_dict(self.actor.state_dict())
    self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)

    self.critic = Critic(self.state_dim, self.action_dim).to(device)
    self.critic_target = Critic(self.state_dim, self.action_dim).to(device)
    self.critic_target.load_state_dict(self.critic.state_dict())
    self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)

    # Replay buffer
    self.buffer = deque(maxlen=buffer_size)
    self.batch_size = batch_size

    # Hyperparameters
    self.gamma = gamma
    self.tau = tau

  def store_transition(self, state, action, reward, next_state, done):
    self.buffer.append((state, action, reward, next_state, done))

  def sample_replay_buffer(self):
    batch = random.sample(self.buffer, min(len(self.buffer), self.batch_size))
    states, actions, rewards, next_states, dones = zip(*batch)
    return (
        torch.FloatTensor(states),
        torch.FloatTensor(actions),
        torch.FloatTensor(rewards).unsqueeze(1),
        torch.FloatTensor(next_states),
        torch.FloatTensor(dones).unsqueeze(1),
    )

  def update(self):
    if len(self.buffer) < self.batch_size:
      return

    states, actions, rewards, next_states, dones = self.sample_replay_buffer()

    # Update Critic
    with torch.no_grad():
      target_actions = self.actor_target(next_states)
      target_q = self.critic_target(next_states, target_actions)
      target_value = rewards + self.gamma * (1 - dones) * target_q

    current_q = self.critic(states, actions)
    critic_loss = nn.MSELoss()(current_q, target_value)
    self.critic_optimizer.zero_grad()
    critic_loss.backward()
    self.critic_optimizer.step()

    # Update Actor
    actor_loss = -self.critic(states, self.actor(states)).mean()
    self.actor_optimizer.zero_grad()
    actor_loss.backward()
    self.actor_optimizer.step()

    # Soft update target networks
    for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
      target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
    for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
      target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

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

    while not done:
      state_tensor = torch.FloatTensor(state).unsqueeze(0)
      action = self.actor(state_tensor).detach().numpy()[0]
      action = np.clip(action, -self.max_action, self.max_action)

      next_state, reward, terminated, truncated, _ = self.env.step(action)
      done = terminated or truncated

      self.store_transition(state, action, reward, next_state, done)
      self.update()

      trajectory.append({
          "state": state.tolist(),
          "action": action.tolist(),
          "reward": reward,
          "next_state": next_state.tolist(),
          "done": done
      })
      episode_reward += reward
      state = next_state

    return episode_reward, trajectory
