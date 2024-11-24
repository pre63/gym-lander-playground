import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random


class ReplayBuffer:
  def __init__(self, buffer_size=100000, batch_size=256):
    self.buffer = deque(maxlen=buffer_size)
    self.batch_size = batch_size

  def store(self, state, action, reward, next_state, done):
    self.buffer.append((state, action, reward, next_state, done))

  def sample(self):
    batch = random.sample(self.buffer, min(len(self.buffer), self.batch_size))
    states, actions, rewards, next_states, dones = zip(*batch)
    return (
        torch.FloatTensor(states),
        torch.FloatTensor(actions),
        torch.FloatTensor(rewards).unsqueeze(1),
        torch.FloatTensor(next_states),
        torch.FloatTensor(dones).unsqueeze(1),
    )

  def size(self):
    return len(self.buffer)


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
  def __init__(self, env, buffer_size=100000, batch_size=256, gamma=0.99, tau=0.005, alpha=0.2, actor_lr=3e-4, critic_lr=3e-4):
    self.env = env
    self.state_dim = env.observation_space.shape[0]
    self.action_dim = env.action_space.shape[0]
    self.max_action = env.action_space.high[0]

    # Networks
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    self.actor = Actor(self.state_dim, self.action_dim, self.max_action).to(device)
    self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)

    self.critic = Critic(self.state_dim, self.action_dim).to(device)
    self.target_critic = Critic(self.state_dim, self.action_dim).to(device)
    self.target_critic.load_state_dict(self.critic.state_dict())
    self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)

    # Mean Actor
    self.mean_actor = Actor(self.state_dim, self.action_dim, self.max_action).to(device)
    self.mean_actor.load_state_dict(self.actor.state_dict())

    # Replay Buffer
    self.replay_buffer = ReplayBuffer(buffer_size, batch_size)

    # Hyperparameters
    self.gamma = gamma
    self.tau = tau
    self.alpha = alpha

  def store_transition(self, state, action, reward, next_state, done):
    self.replay_buffer.store(state, action, reward, next_state, done)

  def soft_update(self, target_net, source_net):
    for target_param, param in zip(target_net.parameters(), source_net.parameters()):
      target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

  def train_step(self):
    if self.replay_buffer.size() < self.replay_buffer.batch_size:
      return

    states, actions, rewards, next_states, dones = self.replay_buffer.sample()

    # Critic update
    with torch.no_grad():
      next_actions = self.mean_actor(next_states)  # Use mean actor for stability
      target_q_values = rewards + self.gamma * (1 - dones) * self.target_critic(next_states, next_actions)

    current_q_values = self.critic(states, actions)
    critic_loss = nn.MSELoss()(current_q_values, target_q_values)

    self.critic_optimizer.zero_grad()
    critic_loss.backward()
    self.critic_optimizer.step()

    # Actor update with mean regularization
    actions = self.actor(states)
    actor_loss = -self.critic(states, actions).mean()

    # Mean regularization
    mean_actor_actions = self.mean_actor(states).detach()
    regularization = nn.MSELoss()(actions, mean_actor_actions)
    total_actor_loss = actor_loss + self.alpha * regularization

    self.actor_optimizer.zero_grad()
    total_actor_loss.backward()
    self.actor_optimizer.step()

    # Soft update of target networks and mean actor
    self.soft_update(self.target_critic, self.critic)
    self.soft_update(self.mean_actor, self.actor)

  def train(self):
    """
    Run an episode of training and return the episode reward and rewards per step.

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
      next_state, reward, terminated, truncated, _ = self.env.step(action)
      done = terminated or truncated

      self.store_transition(state, action, reward, next_state, done)
      self.train_step()

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
