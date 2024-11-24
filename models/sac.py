import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
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
    )
    self.mean_layer = nn.Linear(256, action_dim)
    self.log_std_layer = nn.Linear(256, action_dim)
    self.max_action = max_action

  def forward(self, state):
    x = self.net(state)
    mean = self.mean_layer(x)
    log_std = self.log_std_layer(x).clamp(-20, 2)
    return mean, log_std

  def sample(self, state):
    mean, log_std = self(state)
    std = log_std.exp()
    normal = torch.distributions.Normal(mean, std)
    x_t = normal.rsample()  # Reparameterization trick
    action = torch.tanh(x_t) * self.max_action
    log_prob = normal.log_prob(x_t).sum(dim=-1)
    log_prob -= torch.log(1 - action.pow(2) + 1e-6).sum(dim=-1)
    return action, log_prob


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
  def __init__(self, env: gym.Env, buffer_size=1000000, batch_size=256, gamma=0.99, tau=0.005, alpha=0.2, actor_lr=3e-4, critic_lr=3e-4, target_entropy=None):
    """
    Initialize the Soft Actor-Critic (SAC) model.
    Args:
        env (gym.Env): The environment to train on.
        buffer_size (int): Maximum size of the replay buffer.
        batch_size (int): Number of samples per training batch.
        gamma (float): Discount factor.
        tau (float): Soft update factor for target networks.
        alpha (float): Temperature parameter for entropy regularization.
        actor_lr (float): Learning rate for the actor network.
        critic_lr (float): Learning rate for the critic networks.
        target_entropy (float): Target entropy for the policy.
    """
    self.env = env
    self.state_dim = env.observation_space.shape[0]
    self.action_dim = env.action_space.shape[0]
    self.max_action = env.action_space.high[0]

    # Device setup
    self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Actor
    self.actor = Actor(self.state_dim, self.action_dim, self.max_action).to(self.device)
    self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)

    # Critics
    self.critic_1 = Critic(self.state_dim, self.action_dim).to(self.device)
    self.critic_2 = Critic(self.state_dim, self.action_dim).to(self.device)
    self.critic_target_1 = Critic(self.state_dim, self.action_dim).to(self.device)
    self.critic_target_2 = Critic(self.state_dim, self.action_dim).to(self.device)
    self.critic_target_1.load_state_dict(self.critic_1.state_dict())
    self.critic_target_2.load_state_dict(self.critic_2.state_dict())
    self.critic_optimizer_1 = optim.Adam(self.critic_1.parameters(), lr=critic_lr)
    self.critic_optimizer_2 = optim.Adam(self.critic_2.parameters(), lr=critic_lr)

    # Replay buffer
    self.buffer = deque(maxlen=buffer_size)
    self.batch_size = batch_size

    # Hyperparameters
    self.gamma = gamma
    self.tau = tau
    self.alpha = alpha
    self.target_entropy = target_entropy or -np.prod(env.action_space.shape).item()

  def store_transition(self, state, action, reward, next_state, done):
    self.buffer.append((state, action, reward, next_state, done))

  def sample_replay_buffer(self):
    batch = random.sample(self.buffer, min(len(self.buffer), self.batch_size))
    batch = np.array(batch, dtype=object)  # Convert list of tuples to a numpy array for efficient slicing

    # Efficient slicing for each component
    states = np.stack(batch[:, 0])  # Stack states into a single array
    actions = np.stack(batch[:, 1])  # Stack actions
    rewards = np.array(batch[:, 2], dtype=np.float32).reshape(-1, 1)  # Convert rewards to float and reshape
    next_states = np.stack(batch[:, 3])  # Stack next states
    dones = np.array(batch[:, 4], dtype=np.float32).reshape(-1, 1)  # Convert dones to float and reshape

    # Convert to tensors and move to the correct device
    return (
        torch.from_numpy(states).to(self.device),
        torch.from_numpy(actions).to(self.device),
        torch.from_numpy(rewards).to(self.device),
        torch.from_numpy(next_states).to(self.device),
        torch.from_numpy(dones).to(self.device),
    )

  def update(self):
    if len(self.buffer) < self.batch_size:
      return

    states, actions, rewards, next_states, dones = self.sample_replay_buffer()

    # Update critics
    with torch.no_grad():
      next_actions, next_log_probs = self.actor.sample(next_states)
      target_q1 = self.critic_target_1(next_states, next_actions)
      target_q2 = self.critic_target_2(next_states, next_actions)
      target_q = torch.min(target_q1, target_q2) - self.alpha * next_log_probs
      target_value = rewards + self.gamma * (1 - dones) * target_q

    current_q1 = self.critic_1(states, actions)
    current_q2 = self.critic_2(states, actions)
    critic_loss_1 = nn.MSELoss()(current_q1, target_value)
    critic_loss_2 = nn.MSELoss()(current_q2, target_value)

    self.critic_optimizer_1.zero_grad()
    critic_loss_1.backward()
    self.critic_optimizer_1.step()

    self.critic_optimizer_2.zero_grad()
    critic_loss_2.backward()
    self.critic_optimizer_2.step()

    # Update actor
    actions, log_probs = self.actor.sample(states)
    actor_loss = (self.alpha * log_probs - self.critic_1(states, actions)).mean()

    self.actor_optimizer.zero_grad()
    actor_loss.backward()
    self.actor_optimizer.step()

    # Soft update target networks
    for target_param, param in zip(self.critic_target_1.parameters(), self.critic_1.parameters()):
      target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
    for target_param, param in zip(self.critic_target_2.parameters(), self.critic_2.parameters()):
      target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

  def train(self):
    """
    Train the model for one episode and return the episode reward.
    Returns:
        episode_reward (float): Total reward obtained in the episode.
        trajectory (list): The trajectory of the agent..
    """
    state, _ = self.env.reset()
    state = torch.FloatTensor(state).to(self.device)
    done = False
    episode_reward = 0
    trajectory = []

    while not done:
      state_tensor = state.unsqueeze(0)
      action, _ = self.actor.sample(state_tensor)
      action = action.detach().cpu().numpy()[0]

      next_state, reward, terminated, truncated, _ = self.env.step(action)
      next_state = torch.FloatTensor(next_state).to(self.device)
      done = terminated or truncated

      self.store_transition(state.cpu().numpy(), action, reward, next_state.cpu().numpy(), done)
      self.update()

      trajectory.append({
          "state": state.cpu().numpy().tolist(),
          "action": action.tolist(),
          "reward": reward,
          "next_state": next_state.cpu().numpy().tolist(),
          "done": done
      })
      episode_reward += reward
      state = next_state

    return episode_reward, trajectory
