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
  def __init__(self, env: gym.Env, buffer_size=100000, batch_size=64, gamma=0.99, tau=0.005, actor_lr=1e-3, critic_lr=1e-3, policy_noise=0.2, noise_clip=0.5, policy_freq=2):
    """
    Initialize the Twin Delayed Deep Deterministic Policy Gradient (TD3) model.
    Args:
        env (gym.Env): The environment to train on.
        buffer_size (int): Maximum size of the replay buffer.
        batch_size (int): Number of samples per training batch.
        gamma (float): Discount factor.
        tau (float): Soft update factor for target networks.
        actor_lr (float): Learning rate for the actor network.
        critic_lr (float): Learning rate for the critic network.
        policy_noise (float): Stddev of noise added to target actions.
        noise_clip (float): Max magnitude of noise added to target actions.
        policy_freq (int): Frequency of policy updates relative to critic updates.
    """
    self.parameters = {
        "buffer_size": buffer_size,
        "batch_size": batch_size,
        "gamma": gamma,
        "tau": tau,
        "actor_lr": actor_lr,
        "critic_lr": critic_lr,
        "policy_noise": policy_noise,
        "noise_clip": noise_clip,
        "policy_freq": policy_freq
    }

    self.env = env
    self.state_dim = env.observation_space.shape[0]
    self.action_dim = env.action_space.shape[0]
    self.max_action = env.action_space.high[0]

    # Device setup
    self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Actor and Critic networks
    self.actor = Actor(self.state_dim, self.action_dim, self.max_action).to(self.device)
    self.actor_target = Actor(self.state_dim, self.action_dim, self.max_action).to(self.device)
    self.actor_target.load_state_dict(self.actor.state_dict())
    self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)

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
    self.policy_noise = policy_noise
    self.noise_clip = noise_clip
    self.policy_freq = policy_freq
    self.policy_update_counter = 0

  def store_transition(self, state, action, reward, next_state, done):
    self.buffer.append((state, action, reward, next_state, done))

  def sample_replay_buffer(self):
    batch = random.sample(self.buffer, min(len(self.buffer), self.batch_size))
    states, actions, rewards, next_states, dones = zip(*batch)

    # Convert to numpy arrays for efficiency
    np_states = np.stack(states)  # Use np.stack to ensure consistent array shapes
    np_actions = np.stack(actions)
    np_rewards = np.array(rewards)  # 1D array is sufficient for rewards
    np_next_states = np.stack(next_states)
    np_dones = np.array(dones, dtype=np.float32)  # Convert to float for compatibility

    # Convert to tensors and move to the correct device
    return (
        torch.FloatTensor(np_states).to(self.device),
        torch.FloatTensor(np_actions).to(self.device),
        torch.FloatTensor(np_rewards).unsqueeze(1).to(self.device),  # Add dimension for reward
        torch.FloatTensor(np_next_states).to(self.device),
        torch.FloatTensor(np_dones).unsqueeze(1).to(self.device),  # Add dimension for done flags
    )

  def update(self):
    if len(self.buffer) < self.batch_size:
      return

    states, actions, rewards, next_states, dones = self.sample_replay_buffer()

    # Add noise to target actions
    noise = torch.normal(0, self.policy_noise, size=actions.shape, device=self.device).clamp(-self.noise_clip, self.noise_clip)
    target_actions = (self.actor_target(next_states) + noise).clamp(-self.max_action, self.max_action)

    # Compute target Q-values
    with torch.no_grad():
      target_q1 = self.critic_target_1(next_states, target_actions)
      target_q2 = self.critic_target_2(next_states, target_actions)
      target_q = rewards + self.gamma * (1 - dones) * torch.min(target_q1, target_q2)

    # Update critics
    current_q1 = self.critic_1(states, actions)
    current_q2 = self.critic_2(states, actions)
    critic_loss_1 = nn.MSELoss()(current_q1, target_q)
    critic_loss_2 = nn.MSELoss()(current_q2, target_q)

    self.critic_optimizer_1.zero_grad()
    critic_loss_1.backward()
    self.critic_optimizer_1.step()

    self.critic_optimizer_2.zero_grad()
    critic_loss_2.backward()
    self.critic_optimizer_2.step()

    # Delayed policy updates
    if self.policy_update_counter % self.policy_freq == 0:
      # Update actor
      actor_loss = -self.critic_1(states, self.actor(states)).mean()
      self.actor_optimizer.zero_grad()
      actor_loss.backward()
      self.actor_optimizer.step()

      # Soft update target networks
      for target_param, param in zip(self.critic_target_1.parameters(), self.critic_1.parameters()):
        target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
      for target_param, param in zip(self.critic_target_2.parameters(), self.critic_2.parameters()):
        target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
      for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
        target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    self.policy_update_counter += 1

  def train(self):
    """
    Train the model for one episode and return the episode reward.
    Returns:
        episode_reward (float): Total reward obtained in the episode.
        history (list): The history of the agent..
    """
    state, _ = self.env.reset()
    state = torch.FloatTensor(state).to(self.device)
    done = False
    episode_reward = 0
    history = []

    while not done:
      state_tensor = state.unsqueeze(0)
      action = self.actor(state_tensor).detach().cpu().numpy()[0]
      action = np.clip(action, -self.max_action, self.max_action)

      next_state, reward, terminated, truncated, _ = self.env.step(action)
      next_state = torch.FloatTensor(next_state).to(self.device)
      done = terminated or truncated

      self.store_transition(state.cpu().numpy(), action, reward, next_state.cpu().numpy(), done)
      self.update()

      state = next_state
      history.append({
          "state": state.cpu().numpy().tolist(),
          "action": action.tolist(),
          "reward": reward,
          "episode_reward": episode_reward,
          "next_state": next_state.cpu().numpy().tolist(),
          "done": done
      })
      episode_reward += reward

    return episode_reward, history

  def save(self, filename):
    torch.save({
        "actor": self.actor.state_dict(),
        "critic_1": self.critic_1.state_dict(),
        "critic_2": self.critic_2.state_dict(),
        "actor_optimizer": self.actor_optimizer.state_dict(),
        "critic_optimizer_1": self.critic_optimizer_1.state_dict(),
        "critic_optimizer_2": self.critic_optimizer_2.state_dict(),
        "parameters": self.parameters
    }, filename)

  def load(self, filename):
    checkpoint = torch.load(filename)
    self.actor.load_state_dict(checkpoint["actor"])
    self.critic_1.load_state_dict(checkpoint["critic_1"])
    self.critic_2.load_state_dict(checkpoint["critic_2"])
    self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])
    self.critic_optimizer_1.load_state_dict(checkpoint["critic_optimizer_1"])
    self.critic_optimizer_2.load_state_dict(checkpoint["critic_optimizer_2"])
    self.parameters = checkpoint["parameters"]

  def evaluate(self, render=False):
    """
    Evaluate the model without training and return success, rewards, and frames.
    Args:
        render (bool): Whether to render the environment during evaluation.
    Returns:
        success (bool): Whether the evaluation was successful based on the defined criteria.
        episode_reward (float): Total reward obtained in the episode.
        frames (list): List of frames (always returned, even if empty).
    """
    state, _ = self.env.reset()
    state = torch.FloatTensor(state).to(self.device)
    done = False
    episode_reward = 0
    frames = []

    while not done:
      state_tensor = state.unsqueeze(0)
      with torch.no_grad():
        action = self.actor(state_tensor).detach().cpu().numpy()[0]
      action = np.clip(action, -self.max_action, self.max_action)

      next_state, reward, terminated, truncated, _ = self.env.step(action)
      next_state = torch.FloatTensor(next_state).to(self.device)
      done = terminated or truncated

      if render:
        frame = self.env.render()
        frames.append(frame)

      episode_reward += reward
      state = next_state

    # Define success condition
    success = not terminated and not truncated and episode_reward >= 0

    return success, episode_reward, frames
