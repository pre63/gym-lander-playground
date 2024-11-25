import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
import gymnasium as gym

from success import check_success


class ReplayBuffer:
  def __init__(self, buffer_size=100000, batch_size=256, large_batch_multiplier=4, device="cpu"):
    """
    Initialize a replay buffer with a fixed size.
    Args:
        buffer_size (int): Maximum number of transitions to store.
        batch_size (int): Number of transitions to sample during training.
        device (str): The device ('cpu' or 'cuda') to which tensors will be moved.
    """
    self.buffer = deque(maxlen=buffer_size)
    self.batch_size = batch_size
    self.large_batch_size = batch_size * large_batch_multiplier
    self.device = device

  def store(self, state, action, reward, next_state, done):
    """
    Store a single transition in the buffer.
    Args:
        state (array-like): The state before the action.
        action (array-like): The action taken.
        reward (float): The reward obtained after the action.
        next_state (array-like): The state after the action.
        done (bool): Whether the episode ended.
    """
    self.buffer.append((state, action, reward, next_state, done))

  def sample(self):
    """
    Sample a batch of transitions from the buffer.
    Returns:
        Tuple[torch.Tensor]: Tensors for states, actions, rewards, next states, and done flags.
    """
    batch = random.sample(self.buffer, min(len(self.buffer), self.batch_size))
    states, actions, rewards, next_states, dones = zip(*batch)

    # Convert to numpy arrays for efficient tensor conversion
    np_states = np.array(states)
    np_actions = np.array(actions)
    np_rewards = np.array(rewards, dtype=np.float32).reshape(-1, 1)
    np_next_states = np.array(next_states)
    np_dones = np.array(dones, dtype=np.float32).reshape(-1, 1)

    # Convert to tensors and move to the specified device
    return (
        torch.FloatTensor(np_states).to(self.device),
        torch.FloatTensor(np_actions).to(self.device),
        torch.FloatTensor(np_rewards).to(self.device),
        torch.FloatTensor(np_next_states).to(self.device),
        torch.FloatTensor(np_dones).to(self.device)
    )

  def size(self):
    """
    Get the current size of the buffer.
    Returns:
        int: The number of transitions stored in the buffer.
    """
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

    # Mean Actor
    self.mean_actor = Actor(self.state_dim, self.action_dim, self.max_action).to(self.device)
    self.mean_actor.load_state_dict(self.actor.state_dict())

    # Replay Buffer
    self.replay_buffer = ReplayBuffer(buffer_size, batch_size, device=self.device)

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

    # Sample from replay buffer
    states, actions, rewards, next_states, dones = self.replay_buffer.sample()

    # Move data to device
    states = states.to(self.device)
    actions = actions.to(self.device)
    rewards = rewards.to(self.device)
    next_states = next_states.to(self.device)
    dones = dones.to(self.device)

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
        history (list): The history of the agent.
    """
    state, _ = self.env.reset()
    state = torch.FloatTensor(state).to(self.device)
    done = False
    episode_reward = 0
    history = []

    while not done:
      state_tensor = state.unsqueeze(0)
      action = self.actor(state_tensor).detach().cpu().numpy()[0]
      next_state, reward, terminated, truncated, _ = self.env.step(action)
      next_state = torch.FloatTensor(next_state).to(self.device)
      done = terminated or truncated

      self.store_transition(state.cpu().numpy(), action, reward, next_state.cpu().numpy(), done)
      self.train_step()

      history.append({
          "state": state.cpu().numpy().tolist(),
          "action": action.tolist(),
          "reward": reward,
          "episode_reward": episode_reward,
          "next_state": next_state.cpu().numpy().tolist(),
          "done": done
      })
      episode_reward += reward
      state = next_state

    return episode_reward, history

  def save(self, filename):
    """
    Save the model to a file.
    Args:
        filename (str): The name of the file to save.
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
    Load the model from a file.
    Args:
        filename (str): The name of the file to load.
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
        frames (list): List of frames (always returned, even if empty).
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
        frame = self.env.render()
        frames.append(frame)

      state = next_state

    # Define success condition
    success = check_success(next_state, terminated)

    # Always return frames, even if empty
    return success, episode_reward, frames
