
import numpy as np

import gymnasium as gym
from gymnasium import spaces

import torch
import torch.nn as nn
import torch.optim as optim

from models.base import BaseModel, BaseConvertStableBaselinesModel


class TrueOnlineTDLambdaCompressedReplayModel:
  def __init__(self, state_space, alpha=0.2, lambd=0.8, gamma=1.0, v0=0.5):
    self.alpha = alpha
    self.lambd = lambd
    self.gamma = gamma
    self.state_dim = state_space
    self.weights = np.full(state_space, v0, dtype=float)  # Linear weights for value function
    self.eligibility_trace = np.zeros(state_space)
    self.compressed_experience = {}

  def state_features(self, state):
    """Extract features from the state (identity mapping for now)."""
    return np.array(state)

  def value(self, state):
    """Compute V(s) using the linear approximator."""
    features = self.state_features(state)
    return np.dot(features, self.weights)

  def update(self, state, reward, next_state, done):
    """Update value function using True Online TD(λ)."""
    state_features = self.state_features(state)
    next_state_features = self.state_features(next_state)
    delta = reward + (1 - done) * self.gamma * np.dot(next_state_features, self.weights) - np.dot(state_features, self.weights)
    self.eligibility_trace = (
        self.eligibility_trace * self.gamma * self.lambd + state_features
    )
    self.weights += self.alpha * delta * self.eligibility_trace

  def store_transition(self, state, reward, next_state, done):
    """Store transition in compressed format."""
    state_key = tuple(state)
    if state_key not in self.compressed_experience:
      self.compressed_experience[state_key] = [0.0, 0, np.zeros_like(self.eligibility_trace)]
    self.compressed_experience[state_key][0] += reward  # Cumulative reward
    self.compressed_experience[state_key][1] += 1  # Count of visits
    self.compressed_experience[state_key][2] += self.eligibility_trace.copy()

  def replay(self):
    """Replay stored transitions."""
    for state_key, (reward_sum, count, trace) in self.compressed_experience.items():
      avg_reward = reward_sum / count
      features = self.state_features(state_key)
      delta = avg_reward + self.gamma * np.dot(features, self.weights) - np.dot(features, self.weights)
      self.weights += self.alpha * delta * trace

  def reset_traces(self):
    self.eligibility_trace.fill(0)

  def get_values(self):
    return self.weights  # Return the linear weights as the value function

  def clear_buffer(self):
    self.compressed_experience = {}


class ActorNetwork(nn.Module):
  def __init__(self, state_dim, action_dim, hidden_dim=64, actor_lr=1e-3):
    super(ActorNetwork, self).__init__()

    self.model = nn.Sequential(
        nn.Linear(state_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, action_dim)
    )

    self.log_std = nn.Parameter(torch.zeros(action_dim))  # Learnable log standard deviation

    self.optimizer = optim.Adam(self.parameters(), lr=actor_lr)

  def forward(self, state):
    """Compute mean and std for the policy."""
    mean = self.model(state)
    std = torch.exp(self.log_std)
    return mean, std

  def get_action(self, state):
    state_tensor = torch.tensor(state, dtype=torch.float32)
    mean, std = self.forward(state_tensor)
    dist = torch.distributions.Normal(mean, std)
    action = dist.sample()
    return action.detach().numpy().flatten(), mean.detach().numpy().flatten()

  def update(self, state, action, advantage):
    """Update the policy using the advantage."""
    state_tensor = torch.tensor(state, dtype=torch.float32)
    mean, std = self.forward(state_tensor)
    dist = torch.distributions.Normal(mean, std)

    log_prob = dist.log_prob(torch.tensor(action, dtype=torch.float32))
    actor_loss = -log_prob * advantage  # Maximize expected advantage

    # Reduce actor_loss to a scalar (e.g., sum or mean)
    actor_loss = actor_loss.sum()  # If multiple actions, sum over all dimensions

    self.optimizer.zero_grad()
    actor_loss.backward()
    self.optimizer.step()


class ActorCriticContinuous:
  def __init__(self, actor, critic):
    self.actor = actor
    self.critic = critic

  def get_action(self, state):
    """Get an action from the actor."""
    return self.actor.get_action(state)

  def compute_advantage(self, reward, next_state, state, done, gamma=0.99):
    """Compute the advantage using the critic."""
    state_value = self.critic.value(state)
    next_state_value = self.critic.value(next_state)
    target = reward + gamma * next_state_value * (1 - done)
    advantage = target - state_value
    return advantage, target

  def update(self, state, action, advantage):
    """Update actor and critic."""
    # Update the critic
    reward = advantage + self.critic.value(state)
    done = 0  # Assume not done for critic update
    next_state = state  # Mock next_state for critic update
    self.critic.update(state, reward, next_state, done)

    # Update the actor
    self.actor.update(state, action, advantage)

  def step(self, state, action, reward, next_state, done):
    """Compute the advantage and update the actor and critic."""
    self.critic.store_transition(state, reward, next_state, done=done)
    self.critic.update(state, reward, next_state, done=done)

    advantage, target = self.compute_advantage(reward, next_state, state, done)

    self.update(state, action, advantage)

    self.critic.replay()


class TOTDLR(BaseConvertStableBaselinesModel):
  def __init__(self, env, alpha=0.01, gamma=0.99, lambda_=0.9, env_type="gym", num_envs=1):
    super().__init__(env, env_type, num_envs)
    self.alpha = alpha
    self.gamma = gamma
    self.lambda_ = lambda_

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    critic = TrueOnlineTDLambdaCompressedReplayModel(state_dim, alpha=alpha, lambd=lambda_, gamma=gamma)
    actor = ActorNetwork(state_dim, action_dim)

    self.model = ActorCriticContinuous(actor, critic)

  def train_step(self, state, action, reward, next_state, next_action, done, info):
    self.model.step(state, action, reward, next_state, done)

  def predict(self, state):
    action = self.model.get_action(state)
    return action

  def learn_episode_setup(self):
    self.model.critic.reset_traces()
    self.model.critic.clear_buffer()

  def save(self, filename):
    torch.save(self.model.actor.state_dict(), filename + "_actor.pth")
    np.save(filename + "_critic.npy", self.model.critic.get_values())

  def load(self, filename):
    self.model.actor.load_state_dict(torch.load(filename + "_actor.pth"))
    self.model.critic.weights = np.load(filename + "_critic.npy")


class Model(BaseModel):
  def __init__(self, env_name, num_envs=16, max_episode_steps=5000, reward_strategy="default", **kwargs):
    """
    Initialize the Model class specifically for PPO.
    Args:
        env_name (str): The name of the environment to train on.
        num_envs (int): Number of parallel environments to use.
        kwargs: Additional arguments to pass to the PPO initialization.
    """
    super().__init__(env_name, num_envs, max_episode_steps, reward_strategy, **kwargs)
    self.model = TOTDLR(self.env, env_type=self.env_type, ** self.parameters)


if __name__ == "__main__":
  env = gym.make("MountainCarContinuous-v0", render_mode="human")
  # Initialize critic and actor-critic models
  totdlr = TOTDLR(env, alpha=0.01, gamma=0.99, lambda_=0.9)
  totdlr.learn(1000, max_steps=1000, render_frequency=100)
