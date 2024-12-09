import numpy as np
import gymnasium as gym
from gymnasium import spaces
from itertools import product
from collections import deque

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


from models.base import BaseModel, BaseConvertStableBaselinesModel
from models._ac.buffer import Replay_Buffer
from models._ac.nn import Actor, Critic
from models._ac.noise import OU_Noise


class Agent:

  # Based on the work of Cheng Xiaotian
  # https://github.com/greatwallet/mountain-car/blob/master/train_continuous.py

  def __init__(self,
               n_states,
               n_actions,
               max_episode_steps=1000,
               seed=0,
               buffer_size=1000000,
               batch_size=256,
               mem_seed=72,
               ou_seed=72,
               lr_critic=2e-2,
               lr_actor=3e-3,
               update_every_n_steps=20,
               learning_updates_per_learning_session=10,
               mu=0.0,
               theta=0.15,
               sigma=0.25,
               gamma=0.99,
               clamp_critic=5,
               clamp_actor=5,
               tau_critic=5e-3,
               tau_actor=5e-3,
               win=100,
               score_th=90):

    self.label = f"Agent(αc%.5f_αc%.5f_γ%.5f_e%d)" % (
        lr_critic, lr_actor, gamma, max_episode_steps)

    self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    self.n_states = n_states
    self.n_actions = n_actions

    self.seed = seed
    self.buffer_size = buffer_size
    self.batch_size = batch_size
    self.mem_seed = mem_seed
    self.ou_seed = ou_seed
    self.lr_critic = lr_critic
    self.lr_actor = lr_actor
    self.update_every_n_steps = update_every_n_steps
    self.learning_updates_per_learning_session = learning_updates_per_learning_session
    self.mu = mu
    self.theta = theta
    self.sigma = sigma
    self.gamma = gamma
    self.clamp_critic = clamp_critic
    self.clamp_actor = clamp_actor
    self.tau_critic = tau_critic
    self.tau_actor = tau_actor
    self.win = win
    self.score_th = score_th

    self.memory = Replay_Buffer(buffer_size, batch_size, seed)

    # critic
    self.critic_local = Critic(n_states=n_states, n_actions=n_actions, n_outputs=1).to(self.device)
    self.critic_target = Critic(n_states=n_states, n_actions=n_actions, n_outputs=1).to(self.device)

    self.model_deep_copy(from_model=self.critic_local, to_model=self.critic_target)

    self.optim_critic = optim.Adam(
        self.critic_local.parameters(), lr=lr_critic, eps=1e-4)

    self.memory = Replay_Buffer(buffer_size, batch_size, mem_seed)

    # actor
    self.actor_local = Actor(n_states=n_states, n_actions=n_actions).to(self.device)
    self.actor_target = Actor(n_states=n_states, n_actions=n_actions).to(self.device)
    self.model_deep_copy(from_model=self.actor_local, to_model=self.actor_target)

    self.optim_actor = optim.Adam(
        self.actor_local.parameters(), lr=lr_actor, eps=1e-4)

    # ou noise
    self.ou_noise = OU_Noise(
        size=self.n_actions,
        seed=ou_seed,
        mu=mu,
        theta=theta,
        sigma=sigma
    )

    self.rewards = []
    self.steps = []

  def decay(starting_lr, optimizer, rolling_score_list, score_th):
    """Lowers the learning rate according to how close we are to the solution"""
    if len(rolling_score_list) > 0:
      last_rolling_score = rolling_score_list[-1]
      if last_rolling_score > 0.75 * score_th:
        new_lr = starting_lr / 100.0
      elif last_rolling_score > 0.6 * score_th:
        new_lr = starting_lr / 20.0
      elif last_rolling_score > 0.5 * score_th:
        new_lr = starting_lr / 10.0
      elif last_rolling_score > 0.25 * score_th:
        new_lr = starting_lr / 2.0
      else:
        new_lr = starting_lr
      for g in optimizer.param_groups:
        g['lr'] = new_lr

  def model_deep_copy(self, from_model, to_model):
    """Copies model parameters from from_model to to_model"""
    for to_model, from_model in zip(to_model.parameters(), from_model.parameters()):
      to_model.data.copy_(from_model.data.clone())

  def setup(self):
    self.ou_noise.reset()

  def step(self, state, action, reward, next_state, done):
    # Reset OU noise if a new episode starts
    if done:
      print(f"Episode done. Reward: {reward}")
      self.ou_noise.reset()

    # Add the current transition to the memory
    self.memory.add_experience(state, action, reward, next_state, done)

    # If enough experiences exist, start learning
    if len(self.memory) >= self.batch_size:

      # Sample experiences from memory
      states_numpy, actions_numpy, rewards_numpy, next_states_numpy, dones_numpy = self.memory.sample()
      states = torch.from_numpy(states_numpy).float().to(self.device)
      actions = torch.from_numpy(actions_numpy).float().to(self.device)
      rewards = torch.from_numpy(rewards_numpy).float().to(self.device)
      next_states = torch.from_numpy(next_states_numpy).float().to(self.device)
      dones = torch.from_numpy(dones_numpy).float().unsqueeze(1).to(self.device)

      # ---- Critic Update ----
      # Compute target values
      with torch.no_grad():
        next_actions = self.actor_target(next_states)
        next_values = self.critic_target(next_states, next_actions)
        target_values = rewards + self.gamma * next_values * (1.0 - dones)

      # Compute expected values
      values = self.critic_local(states, actions)

      # Compute loss and optimize critic
      critic_loss = F.mse_loss(values, target_values)
      self.optim_critic.zero_grad()
      critic_loss.backward()
      if self.clamp_critic is not None:
        torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(), self.clamp_critic)
      self.optim_critic.step()

      # Soft update target critic network
      for target_param, local_param in zip(self.critic_target.parameters(), self.critic_local.parameters()):
        target_param.data.copy_(self.tau_critic * local_param.data + (1.0 - self.tau_critic) * target_param.data)

      # ---- Actor Update ----
      # Compute actor loss
      predicted_actions = self.actor_local(states)
      actor_loss = -self.critic_local(states, predicted_actions).mean()

      # Optimize actor
      self.optim_actor.zero_grad()
      actor_loss.backward()
      if self.clamp_actor is not None:
        torch.nn.utils.clip_grad_norm_(self.actor_local.parameters(), self.clamp_actor)
      self.optim_actor.step()

      # Soft update target actor network
      for target_param, local_param in zip(self.actor_target.parameters(), self.actor_local.parameters()):
        target_param.data.copy_(self.tau_actor * local_param.data + (1.0 - self.tau_actor) * target_param.data)

  def get_action(self, state):
    state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)

    self.actor_local.eval()

    with torch.no_grad():
      action = self.actor_local(state).cpu().data.numpy().squeeze(0)

    self.actor_local.train()

    return action

  def save(self, filename):
    torch.save({
        "actor_local": self.actor_local.state_dict(),
        "actor_target": self.actor_target.state_dict(),
        "critic_local": self.critic_local.state_dict(),
        "critic_target": self.critic_target.state_dict(),
        "optim_actor": self.optim_actor.state_dict(),
        "optim_critic": self.optim_critic.state_dict(),
        "rewards": self.rewards,
        "steps": self.steps
    }, filename)

  def load(self, filename):
    checkpoint = torch.load(filename, weights_only=True)
    self.actor_local.load_state_dict(checkpoint["actor_local"])
    self.actor_target.load_state_dict(checkpoint["actor_target"])
    self.critic_local.load_state_dict(checkpoint["critic_local"])
    self.critic_target.load_state_dict(checkpoint["critic_target"])
    self.optim_actor.load_state_dict(checkpoint["optim_actor"])
    self.optim_critic.load_state_dict(checkpoint["optim_critic"])
    self.rewards = checkpoint["rewards"]
    self.steps = checkpoint["steps"]


class AC(BaseConvertStableBaselinesModel):
  def __init__(self, env, alpha=0.01, gamma=0.99, lambda_=0.9, env_type="gym", num_envs=1, max_episode_steps=10000):
    super().__init__(env, env_type, num_envs)
    self.alpha = alpha
    self.gamma = gamma
    self.lambda_ = lambda_

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    self.model = Agent(
        n_states=state_dim,
        n_actions=action_dim,
        max_episode_steps=max_episode_steps
    )

  def train_step(self, state, action, reward, next_state, next_action, done, info):
    self.model.step(state, action, reward, next_state, done)

  def predict(self, state):
    action = self.model.get_action(state)
    action = np.clip(action, self.env.action_space.low, self.env.action_space.high)
    action = np.squeeze(action)
    return action, {}

  def learn_episode_setup(self):
    self.model.setup()

  def save(self, filename):
    self.model.save(filename)

  def load(self, filename):
    self.model.load(filename)


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
    self.model = AC(self.env, env_type=self.env_type, max_episode_steps=max_episode_steps, ** self.parameters)


if __name__ == "__main__":
  env = gym.make("MountainCarContinuous-v0", render_mode="human")
  # Initialize critic and actor-critic models
  totdlr = AC(env, alpha=0.01, gamma=0.99, lambda_=0.9)
  totdlr.learn(1000, max_episode_steps=1000, render_frequency=100)
