import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class Actor(nn.Module):
  def __init__(self, n_states, n_actions):
    super(Actor, self).__init__()
    self.net = nn.Sequential(
        nn.Linear(n_states, 64),
        nn.ReLU(),
        nn.Linear(64, 128),
        nn.ReLU(),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, n_actions),
        nn.Tanh()
    )

  def forward(self, state):
    return self.net(state)


class Critic(nn.Module):
  def __init__(self, n_states, n_actions, n_outputs=1):
    super(Critic, self).__init__()
    self.net = nn.Sequential(
        nn.Linear(n_states + n_actions, 64),
        nn.ReLU(),
        nn.Linear(64, 64),
        nn.ReLU(),
        nn.Linear(64, n_outputs)
    )

  def forward(self, state, action):
    x = torch.cat((state, action), dim=1)
    return self.net(x)
