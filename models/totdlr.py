
from envs.random_walk import RandomWalk
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import matplotlib.pyplot as plt


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


class ActorCriticContinuous:
  def __init__(self, state_space, action_space, critic_model, actor_lr=0.01):
    self.state_space = state_space
    self.action_space = action_space
    self.critic = critic_model
    self.actor_weights = np.random.randn(state_space, 1)  # Match to state dimensions
    self.actor_lr = actor_lr
    self.sigma = 1.0  # Fixed variance for Gaussian policy

  def get_action(self, state):
    state = np.array(state)  # Ensure state is a NumPy array
    mean = np.dot(state, self.actor_weights).item()
    action = np.random.normal(mean, self.sigma)
    return action, mean

  def update_actor(self, state, action, advantage):
    state = np.array(state)  # Ensure state is a NumPy array
    mean = np.dot(state, self.actor_weights).item()
    grad_log_pi = (action - mean) / (self.sigma**2) * state
    self.actor_weights += self.actor_lr * advantage * grad_log_pi[:, np.newaxis]


class ActorCriticTrainer:
  def __init__(self, env, actor_critic):
    self.env = env
    self.ac = actor_critic

  def train(self, episodes=100, replay_frequency=1):
    for episode in range(episodes):
      state, _ = self.env.reset()
      self.ac.critic.reset_traces()
      self.ac.critic.clear_buffer()

      while True:
        action, mean = self.ac.get_action(state)
        next_state, reward, done, _, _ = self.env.step([action])
        next_state = np.array(next_state)  # Use raw state values for continuous spaces

        # Store transition and update critic
        self.ac.critic.store_transition(state, reward, next_state, done)
        self.ac.critic.update(state, reward, next_state, done)

        # Compute advantage
        td_error = reward + (1 - done) * self.ac.critic.value(next_state) - self.ac.critic.value(state)

        # Update actor using advantage
        self.ac.update_actor(state, action, td_error)

        if done:
          break
        state = next_state

      if episode % replay_frequency == 0:
        self.ac.critic.replay()

  def render_value_function(self):
    values = self.ac.critic.get_values()
    plt.plot(values, marker="o", label="Value Function")
    plt.xlabel("State")
    plt.ylabel("Value")
    plt.title("Estimated Value Function")
    plt.grid()
    plt.legend()
    plt.show()


if __name__ == "__main__":
  env = gym.make("MountainCarContinuous-v0", render_mode="human")

  # Use actual state and action dimensions from the environment
  state_space = env.observation_space.shape[0]  # State space size = 2 (position, velocity)
  action_space = env.action_space.shape[0]  # Action space size = 1 (continuous action)

  # Initialize critic and actor-critic models
  critic = TrueOnlineTDLambdaCompressedReplayModel(state_space=state_space, alpha=0.1, lambd=0.9, v0=0.5)
  actor_critic = ActorCriticContinuous(state_space=state_space, action_space=action_space, critic_model=critic)

  # Train the Actor-Critic model
  trainer = ActorCriticTrainer(env, actor_critic)
  trainer.train(episodes=100)  # Train for 100 episodes
  trainer.render_value_function()  # Visualize the critic's value function
