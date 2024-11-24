import gymnasium as gym
from sb3_contrib import TRPO


class Model:
  def __init__(self, env: gym.Env, gamma=0.99, gae_lambda=0.95, target_kl=0.01, net_arch=[64, 64]):
    """
    Initialize the TRPO model with the given environment and hyperparameters.
    Args:
        env (gym.Env): The environment to train on.
        gamma (float): Discount factor for rewards.
        gae_lambda (float): Lambda for Generalized Advantage Estimation (GAE).
        target_kl (float): Target Kullback-Leibler divergence for trust region update.
    """
    self.parameters = {
        "gamma": gamma,
        "gae_lambda": gae_lambda,
        "target_kl": target_kl,
        "net_arch": net_arch
    }

    self.env = env
    self.model = TRPO(
        policy="MlpPolicy",
        env=env,
        verbose=0,
        gamma=gamma,
        gae_lambda=gae_lambda,
        target_kl=target_kl,  # Correctly using target_kl here
        policy_kwargs={
            "net_arch": net_arch
        },
        device="cpu"
    )

  def train(self):
    """
    Train the TRPO model for one episode and return the episode reward and rewards per step.

    Returns:
        episode_reward (float): Total reward obtained in the episode.
        trajectory (list): The trajectory of the agent..
    """
    state, _ = self.env.reset()
    done = False
    episode_reward = 0
    trajectory = []
    frames = []

    while not done:
      action, _ = self.model.predict(state, deterministic=False)
      next_state, reward, terminated, truncated, _ = self.env.step(action)
      done = terminated or truncated

      trajectory.append({
          "state": state.tolist(),
          "action": action.tolist(),
          "reward": reward,
          "next_state": next_state.tolist(),
          "done": done
      })
      episode_reward += reward
      state = next_state

      frames.append(self.env.render())

    # Perform a learning step with a fixed number of timesteps
    self.model.learn(total_timesteps=1000, reset_num_timesteps=False)

    return episode_reward, trajectory, frames
