from stable_baselines3 import PPO

from models.sb import SBase


class Model(SBase):
  def __init__(self, env_name, num_envs=16, max_episode_steps=5000, reward_strategy="default", **kwargs):
    """
    Initialize the Model class specifically for PPO.
    Args:
        env_name (str): The name of the environment to train on.
        num_envs (int): Number of parallel environments to use.
        kwargs: Additional arguments to pass to the PPO initialization.
    """
    super().__init__(env_name, num_envs, max_episode_steps, reward_strategy, **kwargs)
    self.model = PPO("MlpPolicy", self.env, **self.parameters)
