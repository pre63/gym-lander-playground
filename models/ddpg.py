from stable_baselines3 import DDPG

from models.sb import SBase


class Model(SBase):
  def __init__(self, env_name, num_envs=8, max_episode_steps=2000, reward_strategy="default", **kwargs):
    """
    Initialize the Model class specifically for DDPG.
    Args:
        env_name (str): The name of the environment to train on.
        num_envs (int): Number of parallel environments to use.
        kwargs: Additional arguments to pass to the DDPG initialization.
    """
    super().__init__(env_name, num_envs, max_episode_steps, reward_strategy, **kwargs)
    self.model = DDPG("MlpPolicy", self.env, **self.parameters)
