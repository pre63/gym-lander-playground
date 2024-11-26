from sb3_contrib import TRPO

from models.sb import SBase


class Model(SBase):
  def __init__(self, env_name, num_envs=8, max_episode_steps=2000, reward_strategy="default", **kwargs):
    """
    Initialize the Model class specifically for TRPO.
    Args:
        env_name (str): The name of the environment to train on.
        num_envs (int): Number of parallel environments to use.
        kwargs: Additional arguments to pass to the TRPO initialization.
    """
    super().__init__(env_name, num_envs, max_episode_steps, reward_strategy, **kwargs)
    self.model = TRPO("MlpPolicy", self.env, **self.parameters)
