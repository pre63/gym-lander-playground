"""
Parallel training script

for model in ppo sac td3 totd trpo ddpg laber mac 
    for episodes in 10 100 200 500 1000 2000
        for strategy in default proximity energy_efficient combined
            python playground.py $model $episodes $strategy &
        end
    end
end
wait


"""

import os
import sys
import importlib
import json
from datetime import datetime
import gymnasium as gym
from reward import REWARD_STRATEGIES


def create_results_folder(model_name, config):
  timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
  folder_name = f"results/{model_name}-{timestamp}"
  os.makedirs(folder_name, exist_ok=True)

  # Save configuration to disk
  with open(os.path.join(folder_name, "config.json"), "w") as f:
    json.dump(config, f, indent=4)

  return folder_name


def save_results_to_disk(results, folder_name):
  with open(os.path.join(folder_name, "results.json"), "w") as f:
    json.dump(results, f, indent=4)


def save_best_episode(folder_name, best_episode):
  with open(os.path.join(folder_name, "best_episode.json"), "w") as f:
    json.dump(best_episode, f, indent=4)


def load_model(model_name):
  model_path = f"models.{model_name}"
  try:
    module = importlib.import_module(model_path)
    return module.Model
  except ModuleNotFoundError:
    print(f"Error: Model '{model_name}' not found in 'models' folder.")
    sys.exit(1)


def wrap_environment(env, reward_strategy):
  """
  Wrap the environment with the selected reward strategy.
  """
  class RewardWrapper(gym.Wrapper):
    def __init__(self, env, reward_fn):
      super().__init__(env)
      self.reward_fn = reward_fn

    def step(self, action):
      state, reward, terminated, truncated, info = self.env.step(action)
      custom_reward = self.reward_fn(state, reward, action, terminated or truncated, info)
      return state, custom_reward, terminated, truncated, info

  return RewardWrapper(env, reward_strategy)


def main():
  if len(sys.argv) < 4:
    print("Usage: python playground.py [model] [episodes] [reward_strategy] [optional_env]")
    sys.exit(1)

  model_name = sys.argv[1]
  try:
    num_episodes = int(sys.argv[2])
  except ValueError:
    print("Error: Number of episodes must be an integer.")
    sys.exit(1)

  reward_strategy_name = sys.argv[3] if len(sys.argv) > 3 else "default"
  reward_strategy = REWARD_STRATEGIES.get(reward_strategy_name)
  if reward_strategy is None:
    print(f"Error: Reward strategy '{reward_strategy_name}' not found.")
    sys.exit(1)

  # Use LunarLanderContinuous-v3 as the default environment, allow override
  env_name = sys.argv[4] if len(sys.argv) > 4 else "LunarLanderContinuous-v3"

  config = {
      "model": model_name,
      "episodes": num_episodes,
      "reward_strategy": reward_strategy_name,
      "environment": env_name,
  }

  results_folder = create_results_folder(model_name, config)
  ModelClass = load_model(model_name)

  try:
    env = gym.make(env_name)
  except gym.error.Error as e:
    print(f"Error: Unable to create environment '{env_name}'.\n{e}")
    sys.exit(1)

  # Wrap environment with selected reward strategy
  env = wrap_environment(env, reward_strategy)

  model = ModelClass(env)

  run_model(model, num_episodes, results_folder, env)


def run_model(model, num_episodes, results_folder, env):
  all_rewards = []
  best_episode = {"episode": None, "reward": float("-inf"), "trajectory": []}

  start_time = datetime.now()

  for episode in range(num_episodes):
    # Get data from model.train() which handles the episode logic
    episode_reward, trajectory = model.train()
    all_rewards.append(episode_reward)

    # Update best episode if this one has a higher reward
    if episode_reward > best_episode["reward"]:
      best_episode.update({
          "episode": episode,
          "reward": episode_reward,
          "trajectory": trajectory,
      })

    print(f"Episode {episode + 1}/{num_episodes}, Reward: {episode_reward}")

  end_time = datetime.now()

  # Save results
  results = {
      "all_rewards": all_rewards,
      "average_reward": sum(all_rewards) / len(all_rewards),
      "variance_in_rewards": sum((x - sum(all_rewards) / len(all_rewards)) ** 2 for x in all_rewards) / len(all_rewards),
      "average_time": (end_time - start_time).total_seconds() / num_episodes,
  }

  save_results_to_disk(results, results_folder)
  save_best_episode(results_folder, best_episode)

  env.close()
  print(f"Results saved in folder: {results_folder}")
  print(f"\nTo replay the best episode, run:\n")
  print(f"    python replay.py {results_folder}")
  print(f"\n")


if __name__ == "__main__":
  main()
