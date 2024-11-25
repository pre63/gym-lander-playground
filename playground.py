import os
import sys
import numpy as np
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


def save_best_episode(folder_name, frames):
  if frames:
    np.savez_compressed(os.path.join(folder_name, "best_episode.npz"), frames=np.array(frames))


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


def evaluate_model(model, trials=1000):
  """
  Evaluate the model over multiple trials and collect statistics.
  Args:
      model: The trained model to evaluate.
      trials (int): Number of evaluation trials.
  Returns:
      stats (dict): A dictionary containing evaluation metrics.
  """
  all_rewards = []
  success_count = 0

  for _ in range(trials):
    success, episode_reward, _ = model.evaluate(render=False)
    all_rewards.append(episode_reward)

    if success:
      success_count += 1

  stats = {
      "eval_average_reward": np.mean(all_rewards),
      "eval_variance_reward": np.var(all_rewards),
      "eval_success_rate": success_count / trials,
      "eval_rewards": all_rewards
  }

  return stats


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
    env = gym.make(env_name, render_mode="rgb_array")
  except gym.error.Error as e:
    print(f"Error: Unable to create environment '{env_name}'.\n{e}")
    sys.exit(1)

  # Wrap environment with selected reward strategy
  env = wrap_environment(env, reward_strategy)

  model = ModelClass(env)

  run_model(model, num_episodes, results_folder, env)


def run_model(model, num_episodes, results_folder, env, evaluation_episodes=100):
  all_rewards = []
  best_episode = {"episode": None, "reward": float("-inf"), "history": []}

  start_time = datetime.now()

  for episode in range(num_episodes):
    episode_reward, history = model.train()
    all_rewards.append(episode_reward)

    # Update best episode if this one has a higher reward
    if episode_reward > best_episode["reward"]:
      best_episode.update({
          "episode": episode,
          "reward": episode_reward,
          "history": history,
      })

    print(f"Episode {episode + 1}/{num_episodes}, Reward: {episode_reward}")
  end_time = datetime.now()

  # Evaluate the model after training
  stats = evaluate_model(model, trials=evaluation_episodes)

  # Run one evaluation with rendering to capture frames for the best episode
  success, episode_reward, frames = model.evaluate(render=True)
  save_best_episode(results_folder, frames)

  # Combine training and evaluation results
  results = {
      "all_rewards": all_rewards,
      "average_reward": sum(all_rewards) / len(all_rewards),
      "variance_in_rewards": sum((x - sum(all_rewards) / len(all_rewards)) ** 2 for x in all_rewards) / len(all_rewards),
      "average_time": (end_time - start_time).total_seconds() / num_episodes,
      "eval_average_reward": stats["eval_average_reward"],
      "eval_variance_reward": stats["eval_variance_reward"],
      "eval_success_rate": stats["eval_success_rate"],
      "eval_rewards": stats["eval_rewards"],
  }

  save_results_to_disk(results, results_folder)

  print("\nAverage Reward:", results["average_reward"])
  print("Variance in Reward:", results["variance_in_rewards"])
  print("Average Time:", results["average_time"])
  print("Evaluation Average Reward:", results["eval_average_reward"])
  print("Evaluation Variance Reward:", results["eval_variance_reward"])
  print("Evaluation Success Rate:", results["eval_success_rate"])

  print(f"Results saved in folder: {results_folder}")
  print(f"\nTo replay the best episode, run:\n")
  print(f"    python replay.py {results_folder}")
  print(f"\n")

  # Save the model
  model.save(os.path.join(results_folder, "model"))


if __name__ == "__main__":
  main()
