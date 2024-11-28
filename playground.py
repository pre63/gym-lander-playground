import os
import sys
import numpy as np
import importlib
import json
from datetime import datetime

import gymnasium as gym
from gymnasium.wrappers import TimeLimit

from evaluate import evaluate_model, render_sample
from models.utils import load_model


def create_results_folder(model_name, config):
  timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
  folder_name = f"results/{model_name}-{timestamp}"
  os.makedirs(folder_name, exist_ok=True)

  # Save configuration to disk
  with open(os.path.join(folder_name, "config.json"), "w") as f:
    json.dump(config, f, indent=4)

  return folder_name


def save_results_to_disk(results, folder_name):
  os.makedirs(folder_name, exist_ok=True)
  with open(os.path.join(folder_name, "results.json"), "w") as f:
    json.dump(results, f, indent=4)


def run_model(model, total_timesteps, results_folder, trials=200):
  start_time = datetime.now()

  model.learn(total_timesteps=total_timesteps, progress_bar=True)

  end_time = datetime.now()

  model.save(os.path.join(results_folder, "model"))

  results = {
      "average_time": (end_time - start_time).total_seconds() / total_timesteps,
      "gen": "V4"
  }

  save_results_to_disk(results, results_folder)

  evaluate_model(results_folder, trials)

  render_sample(results_folder)


def main():
  if len(sys.argv) < 4:
    print("Usage: python playground.py [model] [timesteps] [reward_strategy] [optional_env]")
    sys.exit(1)

  model_name = sys.argv[1]

  total_timesteps = 1000 if len(sys.argv) < 3 else int(sys.argv[2])

  reward_strategy_name = sys.argv[3] if len(sys.argv) > 3 else "default"

  env_name = sys.argv[4] if len(sys.argv) > 4 else "LunarLanderContinuous-v3"

  ModelClass = load_model(model_name)

  model = ModelClass(env_name, num_envs=16, max_episode_steps=5000, reward_strategy=reward_strategy_name)

  config = {
      "model": model_name,
      "timesteps": total_timesteps,
      "reward_strategy": reward_strategy_name,
      "environment": env_name,
      "model_params": model.parameters
  }

  results_folder = create_results_folder(model_name, config)

  run_model(model, total_timesteps, results_folder)


if __name__ == "__main__":
  main()
