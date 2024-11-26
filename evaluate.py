import os
import sys
import json
import numpy as np
import gymnasium as gym
from playground import load_model


def evaluate_model(results_folder, evaluation_trials):
  """
  Evaluate a trained model from the specified results folder.

  Args:
      results_folder (str): Path to the folder containing the model and config.
      evaluation_trials (int): Number of evaluation trials to run.

  Returns:
      None
  """
  # Load the configuration
  config_path = os.path.join(results_folder, "config.json")
  if not os.path.exists(config_path):
    print(f"Error: Configuration file not found at {config_path}")
    sys.exit(1)

  with open(config_path, "r") as f:
    config = json.load(f)

  model_name = config["model"]
  env_name = config["environment"]

  print(f"Evaluating model: {model_name}")
  print(f"Environment: {env_name}")
  print(f"Number of evaluation trials: {evaluation_trials}")

  # Load the environment
  try:
    env = gym.make(env_name, render_mode="rgb_array")
  except gym.error.Error as e:
    print(f"Error: Unable to create environment '{env_name}'.\n{e}")
    sys.exit(1)

  # Load the model
  model_class = load_model(model_name)
  model = model_class(env)
  model_path = os.path.join(results_folder, "model")
  model.load(model_path)

  # Run evaluation
  all_rewards = []
  success_count = 0
  print(f"\nRunning {evaluation_trials} evaluation trials...\n")
  for i in range(evaluation_trials):
    success, episode_reward, _ = model.evaluate(render=False)
    all_rewards.append(episode_reward)
    if success:
      success_count += 1
    print(f"Trial {i + 1}/{evaluation_trials}: Reward: {episode_reward}, {'Landed' if success else 'Crash'}")

  # Compute statistics
  average_reward = np.mean(all_rewards)
  variance_reward = np.var(all_rewards)
  success_rate = success_count / evaluation_trials

  print("\nEvaluation Results:")
  print(f"  Average Reward: {average_reward}")
  print(f"  Reward Variance: {variance_reward}")
  print(f"  Success Count: {success_count}")
  print(f"  Success Rate: {success_rate}")

  # Save results
  results_json_file = os.path.join(results_folder, "results.json")
  with open(results_json_file, "r") as f:
    results = json.load(f)
    results["eval_average_reward"] = average_reward
    results["eval_variance_reward"] = variance_reward
    results["eval_success_rate"] = success_rate
    results["eval_rewards"] = all_rewards

  with open(results_json_file, "w") as f:
    json.dump(results, f, indent=4)


if __name__ == "__main__":
  if len(sys.argv) < 3:
    print("Usage: python evaluate.py [evaluation_trials] [results_folder]")
    sys.exit(1)

  try:
    evaluation_trials = int(sys.argv[1])
  except ValueError:
    print("Error: Evaluation trials must be an integer.")
    sys.exit(1)

  results_folder = sys.argv[2]
  if not os.path.exists(results_folder):
    print(f"Error: Results folder '{results_folder}' does not exist.")
    sys.exit(1)

  evaluate_model(results_folder, evaluation_trials)
