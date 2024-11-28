import os
import sys
import json
import numpy as np
import gymnasium as gym

from models.utils import load_model
from success import check_success
from telemetry import add_telemetry_overlay, add_success_failure_to_frames


def get_env_and_model(results_folder):
  # Load the configuration
  config_path = os.path.join(results_folder, "config.json")
  if not os.path.exists(config_path):
    print(f"Error: Configuration file not found at {config_path}")
    sys.exit(1)

  with open(config_path, "r") as f:
    config = json.load(f)

  model_name = config["model"]
  env_name = config["environment"]

  # Load the model
  model_class = load_model(model_name)
  model = model_class(env_name, env_type="gym")
  model_path = os.path.join(results_folder, "model")
  model.load(model_path)

  return model.env, model, model_name, env_name


def render_sample(results_folder):
  print("Rendering sample...")

  def save(frames):
    if frames:
      np.savez_compressed(os.path.join(results_folder, "best_episode.npz"), frames=np.array(frames))

  def render():
    state, _ = env.reset()
    episode_reward = 0
    frames = []
    done = False

    while not done:
      action, _ = model.predict(state)
      next_state, reward, terminated, truncated, info = env.step(action)
      done = terminated or truncated

      frame = env.render()
      frame = add_telemetry_overlay(frame, next_state)
      frames.append(frame)

      episode_reward += reward
      state = next_state

    success = check_success(next_state, terminated)
    frames = add_success_failure_to_frames(frames, success)

    return success, episode_reward, frames

  env, model, model_name, env_name = get_env_and_model(results_folder)

  success = False
  prev_rewards = 0

  samples = []

  attempts = 0
  while (not success) and attempts < 10:
    attempts += 1
    success, episode_reward, frames = render()
    samples.append((success, episode_reward, frames))

  success, prev_rewards, frames = max(samples, key=lambda x: x[1])
  save(frames)

  print(f"Results saved in folder: {results_folder}, with {attempts} attempts.")
  print(f"Success: {success}, Reward: {prev_rewards}")
  print(f"\nTo replay the best episode, run:\n")
  print(f"    python replay.py {results_folder}")
  print(f"\n")


def evaluate_model(results_folder, evaluation_trials):
  """
  Evaluate a trained model from the specified results folder.

  Args:
      results_folder (str): Path to the folder containing the model and config.
      evaluation_trials (int): Number of evaluation trials to run.

  Returns:
      None
  """
  env, model, model_name, env_name = get_env_and_model(results_folder)

  print(f"Evaluating model: {model_name}")
  print(f"Environment: {env_name}")
  print(f"Number of evaluation trials: {evaluation_trials}")

  # Run evaluation
  all_rewards = []
  success_count = 0

  for i in range(evaluation_trials):
    state, _ = env.reset()
    done = False
    rewards = []
    while not done:
      action, _ = model.predict(state)
      state, reward, terminated, truncated, info = env.step(action)
      done = terminated or truncated

      rewards.append(reward)

    success = check_success(state, terminated)
    episode_reward = np.sum(reward)

    all_rewards.append(episode_reward)
    if success:
      success_count += 1

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
  os.makedirs(results_folder, exist_ok=True)

  results_json_file = os.path.join(results_folder, "results.json")
  with open(results_json_file, "r") as f:
    results = json.load(f)
    results["average_reward"] = average_reward
    results["variance_reward"] = variance_reward
    results["success_rate"] = success_rate

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
