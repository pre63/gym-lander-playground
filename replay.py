import os
import sys
import json
import gymnasium as gym


def load_results(results_folder):
  """
  Load results, configuration, and best episode data from the specified folder.
  Args:
      results_folder (str): Path to the folder containing results.
  Returns:
      config (dict): Configuration used for training.
      best_episode (dict): Best episode data.
  """
  try:
    with open(os.path.join(results_folder, "config.json"), "r") as f:
      config = json.load(f)
    with open(os.path.join(results_folder, "best_episode.json"), "r") as f:
      best_episode = json.load(f)
    return config, best_episode
  except FileNotFoundError as e:
    print(f"Error: {e}")
    sys.exit(1)


def replay_best_episode(results_folder):
  """
  Replay the best episode using the saved configuration and results.
  Args:
      results_folder (str): Path to the folder containing results.
  """
  config, best_episode = load_results(results_folder)
  env_name = config["environment"]

  print(f"Replaying Best Episode from Episode: {best_episode['episode']} with Reward: {best_episode['reward']}")

  # Create the environment with rendering
  env = gym.make(env_name, render_mode="human")  # Requires an environment supporting "human" render mode

  trajectory = best_episode["trajectory"]

  state = trajectory[0]["state"]  # Initial state
  env.reset()  # Reset environment (required before stepping)
  env.env.state = state  # Manually set the initial state if supported

  for step in trajectory:
    action = step["action"]
    reward = step["reward"]
    done = step["done"]

    if done:
      print(f"Reward: {reward}, Done")
    else:
      print(f"Reward: {reward}")

    env.render()
    _, _, _, _, _ = env.step(action)

    if done:
      break

  env.close()
  print("Replay completed.")


def main():
  if len(sys.argv) < 2:
    print("Usage: python replay.py [path-to-result-folder]")
    sys.exit(1)

  results_folder = sys.argv[1]
  if not os.path.exists(results_folder):
    print(f"Error: Folder '{results_folder}' does not exist.")
    sys.exit(1)

  replay_best_episode(results_folder)


if __name__ == "__main__":
  main()
