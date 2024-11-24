import os
import sys
import numpy as np
import subprocess


def find_best_episode_file(results_folder):
  """
  Find the best_episode.npz file in the given results folder.
  Args:
      results_folder (str): Path to the folder containing the results.
  Returns:
      str: Path to the best_episode.npz file.
  """
  best_episode_file = os.path.join(results_folder, "best_episode.npz")
  if not os.path.exists(best_episode_file):
    print(f"Error: File '{best_episode_file}' does not exist in the specified folder.")
    sys.exit(1)
  return best_episode_file


def replay_best_episode(results_folder):
  """
  Replay the best episode using the frames saved in the .npz file and save the video in the results folder.
  Args:
      results_folder (str): Path to the folder containing the results.
  """
  best_episode_file = find_best_episode_file(results_folder)

  # Load the RGB frames
  data = np.load(best_episode_file)
  if "frames" not in data:
    print("Error: No 'frames' key found in the best_episode.npz file.")
    sys.exit(1)

  frames = data["frames"]
  output_video = os.path.join(results_folder, "best_episode.mp4")

  try:
    height, width, _ = frames[0].shape
    input_file = os.path.join(results_folder, "temp_frames.raw")

    # Save raw RGB frames to a binary file
    with open(input_file, "wb") as f:
      for frame in frames:
        f.write(frame.tobytes())

    # Use FFmpeg to convert the raw RGB file into a playable MP4 video
    # The key change here is explicitly specifying the duration and proper frame data
    subprocess.run([
        "ffmpeg",
        "-y",  # Overwrite existing files
        "-f", "rawvideo",
        "-pixel_format", "rgb24",
        "-video_size", f"{width}x{height}",
        "-framerate", "30",  # Ensures the correct video length
        "-i", input_file,
        "-vcodec", "libx264",
        "-pix_fmt", "yuv420p",  # Ensure compatibility
        "-loglevel", "error",  # Suppress verbose output
        output_video
    ], check=True)

    print(f"Video saved at: {output_video}")

    # Play the video using an external player (e.g., ffplay)
    subprocess.run(["ffplay", "-autoexit", "-loglevel", "error", output_video], check=True)

  except Exception as e:
    print(f"Error during replay: {e}")
    sys.exit(1)
  finally:
    # Cleanup temporary raw file
    if os.path.exists(input_file):
      os.remove(input_file)


def main():
  if len(sys.argv) < 2:
    print("Usage: python replay.py [path-to-results-folder]")
    sys.exit(1)

  results_folder = sys.argv[1]
  if not os.path.isdir(results_folder):
    print(f"Error: Folder '{results_folder}' does not exist.")
    sys.exit(1)

  replay_best_episode(results_folder)


if __name__ == "__main__":
  main()
