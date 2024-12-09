import sys
import gymnasium as gym

from playground import create_results_folder, run_model
from reward import REWARD_STRATEGIES

# ddpg.py (Deep Deterministic Policy Gradient)
# laber.py (Likely supports continuous actions due to enhancements over DDPG/SAC)
# mac.py (Mean Actor-Critic, continuous if used similarly to SAC/TD3)
# ppo.py (Proximal Policy Optimization - can handle continuous actions)
# sac.py (Soft Actor-Critic, explicitly for continuous action spaces)
# td3.py (Twin Delayed DDPG, designed for continuous actions)
# trpo.py (Trust Region Policy Optimization, can handle continuous actions)

from models.ddpg import Model as DDPG
from models.laber import Model as LABER
from models.mac import Model as MAC
from models.ppo import Model as PPO
from models.sac import Model as SAC
from models.td3 import Model as TD3
from models.trpo import Model as TRPO
from models.totd import Model as TOTD


def initiate_models(model_name, env_name, reward_strategy):
  if model_name == "PPO":
    return [
        ("PPO", PPO(env_name, reward_strategy=reward_strategy, gamma=0.99, gae_lambda=0.95, clip_range=0.2, ent_coef=0.01)),  # Standard PPO parameters.
        ("PPO", PPO(env_name, reward_strategy=reward_strategy, gamma=0.98, gae_lambda=0.92, clip_range=0.15, ent_coef=0.02)),  # Increased entropy for exploration.
        ("PPO", PPO(env_name, reward_strategy=reward_strategy, gamma=0.97, gae_lambda=0.9, clip_range=0.25, ent_coef=0.005)),  # Exploitation-focused with larger policy updates.
        ("PPO", PPO(env_name, reward_strategy=reward_strategy, gamma=0.96, gae_lambda=0.88, clip_range=0.18, ent_coef=0.015)),  # Balances exploration and advantage estimation.
    ]
  if model_name == "SAC":
    return [
        ("SAC", SAC(env_name, reward_strategy=reward_strategy, buffer_size=1000000, batch_size=256, gamma=0.99, tau=0.005, learning_rate=0.2)),  # Stable entropy regularization.
        ("SAC", SAC(env_name, reward_strategy=reward_strategy, buffer_size=500000, batch_size=128, gamma=0.98, tau=0.01, learning_rate=0.15)),  # Balanced exploration-exploitation.
        ("SAC", SAC(env_name, reward_strategy=reward_strategy, buffer_size=750000, batch_size=64, gamma=0.97, tau=0.02, learning_rate=0.1)),  # Rapid updates for dynamic rewards.
        ("SAC", SAC(env_name, reward_strategy=reward_strategy, buffer_size=800000, batch_size=256, gamma=0.96, tau=0.01, learning_rate=0.25)),  # Larger buffer for smooth learning.
    ]
  if model_name == "TD3":
    return [
        ("TD3", TD3(env_name, reward_strategy=reward_strategy, buffer_size=500000, batch_size=128, gamma=0.99, tau=0.01)),  # Standard TD3 setup.
        ("TD3", TD3(env_name, reward_strategy=reward_strategy, buffer_size=1000000, batch_size=256, gamma=0.98, tau=0.005)),  # Conservative exploration noise.
        ("TD3", TD3(env_name, reward_strategy=reward_strategy, buffer_size=250000, batch_size=64, gamma=0.97, tau=0.02)),  # Aggressive learning with reduced noise.
        ("TD3", TD3(env_name, reward_strategy=reward_strategy, buffer_size=750000, batch_size=128, gamma=0.96, tau=0.01)),  # Balanced learning and exploration.
    ]
  if model_name == "TRPO":
    return [
        ("TRPO", TRPO(env_name, reward_strategy=reward_strategy, gamma=0.99, gae_lambda=0.95, target_kl=0.01)),  # Conservative KL, high discount for long-term planning.
        ("TRPO", TRPO(env_name, reward_strategy=reward_strategy, gamma=0.98, gae_lambda=0.90, target_kl=0.02)),  # Balanced KL and advantage estimation for dynamic phases.
        ("TRPO", TRPO(env_name, reward_strategy=reward_strategy, gamma=0.97, gae_lambda=0.92, target_kl=0.015)),  # Moderate gamma and target KL for faster adaptation.
        ("TRPO", TRPO(env_name, reward_strategy=reward_strategy, gamma=0.96, gae_lambda=0.85, target_kl=0.03)),  # Aggressive KL for larger policy updates, faster convergence.
    ]
  if model_name == "DDPG":
    return [
        ("DDPG", DDPG(env_name, reward_strategy=reward_strategy, buffer_size=500000, batch_size=128, gamma=0.98, tau=0.01, learning_rate=5e-4)),  # Balanced parameters for general performance.
        ("DDPG", DDPG(env_name, reward_strategy=reward_strategy, buffer_size=1000000, batch_size=256, gamma=0.99, tau=0.005, learning_rate=1e-4)),  # Conservative learning for long-term stability.
        ("DDPG", DDPG(env_name, reward_strategy=reward_strategy, buffer_size=250000, batch_size=64, gamma=0.97, tau=0.02, learning_rate=1e-3)),  # Smaller buffer and aggressive learning for faster updates.
        ("DDPG", DDPG(env_name, reward_strategy=reward_strategy, buffer_size=750000, batch_size=256, gamma=0.96, tau=0.01, learning_rate=2e-4)),  # Hybrid exploration-exploitation parameters.
    ]


def main():
  if len(sys.argv) < 2:
    print("Usage: python suite.py [episodes] [stratery] [optional_env]")
    sys.exit(1)

  try:
    total_timesteps = int(sys.argv[1])
  except ValueError:
    print("Error: Number of episodes must be an integer.")
    sys.exit(1)

  reward_strategy_name = "default" if len(sys.argv) < 3 else sys.argv[2]

  env_name = "LunarLanderContinuous-v3" if len(sys.argv) < 4 else sys.argv[3]

  model_names = ["PPO", "DDPG", "SAC", "TRPO", "TD3"]
  for model_name in model_names:
    models = initiate_models(model_name, env_name, reward_strategy_name)

    for model_name, model in models:
      print(f"Training {model_name} for {total_timesteps} episodes on {env_name} environment.")

      config = {
          "model": model_name,
          "reward_strategy": reward_strategy_name,
          "environment": env_name,
          "model_params": model.parameters,
          "timesteps": total_timesteps,
      }

      results_folder = create_results_folder(model_name, config)

      run_model(model, total_timesteps, results_folder)


if __name__ == "__main__":
  main()
