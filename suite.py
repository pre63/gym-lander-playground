import sys
import gymnasium as gym

from playground import create_results_folder, run_model, wrap_environment
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


def initiate_models(env):
  return [
      # PPO
      ("PPO", PPO(env, gamma=0.99, gae_lambda=0.95, clip_range=0.2, ent_coef=0.01)),  # Standard PPO parameters.
      ("PPO", PPO(env, gamma=0.98, gae_lambda=0.92, clip_range=0.15, ent_coef=0.02)),  # Increased entropy for exploration.
      ("PPO", PPO(env, gamma=0.97, gae_lambda=0.9, clip_range=0.25, ent_coef=0.005)),  # Exploitation-focused with larger policy updates.
      ("PPO", PPO(env, gamma=0.96, gae_lambda=0.88, clip_range=0.18, ent_coef=0.015)),  # Balances exploration and advantage estimation.

      # # TOTD
      # ("TOTD", TOTD(env, gamma=0.98, lambda_=0.9)),  # Balanced gamma, medium trace decay.
      # ("TOTD", TOTD(env, gamma=0.99, lambda_=0.95)),  # Emphasizes long-term rewards, slower decay for better credit assignment.
      # ("TOTD", TOTD(env, gamma=0.97, lambda_=0.8)),  # Shorter traces for rapid adjustments, more reactive in dynamic tasks.
      # ("TOTD", TOTD(env, gamma=0.96, lambda_=0.85)),  # Combines shorter traces with slightly less long-term focus.

      # SAC
      ("SAC", SAC(env, buffer_size=1000000, batch_size=256, gamma=0.99, tau=0.005, alpha=0.2, actor_lr=3e-4, critic_lr=3e-4)),  # Stable entropy regularization.
      ("SAC", SAC(env, buffer_size=500000, batch_size=128, gamma=0.98, tau=0.01, alpha=0.15, actor_lr=1e-3, critic_lr=1e-3)),  # Balanced exploration-exploitation.
      ("SAC", SAC(env, buffer_size=750000, batch_size=64, gamma=0.97, tau=0.02, alpha=0.1, actor_lr=5e-4, critic_lr=5e-4)),  # Rapid updates for dynamic rewards.
      ("SAC", SAC(env, buffer_size=800000, batch_size=256, gamma=0.96, tau=0.01, alpha=0.25, actor_lr=2e-4, critic_lr=2e-4)),  # Larger buffer for smooth learning.

      # TD3
      ("TD3", TD3(env, buffer_size=500000, batch_size=128, gamma=0.99, tau=0.01, actor_lr=5e-4, critic_lr=5e-4, policy_noise=0.1, noise_clip=0.25, policy_freq=2)),  # Standard TD3 setup.
      ("TD3", TD3(env, buffer_size=1000000, batch_size=256, gamma=0.98, tau=0.005, actor_lr=1e-4, critic_lr=1e-4, policy_noise=0.2, noise_clip=0.3, policy_freq=3)),  # Conservative exploration noise.
      ("TD3", TD3(env, buffer_size=250000, batch_size=64, gamma=0.97, tau=0.02, actor_lr=1e-3, critic_lr=1e-3, policy_noise=0.05, noise_clip=0.1, policy_freq=1)),  # Aggressive learning with reduced noise.
      ("TD3", TD3(env, buffer_size=750000, batch_size=128, gamma=0.96, tau=0.01, actor_lr=2e-4, critic_lr=2e-4, policy_noise=0.15, noise_clip=0.2, policy_freq=2)),  # Balanced learning and exploration.

      # TRPO
      ("TRPO", TRPO(env, gamma=0.99, gae_lambda=0.95, target_kl=0.01)),  # Conservative KL, high discount for long-term planning.
      ("TRPO", TRPO(env, gamma=0.98, gae_lambda=0.90, target_kl=0.02)),  # Balanced KL and advantage estimation for dynamic phases.
      ("TRPO", TRPO(env, gamma=0.97, gae_lambda=0.92, target_kl=0.015)),  # Moderate gamma and target KL for faster adaptation.
      ("TRPO", TRPO(env, gamma=0.96, gae_lambda=0.85, target_kl=0.03)),  # Aggressive KL for larger policy updates, faster convergence.

      # DDPG
      ("DDPG", DDPG(env, buffer_size=500000, batch_size=128, gamma=0.98, tau=0.01, learning_rate=5e-4)),  # Balanced parameters for general performance.
      ("DDPG", DDPG(env, buffer_size=1000000, batch_size=256, gamma=0.99, tau=0.005, learning_rate=1e-4)),  # Conservative learning for long-term stability.
      ("DDPG", DDPG(env, buffer_size=250000, batch_size=64, gamma=0.97, tau=0.02, learning_rate=1e-3)),  # Smaller buffer and aggressive learning for faster updates.
      ("DDPG", DDPG(env, buffer_size=750000, batch_size=256, gamma=0.96, tau=0.01, learning_rate=2e-4)),  # Hybrid exploration-exploitation parameters.

      # # LABER
      # ("LABER", LABER(env, buffer_size=200000, batch_size=256, alpha=0.2, gamma=0.97, tau=0.01)),  # Balanced prioritization and discount.
      # ("LABER", LABER(env, buffer_size=300000, batch_size=128, alpha=0.1, gamma=0.96, tau=0.005)),  # Emphasizes exploitation with conservative updates.
      # ("LABER", LABER(env, buffer_size=150000, batch_size=64, alpha=0.15, gamma=0.98, tau=0.02)),  # Prioritizes shorter-term adjustments, rapid replay.
      # ("LABER", LABER(env, buffer_size=400000, batch_size=128, alpha=0.25, gamma=0.95, tau=0.01)),  # Larger buffer and prioritization for high-dimensional states.

      # # MAC
      # ("MAC", MAC(env, buffer_size=300000, batch_size=64, actor_lr=1e-3, critic_lr=1e-3, tau=0.005)),  # Rapid updates for dynamic control.
      # ("MAC", MAC(env, buffer_size=500000, batch_size=128, actor_lr=5e-4, critic_lr=5e-4, tau=0.01)),  # Balanced parameters for steady performance.
      # ("MAC", MAC(env, buffer_size=250000, batch_size=32, actor_lr=1e-4, critic_lr=1e-4, tau=0.02)),  # Smaller memory for faster response times.
      # ("MAC", MAC(env, buffer_size=400000, batch_size=128, actor_lr=3e-4, critic_lr=3e-4, tau=0.01)),  # Larger replay memory for robust long-term learning.
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

  try:
    env = gym.make(env_name, render_mode="rgb_array")
  except gym.error.Error as e:
    print(f"Error: Unable to create environment '{env_name}'.\n{e}")
    sys.exit(1)

  # Wrap environment with selected reward strategy
  reward_strategy = REWARD_STRATEGIES.get(reward_strategy_name)
  env = wrap_environment(env, reward_strategy)

  models = initiate_models(env)

  for model_name, model in models:
    print(f"Training {model_name} for {total_timesteps} episodes on {env_name} environment.")

    config = {
        "model": model_name,
        "reward_strategy": reward_strategy_name,
        "environment": env_name,
        "model_params": model.parameters
    }

    results_folder = create_results_folder(model_name, config)

    run_model(model, total_timesteps, results_folder)


if __name__ == "__main__":
  main()
