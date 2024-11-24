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
      ("TOTD", TOTD(env)),
      ("TRPO", TRPO(env)),
      ("DDPG", DDPG(env)),
      ("LABER", LABER(env)),
      ("MAC", MAC(env)),
      ("PPO", PPO(env)),
      ("SAC", SAC(env)),
      ("TD3", TD3(env))
  ]


def main():
  if len(sys.argv) < 2:
    print("Usage: python suite.py [episodes] [optional_env]")
    sys.exit(1)

  try:
    num_episodes = int(sys.argv[1])
  except ValueError:
    print("Error: Number of episodes must be an integer.")
    sys.exit(1)

  # Use LunarLanderContinuous-v3 as the default environment, allow override
  env_name = sys.argv[2] if len(sys.argv) > 2 else "LunarLanderContinuous-v3"

  reward_strategy_name = "default" if len(sys.argv) < 4 else sys.argv[3]

  try:
    env = gym.make(env_name)
  except gym.error.Error as e:
    print(f"Error: Unable to create environment '{env_name}'.\n{e}")
    sys.exit(1)

  # Wrap environment with selected reward strategy
  reward_strategy = REWARD_STRATEGIES.get(reward_strategy_name)
  env = wrap_environment(env, reward_strategy)

  models = initiate_models(env)

  for model_name, model in models:
    print(f"Training {model_name} for {num_episodes} episodes on {env_name} environment.")

    config = {
        "model": model_name,
        "episodes": num_episodes,
        "reward_strategy": reward_strategy_name,
        "environment": env_name,
    }

    results_folder = create_results_folder(model_name, config)
    run_model(model, num_episodes, results_folder, env)


if __name__ == "__main__":
  main()
