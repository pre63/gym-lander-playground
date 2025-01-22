# Gymnasium Lunar Lander Evaluations (MAC, SARSA, AC)++

This repository provides a framework for exploring and comparing reinforcement learning (RL) algorithms in continuous action spaces, focusing on both well-established methods like PPO and SAC and newer approaches such as Mean Actor-Critic (MAC) and Deep True Online TD(λ)-Replay (TOTDLR). By integrating these algorithms into a consistent evaluation pipeline, the repository enables detailed analysis of their performance across environments, particularly the LunarLanderContinuous-v3 task.

## Overview

Reinforcement learning in continuous control settings poses unique challenges, particularly around stability, scalability, and efficiency. This repository combines standard baselines with experimental approaches to explore these aspects in greater depth. Algorithms like MAC and TOTDLR extend existing paradigms by incorporating mechanisms for variance reduction and replay, offering potentially improved performance in complex tasks. The LunarLanderContinuous-v3 environment provides a well-suited testing ground for these techniques, balancing complexity with interpretability.

The inclusion of models like MAC, which explicitly leverages action value representations to reduce variance, and TOTDLR, which combines eligibility traces with replay for efficient planning, highlights efforts to bridge gaps between theoretical potential and practical application in RL. Established baselines such as PPO, SAC, and TD3 provide essential benchmarks for contextualizing results.

## Algorithms

### Experimental Models

- **Mean Actor-Critic (MAC):** A policy gradient algorithm that uses all action values to compute gradients rather than relying solely on executed actions. This approach reduces gradient variance, improving learning stability and sample efficiency. The implementation follows Allen et al. ([paper](https://doi.org/10.48550/arXiv.1709.00503)).
  
- **Deep True Online TD(λ)-Replay (TOTDLR):** Combines eligibility traces with experience replay for model-free planning. This hybrid approach balances real-time learning with memory-based updates, as described by Altahhan ([paper](https://ieeexplore.ieee.org/document/9206608)).

### Baseline Models

- **PPO (Proximal Policy Optimization):** A widely-used policy gradient algorithm that ensures stability with clipped updates.
- **SAC (Soft Actor-Critic):** An entropy-regularized method designed for efficient exploration in continuous control tasks.
- **TD3 (Twin Delayed DDPG):** Optimized for deterministic policy updates in continuous spaces.
- **TRPO (Trust Region Policy Optimization):** Constrains policy updates for stability and convergence.
- **SARSA and Actor-Critic (AC):** Fundamental on-policy algorithms, including adaptations for continuous spaces.

## Features

The repository simplifies the evaluation of RL models by automating key processes such as training, logging, and replay. Training results include performance metrics, configuration details, and data from the best-performing episode, all saved in a structured format. Users can easily replay episodes to visually inspect the agent’s behavior, providing insights into how different models approach specific tasks.

The LunarLanderContinuous-v3 environment serves as the primary testing domain, but the framework supports any Gymnasium-compatible environment. This flexibility allows users to extend experiments to other domains, such as robotic control or high-dimensional state spaces.

## Usage

To train a model, run:
```bash
python playground.py ppo 10
```
This trains PPO for 10 episodes on the default environment. Results are saved in a timestamped folder. To replay the best episode:
```bash
python replay.py results/ppo-20231123-123456
```

For analysis, the included Jupyter Notebook provides tools to compare metrics like average rewards and stability across experiments.

## Discussion

Algorithms like MAC and TOTDLR offer interesting extensions to traditional RL paradigms. MAC’s ability to reduce gradient variance can be especially valuable in environments with sparse or noisy rewards, where sample efficiency is critical. TOTDLR, by combining eligibility traces and replay, demonstrates the potential to improve learning stability and long-term planning. These techniques complement established methods like PPO and SAC, which excel in balancing stability and exploration.

The LunarLanderContinuous-v3 environment provides a practical testbed for evaluating these algorithms. Its continuous state and action spaces, coupled with clear success criteria, enable detailed comparisons across methods. While TOTDLR and MAC show promise, further work is needed to refine their implementations, particularly in addressing stability and scalability in more complex environments.

## Future Directions

There are several avenues for building on this work:
- **Stability Improvements:** Further efforts are needed to enhance the numerical stability of experimental models like TOTDLR, particularly in environments with high-dimensional state spaces or sparse rewards.
- **Dynamic Action Spaces:** Adapting these models to handle environments where the available actions change dynamically could extend their applicability.
- **Robotics and Mujoco Tasks:** Expanding evaluations to robotic control environments or Mujoco-based tasks could provide insights into their performance in real-world-inspired domains.
- **MAC for Continuous Domains:** While MAC is designed for discrete actions, extending it to continuous spaces could further reduce variance in policy gradients and improve learning efficiency.
- **Noisy Environments:** Exploring how these models perform in environments with high observation or reward noise would offer a more comprehensive understanding of their robustness.

## Conclusion
This repository provides tools and implementations for experimenting with RL models in continuous control environments. By combining established baselines with experimental techniques, it lays a foundation for further exploration and refinement of RL methods. Researchers and practitioners can use it to test new ideas, compare algorithms, and gain insights into the complexities of reinforcement learning in dynamic settings.
