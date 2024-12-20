# Gym-Lander-Playground

This project provides a framework to train and evaluate reinforcement learning (RL) models using the Gymnasium library. It supports comparing models on metrics such as success rate, learning stability, and sample efficiency.


## Future Work
- Trial Gymnasium robotics environments.
- 

## Features

- Train reinforcement learning models with ease using a generic interface.
- Save training results, including performance metrics and the best episode.
- Replay the best episode for visual inspection of model performance.
- Preconfigured with PPO as the default starting model.
- Supports loading and evaluating models on different Gymnasium environments.


## Installation
```bash
pip install -r requirements.txt
```

## Usage

### Training a Model

To train the default PPO model on the default environment (`LunarLanderContinuous-v3`), run:
```bash
python playground.py ppo 10
```

- `ppo`: The RL model to use (defaults to PPO).
- `10`: Number of training episodes.

To train on a different environment, specify the environment name as a third argument:
```bash
python playground.py ppo 10 CartPole-v1
```

Training results will be saved in a folder named `[model]-[datetime]` (e.g., `ppo-20231123-123456`).

### Replaying the Best Episode

To replay the best episode, use:
```bash
python replay.py [path-to-result-folder]
```

Example:
```bash
python replay.py results/ppo-20231123-123456
```

### Analyzing Results

To generate a sortable table of results across all experiments, open a Jupyter Notebook and use the provided code snippet in the repository. This will create an overview of metrics like average reward, variance, and the best episode reward for each run.

## Example Workflow

1. Train the PPO model on a Gymnasium environment:
    ```bash
    python playground.py ppo 20
    ```

2. Inspect the generated folder (e.g., `results/ppo-20231123-123456`) for:
   - `config.json`: Configuration used during training.
   - `results.json`: Training performance metrics.
   - `best_episode.json`: Details of the best episode.

3. Replay the best episode:
    ```bash
    python replay.py results/ppo-20231123-123456
    ```

4. Analyze results across runs by using the Jupyter Notebook cell provided.

## Customization

This framework allows you to:
- Experiment with different Gymnasium environments by specifying the environment name when running `playground.py`.
- Add new RL models by creating Python files in the `models/` directory. Implement a `Model` class with the required `train()` method.
- Modify hyperparameters for PPO by editing the `models/ppo.py` file.

## Dependencies

All dependencies are listed in `requirements.txt`. Install them using:
```bash
pip install -r requirements.txt
```