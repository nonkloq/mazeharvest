# MazeHarvest

MazeHarvest is a partially observable, stochastic reinforcement learning environment. For a detailed description, refer to [homegym/README.md](./homegym/README.md). In summary, the agent must rely on sensory data like ray perception and heuristics to accomplish the following tasks:

- **Kill hostile moles** that hunt the agent.
- **Harvest toxic plants** to reduce environmental toxicity and survive.
- **Navigate complex mazes** in advanced settings.

To install `homegym`, clone this repository and follow the instructions in [homegym/README.md](./homegym/README.md).

---

## Experiments 

I have Implemented few RL algorithms using Pytorch in the [experiments/algos](./experiments/algos) directory to solve this environment. Each folder contains these two file:

- **toy_test.py**: Inherits the base algorithm class to test it on gymnasium environments and logs the test results in `ttresults.md`.

- **ttresults.md**: Logs with:
    Hyperparameters used during training.
    Descriptions of the algorithm and its implementation.
    Test results: performance across different environments.
    Evaluation scores for 100 episodes.

**Test environments**: The algorithms are tested on Gymnasium toy environments to ensure the implmentation robustness.

### Implemented Algorithms

| Algorithm     | Folder         | Description                                                                 |
|---------------|----------------|-----------------------------------------------------------------------------|
| **PPO**       | `algos/ppo`    | Simple implementation of Proximal Policy Optimization (PPO) with Generalized Advantage Estimation (GAE). |
| **Rainbow DQN** | `algos/rdqn`  | Combines C51, N-Step Learning (random M to N steps, instead of fixed N steps), Prioritized Experience Replay (PER), Double Q-Learning, and Dueling Network Architecture. |

---

### Utilities

The [experiments/eutils](./experiments/eutils) directory contains utility functions for:

- Processing observations.
- Training & Testing models for this environment.

---

### Training Scripts

Each training script follows the naming convention `*_train.py`, where `*` corresponds to the algorithm name. For example, `ppo_train.py` is the script to train the MazeHarvest environment using PPO.
