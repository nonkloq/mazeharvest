# MazeHarvest

MazeHarvest is a partially observable, stochastic reinforcement learning environment designed to train autonomous navigation agents. For a detailed description, refer to [homegym/README.md](./homegym/README.md). In summary, the agent must rely on sensory data like ray perception and heuristics to accomplish the following tasks:

- **Navigate complex mazes** using heuristic information and sensory inputs.
- **Interact with dynamic obstacles** such as walls and hostile entities.
- **Reach target locations efficiently** while adapting to environmental challenges.

To install `homegym`, follow the steps below in the [Installation](#installation) section.

<!-- <img src="./experiments/aperf/rdqn_vision_net.gif" width="500"> -->

<video src="https://github.com/user-attachments/assets/68a4458f-2c49-4519-9269-16c9782d2ea5" width="540" height="340"></video>

<details>
  <summary>Agent Details</summary>
  
  > LSTM-PPO Agent, Recurrent Policy with episodic memory.
</details>

---

## Experiments 

Implementation of few RL algorithms using Pytorch in the [experiments/algos](./experiments/algos) directory to solve this environment. Each folder contains these two file:

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
| **LSTM PPO** | `alogs/lstmppo` | Proximal Policy Optimization with an LSTM layer, truncated BPTT and previous action encoding in observation.|

--- 
## Installation

### Install MazeHarvest Environment

To install only the MazeHarvest environment, follow these steps:

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/nonkloq/mazeharvest.git
   ```

2. **Navigate to the homegym directory**:
   ```bash
   cd homegym
   ```

3. **Install the package**:
   ```bash
   pip install -e .
   ```

### Setup for Experiments

To setup the environment for running experiments with MazeHarvest, follow these steps:

1. **Create a Conda Environment**:
   ```bash
   conda create --name mazeharvest_env python=3.12
   conda activate mazeharvest_env
   ```

2. **Install PyTorch**:
   Visit the [official PyTorch website](https://pytorch.org/get-started/previous-versions/) to select the required version >=2.6.0 for your system, whether you need CPU or GPU support. Follow the instructions provided there to install PyTorch.
   ```bash
   # All experiments can be run on CPU
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
   ```

3. **Install Additional Requirements**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Install Homegym**:
   ```bash
   cd homegym
   pip install -e .
   ```

With this setup, you can run reinforcement learning experiments with all the implemented RL algorithms, which is also required for both training and testing.

### Training Scripts

Each training script follows the naming convention `*_train.py` in the experiments directory, where `*` corresponds to the algorithm name. For example, `ppo_train.py` is the script to train the MazeHarvest environment using PPO. You can modify the hyperparameters and environment configurations to start training.

### Testing Scripts

- `experiments/runsim2d.py` runs the trained agent and renders using pygame for 2D visualization.
- `experiments/runsim3d.py` is similar to `runsim2d.py` but uses panda3d for 3D rendering.

Currently, the best agent is LSTM-PPO, and the runsim-testing scripts are specifically made for this agent. You can modify the environment configurations to test different variations of the environments in these scripts.

## License

This project is licensed under the MIT License.
