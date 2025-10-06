# Policy Discovery – Reinforcement Learning Assignments

This repository contains implementations and experiments for reinforcement learning assignments, covering **Dynamic Programming (DP)**, **Tabular RL (Model-Based, Q-Learning, and Bandits)**, and **Deep Q-Learning (DQN)**.

---

## Project Structure

```bash
Policy-Discovery/
├── dqn/                    # Part III – Deep Q-Learning
│   ├── DQN.py
│   ├── TestDQN.py
│   ├── outputs/
│   │   ├── part3_minibatch_sweep.csv
│   │   ├── part3_minibatch_sweep.png
│   │   ├── part3_target_update_sweep.csv
│   │   └── part3_target_update_sweep.png
│   └── utils/
├── mdp/                    # Part I – Dynamic Programming for MDPs
│   ├── MDP.py
│   ├── TestMDP.py
│   └── TestMDPMaze.py
├── rl/                     # Part II – Tabular RL
│   ├── RL.py
│   ├── RL2.py
│   ├── TestRL.py
│   ├── TestRL2.py
│   ├── TestRLMaze.py
│   ├── results/
│   │   ├── grid_world.png
│   │   └── multi_arm_bandit.png
│   ├── rl_maze_avg_returns.csv
│   └── rl_maze_avg_returns.png
├── README.md
└── requirements.txt
```

---

## Part I – Dynamic Programming for MDPs

### File: `mdp/MDP.py`

Implements classical planning algorithms for known MDPs:

* Value Iteration (VI)
* Policy Iteration (PI)
* Modified Policy Iteration (MPI)

### Part I Experiments

* `TestMDP.py` — Solves a toy 4-state MDP from lecture slides.
* `TestMDPMaze.py` — Solves a 4×4 maze with stochastic transitions.

### Part I Run

```bash
python -m mdp.TestMDP
python -m mdp.TestMDPMaze
```

---

## Part II – Tabular RL

This part covers several model-free and model-based tabular reinforcement learning algorithms.

### Files: `rl/RL.py`, `rl/RL2.py`

Implements the following algorithms:

* **Q-Learning** with ε-greedy and Boltzmann exploration (`RL.py`)
* **Model-Based RL** with planning via Value Iteration (`RL2.py`)
* **Multi-Arm Bandit Algorithms** (`RL2.py`):
  * Epsilon-Greedy
  * Thompson Sampling
  * Upper Confidence Bound (UCB)

### Part II Experiments

* `TestRLMaze.py` — Compares Q-Learning performance in the maze domain with different ε values.
* `TestRL2.py` — Conducts two main experiments:
    1. Compares the performance of the three multi-arm bandit algorithms.
    2. Compares the learning curves of Model-Based RL vs. Q-Learning.

### Part II Run

```bash
python -m rl.TestRLMaze
python -m rl.TestRL2
```

---

## Part III – Deep Q-Learning (DQN)

### File: `dqn/DQN.py`

Implements a Deep Q-Network agent trained on the `CartPole-v1` environment.

### Features

* Replay Buffer (Experience Replay)
* Target Network for stable Q-learning
* ε-greedy exploration with annealing
* MLP architecture using PyTorch

### Part III Experiments

* `TestDQN.py` — Training and evaluation driver for the DQN agent.
* Performs sweeps over hyperparameters like target update frequency and minibatch size.
* Saves learning curves and logs to `dqn/outputs/`.

### Part III Run

```bash
python -m dqn.TestDQN
```

---

## Results (Summary)

* **Part I (DP)**: VI, PI, and MPI all converge to the same optimal policies and value functions on the provided MDPs.
* **Part II (Tabular RL)**:
  * **Q-Learning**: Shows effective learning in the maze with small ε values (0.05–0.1); higher exploration can degrade performance.
  * **Bandits**: Thompson Sampling and UCB significantly outperform the simple Epsilon-Greedy strategy, with Thompson Sampling showing the best performance.
  * **Model-Based vs. Q-Learning**: Model-Based RL exhibits high sample efficiency (very fast initial learning) but can suffer from performance collapse due to exploiting a flawed model. Q-Learning is slower and less efficient but provides more stable and robust learning.
* **Part III (DQN)**: Achieves stable performance on `CartPole-v1` with proper hyperparameter tuning. Larger minibatches and a moderate target network update frequency (e.g., every 10 episodes) lead to the best results.

---

### Notes

* Uses `Gymnasium` instead of `Gym`. For rendering, use `render_mode="human"` in `gym.make()`.
* Set seeds via `dqn/utils/seed.py` for reproducibility in the DQN experiments.
* Plots and logs are written into each part’s respective `outputs/` or `results/` directory.
