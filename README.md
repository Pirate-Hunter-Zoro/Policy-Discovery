# Policy Discovery – Reinforcement Learning Assignment

This repository contains implementations and experiments for a reinforcement learning assignment, covering **Dynamic Programming (DP)**, **Tabular RL (Q-Learning)**, and **Deep Q-Learning (DQN)**.  
It is structured into three main parts, each corresponding to a core set of algorithms and experiments.

---

## Project Structure

```bash
Policy-Discovery/
├── mdp/                # Part I – Dynamic Programming for MDPs
│   ├── MDP.py
│   ├── TestMDP.py
│   ├── TestMDPMaze.py
│   └── __init__.py
├── rl/                 # Part II – Tabular RL (Q-Learning)
│   ├── RL.py
│   ├── TestRL.py
│   ├── TestRLMaze.py
│   └── __init__.py
├── dqn/                # Part III – Deep Q-Learning
│   ├── DQN.py
│   ├── TestDQN.py
│   ├── outputs/        # Saved plots and CSV logs
│   └── utils/          # Shared utilities
│       ├── buffers.py
│       ├── envs.py
│       ├── seed.py
│       ├── torch.py
│       └── common.py
├── requirements.txt
└── README.md
```

## Part I – Dynamic Programming for MDPs

### File: mdp/MDP.py

Implements classical planning algorithms:
 • Value Iteration (VI)
 • Policy Iteration (PI)
 • Modified Policy Iteration (MPI)
 • Policy Evaluation (exact and partial)
 • Policy Extraction

Experiments:
 • TestMDP.py — toy 4-state MDP (from Lecture 2a, slides 13–14).
 • TestMDPMaze.py — 4×4 slip maze with goal, bad, and absorbing end state.

Run:

```bash
python -m mdp.TestMDP
python -m mdp.TestMDPMaze
```

⸻

## Part II – Tabular RL (Q-Learning)

### File: rl/RL.py

Implements model-free Q-Learning with:
 • ε-greedy exploration
 • Boltzmann (softmax) exploration
 • Sampling of rewards and next states

Experiments:
 • TestRL.py — sanity check Q-Learning on the toy MDP.
 • TestRLMaze.py — Q-Learning in the maze domain with different ε values.
Produces learning curves (rl_maze_avg_returns.png, CSV logs).

Run:

```bash
python -m rl.TestRL
python -m rl.TestRLMaze
```

⸻

## Part III – Deep Q-Learning (DQN)

### File: dqn/DQN.py

Implements a Deep Q-Network agent trained on CartPole-v1.

Features:
 • Replay buffer (experience replay)
 • Target network updates
 • ε-greedy exploration with annealing
 • Torch MLP (2 hidden layers, 512 units each)

Utilities (dqn/utils/):
 • envs.py — play episodes, handle Gymnasium API
 • buffers.py — replay buffer implementation
 • seed.py — reproducibility helpers
 • torch.py — Torch helper functions

Experiments:
 • TestDQN.py — training/evaluation driver for DQN
 • Sweeps over target update frequency and minibatch size
 • Saves learning curves (.png) and per-episode averages (.csv) to dqn/outputs/.

Run:

```bash
python -m dqn.TestDQN
```

⸻

## Results (Summary)

 • Part I (DP): VI, PI, and MPI all converge to the same optimal policies and values on both toy MDP and maze.
 • Part II (Q-Learning): Effective learning with small ε (0.05–0.1); high ε (0.3–0.5) degrades performance.
 • Part III (DQN): Stable CartPole performance when tuned:
 • Best with target update ≈ 10 episodes.
 • Larger minibatches (50–100) yield smoother, faster learning.

⸻

Notes
 • Uses Gymnasium instead of Gym; for rendering, use render_mode="human" in gym.make().
 • Reward sampling in Part II can be stochastic, but experiments use deterministic means.
 • Set seeds via dqn/utils/seed.py for reproducibility.
 • Plots and logs are written into each part’s outputs/ directory.
