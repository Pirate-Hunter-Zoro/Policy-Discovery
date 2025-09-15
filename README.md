# Policy Discovery – Reinforcement Learning Assignment

This repository contains implementations and experiments for a reinforcement learning assignment covering **Dynamic Programming**, **Tabular RL**, and **Deep Q-Learning**.

---

## Project Structure

```bash
Policy-Discovery/
├── mdp/                # Part 1 – MDP algorithms
│   ├── MDP.py
│   ├── TestMDP.py
│   ├── TestMDPMaze.py
│   └── __init__.py
├── rl/                 # Part 2 – Tabular RL (Q-learning)
│   ├── RL.py
│   ├── TestRL.py
│   ├── TestRLMaze.py
│   └── __init__.py
├── dqn/                # Part 3 – Deep Q-Networks
│   ├── DQN.py
│   ├── utils/
│   │   ├── buffers.py
│   │   ├── common.py
│   │   ├── envs.py
│   │   ├── seed.py
│   │   └── torch.py
│   └── __init__.py
├── requirements.txt
└── README.md
```

---

## Installation

Clone this repository and create a virtual environment:

```bash
git clone <your-repo>
cd Policy-Discovery
conda create -n rl python=3.10
conda activate rl
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Dependencies include:

- numpy
- matplotlib
- tqdm
- torch
- gymnasium (drop-in replacement for gym)

---

## Part 1 – Markov Decision Processes

**File:** `mdp/MDP.py`  
Implements classical planning algorithms:

- Value Iteration
- Policy Iteration
- Modified Policy Iteration
- Policy Evaluation (exact and partial)
- Policy Extraction

**Tests:**

- `TestMDP.py` – runs algorithms on a toy 4-state MDP (Lecture 2a, slides 13–14).
- `TestMDPMaze.py` – larger gridworld with goal/bad states and absorbing end state.

**Run:**

```bash
python -m mdp.TestMDP
python -m mdp.TestMDPMaze
```

---

## Part 2 – Tabular RL (Q-Learning)

**File:** `rl/RL.py`  
Implements model-free RL:

- Q-Learning with:
  - ε-greedy exploration
  - Boltzmann/softmax exploration
- Reward/state sampling via `sampleRewardAndNextState`.

**Tests:**

- `TestRL.py` – sanity check Q-Learning on a toy MDP.
- `TestRLMaze.py` – Q-Learning in the maze domain with varying ε values.

**Run:**

```bash
python -m rl.TestRL
python -m rl.TestRLMaze
```

---

## Part 3 – Deep Q-Learning

**File:** `dqn/DQN.py`  
Implements a Deep Q-Network agent trained on CartPole-v1.

Key features:

- Replay buffer (experience replay)
- Target network updates
- ε-greedy exploration with annealing
- Torch MLP for Q(s,a)

**Utilities (`dqn/utils/`):**

- `envs.py` – episode play loops (with Gymnasium API)
- `buffers.py` – replay buffer
- `seed.py` – reproducibility helpers
- `torch.py` – Torch helper functions

**Run training:**

```bash
python -m dqn.DQN
```

This trains DQN across multiple random seeds and plots the average reward curve.

---

## Results

- Part 1: All DP algorithms converge to consistent value functions and policies on the toy MDP and maze.  
- Part 2: Q-Learning learns effective policies under different exploration rates (ε).  
- Part 3: DQN achieves stable CartPole balancing performance, with reward curves approaching the 200-step cap.

---

## Notes

- Gym has been replaced by Gymnasium; ensure environments are created with the updated API.
- Rendering requires `render_mode="human"` when calling `gym.make`.
- Reward sampling in Part 2 is stochastic (e.g., Gaussian), so policies may vary across runs.
- For reproducibility, set seeds via `utils/seed`.
