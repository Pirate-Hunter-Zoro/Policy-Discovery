#!/usr/bin/env python3
"""
TestRLMaze.py
Assignment 1 Part II: Q-learning on the 4x4 slip maze with absorbing end.

Requirements satisfied:
- Maze is same as mdp/TestMDPMaze.py builder (|S|=17, |A|=4).
- Initial state s0 = 0.
- Initial Q = 0 for all state-action pairs.
- 200 episodes, 100 steps per episode.
- For eps ∈ {0.05, 0.1, 0.3, 0.5}, compute the average (over 100 trials)
  of the cumulative discounted reward per episode (length 100).
- Produce a single plot with 4 curves and save CSV with the averages.

Outputs:
  rl_maze_avg_returns.csv
  rl_maze_avg_returns.png
"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from mdp.TestMDPMaze import build_maze_mdp
from rl.RL import RL

np.set_printoptions(precision=6, suppress=True)

def sample_reward(mean):
    # Deterministic sampling from reward "distribution" (use the mean).
    return float(mean)

def run_trials(epsilon, n_trials, n_episodes, n_steps, seed0=0):
    mdp = build_maze_mdp()
    all_returns = np.zeros((n_trials, n_episodes), dtype=float)
    for t in range(n_trials):
        np.random.seed(seed0 + t)
        rl = RL(mdp, sample_reward)
        Q0 = np.zeros((mdp.nActions, mdp.nStates), dtype=float)
        Q, policy, ep_returns = rl.qLearning(
            s0=0,
            initialQ=Q0,
            nEpisodes=n_episodes,
            nSteps=n_steps,
            epsilon=epsilon,
            temperature=0.0
        )
        all_returns[t, :] = ep_returns
    return all_returns

def main():
    out_dir = Path("./rl")
    epsilons = [0.05, 0.1, 0.3, 0.5]
    n_trials = 100
    n_episodes = 200
    n_steps = 100

    avg_by_eps = {}
    for eps in epsilons:
        returns = run_trials(eps, n_trials, n_episodes, n_steps, seed0=1337)
        avg = returns.mean(axis=0)
        avg_by_eps[eps] = avg
        print(f"Epsilon={eps:0.2f}  avg cumulative discounted reward (episode 0..{n_episodes-1}) computed.")

    # Save CSV
    csv_path = out_dir / "rl_maze_avg_returns.csv"
    with open(csv_path, "w") as f:
        header = "episode," + ",".join([f"eps_{str(eps).replace('.','p')}" for eps in epsilons]) + "\n"
        f.write(header)
        for ep in range(n_episodes):
            row = [str(ep)] + [f"{avg_by_eps[eps][ep]:.6f}" for eps in epsilons]
            f.write(",".join(row) + "\n")
    print(f"Saved CSV: {csv_path}")

    # Plot
    plt.figure()
    x = np.arange(n_episodes)
    for eps in epsilons:
        plt.plot(x, avg_by_eps[eps], label=f"epsilon={eps}")
    plt.xlabel("Episode")
    plt.ylabel("Avg cumulative discounted reward (n=100, 100 steps)")
    plt.title("Q-learning on Maze: effect of epsilon")
    plt.legend(loc="best")
    png_path = out_dir / "rl_maze_avg_returns.png"
    plt.tight_layout()
    plt.savefig(png_path, dpi=150)
    print(f"Saved plot: {png_path}")

if __name__ == "__main__":
    main()
