#!/usr/bin/env python3
"""
TestRL.py
Sanity tests for RL.qLearning on a tiny toy MDP.

- Builds a small 4-state, 2-action MDP (same structure as TestMDP).
- Runs a short Q-learning session and prints the learned policy and a slice of Q.
This file is for quick compile/run checks; Part II grading focuses on TestRLMaze.py.
"""
import numpy as np
from mdp.MDP import MDP
from rl.RL import RL

np.set_printoptions(precision=6, suppress=True)

def sample_reward(mean):
    # Deterministic reward sampling (returns the mean). Keeps focus on transition stochasticity.
    return float(mean)

def build_toy_mdp():
    # Same toy as TestMDP.py
    T = np.array([
        # action 0
        [
            [0.5, 0.5, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.5, 0.5, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
        ],
        # action 1
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.5, 0.0, 0.0, 0.5],
            [0.5, 0.0, 0.5, 0.0],
            [0.0, 0.0, 0.5, 0.5],
        ],
    ], dtype=float)
    R = np.array([
        [0.0, 0.0, 10.0, 10.0],
        [0.0, 0.0, 10.0, 10.0],
    ], dtype=float)
    gamma = 0.9
    return MDP(T, R, gamma)

def main():
    np.random.seed(123)
    mdp = build_toy_mdp()
    rl = RL(mdp, sample_reward)
    Q0 = np.zeros((mdp.nActions, mdp.nStates), dtype=float)

    Q, policy, ep_returns = rl.qLearning(
        s0=0, initialQ=Q0, nEpisodes=100, nSteps=100, epsilon=0.1, temperature=0.0
    )

    print("Final greedy policy (toy):", policy)
    print("Q[:, :]:\n", np.round(Q, 4))
    print("Episode returns (first 10):", np.round(ep_returns[:10], 4))

if __name__ == "__main__":
    main()
