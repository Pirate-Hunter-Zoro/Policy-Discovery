#!/usr/bin/env python3
"""
TestDQN.py
Generates both figures requested in Assignment 1, Part III (CartPole DQN).

Experiments:
1) Target network update frequency sweep: {1, 10 (default), 50, 100} episodes.
2) Minibatch size sweep: {1, 10 (default), 50, 100}.

For each setting:
- Run 5 seeds (SEEDS in dqn.DQN), EPISODES=300.
- Plot the average cumulative reward of the last 25 test episodes vs episode index.
- Save per-episode mean curves to CSV.
- Save figures as PNG.

Assumptions:
- Your code is in a package `dqn` with `DQN.py` using relative imports (.utils.*).
- Run this script from REPO ROOT (so that `import dqn.DQN as DQN` works).

Usage:
    python -m dqn.TESTDQN
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import importlib

# Import your DQN module as a package module (dqn/DQN.py)
DQN = importlib.import_module("dqn.DQN")

OUTDIR = Path("dqn/outputs")
OUTDIR.mkdir(parents=True, exist_ok=True)

def run_average_over_seeds(run_label):
    """Helper to run DQN.train over DQN.SEEDS and return a 2D array [nSeeds x episodes]."""
    curves = []
    for seed in DQN.SEEDS:
        print(f"[{run_label}] Training seed={seed}")
        curve = DQN.train(seed)
        curves.append(curve)
    curves = np.array(curves, dtype=float)
    return curves

def save_csv(mean_curves, x, header_labels, path):
    path = Path(path)
    with path.open("w") as f:
        f.write("episode," + ",".join(header_labels) + "\n")
        for i, ep in enumerate(x):
            row = [str(ep)] + [f"{mean_curves[j][i]:.6f}" for j in range(len(header_labels))]
            f.write(",".join(row) + "\n")
    print(f"Saved CSV: {path}")

def plot_curves(x, mean_curves, labels, title, out_png):
    plt.figure()
    for mean, label in zip(mean_curves, labels):
        plt.plot(x, mean, label=label)
    plt.xlabel("Episode")
    plt.ylabel("Average cumulative reward of last 25 episodes")
    plt.title(title)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()
    print(f"Saved plot: {out_png}")

def sweep_target_update():
    # Keep everything default except TARGET_UPDATE_FREQ and EPISODES
    DQN.EPISODES = 300
    DQN.SEEDS = [1,2,3,4,5]

    freqs = [1, 10, 50, 100]
    labels = [f"target_update={f}" for f in freqs]
    means = []

    for f in freqs:
        DQN.TARGET_UPDATE_FREQ = f
        curves = run_average_over_seeds(f"target={f}")
        means.append(curves.mean(axis=0))

    x = np.arange(len(means[0]))
    save_csv(means, x, labels, OUTDIR / "part3_target_update_sweep.csv")
    plot_curves(x, means, labels, "DQN CartPole: Target network update frequency", OUTDIR / "part3_target_update_sweep.png")

def sweep_minibatch():
    # Keep everything default except MINIBATCH_SIZE and EPISODES
    DQN.EPISODES = 300
    DQN.SEEDS = [1,2,3,4,5]

    batches = [1, 10, 50, 100]
    labels = [f"minibatch={b}" for b in batches]
    means = []

    for b in batches:
        DQN.MINIBATCH_SIZE = b
        DQN.TARGET_UPDATE_FREQ = 10 # Best one based on results
        curves = run_average_over_seeds(f"batch={b}")
        means.append(curves.mean(axis=0))

    x = np.arange(len(means[0]))
    save_csv(means, x, labels, OUTDIR / "part3_minibatch_sweep.csv")
    plot_curves(x, means, labels, "DQN CartPole: Minibatch size", OUTDIR / "part3_minibatch_sweep.png")

def main():
    print("=== Part III: Generating target update frequency sweep ===")
    sweep_target_update()
    print("=== Part III: Generating minibatch size sweep ===")
    sweep_minibatch()
    print("Done. Artifacts saved under dqn/outputs/.")

if __name__ == "__main__":
    main()
