# mdp/TestMDP.py
import numpy as np
from mdp.MDP import MDP

np.set_printoptions(precision=6, suppress=True)

def build_toy_mdp():
    """
    The simple 4-state MDP from Lecture 2a Slides 13–14.
    |S|=4, |A|=2
    """
    # Transition: A x S x S'
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

    # Reward: A x S
    R = np.array([
        [0.0, 0.0, 10.0, 10.0],
        [0.0, 0.0, 10.0, 10.0],
    ], dtype=float)

    gamma = 0.9
    return MDP(T, R, gamma)

def print_header(title):
    print("=" * 80)
    print(title)
    print("=" * 80)

def main():
    mdp = build_toy_mdp()
    nS = mdp.nStates

    # Value Iteration (tol=0.01, V0=0)
    print_header("TOY MDP — VALUE ITERATION (tol=0.01, V0=0)")
    V0 = np.zeros(nS)
    V, iters, eps = mdp.valueIteration(initialV=V0, tolerance=0.01)
    print("Iterations:", iters)
    print("Final epsilon (∞-norm):", float(eps))
    print("Value function V*:\n", V)
    pi_star = mdp.extractPolicy(V)
    print("Greedy policy π*(V*):", pi_star)
    print()

    # Policy Iteration (π0 = all action 0)
    print_header("TOY MDP — POLICY ITERATION (π0 = all-zeros)")
    pi0 = np.zeros(nS, dtype=int)
    pi_opt, V_pi, outer_iters = mdp.policyIteration(initialPolicy=pi0)
    print("Outer iterations:", outer_iters)
    print("Optimal policy π*:", pi_opt)
    print("Value function V^{π*}:\n", V_pi)
    print()

    # Partial Policy Evaluation (one sweep, to show mechanics)
    print_header("TOY MDP — PARTIAL POLICY EVALUATION (1 sweep from V0=0)")
    V_part, pe_iters, pe_eps = mdp.evaluatePolicyPartially(policy=pi_opt,
                                                           initialV=np.zeros(nS),
                                                           nIterations=1)
    print("Sweeps performed:", pe_iters)
    print("Epsilon (∞-norm after sweep):", float(pe_eps))
    print("V after 1 sweep:\n", V_part)
    print()

    # Modified Policy Iteration (single run example)
    print_header("TOY MDP — MODIFIED POLICY ITERATION (nEvalIterations=5)")
    mpi_pi, mpi_V, mpi_outer, mpi_eps = mdp.modifiedPolicyIteration(
        initialPolicy=pi0,
        initialV=np.zeros(nS),
        nEvalIterations=5,
        tolerance=0.01
    )
    print("Outer iterations:", mpi_outer)
    print("Final outer epsilon:", float(mpi_eps))
    print("Final policy:", mpi_pi)
    print("Final V:\n", mpi_V)
    print()

    # MPI sweep K=1..10: report outer iterations to converge
    print_header("TOY MDP — MPI SWEEP (K = 1..10) → OUTER ITERS TO CONVERGE")
    print("K\tOuterIters\tFinalOuterEps")
    for K in range(1, 11):
        piK, VK, outerK, epsK = mdp.modifiedPolicyIteration(
            initialPolicy=pi0,
            initialV=np.zeros(nS),
            nEvalIterations=K,
            tolerance=0.01
        )
        print(f"{K}\t{outerK}\t\t{float(epsK):.6f}")

if __name__ == "__main__":
    main()