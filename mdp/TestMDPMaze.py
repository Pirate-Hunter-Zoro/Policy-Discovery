# mdp/TestMDPMaze.py
import numpy as np
from mdp.MDP import MDP

np.set_printoptions(precision=6, suppress=True)

def build_maze_mdp():
    """
    4x4 grid (0..15), goal=14, bad=9, absorbing end=16.
    |S| = 17 (includes end), |A| = 4 (up/down/left/right).
    Transition model with slip (a=0.8 intended, b=0.1 lateral).
    Rewards: -1 per step, +100 at goal, -70 at bad, 0 at end.
    gamma = 0.95
    """
    T = np.zeros([4, 17, 17], dtype=float)
    a = 0.8  # intended move
    b = 0.1  # lateral move

    # ----- Up (0)
    T[0,0,0]=a+b;    T[0,0,1]=b
    T[0,1,0]=b;      T[0,1,1]=a;      T[0,1,2]=b
    T[0,2,1]=b;      T[0,2,2]=a;      T[0,2,3]=b
    T[0,3,2]=b;      T[0,3,3]=a+b
    T[0,4,4]=b;      T[0,4,0]=a;      T[0,4,5]=b
    T[0,5,4]=b;      T[0,5,1]=a;      T[0,5,6]=b
    T[0,6,5]=b;      T[0,6,2]=a;      T[0,6,7]=b
    T[0,7,6]=b;      T[0,7,3]=a;      T[0,7,7]=b
    T[0,8,8]=b;      T[0,8,4]=a;      T[0,8,9]=b
    T[0,9,8]=b;      T[0,9,5]=a;      T[0,9,10]=b
    T[0,10,9]=b;     T[0,10,6]=a;     T[0,10,11]=b
    T[0,11,10]=b;    T[0,11,7]=a;     T[0,11,11]=b
    T[0,12,12]=b;    T[0,12,8]=a;     T[0,12,13]=b
    T[0,13,12]=b;    T[0,13,9]=a;     T[0,13,14]=b
    T[0,14,16]=1.0
    T[0,15,11]=a;    T[0,15,14]=b;    T[0,15,15]=b
    T[0,16,16]=1.0

    # ----- Down (1)
    T[1,0,0]=b;      T[1,0,4]=a;      T[1,0,1]=b
    T[1,1,0]=b;      T[1,1,5]=a;      T[1,1,2]=b
    T[1,2,1]=b;      T[1,2,6]=a;      T[1,2,3]=b
    T[1,3,2]=b;      T[1,3,7]=a;      T[1,3,3]=b
    T[1,4,4]=b;      T[1,4,8]=a;      T[1,4,5]=b
    T[1,5,4]=b;      T[1,5,9]=a;      T[1,5,6]=b
    T[1,6,5]=b;      T[1,6,10]=a;     T[1,6,7]=b
    T[1,7,6]=b;      T[1,7,11]=a;     T[1,7,7]=b
    T[1,8,8]=b;      T[1,8,12]=a;     T[1,8,9]=b
    T[1,9,8]=b;      T[1,9,13]=a;     T[1,9,10]=b
    T[1,10,9]=b;     T[1,10,14]=a;    T[1,10,11]=b
    T[1,11,10]=b;    T[1,11,15]=a;    T[1,11,11]=b
    T[1,12,12]=a+b;  T[1,12,13]=b
    T[1,13,12]=b;    T[1,13,13]=a;    T[1,13,14]=b
    T[1,14,16]=1.0
    T[1,15,14]=b;    T[1,15,15]=a+b
    T[1,16,16]=1.0

    # ----- Left (2)
    T[2,0,0]=a+b;    T[2,0,4]=b
    T[2,1,1]=b;      T[2,1,0]=a;      T[2,1,5]=b
    T[2,2,2]=b;      T[2,2,1]=a;      T[2,2,6]=b
    T[2,3,3]=b;      T[2,3,2]=a;      T[2,3,7]=b
    T[2,4,0]=b;      T[2,4,4]=a;      T[2,4,8]=b
    T[2,5,1]=b;      T[2,5,4]=a;      T[2,5,9]=b
    T[2,6,2]=b;      T[2,6,5]=a;      T[2,6,10]=b
    T[2,7,3]=b;      T[2,7,6]=a;      T[2,7,11]=b
    T[2,8,4]=b;      T[2,8,8]=a;      T[2,8,12]=b
    T[2,9,5]=b;      T[2,9,8]=a;      T[2,9,13]=b
    T[2,10,6]=b;     T[2,10,9]=a;     T[2,10,14]=b
    T[2,11,7]=b;     T[2,11,10]=a;    T[2,11,15]=b
    T[2,12,8]=b;     T[2,12,12]=a+b
    T[2,13,9]=b;     T[2,13,12]=a;    T[2,13,13]=b
    T[2,14,16]=1.0
    T[2,15,11]=a;    T[2,15,14]=b;    T[2,15,15]=b
    T[2,16,16]=1.0

    # ----- Right (3)
    T[3,0,0]=b;      T[3,0,1]=a;      T[3,0,4]=b
    T[3,1,1]=b;      T[3,1,2]=a;      T[3,1,5]=b
    T[3,2,2]=b;      T[3,2,3]=a;      T[3,2,6]=b
    T[3,3,3]=a+b;    T[3,3,7]=b
    T[3,4,0]=b;      T[3,4,5]=a;      T[3,4,8]=b
    T[3,5,1]=b;      T[3,5,6]=a;      T[3,5,9]=b
    T[3,6,2]=b;      T[3,6,7]=a;      T[3,6,10]=b
    T[3,7,3]=b;      T[3,7,7]=a;      T[3,7,11]=b
    T[3,8,4]=b;      T[3,8,9]=a;      T[3,8,12]=b
    T[3,9,5]=b;      T[3,9,10]=a;     T[3,9,13]=b
    T[3,10,6]=b;     T[3,10,11]=a;    T[3,10,14]=b
    T[3,11,7]=b;     T[3,11,11]=a;    T[3,11,15]=b
    T[3,12,8]=b;     T[3,12,13]=a;    T[3,12,12]=b
    T[3,13,9]=b;     T[3,13,14]=a;    T[3,13,13]=b
    T[3,14,16]=1.0
    T[3,15,11]=b;    T[3,15,15]=a+b
    T[3,16,16]=1.0

    # Rewards: A x S
    R = -1.0 * np.ones([4, 17], dtype=float)
    R[:, 14] = 100.0  # goal
    R[:, 9]  = -70.0  # bad
    R[:, 16] = 0.0    # end (absorbing)

    gamma = 0.95
    return MDP(T, R, gamma)

def print_header(title):
    print("=" * 100)
    print(title)
    print("=" * 100)

def check_stochastic_rows(T):
    ok = np.allclose(T.sum(axis=2), 1.0, atol=1e-8)
    if not ok:
        bad = np.where(~np.isclose(T.sum(axis=2), 1.0, atol=1e-8))
        raise ValueError(f"Non-stochastic rows at indices {bad}")
    return ok

def main():
    mdp = build_maze_mdp()
    nS = mdp.nStates

    # Sanity: transitions are stochastic
    check_stochastic_rows(mdp.T)

    # Value Iteration (tol=0.01, V0=0)
    print_header("MAZE — VALUE ITERATION (tol=0.01, V0=0)")
    V0 = np.zeros(nS)
    V_star, vi_iters, vi_eps = mdp.valueIteration(initialV=V0, tolerance=0.01)
    print("Iterations:", vi_iters)
    print("Final epsilon (∞-norm):", float(vi_eps))
    print("V* (first 17 entries):\n", V_star)
    pi_star = mdp.extractPolicy(V_star)
    print("Greedy policy π*(V*):\n", pi_star)
    print()

    # Policy Iteration (π0 = all action 0)
    print_header("MAZE — POLICY ITERATION (π0 = all-zeros)")
    pi0 = np.zeros(nS, dtype=int)
    pi_opt, V_pi, outer_iters = mdp.policyIteration(initialPolicy=pi0)
    print("Outer iterations:", outer_iters)
    print("Optimal policy π*:\n", pi_opt)
    print("V^{π*}:\n", V_pi)
    print()

    # Modified Policy Iteration: report OUTER iterations for K=1..10
    print_header("MAZE — MPI SWEEP (K = 1..10) → OUTER ITERS TO CONVERGE")
    print("K\tOuterIters\tFinalOuterEps")
    for K in range(1, 11):
        mpi_pi, mpi_V, mpi_outer, mpi_eps = mdp.modifiedPolicyIteration(
            initialPolicy=pi0,
            initialV=np.zeros(nS),
            nEvalIterations=K,
            tolerance=0.01
        )
        print(f"{K}\t{mpi_outer}\t\t{float(mpi_eps):.6f}")

if __name__ == "__main__":
    main()