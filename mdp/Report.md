
# Dynamic Programming for MDPs

## Overview

This report implements and evaluates **Value Iteration (VI)**, **Policy Iteration (PI)**, and **Modified Policy Iteration (MPI)** on two environments:

1. A **toy 4‑state, 2‑action MDP** (Lecture 2a, slides 13–14 style).
2. A **4×4 slip maze** with an absorbing end state (|S|=17), rewards: step −1, goal +100 at state 14, bad −70 at state 9, end 0 at state 16, discount γ=0.95.

All algorithms use dense transition (`T ∈ ℝ^{|A|×|S|×|S|}`) and reward (`R ∈ ℝ^{|A|×|S|}`) tensors.

---

## Methods

- **Value Iteration** uses tolerance **0.01** and **V₀ = 0**.
- **Policy Iteration** starts from **π₀ = 0** (action 0 in all states).
- **Modified Policy Iteration** sweeps **K ∈ {1,…,10}** partial‑evaluation iterations with **V₀ = 0**, **π₀ = 0**, tolerance **0.01**; report the **outer** iterations until convergence and the final outer ε∞.

---

## Results — Toy MDP (|S|=4, |A|=2, γ=0.9)

### Value Iteration

- **Iterations:** 58  
- **Final ε∞:** 0.009860138819838937  
- **V\*** = `[31.496363, 38.515275, 43.935435, 54.112858]`  
- **Greedy π(V\*)** = `[0, 1, 1, 1]`

### Policy Iteration (π₀ = all zeros)

- **Outer iterations:** 2  
- **Optimal policy π\*:** `[0, 1, 1, 1]`  
- **V^{π\*}:** `[31.585104, 38.604016, 44.024176, 54.201599]`

### Modified Policy Iteration — outer iterations vs K

| K | Outer iters | Final outer ε∞ |
|---:|------------:|----------------:|
| 1  | 58 | 0.009860 |
| 2  | 33 | 0.009956 |
| 3  | 24 | 0.009317 |
| 4  | 19 | 0.009577 |
| 5  | 16 | 0.009237 |
| 6  | 14 | 0.008561 |
| 7  | 13 | 0.005629 |
| 8  | 12 | 0.004479 |
| 9  | 11 | 0.004336 |
| 10 | 10 | 0.005123 |

**Observation.** Outer iterations **decrease** monotonically (with minor noise in ε) as **K** increases, transitioning from VI‑like behavior (K=1) toward PI‑like behavior (large K). Policies and values across VI/PI/MPI match closely (minor numerical differences).

---

## Results — Maze (|S|=17, |A|=4, γ=0.95)

### Value Iteration Results (tol=0.01, V₀=0)

- **Iterations:** 21  
- **Final ε∞:** 0.004895425288879096  
- **V\*** (states 0..16):  
`[57.40982, 62.355017, 67.77964, 64.962844, 58.620881, 62.31803, 74.585942, 70.201532, 63.331494, 7.47047, 82.890544, 75.588364, 75.796541, 83.657113, 100.0, 72.869372, 0.0]`  
- **Greedy π(V\*)** = `[3, 3, 1, 1, 1, 3, 1, 1, 1, 1, 1, 2, 3, 3, 0, 0, 0]`

### Policy Iteration Results (π₀ = all zeros)

- **Outer iterations:** 5  
- **Optimal policy π\*:** `[3, 3, 1, 1, 1, 3, 1, 1, 1, 1, 1, 2, 3, 3, 0, 0, 0]`  
- **V^{π\*}:**  
`[57.415792, 62.358233, 67.781387, 64.964711, 58.621411, 62.31944, 74.586458, 70.20241, 63.331618, 7.470531, 82.890634, 75.588773, 75.796597, 83.657128, 100.0, 72.87013, 0.0]`

### Modified Policy Iteration Results — outer iterations vs K

| K | Outer iters | Final outer ε∞ |
|---:|------------:|----------------:|
| 1  | 21 | 0.004895 |
| 2  | 12 | 0.008581 |
| 3  | 10 | 0.002041 |
| 4  | 8  | 0.006317 |
| 5  | 8  | 0.000475 |
| 6  | 7  | 0.001489 |
| 7  | 7  | 0.000258 |
| 8  | 7  | 0.000041 |
| 9  | 6  | 0.001308 |
| 10 | 7  | 0.000001 |

**Observation.** As **K** increases, MPI typically needs **fewer outer iterations**, trending toward PI (finite evaluation to near‑exact). Small non‑monotonicities in ε∞ reflect the different stopping surfaces across K; nonetheless, policies match VI/PI: `[3,3,1,1,1,3,1,1,1,1,1,2,3,3,0,0,0]` and values are numerically consistent. Note that as K approcahes infinity, it becomes closer to directly solving for the state values given the policy, and then iterating the policy (which is the definition of PI). And of course when K=1, this is precisely VI.

---

## Discussion

- **VI vs PI.** VI applies a max‑Bellman backup each sweep and is a **γ‑contraction**, hence convergence to V\* is guaranteed; it required **21** iterations on the maze with tol=0.01. PI alternates **exact** policy evaluation with greedy improvement; it converged in **5 outer iterations**, yielding the same optimal policy/value up to numerical tolerance.
- **MPI behavior.** MPI interpolates between VI and PI by using **K** evaluation sweeps before each improvement. For both toy and maze:
  - **Small K** → more outer iterations (VI‑like).  
  - **Large K** → fewer outer iterations (PI‑like).  
  The empirical curves match the theoretical expectation that increasing K improves policy evaluation accuracy per outer step, reducing the number of policy improvements needed. Diminishing returns appear after moderate K (e.g., K≈6–8 on the maze).
- **Consistency.** VI/PI/MPI yield **matching greedy policies and very close value functions**, validating the implementation.

---

## Reproducibility

- **Commands**
  - Toy: `python -m mdp.TestMDP`
  - Maze: `python -m mdp.TestMDPMaze`
- **Key settings**: tol=0.01; V₀=0; π₀=0; MPI K∈{1..10}.  
- **Environment**: NumPy; dense `T` and `R` as described; no randomness used in DP (maze slip is in `T`).
