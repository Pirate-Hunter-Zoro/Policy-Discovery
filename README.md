# Assignment 1  

**Out:** Jan 10  
**Due:** Jan 21 (11:59 pm)  
**Total:** 10 points  

---

## Part I — Value Iteration, Policy Iteration, Modified Policy Iteration (4 points)

In this part, you will implement **value iteration**, **policy iteration**, and **modified policy iteration** for Markov Decision Processes (MDPs) in Python.

- Fill in the functions in the skeleton code `MDP.py`.  
- Use `TestMDP.py` to verify correctness (contains the simple MDP example from Lecture 2a, Slides 13–14).  
- Run with:  

```bash
python TestMDP.py

 • Add print statements to check that outputs make sense.

Provided files:
 • Skeleton code: MDP.py
 • Simple MDP for testing: TestMDP.py

Submission requirements:
 1. Submit your Python code.
 2. Test your code with the maze problem (TestMDPmaze.py).
 3. Report results:
 • Value Iteration: Policy, value function, and number of iterations with tolerance = 0.01, starting from all-zero value function. [1 point]
 • Policy Iteration: Policy, value function, and number of iterations when starting with a policy that chooses action 0 in all states. [1 point]
 • Modified Policy Iteration: Number of iterations required for convergence when partial policy evaluation runs from 1 to 10 iterations, with tolerance = 0.01, initial all-zero value function and policy choosing action 0 everywhere. Discuss the impact compared to value iteration and policy iteration. [2 points]

⸻

Part II — Q-Learning (3 points)

In this part, you will implement the Q-learning algorithm in Python.
 • Fill in the functions in the skeleton code RL.py.
 • RL.py requires MDP.py from Part I, so keep them in the same directory.
 • Use TestRL.py to test your functions.
 • Run with:

python TestRL.py

Provided files:
 • Skeleton code: RL.py
 • Simple RL test: TestRL.py

Submission requirements:
 1. Submit your Python code.
 2. Test your code with the maze problem (TestRLmaze.py).
 3. Report results:
 • Produce a graph:
 • x-axis: Episode # (0–200)
 • y-axis: Average (100 trials) of cumulative discounted rewards per episode (100 steps).
 • Curves: ε = 0.05, 0.1, 0.3, 0.5. [1 point]
 • Initial state = 0, Q-function initialized to 0 for all state-action pairs.
 • Explain the impact of ε on training rewards, Q-values, and resulting policy. [2 points]

⸻

Part III — Deep Q-Network (3 points)

In this part, you will train a Deep Q-Network (DQN) to solve the CartPole problem from OpenAI Gym. Since the problem has continuous states, use a neural network for the Q-function.

Setup steps:
 1. Read about the CartPole problem in OpenAI Gym.
 2. Install PyTorch and complete a basic tutorial.
 3. Use the provided code (Part 3) to train a DQN on CartPole.

Submission requirements:
 1. Target Network Experiments:
 • Modify the code to produce a graph:
 • x-axis: Episodes (up to 300)
 • y-axis: Average cumulative reward of last 25 episodes.
 • Curves: Update target network every 1, 10 (default), 50, 100 episodes.
 • Average results over 5 runs (different seeds). [1 point]
 • Explain the impact of the target network and relate it to value iteration. [1 point]
 2. Replay Buffer Experiments:
 • Modify the code to produce a graph:
 • x-axis: Episodes (up to 300)
 • y-axis: Average cumulative reward of last 25 episodes.
 • Curves: Mini-batch sizes of 1, 10 (default), 50, 100.
 • Average results over 5 runs (different seeds). [1 point]
 • Explain the impact of the replay buffer and relate it to exact gradient descent. [1 point]
