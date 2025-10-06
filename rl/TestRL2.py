import numpy as np
from mdp.MDP import MDP
from rl.RL2 import RL2
from rl.RL import RL
import matplotlib.pyplot as plt
import os
from pathlib import Path

N_TRIALS = 1000
N_ITERATIONS = 200

def sampleBernoulli(mean):
    ''' function to obtain a sample from a Bernoulli distribution

    Input:
    mean -- mean of the Bernoulli
    
    Output:
    sample -- sample (0 or 1)
    '''

    if np.random.rand(1) < mean: return 1
    else: return 0


# Multi-arm bandit problems (3 arms with probabilities 0.3, 0.5 and 0.7)
T = np.array([[[1]],[[1]],[[1]]])
R = np.array([[0.3],[0.5],[0.7]])
discount = 0.999
mdp = MDP(T,R,discount)
banditProblem = RL2(mdp,sampleBernoulli)

results = np.zeros((3, N_TRIALS, N_ITERATIONS))

for i in range(N_TRIALS):
    _, all_rewards = banditProblem.epsilonGreedyBandit(nIterations=N_ITERATIONS)
    results[0,i,:] = all_rewards
    _, all_rewards = banditProblem.thompsonSamplingBandit(prior=np.ones([mdp.nActions,2]), nIterations=N_ITERATIONS)
    results[1,i,:] = all_rewards
    _, all_rewards = banditProblem.UCBbandit(nIterations=N_ITERATIONS)
    results[2,i,:] = all_rewards
# Now average over our trials
avg_rewards_per_iteration_over_trials = np.mean(results, axis=1)
algorithm_labels = ["Epsilon Greedy Bandit", "Thompson Sampling Bandit", "UCB Bandit"]
plt.title("Multi-Arm Bandit Results")
plt.xlabel("Iterations")
plt.ylabel("Average Reward")
os.makedirs(Path("rl/results"), exist_ok=True)
for i in range(3):
    alg_avg_results = avg_rewards_per_iteration_over_trials[i]
    plt.plot(alg_avg_results, label=algorithm_labels[i])
plt.legend()
plt.savefig("rl/results/multi_arm_bandit.png")
plt.close()


# Model based RL

''' Construct simple MDP as described in Lecture 2a Slides 13-14'''
T = np.array([[[0.5,0.5,0,0],[0,1,0,0],[0.5,0.5,0,0],[0,1,0,0]],[[1,0,0,0],[0.5,0,0,0.5],[0.5,0,0.5,0],[0,0,0.5,0.5]]])
R = np.array([[0,0,10,10],[0,0,10,10]])
discount = 0.9        

N_TRIALS = 100
N_EPISODES = 200

model_based_results = np.zeros((N_TRIALS, N_EPISODES))
q_learning_results = np.zeros((N_TRIALS, N_EPISODES))

for i in range(N_TRIALS):
    mdp = MDP(T,R,discount)
    rl = RL(mdp, sampleReward=np.random.normal)
    rlProblem = RL2(mdp,np.random.normal)
    default_T = np.ones([mdp.nActions, mdp.nStates, mdp.nStates]) / mdp.nStates
    default_R = np.zeros([mdp.nActions, mdp.nStates])
    [_, _, cumulative_rewards] = rlProblem.modelBasedRL(s0=0, defaultT=default_T, initialR=default_R, nEpisodes=N_EPISODES, nSteps=100, epsilon=0.05)
    model_based_results[i, :] = cumulative_rewards
    [_, _, cumulative_rewards] = rl.qLearning(s0=0, initialQ=np.zeros((mdp.nActions,mdp.nStates)),nEpisodes=N_EPISODES, nSteps=100, epsilon=0.05)
    q_learning_results[i, :] = cumulative_rewards

# Find average results per episode over all trials
plt.title("Grid World Results")
plt.xlabel("Episodes")
plt.ylabel("Average Cumulative Discounted Reward")
model_based_results_avg = np.mean(model_based_results, axis=0)
q_learning_results_avg = np.mean(q_learning_results, axis=0)
plt.plot(model_based_results_avg, label="Model Based RL")
plt.plot(q_learning_results_avg, label="Q-Learning RL")
plt.legend()
plt.savefig("rl/results/grid_world.png")
plt.close()