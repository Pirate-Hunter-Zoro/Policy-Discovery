import numpy as np
import random
import math

class RL2:
    def __init__(self,mdp,sampleReward):
        '''Constructor for the RL class

        Inputs:
        mdp -- Markov decision process (T, R, discount)
        sampleReward -- Function to sample rewards (e.g., bernoulli, Gaussian).
        This function takes one argument: the mean of the distributon and 
        returns a sample from the distribution.
        '''

        self.mdp = mdp
        self.sampleReward = sampleReward

    def sampleRewardAndNextState(self,state,action):
        '''Procedure to sample a reward and the next state
        reward ~ Pr(r)
        nextState ~ Pr(s'|s,a)

        Inputs:
        state -- current state
        action -- action to be executed

        Outputs: 
        reward -- sampled reward
        nextState -- sampled next state
        '''

        reward = self.sampleReward(self.mdp.R[action,state])
        cumProb = np.cumsum(self.mdp.T[action,state,:])
        nextState = np.where(cumProb >= np.random.rand(1))[0][0]
        return [reward,nextState]

    def modelBasedRL(self,s0,defaultT,initialR,nEpisodes,nSteps,epsilon=0):
        '''Model-based Reinforcement Learning with epsilon greedy 
        exploration.  This function should use value iteration,
        policy iteration or modified policy iteration to update the policy at each step

        Inputs:
        s0 -- initial state
        defaultT -- default transition function when a state-action pair has not been vsited
        initialR -- initial estimate of the reward function
        nEpisodes -- # of episodes (one episode consists of a trajectory of nSteps that starts in s0
        nSteps -- # of steps per episode
        epsilon -- probability with which an action is chosen at random

        Outputs: 
        V -- final value function
        policy -- final policy
        '''

        # temporary values to ensure that the code compiles until this
        # function is coded
        policy = np.zeros(self.mdp.nStates,int)
        state_action_pair_counts = {}
        transition_counts = {}
        T_hat = defaultT
        R_hat = initialR
        for s in range(self.mdp.nStates):
            for a in range(self.mdp.nActions):
                state_action_pair_counts[(s,a)] = 0
                transition_counts[(s,a)] = {}
                for s_prime in range(self.mdp.nStates):
                    transition_counts[(s,a)][s_prime] = 0
        
        rnd = random.Random(42)
        cumulative_discounted_rewards = []
        for i in range(nEpisodes):
            current_state = s0
            total_discounted_reward = 0
            for step in range(nSteps):
                if rnd.random() < epsilon:
                    # Random action
                    next_action = rnd.randint(0,self.mdp.nActions-1)
                else:
                    next_action = policy[current_state]
                reward_and_next_state = self.sampleRewardAndNextState(current_state, next_action)
                reward = reward_and_next_state[0]
                next_state = reward_and_next_state[1]
                state_action_pair_counts[(current_state,next_action)] += 1
                transition_counts[(current_state,next_action)][next_state] += 1
                T_hat[next_action][current_state] = [transition_counts[(current_state,next_action)][s_prime]/state_action_pair_counts[(current_state,next_action)] for s_prime in range(self.mdp.nStates)]
                old_avg_reward = R_hat[next_action][current_state]
                R_hat[next_action][current_state] = (step*old_avg_reward + reward)/(state_action_pair_counts[((current_state,next_action))])
                
                current_state = next_state
                total_discounted_reward += reward * (self.mdp.discount ** i)
                
            # Now that our model has been updated over the episode, run value iteration to get our hands on a new policy
            V = self._valueIterationHelper(np.zeros(self.mdp.nStates), T_hat, R_hat)
            policy = self._extractPolicyHelper(V, T_hat, R_hat)
            cumulative_discounted_rewards.append(total_discounted_reward)

        return [V,policy,cumulative_discounted_rewards]    
    
    def _valueIterationHelper(self, initialV, T_hat, R_hat, nIterations=np.inf, tolerance=0.01):
        new_V = initialV.copy()
        nIterations_done = 0
        epsilon = np.inf
        while nIterations_done < nIterations and epsilon > tolerance:
            initialV_for_iter = new_V
            new_V_for_iter = -np.inf * np.ones(len(initialV)) # Start with the lowest possible value so that any action can improve it
            for a in range(self.mdp.nActions):
                T_actions = T_hat[a] # Give us the probability transition matrix for action a
                R_actions = R_hat[a] # Give us the reward vector for action a from each state
                V_for_a = R_actions + self.mdp.discount * np.dot(T_actions, initialV_for_iter) # Updates the value of all states based on the reward achieved from this action plus the weighted average of the values of the next states
                new_V_for_iter = np.maximum(new_V_for_iter, V_for_a) # Each state gets a chance to improve

            new_V = new_V_for_iter
            epsilon = np.linalg.norm(new_V_for_iter - initialV_for_iter, np.inf)
            nIterations_done += 1
            
        return new_V
        
    def _extractPolicyHelper(self, V, T_hat, R_hat):
        policy = np.zeros(len(V), dtype=int)
        for i in range(len(V)):
            # Determine the best action in state i
            record = -math.inf
            rewards_for_state = R_hat[:, i] # Get the rewards for all actions in state i
            transition_for_state = T_hat[:, i, :] # Get the transition probabilities for all actions from this state
            future_state_values = np.dot(transition_for_state, V) # Get the expected value of the next state for each action
            for a in range(self.mdp.nActions):
                value_for_a = rewards_for_state[a] + self.mdp.discount * future_state_values[a]
                if value_for_a > record:
                    record = value_for_a
                    policy[i] = a

        return policy 

    def epsilonGreedyBandit(self,nIterations):
        '''Epsilon greedy algorithm for bandits (assume no discount factor).  Use epsilon = 1 / # of iterations.

        Inputs:
        nIterations -- # of arms that are pulled

        Outputs: 
        empiricalMeans -- empirical average of rewards for each arm (array of |A| entries)
        '''
        empiricalMeans = np.zeros(self.mdp.nActions)
        n_pulls = np.zeros(self.mdp.nActions)
        rewards = []
        rnd = random.Random(42)
        for t in range(nIterations):
            epsilon = 1/(t+1)
            if rnd.random() < epsilon:
                # Random action
                next_action = rnd.randint(0, self.mdp.nActions-1)
            else:
                # Greedy
                next_action = np.argmax(empiricalMeans)
            true_mean_reward = self.mdp.R[next_action, 0] # In multi-arm bandit, there's only one state
            reward = self.sampleReward(true_mean_reward)
            n_pulls[next_action] += 1
            empiricalMeans[next_action] = (empiricalMeans[next_action]*(n_pulls[next_action]-1)+reward)/n_pulls[next_action]
            rewards.append(reward)

        return empiricalMeans, rewards

    def thompsonSamplingBandit(self,prior,nIterations,k=1):
        '''Thompson sampling algorithm for Bernoulli bandits (assume no discount factor)

        Inputs:
        prior -- initial beta distribution over the average reward of each arm (|A|x2 matrix such that prior[a,0] is the alpha hyperparameter for arm a and prior[a,1] is the beta hyperparameter for arm a)  
        nIterations -- # of arms that are pulled
        k -- # of sampled average rewards

        Outputs: 
        empiricalMeans -- empirical average of rewards for each arm (array of |A| entries)
        '''

        empiricalMeans = np.zeros(self.mdp.nActions)
        alphas = [prior[a][0] for a in range(self.mdp.nActions)]
        betas = [prior[a][1] for a in range(self.mdp.nActions)]
        n_pulls = np.zeros(self.mdp.nActions)
        rewards = []
        
        for _ in range(nIterations):
            arm_sample_values = np.array([np.random.beta(alphas[a],betas[a]) for a in range(self.mdp.nActions)])
            next_pull = np.argmax(arm_sample_values)
            reward = self.sampleReward(self.mdp.R[next_pull, 0]) # Again only one state but we have discrete rewards
            if reward == 1:
                # Success
                alphas[next_pull] += 1
            else:
                betas[next_pull] += 1
            n_pulls[next_pull] += 1
            empiricalMeans[next_pull] = (empiricalMeans[next_pull]*(n_pulls[next_pull]-1)+reward)/n_pulls[next_pull]
            rewards.append(reward)

        return empiricalMeans, rewards

    def UCBbandit(self,nIterations):
        '''Upper confidence bound algorithm for bandits (assume no discount factor)

        Inputs:
        nIterations -- # of arms that are pulled

        Outputs: 
        empiricalMeans -- empirical average of rewards for each arm (array of |A| entries)
        '''

        empiricalMeans = np.zeros(self.mdp.nActions)
        n_pulls = np.zeros(self.mdp.nActions)
        rewards = []
        # Pull all arms once to avoid division by zero
        for a in range(self.mdp.nActions):
            n_pulls[a] += 1
            reward = self.sampleReward(self.mdp.R[a, 0])
            empiricalMeans[a] = reward
            rewards.append(reward)
        
        for t in range(self.mdp.nActions, nIterations):
            max_score = float('-inf')
            best_arm = None
            for a in range(self.mdp.nActions):
                score = empiricalMeans[a] + math.sqrt(2*math.log(t)/n_pulls[a])
                if score > max_score:
                    max_score = score
                    best_arm = a
            # Now pull the best arm
            n_pulls[best_arm] += 1
            reward = self.sampleReward(self.mdp.R[best_arm, 0])
            empiricalMeans[best_arm] = (empiricalMeans[best_arm]*(n_pulls[best_arm]-1)+reward)/n_pulls[best_arm]
            rewards.append(reward)

        return empiricalMeans, rewards