import numpy as np

class RL:
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

    def qLearning(self,s0,initialQ,nEpisodes,nSteps,epsilon=0,temperature=0):
        '''qLearning algorithm.  Epsilon exploration and Boltzmann exploration
        are combined in one procedure by sampling a random action with 
        probabilty epsilon and performing Boltzmann exploration otherwise.  
        When epsilon and temperature are set to 0, there is no exploration.

        Inputs:
        s0 -- initial state
        initialQ -- initial Q function (|A|x|S| array)
        nEpisodes -- # of episodes (one episode consists of a trajectory of nSteps that starts in s0
        nSteps -- # of steps per episode
        epsilon -- probability with which an action is chosen at random
        temperature -- parameter that regulates Boltzmann exploration

        Outputs: 
        Q -- final Q function (|A|x|S| array)
        policy -- final policy
        '''

        # temporary values to ensure that the code compiles until this
        # function is coded
        Q = initialQ.copy()
        policy = np.zeros(self.mdp.nStates,int)
        gamma = self.mdp.discount
        alpha = 0.1
        actions = np.arange(self.mdp.nActions)
        
        episode_returns = np.zeros(nEpisodes, dtype=float)
        for ep in range(nEpisodes):
            s = s0
            G = 0.0
            gamma_pow = 1.0
            for _ in range(nSteps):
                u = np.random.rand()
                
                # Action selection
                if u < epsilon:
                    # Pick uniform random action
                    a = np.random.randint(self.mdp.nActions)
                elif temperature > 0:
                    qcol = Q[:, s]
                    prefs = qcol / temperature
                    # Now exponentiate to turn our list of actions from this state into a probability distribution based on reward
                    prefs -= prefs.max() # For numerical stability to avoid blowing up when exponentiating
                    probs = np.exp(prefs) / np.sum(np.exp(prefs))
                    u = np.random.rand()
                    cum_probs = np.cumsum(probs)
                    a = np.where(cum_probs >= u)[0][0]
                else:
                    qcol = Q[:, s]
                    m = qcol.max()
                    actions_accomplishing_max = []
                    for i in range(len(qcol)):
                        if qcol[i] == m:
                            actions_accomplishing_max.append(i)
                    a = actions_accomplishing_max[0]
                    
                # Environment step
                reward, s_next = self.sampleRewardAndNextState(s, a)
                
                # Q-learning update
                td_target = reward + gamma * np.max(Q[:, s_next]) # Learned from taking this action
                Q[a, s] = (1 - alpha)*Q[a, s] + alpha*td_target
                
                # Accumulate discounted return
                G += gamma_pow * reward
                episode_returns[ep] = G
                gamma_pow *= gamma
                
                # Advance state
                s = s_next

        # Extract final policy from Q - which means being greedy at each state
        for s in range(self.mdp.nStates):
            qcol = Q[:, s]
            m = qcol.max()
            actions_accomplishing_max = []
            for i in range(len(qcol)):
                if abs(qcol[i] - m) <= 1e-12:
                    actions_accomplishing_max.append(i)
            a = actions_accomplishing_max[0]
            policy[s] = a

        return [Q,policy,episode_returns]