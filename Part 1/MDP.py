import math
import numpy as np

class MDP:
    '''A simple MDP class.  It includes the following members'''

    def __init__(self,T,R,discount):
        '''Constructor for the MDP class

        Inputs:
        T -- Transition function: |A| x |S| x |S'| array
        R -- Reward function: |A| x |S| array
        discount -- discount factor: scalar in [0,1)

        The constructor verifies that the inputs are valid and sets
        corresponding variables in a MDP object'''

        assert T.ndim == 3, "Invalid transition function: it should have 3 dimensions"
        self.nActions = T.shape[0]
        self.nStates = T.shape[1]
        assert T.shape == (self.nActions,self.nStates,self.nStates), "Invalid transition function: it has dimensionality " + repr(T.shape) + ", but it should be (nActions,nStates,nStates)"
        assert (abs(T.sum(2)-1) < 1e-5).all(), "Invalid transition function: some transition probability does not equal 1"
        self.T = T
        assert R.ndim == 2, "Invalid reward function: it should have 2 dimensions" 
        assert R.shape == (self.nActions,self.nStates), "Invalid reward function: it has dimensionality " + repr(R.shape) + ", but it should be (nActions,nStates)"
        self.R = R
        assert 0 <= discount < 1, "Invalid discount factor: it should be in [0,1)"
        self.discount = discount
        
    def valueIteration(self,initialV,nIterations=np.inf,tolerance=0.01):
        '''Value iteration procedure
        V <-- max_a R^a + gamma T^a V

        Inputs:
        initialV -- Initial value function: array of |S| entries
        nIterations -- limit on the # of iterations: scalar (default: infinity)
        tolerance -- threshold on ||V^n-V^n+1||_inf: scalar (default: 0.01)

        Outputs: 
        V -- Value function: array of |S| entries
        iterId -- # of iterations performed: scalar
        epsilon -- ||V^n-V^n+1||_inf: scalar'''
        
        # temporary values to ensure that the code compiles until this
        # function is coded
        new_V = initialV.copy()
        assert new_V.shape == (self.nStates,), "Invalid initial value function: it has dimensionality " + repr(new_V.shape) + ", but it should be (nStates,)"
        nIterations_done = 0
        epsilon = np.inf
        while nIterations_done < nIterations and epsilon > tolerance:
            initialV_for_iter = new_V
            new_V_for_iter = -np.inf * np.ones(self.nStates) # Start with the lowest possible value so that any action can improve it
            for a in range(self.nActions):
                T_actions = self.T[a] # Give us the probability transition matrix for action a
                R_actions = self.R[a] # Give us the reward vector for action a from each state
                V_for_a = R_actions + self.discount * np.dot(T_actions, initialV_for_iter) # Updates the value of all states based on the reward we get from this action plus the weighted average of the values of the next states
                new_V_for_iter = np.maximum(new_V_for_iter, V_for_a) # Each state gets a chance to improve

            new_V = new_V_for_iter
            epsilon = np.linalg.norm(new_V_for_iter - initialV_for_iter, np.inf)
            nIterations_done += 1
            
        epsilon = epsilon.item() # Convert from 0-dim array to scalar
        return (new_V,nIterations_done,epsilon)

    def extractPolicy(self,V):
        '''Procedure to extract a policy from a value function
        pi <-- argmax_a R^a + gamma T^a V

        Inputs:
        V -- Value function: array of |S| entries

        Output:
        policy -- Policy: array of |S| entries'''

        # temporary values to ensure that the code compiles until this
        # function is coded
        assert V.shape == (self.nStates,), "Invalid value function: it has dimensionality " + repr(V.shape) + ", but it should be (nStates,)"
        policy = np.zeros(self.nStates, dtype=int)
        for i in range(self.nStates):
            # Determine the best action in state i
            record = -math.inf
            rewards_for_state = self.R[:, i] # Get the rewards for all actions in state i
            transition_for_state = self.T[:, i, :] # Get the transition probabilities for all actions from this state
            future_state_values = np.dot(transition_for_state, V) # Get the expected value of the next state for each action
            for a in range(self.nActions):
                value_for_a = rewards_for_state[a] + self.discount * future_state_values[a]
                if value_for_a > record:
                    record = value_for_a
                    policy[i] = a

        return policy 

    def evaluatePolicy(self,policy):
        '''Evaluate a policy by solving a system of linear equations
        V^pi = R^pi + gamma T^pi V^pi

        Input:
        policy -- Policy: array of |S| entries

        Ouput:
        V -- Value function: array of |S| entries'''

        # Turn policy into a numpy array if it isn't already
        policy = np.array(policy, dtype=int)
        assert policy.shape == (self.nStates,), "Invalid policy: it has dimensionality " + repr(policy.shape) + ", but it should be (nStates,)"
        I = np.zeros((self.nStates,self.nStates))
        for i in range(self.nStates):
            I[i,i] = 1.0
        P_pi = np.zeros((self.nStates,self.nStates)) # Transition matrix for the policy - since we KNOW what actions will be taken at each state
        for s in range(self.nStates):
            a = policy[s]
            P_pi[s,:] = self.T[a,s,:]
            assert abs(P_pi[s,:].sum() - 1) < 1e-5, "Invalid transition probabilities for state " + repr(s) + ": they sum to " + repr(P_pi[s,:].sum()) + ", but they should sum to 1"
        print("Transition matrix for policy:\n" + repr(P_pi))
        
        R_pi = np.zeros(self.nStates) # We know the reward at each state according to the policy because we know what action we will take
        for s in range(self.nStates):
            a = policy[s]
            R_pi[s] = self.R[a,s]
        print("Reward vector for policy at each state: " + repr(R_pi))
            
        A = I - self.discount * P_pi
        try:
            A_inv = np.linalg.inv(A) # n_states by n_states
        except np.linalg.LinAlgError:
            raise ValueError("The system of linear equations is not solvable")
        V = np.dot(A_inv, R_pi) # n_states by 1

        return V
        
    def policyIteration(self,initialPolicy,nIterations=np.inf):
        '''Policy iteration procedure: alternate between policy
        evaluation (solve V^pi = R^pi + gamma T^pi V^pi) and policy
        improvement (pi <-- argmax_a R^a + gamma T^a V^pi).

        Inputs:
        initialPolicy -- Initial policy: array of |S| entries
        nIterations -- limit on # of iterations: scalar (default: inf)

        Outputs: 
        policy -- Policy: array of |S| entries
        V -- Value function: array of |S| entries
        iterId -- # of iterations peformed by modified policy iteration: scalar'''

        # temporary values to ensure that the code compiles until this
        # function is coded
        assert len(initialPolicy) == self.nStates, "Invalid initial policy: it has length " + repr(len(initialPolicy)) + ", but it should be nStates"
        policy = np.array(initialPolicy, dtype=int)
        V = np.zeros(self.nStates)
        iterId = 0
        while iterId < nIterations:
            old_policy = policy.copy()
            V = self.evaluatePolicy(policy)
            # Now that state values are updated, policy could change
            policy = self.extractPolicy(V)
            iterId += 1
            if (policy == old_policy).all():
                break

        return (policy,V,iterId)
            
    def evaluatePolicyPartially(self,policy,initialV,nIterations=np.inf,tolerance=0.01):
        '''Partial policy evaluation:
        Repeat V^pi <-- R^pi + gamma T^pi V^pi

        Inputs:
        policy -- Policy: array of |S| entries
        initialV -- Initial value function: array of |S| entries
        nIterations -- limit on the # of iterations: scalar (default: infinity)
        tolerance -- threshold on ||V^n-V^n+1||_inf: scalar (default: 0.01)

        Outputs: 
        V -- Value function: array of |S| entries
        iterId -- # of iterations performed: scalar
        epsilon -- ||V^n-V^n+1||_inf: scalar'''

        # temporary values to ensure that the code compiles until this
        # function is coded
        V = np.zeros(self.nStates)
        iterId = 0
        epsilon = 0

        return [V,iterId,epsilon]

    def modifiedPolicyIteration(self,initialPolicy,initialV,nEvalIterations=5,nIterations=np.inf,tolerance=0.01):
        '''Modified policy iteration procedure: alternate between
        partial policy evaluation (repeat a few times V^pi <-- R^pi + gamma T^pi V^pi)
        and policy improvement (pi <-- argmax_a R^a + gamma T^a V^pi)

        Inputs:
        initialPolicy -- Initial policy: array of |S| entries
        initialV -- Initial value function: array of |S| entries
        nEvalIterations -- limit on # of iterations to be performed in each partial policy evaluation: scalar (default: 5)
        nIterations -- limit on # of iterations to be performed in modified policy iteration: scalar (default: inf)
        tolerance -- threshold on ||V^n-V^n+1||_inf: scalar (default: 0.01)

        Outputs: 
        policy -- Policy: array of |S| entries
        V -- Value function: array of |S| entries
        iterId -- # of iterations peformed by modified policy iteration: scalar
        epsilon -- ||V^n-V^n+1||_inf: scalar'''
        
        assert len(initialPolicy) == self.nStates, "Invalid initial policy: it has length " + repr(len(initialPolicy)) + ", but it should be nStates"
        assert initialV.shape == (self.nStates,), "Invalid initial value function: it has dimensionality " + repr(initialV.shape) + ", but it should be (nStates,)"

        # temporary values to ensure that the code compiles until this
        # function is coded
        policy = np.zeros(self.nStates)
        V = np.zeros(self.nStates)
        iterId = 0
        epsilon = 0

        return [policy,V,iterId,epsilon]
        