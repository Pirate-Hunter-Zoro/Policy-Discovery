from .MDP import *

''' Construct simple MDP as described in Lecture 2a Slides 13-14'''
# Transition function: |A| x |S| x |S'| array
T = np.array([[[0.5,0.5,0,0],[0,1,0,0],[0.5,0.5,0,0],[0,1,0,0]],[[1,0,0,0],[0.5,0,0,0.5],[0.5,0,0.5,0],[0,0,0.5,0.5]]])
# Reward function: |A| x |S| array
R = np.array([[0,0,10,10],[0,0,10,10]])
# Discount factor: scalar in [0,1)
discount = 0.9        
# MDP object
mdp = MDP(T,R,discount)

'''Test each procedure'''
[V,nIterations,epsilon] = mdp.valueIteration(initialV=np.zeros(mdp.nStates))
print("Value Iteration Results:")
print(V)
print(nIterations)
print(epsilon)
print("-------------------------------")

policy = mdp.extractPolicy(V)
V = mdp.evaluatePolicy(policy)
print("Policy Extraction and Evaluation Results:")
print(policy)
print(V)
print("-------------------------------")

[policy,V,iterId] = mdp.policyIteration(np.array([0,0,0,0]))
print("Policy Iteration Results:")
print(policy)
print(V)
print(iterId)
print("-------------------------------")

[V,iterId,epsilon] = mdp.evaluatePolicyPartially(policy,np.array([0,0,0,0]), nIterations=1)
print("Partial Policy Evaluation Results:")
print(V)
print(iterId)
print(epsilon)
print("-------------------------------")

[policy,V,iterId,outer_epsilon] = mdp.modifiedPolicyIteration(np.array([1,0,1,0]),np.array([0,10,0,13]), nIterations=1)
print("Modified Policy Iteration Results:")
print(policy)
print(V)
print(iterId)
print(outer_epsilon)
print("-------------------------------")