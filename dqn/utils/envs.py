# Play an episode according to a given policy
# env: environment
# policy: function(env, state)
# render: whether to render the episode or not (default - False)
def play_episode(env, policy, render = False):
    states, actions, rewards = [], [], []
    
    s, _ = env.reset()
    done = False
    while not done:
        if render: env.render()
        a = policy(env, s)
        s2, r, terminated, truncated, _ = env.step(a)
        done = terminated or truncated
       
        states.append(s)
        actions.append(a)
        rewards.append(r)
        s = s2
    return states, actions, rewards

# Play an episode according to a given policy and add 
# to a replay buffer
# env: environment
# policy: function(env, state)
def play_episode_rb(env, policy, buf, render=False):
    states, actions, rewards = [], [], []
    
    s, _ = env.reset()
    done = False
    while not done:
        if render: env.render()
        a = policy(env, s)
        s2, r, terminated, truncated, _ = env.step(a)
        done = terminated or truncated
        
        buf.add(s, a, r, s2, float(done))
        
        states.append(s)
        actions.append(a)
        rewards.append(r)
        s = s2
    return states, actions, rewards
