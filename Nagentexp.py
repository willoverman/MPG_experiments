import numpy as np
import matplotlib.pyplot as plt

N = 2
M = 2
S = 2


def projection_simplex_sort(v, z=1):
    if v.sum() == z and np.alltrue(v >= 0):
        # best projection: itself!
        return v
    n_features = v.shape[0]
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u) - z
    ind = np.arange(n_features) + 1
    cond = u - cssv / ind > 0
    rho = ind[cond][-1]
    theta = cssv[cond][-1] / float(rho)
    w = np.maximum(v - theta, 0)
    return w

def get_reward(state, acts):
    if state == 0:
        if acts[0]== 0 and acts[1] == 0:
            return [5,2]
        elif acts[0] == 0 and acts[1] == 1:
            return [-1,-2]
        elif acts[0] == 1 and acts[1] == 0:
            return [-5,-4]
        else:
            return [1,4]

    elif state == 1:
        if acts[0] == 0 and acts[1] == 0:
            return [32,29]
        elif acts[0] == 0 and acts[1] == 1:
            return [-28,-29]
        elif acts[0] == 1 and acts[1] == 0:
            return [-32,-31]
        else:
            return [28,31]
    else:
        return [0,0]

def get_next_state(state, acts):
    if acts[0] != acts[1]:
        if state == 0:
            return 1
        return 0
    return state

def pick_action(prob_dist):
    acts = [i for i in range(len(prob_dist))]
    action = np.random.choice(acts, 1, p = prob_dist)
    return action

def visit_dist(state, policy, gamma, T):
    visit_states = {st: np.zeros(T) for st in range(S)}        

    for i in range(10):
        curr_state = state
        visit_states[curr_state][0] += 1
        for t in range(1,T):
            actions = [pick_action(policy[state, i]) for i in range(N)]
            curr_state = get_next_state(curr_state, actions)
            visit_states[curr_state][t] += 1
    
    # This is the un-normalized distribution. The normalizing constant would be (1-gamma^T)/(1-gamma) (or 1-gamma^{T+1}/1-gamma) depending
    # on where the index ends. But according to the formula in Kakade, the normalizing constant appears in the derivative of the value function
    # in which we are interested in, so it cancels out. We probably need to double-check this though.
    dist = [np.dot(v/10,gamma**np.arange(T)) for (k,v) in visit_states.items()]
    return dist 

def Q_function(agent, state, action, policy, gamma, T, samples):
    tot_reward = 0
    for i in range(samples):
        actions = [pick_action(policy[state, i]) for i in range(N)]
        actions[agent] = action
        curr_state = state
        reward = get_reward(state, actions)
        curr_state = get_next_state(curr_state, actions)
        tot_reward += reward[agent]

    return (tot_reward / samples)


def policy_gradient(mu, max_iters, gamma, eta, T, samples):
    policy = {}
    for s in range(S):
        for i in range(N):
            policy[s, i] = [1/M for i in range(M)]
            
    for t in range(max_iters):

        a_dist = M *[[0]] #i didnt know what to call it, just intermediate dist values
        for st in range(S):
            a_dist[st] = visit_dist(st, policy, gamma, T)
        
        b_dist = M * [0] #the oens we actually use
        for st in range(S):
            b_dist[st] = np.dot(a_dist[st], mu)

        grads = np.zeros((N, S, M))
        for agent in range(N):
            for st in range(S):
                for act in range(M):
                    grads[agent, st, act] = b_dist[st] * Q_function(agent, st, act, policy, gamma, T, samples)

        for agent in range(N):
            for st in range(S):
                policy[st, agent] = projection_simplex_sort(np.add(policy[st, agent], eta * grads[agent,st]), z=1)

    return policy

print(policy_gradient([1, 0],100,0.99,0.001,10,10))




