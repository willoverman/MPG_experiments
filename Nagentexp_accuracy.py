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

def value_function(policy, gamma, T):
    value_fun = {(s,i):0 for s in range(S) for i in range(N)}
    for state in range(S):
        for k in range(10):
            curr_state = state
            for t in range(T):
                actions = [pick_action(policy[state, i]) for i in range(N)]
                rewards = get_reward(curr_state, actions)
                for i in range(N):
                    value_fun[state,i] += (gamma**t)*rewards[i]
                curr_state = get_next_state(curr_state, actions)
    value_fun.update((x,v/10) for (x,v) in value_fun.items())
    return value_fun

def current_accuracy(mu,policy,policy_star,gamma,T):
    v_star = value_function(policy_star,gamma,T)
    v_pi   = value_function(policy,gamma,T)
    total_dif = N * [0]
    for agent in range(N):
        for s in range(S):
            total_dif[agent] += mu[s]*(value_function(policy,gamma,T)[s,agent]-value_function(policy_star,gamma,T)[s,agent])
    return np.max(total_dif)

def Q_function(agent, state, action, policy, value_fun, gamma, T, samples):
    tot_reward = 0
    for i in range(samples):
        actions = [pick_action(policy[state, i]) for i in range(N)]
        actions[agent] = action
        reward = get_reward(state, actions)
        tot_reward += reward[agent]
        next_state  = get_next_state(state, actions)
        tot_reward += value_fun[next_state,agent] #tot_reward += value_function(policy, gamma, T)[next_state, agent]
    return (tot_reward / samples)


def policy_gradient(mu, max_iters, gamma, eta, T, samples, epsilon):
    policy = {}
    policy_star = {(0, 0): np.array([1., 0.]), (0, 1): np.array([0., 1.]), (1, 0): np.array([1., 0.]), (1, 1): np.array([1., 0.])}
    ca = []
    
    for s in range(S):
        for i in range(N):
            policy[s, i] = [1/M for i in range(M)]
            
    for t in range(max_iters):

        a_dist = M *[[0]] #i didnt know what to call it, just intermediate dist values
        for st in range(S):
            a_dist[st] = visit_dist(st, policy, gamma, T)
        
        b_dist = M * [0] #the ones we actually use
        for st in range(S):
            b_dist[st] = np.dot(a_dist[st], mu)

        v_pi  = value_function(policy,gamma,T)
        grads = np.zeros((N, S, M))
        for agent in range(N):
            for st in range(S):
                for act in range(M):
                    grads[agent, st, act] = b_dist[st] * Q_function(agent, st, act, policy, v_pi, gamma, T, samples)

        for agent in range(N):
            for st in range(S):
                policy[st, agent] = projection_simplex_sort(np.add(policy[st, agent], eta * grads[agent,st]), z=1)
        
#         print(policy)
        
        ca.append(current_accuracy(mu,policy,policy_star,gamma,T))
        if ca[t] < epsilon:
            iter = t 
            break

    return policy,iter,ca

policy_gradient([1, 0],60,0.99,0.001,10,5,0.01)
