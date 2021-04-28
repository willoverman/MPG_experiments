import numpy as np
import matplotlib.pyplot as plt

N = 6
M = 10
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

def get_reward(state, actions):
    return 0

def get_next_state(state, actions):
    return state

def pick_action(prob_dist):
    acts = [i for i in range(len(prob_dist))]
    action = np.random.choice(acts, 1, p = prob_dist)
    return action

def visit_dist(state, policy, gamma, T):
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


def policy_gradient(mu, max_iters, gamma, eta, T):
    joint_policy = {}
    for s in range(S):
        for i in range(N):
            joint_policy[s, i] = [1/M for i in range(M)]

    # CHANGE THIS 
    # dist = {0:mu[0], 1:mu[1]} 
            
    for t in range(max_iters):
        #dist0 = visit_dist(0, joint_policy, gamma, T)
        #dist1 = visit_dist(1, joint_policy, gamma, T)
        #print(dist0)       
        #print(dist1)       
        #dist[0] = mu[0]*dist0[0]+mu[1]*dist1[0]
        #dist[1] = mu[0]*dist0[1]+mu[1]*dist1[1]
        #print(dist)        
        
        agent1grad00 = dist[0] * Q_function1(0, 0, joint_policy, gamma, T) 
        agent1grad01 = dist[0] * Q_function1(0, 1, joint_policy, gamma, T)
        agent1grad10 = dist[1] * Q_function1(1, 0, joint_policy, gamma, T)
        agent1grad11 = dist[1] * Q_function1(1, 1, joint_policy, gamma, T)
        agent2grad00 = dist[0] * Q_function2(0, 0, joint_policy, gamma, T) 
        agent2grad01 = dist[0] * Q_function2(0, 1, joint_policy, gamma, T)
        agent2grad10 = dist[1] * Q_function2(1, 0, joint_policy, gamma, T)
        agent2grad11 = dist[1] * Q_function2(1, 1, joint_policy, gamma, T)

        joint_policy[0,0] = projection_simplex_sort(np.add(joint_policy[0,0], eta * np.array([agent1grad00, agent1grad01])), z=1)
        joint_policy[0,1] = projection_simplex_sort(np.add(joint_policy[0,1], eta * np.array([agent2grad00, agent2grad01])), z=1)
        joint_policy[1,0] = projection_simplex_sort(np.add(joint_policy[1,0], eta * np.array([agent1grad10, agent1grad11])), z=1)
        joint_policy[1,1] = projection_simplex_sort(np.add(joint_policy[1,1], eta * np.array([agent2grad10, agent2grad11])), z=1)

        #print(joint_policy)

    return joint_policy





