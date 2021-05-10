import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

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

    for i in range(15):
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
    repeat = 20
    value_fun = {(s,i): 0 for s in range(S) for i in range(N)}
    for state in range(S):
        for k in range(repeat):
            curr_state = state
            for t in range(T):
                actions = [pick_action(policy[curr_state, i]) for i in range(N)]
                rewards = get_reward(curr_state, actions)
                for i in range(N):
                    value_fun[state,i] += (gamma**t)*rewards[i]
                curr_state = get_next_state(curr_state, actions)
    value_fun.update((x,v/repeat) for (x,v) in value_fun.items())
    return value_fun

def eval_value_pi(mu,value_fun):
    vs_pi = N * [0]
    for agent in range(N):
        for s in range(S):
            vs_pi[agent] += mu[s]*value_fun[s,agent]
    return vs_pi
    
def get_agent_policy(agent,policy):
    agent_pol = {s:policy[s,agent] for s in range(S)}  
    return agent_pol
    
def current_accuracy(mu,v_pi,v_star,gamma,T):
    total_dif = N * [0]
    for agent in range(N):
        for s in range(S):
            total_dif[agent] += mu[s]*(v_star[s,agent]-v_pi[s,agent])
    return np.max(total_dif)

def Q_function(agent, state, action, policy, v_pi, samples):
    tot_reward = 0
    for i in range(samples):
        actions = [pick_action(policy[state, a]) for a in range(N)]
        actions[agent] = action
        reward = get_reward(state, actions)
        tot_reward += reward[agent]
        curr_state  = get_next_state(state, actions)
        tot_reward += v_pi[curr_state,agent] #tot_reward += value_function(policy, gamma, T)[next_state, agent]
    return tot_reward / samples

def policy_gradient(mu, max_iters, gamma, eta, T, samples, epsilon):
    policy = {(s,i): M*[1/M] for s in range(S) for i in range(N)}
    v_pi = value_function(policy,gamma,T)
    v_curr_policy = eval_value_pi(mu,v_pi)
    val_accuracy  = []
    jpol_accuracy = []
    apol_accuracy = []
    itr = max_iters
    
    for t in range(max_iters):
        b_dist = M * [0] #the ones we actually use
        for st in range(S):
            a_dist = visit_dist(st, policy, gamma, T)
            b_dist[st] = np.dot(a_dist, mu)

        # Keep last policy to determine L2 or L1 distance with current.
        last_policy   = dict.copy(policy)
        v_last_policy = v_curr_policy
        
        grads = np.zeros((N, S, M))
        for st in range(S):
            for agent in range(N):
                for act in range(M):
                    grads[st,agent,act] = b_dist[st] * Q_function(agent, st, act, policy, v_pi, samples)
        
        for st in range(S):
            for agent in range(N):
                policy[st, agent] = projection_simplex_sort(np.add(policy[st, agent], eta * grads[st,agent]), z=1)
                 
        # Measure the current accuracy in different ways
        # 1. Value Function
        # These are two arrays with N coordinates: in the i-th coordinate, they contain the value of agent i 
        # for the initial distribution mu. Then, calculate differences for each player and keep the max        
        v_pi = value_function(policy,gamma,T)
        v_curr_policy = eval_value_pi(mu,v_pi)
        # Note the abs value: the next quantity may be positive or negative but we care about convergence.
        val_accuracy.append(np.max(np.abs(np.subtract(v_curr_policy,v_last_policy))))
        
        if val_accuracy[t] <= epsilon:
            itr = t
            break
        
        # 2. Joint Policy in l2 (or l1 by changing 2->1) 
        jpol_accuracy.append(np.linalg.norm(np.subtract(list(policy.values()),list(last_policy.values())),2))

#         if jpol_accuracy[t] <= epsilon:
#             iter = t
#             break
        
        # 3. Agent-wise policy in l2 (or l1 by changing 2->1)
        acu = N * [np.inf]
        for agent in range(N):
            curr_pol   = get_agent_policy(agent,policy)
            last_pol   = get_agent_policy(agent,last_policy)
            acu[agent] = np.linalg.norm(np.subtract(list(curr_pol.values()),list(last_pol.values())),2)
        apol_accuracy.append(np.max(acu))      

        if apol_accuracy[t] <= epsilon:
            itr = t
            break

#         print(policy)
        
    return policy,itr,val_accuracy,jpol_accuracy,apol_accuracy


#differing learning rates, plotting iters on x axis and covnergence on y
plt.figure()
for lr in [.001, .0001]:
    policy, itr, va, jpa, apa = policy_gradient([1, 0],100,0.99,lr,10,5,0.0000001)
    plt.plot(va, label=lr)
plt.legend()
plt.show()

plt.figure()
policy, itr, va, jpa, apa  = policy_gradient([1, 0],100,0.99,.0005,10,5,0.0000001)
x = [i for i in range(itr+1)]
#curve = np.poly1d(np.polyfit(x, ca, 2))
#plt.plot(x, curve(x), 'o', x, ca, '-')
plt.plot(x, va, '-')
plt.show()

#print(policy_gradient([1, 0],100,0.99,0.001,10,5,0.01))
