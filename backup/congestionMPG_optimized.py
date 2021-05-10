from congestion_games import *
import matplotlib.pyplot as plt
import numpy as np
import copy
import statistics
import seaborn as sns; sns.set()
from time import process_time

myp_start = process_time()

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

safe_state = CongGame(4,1,[[1,0],[2,0],[4,0],[6,0]])
bad_state = CongGame(4,1,[[1,-100],[2,-100],[4,-100],[6,-100]])

N = safe_state.n
M = safe_state.num_actions 
S = 2

#dictionary associating each action (value) to an integer (key)
act_dic = {}
counter = 0
for act in safe_state.actions:
	act_dic[counter] = act 
	counter += 1

state_dic = {0: safe_state, 1: bad_state}
selected_profiles={}

def get_next_state(state, actions):
	acts_from_ints = [act_dic[i] for i in actions]
	density = state_dic[state].get_counts(acts_from_ints)
	max_density = max(density)
	if state == 0 and max_density > N/2 or state == 1 and max_density > N/4:
		return 1
	return 0

def pick_action(prob_dist):
    acts = [i for i in range(len(prob_dist))]
    action = np.random.choice(acts, 1, p = prob_dist)
    return action[0]

def visit_dist(state, policy, gamma, T,samples):
    visit_states = {st: np.zeros(T) for st in range(S)}        
    for i in range(samples):
        curr_state = state
        visit_states[curr_state][0] += 1
        for t in range(1,T):
            actions = [pick_action(policy[state, i]) for i in range(N)]
            curr_state = get_next_state(curr_state, actions)
            visit_states[curr_state][t] += 1
    dist = [np.dot(v/samples,gamma**np.arange(T)) for (k,v) in visit_states.items()]
    return dist 

def value_function(policy, gamma, T,samples):
    value_fun = {(s,i):0 for s in range(S) for i in range(N)}
    for state in range(S):
        for k in range(samples):
            curr_state = state
            for t in range(T):
                actions = [pick_action(policy[curr_state, i]) for i in range(N)]
                q = tuple(actions+[state])
                rewards = selected_profiles.setdefault(q,get_reward(state_dic[state], [act_dic[i] for i in actions]))                  
                for i in range(N):
                    value_fun[state,i] += (gamma**t)*rewards[i]
                curr_state = get_next_state(curr_state, actions)
    value_fun.update((x,v/samples) for (x,v) in value_fun.items())
    return value_fun

def Q_function(agent, state, action, policy, value_fun, samples):
    tot_reward = 0
    for i in range(samples):
        actions = [pick_action(policy[state, i]) for i in range(N)]
        actions[agent] = action
        q = tuple(actions+[state])
        rewards = selected_profiles.setdefault(q,get_reward(state_dic[state], [act_dic[i] for i in actions]))
        tot_reward += rewards[agent] + value_fun[get_next_state(state, actions), agent]
    return (tot_reward / samples)

def policy_gradient(mu, max_iters, gamma, eta, T, samples):
    policy = {}
    for s in range(S):
        for i in range(N):
            policy[s, i] = [1/M for i in range(M)]

    policy_hist = []
    policy_hist.append(copy.deepcopy(policy))

    for t in range(max_iters):

        b_dist = M * [0] #the ones we actually use
        for st in range(S):
            a_dist = visit_dist(st, policy, gamma, T, samples)
            b_dist[st] = np.dot(a_dist, mu)
            
        grads = np.zeros((N, S, M))
        
        value_fun = value_function(policy, gamma, T, samples)        
        for agent in range(N):
            for st in range(S):
                for act in range(M):
                    grads[agent, st, act] = b_dist[st] * Q_function(agent, st, act, policy, value_fun, samples)

        for agent in range(N):
            for st in range(S):
                policy[st, agent] = projection_simplex_sort(np.add(policy[st, agent], eta * grads[agent,st]), z=1)
        policy_hist.append(copy.deepcopy(policy))
    return policy_hist

def policy_accuracy(policy_pi, policy_star):
    total_dif = N * [0]
    for agent in range(N):
        for state in range(S):
            total_dif[agent] += np.sum(np.abs((policy_pi[state, agent] - policy_star[state, agent])))
    return np.max(total_dif)

def get_accuracies(policy_hist):
    fin = policy_hist[-1]
    accuracies = []
    for i in range(len(policy_hist)):
        this_acc = policy_accuracy(policy_hist[i], fin)
        accuracies.append(this_acc)
    return accuracies

def full_experiment(runs,iters,T,samples):
    plot_accuracies = []
    for k in range(runs):
        policy_hist = policy_gradient([1, 0],iters,0.99,0.001,T,samples)
        plot_accuracies.append(get_accuracies(policy_hist))

    pmean = list(map(statistics.mean, zip(*plot_accuracies)))
    pstdv = list(map(statistics.stdev, zip(*plot_accuracies)))
    clrs = sns.color_palette("husl", 3)
    piters = list(range(iters+1))
    fig = plt.figure(figsize=(6,4))
    ax = sns.lineplot(piters, pmean, color = clrs[0],label= 'Mean L1-accuracy')
    ax.fill_between(piters, np.subtract(pmean,pstdv), np.add(pmean,pstdv), alpha=0.3, facecolor=clrs[0],label="1-standard deviation")
    ax.legend()
    plt.grid(linewidth=0.6)
    plt.gca().set(xlabel='Iterations',ylabel='L1-accuracy', title='Policy Gradient: {} Runs'.format(runs))
    plt.show()
    fig.savefig('experiment_{}{}{}{}.png'.format(runs,iters,T,samples),bbox_inches='tight')
    return fig

fig = full_experiment(5,4,1,1)

myp_end = process_time()
elapsed_time = myp_end - myp_start
print(elapsed_time)
