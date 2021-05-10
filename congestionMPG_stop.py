from congestion_games import *
import matplotlib.pyplot as plt
import numpy as np
import copy
import statistics
import seaborn as sns; sns.set()
from time import process_time

myp_start = process_time()

def projection_simplex_sort(v, z=1):
	# Courtesy: EdwardRaff/projection_simplex.py
    if v.sum() == z and np.alltrue(v >= 0):
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

# Define the states and some necessary info
safe_state = CongGame(8,1,[[1,0],[2,0],[4,0],[6,0]])
bad_state = CongGame(8,1,[[1,-100],[2,-100],[4,-100],[6,-100]])
state_dic = {0: safe_state, 1: bad_state}
N = safe_state.n
M = safe_state.num_actions 
S = 2

# Dictionary to store the action profiles and rewards to
selected_profiles = {}

# Dictionary associating each action (value) to an integer (key)
act_dic = {}
counter = 0
for act in safe_state.actions:
	act_dic[counter] = act 
	counter += 1

def get_next_state(state, actions):
	acts_from_ints = [act_dic[i] for i in actions]
	density = state_dic[state].get_counts(acts_from_ints)
	max_density = max(density)
	if state == 0 and max_density > N/2 or state == 1 and max_density > N/4:
	# if state == 0 and max_density > N/2 and np.random.uniform() > 0.2 or state == 1 and max_density > N/4 and np.random.uniform() > 0.1:
		return 1
	return 0

def pick_action(prob_dist):
    # np.random.choice(range(len(prob_dist)), 1, p = prob_dist)[0]
    acts = [i for i in range(len(prob_dist))]
    action = np.random.choice(acts, 1, p = prob_dist)
    return action[0]

def visit_dist(state, policy, gamma, T,samples):
    # This is the unnormalized visitation distribution. Since we take finite trajectories, the normalization constant is (1-gamma**T)/(1-gamma).
    visit_states = {st: np.zeros(T) for st in range(S)}        
    for i in range(samples):
        curr_state = state
        for t in range(T):
            visit_states[curr_state][t] += 1
	    actions = [pick_action(policy[curr_state, i]) for i in range(N)]
            curr_state = get_next_state(curr_state, actions)
    dist = [np.dot(v/samples,gamma**np.arange(T)) for (k,v) in visit_states.items()]
    return dist 

def value_function(policy, gamma, T,samples):
    value_fun = {(s,i):0 for s in range(S) for i in range(N)}
    for k in range(samples):
        for state in range(S):
            curr_state = state
            for t in range(T):
                actions = [pick_action(policy[curr_state, i]) for i in range(N)]
                q = tuple(actions+[curr_state])
                rewards = selected_profiles.setdefault(q,get_reward(state_dic[curr_state], [act_dic[i] for i in actions]))                  
                for i in range(N):
                    value_fun[state,i] += (gamma**t)*rewards[i]
                curr_state = get_next_state(curr_state, actions)
    value_fun.update((x,v/samples) for (x,v) in value_fun.items())
    return value_fun

def Q_function(agent, state, action, policy, gamma, value_fun, samples):
    tot_reward = 0
    for i in range(samples):
        actions = [pick_action(policy[state, i]) for i in range(N)]
        actions[agent] = action
        q = tuple(actions+[state])
        rewards = selected_profiles.setdefault(q,get_reward(state_dic[state], [act_dic[i] for i in actions]))
        tot_reward += rewards[agent] + gamma*value_fun[get_next_state(state, actions), agent]
    return (tot_reward / samples)

def policy_accuracy(policy_pi, policy_star):
    total_dif = N * [0]
    for agent in range(N):
        for state in range(S):
            total_dif[agent] += np.sum(np.abs((policy_pi[state, agent] - policy_star[state, agent])))
    return np.max(total_dif)

def policy_gradient(mu, max_iters, gamma, eta, T, samples):
    policy = {}
    for s in range(S):
        for i in range(N):
            policy[s, i] = [1/M for i in range(M)]

    policy_hist = []
    policy_hist.append(copy.deepcopy(policy))

    for t in range(max_iters):
        print(t)

        b_dist = M * [0] #the ones we actually use
        for st in range(S):
            a_dist = visit_dist(st, policy, gamma, T, samples)
            b_dist[st] = np.dot(a_dist, mu)
            
        grads = np.zeros((N, S, M))
        
        value_fun = value_function(policy, gamma, T, samples)        
        for agent in range(N):
            for st in range(S):
                for act in range(M):
                    grads[agent, st, act] = b_dist[st] * Q_function(agent, st, act, policy, gamma, value_fun, samples)

        for agent in range(N):
            for st in range(S):
                policy[st, agent] = projection_simplex_sort(np.add(policy[st, agent], eta * grads[agent,st]), z=1)
        policy_hist.append(copy.deepcopy(policy))

        if policy_accuracy(policy_hist[t], policy_hist[t-1]) < 10e-16:
            return policy_hist

    return policy_hist


def get_accuracies(policy_hist):
    fin = policy_hist[-1]
    accuracies = []
    for i in range(len(policy_hist)):
        this_acc = policy_accuracy(policy_hist[i], fin)
        accuracies.append(this_acc)
    return accuracies

def full_experiment(runs,iters,T,samples):
    raw_accuracies = []
    for k in range(runs):
        policy_hist = policy_gradient([1, 0],iters,0.99,0.01,T,samples)
        raw_accuracies.append(get_accuracies(policy_hist))

    max_length = 0
    for j in range(runs):
        max_length = max(max_length, len(raw_accuracies[j]))

    plot_accuracies = np.zeros((runs, max_length))

    for j in range(runs):
        j_len = len(raw_accuracies[j])
        plot_accuracies[j][:j_len] = raw_accuracies[j]

    pmean = list(map(statistics.mean, zip(*plot_accuracies)))
    pstdv = list(map(statistics.stdev, zip(*plot_accuracies)))
    fig, ax = plt.subplots()
    clrs = sns.color_palette("husl", 2)
    with sns.axes_style("darkgrid"):
        piters = list(range(max_length))
        ax.plot(piters, pmean, c = clrs[1],label= 'Mean L1-accuracy')
        ax.fill_between(piters, np.subtract(pmean,pstdv), np.add(pmean,pstdv), alpha=0.3, facecolor=clrs[1],label="1-standard deviation")
        ax.legend()
        plt.grid(linewidth=0.2)
        plt.xlabel('Iterations')
        plt.ylabel('L1-accuracy')
        plt.title('Policy Gradient: {} Runs'.format(runs))
    plt.show()
    fig.savefig('experiment_{}{}{}{}.png'.format(runs,iters,T,samples))
    return fig

fig = full_experiment(10,200,30,10)

myp_end = process_time()
elapsed_time = myp_end - myp_start
print(elapsed_time)
