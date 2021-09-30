from congestion_games import *
import matplotlib.pyplot as plt
import itertools
import numpy as np
import copy
import statistics
import seaborn as sns; sns.set()
from time import process_time
from collections import defaultdict

myp_start = process_time()
selected_profiles = {}

NUM_LAYERS = 2
NUM_UNITS = 2
NUM_AGENTS = 8
S = NUM_UNITS**NUM_AGENTS
N = NUM_AGENTS
M = NUM_UNITS

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

def get_reward(actions):
    rewards = []
    for i in range(NUM_AGENTS):
        rewards.append(1/actions.count(actions[i]))
    return rewards

def num_to_state(num, num_agents, num_units):
    nums = [0]*num_agents
    for i in range(num_agents):
        num, r = divmod(num, num_units)
        nums[i] = r
    return nums

# make each state into a distinct number
def state_to_num(state, num_agents, num_units):
    num = 0
    for i in range(num_agents):
        num += state[i]*num_units**i
    return num

def pick_action(prob_dist):
    # np.random.choice(range(len(prob_dist)), 1, p = prob_dist)[0]
    acts = [i for i in range(len(prob_dist))]
    action = np.random.choice(acts, 1, p = prob_dist)
    return action[0]

def visit_dist(state, policy, gamma, T, samples):
    # This is the unnormalized visitation distribution. Since we take finite trajectories, the normalization constant is (1-gamma**T)/(1-gamma).
    visit_states = {st: np.zeros(T) for st in range(S)}        
    for i in range(samples):
        curr_state = state
        for t in range(T):
            visit_states[curr_state][t] += 1
            actions = [pick_action(policy[curr_state, i]) for i in range(N)]
            curr_state = state_to_num(actions, NUM_AGENTS, NUM_UNITS)
    dist = [np.dot(v/samples,gamma**np.arange(T)) for (k,v) in visit_states.items()]
    return dist 

# For now we ignore layer because all the same
def value_function(policy, gamma, T, samples):
    value_fun = defaultdict(lambda: 0)
    for k in range(samples):
        for layer in range(NUM_LAYERS):
            for state_num in range(NUM_UNITS**NUM_AGENTS):
                for t in range(T - layer):
                    actions = [pick_action(policy[state_num, i]) for i in range(NUM_AGENTS)]
                    q = tuple(actions+[state_num])
                    rewards = selected_profiles.setdefault(q,get_reward(actions))                  
                    for i in range(NUM_AGENTS):
                        value_fun[state_num,i] += (gamma**t)*rewards[i]

                    state_num = state_to_num(actions, NUM_AGENTS, NUM_UNITS) # need to include layers info later

    value_fun.update((x,v/samples) for (x,v) in value_fun.items())
    return value_fun

def Q_function(agent, state, action, policy, gamma, value_fun, samples):
    tot_reward = 0
    for i in range(samples):
        actions = [pick_action(policy[state, i]) for i in range(NUM_AGENTS)]
        actions[agent] = action
        q = tuple(actions+[state])
        rewards = selected_profiles.setdefault(q,get_reward(actions))
        tot_reward += rewards[agent] + gamma*value_fun[state_to_num(actions, NUM_AGENTS, NUM_UNITS), agent]
    return (tot_reward / samples)

def policy_accuracy(policy_pi, policy_star):
    total_dif = NUM_AGENTS * [0]
    for agent in range(NUM_AGENTS):
        for state in range(NUM_UNITS**NUM_AGENTS):
            total_dif[agent] += np.sum(np.abs((policy_pi[state, agent] - policy_star[state, agent])))
	  # total_dif[agent] += np.sqrt(np.sum((policy_pi[state, agent] - policy_star[state, agent])**2))
    return np.sum(total_dif) / N

def policy_gradient(mu, max_iters, gamma, eta, T, samples):

    policy = {(s,i): [1/M]*M for s in range(S) for i in range(N)}
    policy_hist = [copy.deepcopy(policy)]

    for t in range(max_iters):

        print(t)

        b_dist = S * [0]
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
      # if policy_accuracy(policy_hist[t+1], policy_hist[t]) < 10e-16: (it makes a difference, not when t=0 but from t=1 onwards.)
            return policy_hist

    return policy_hist




def get_accuracies(policy_hist):
    fin = policy_hist[-1]
    accuracies = []
    for i in range(len(policy_hist)):
        this_acc = policy_accuracy(policy_hist[i], fin)
        accuracies.append(this_acc)
    return accuracies

def full_experiment(runs, iters, eta, T, samples):

#     densities = np.zeros((NUM_UNITS**NUM_AGENTS,M))

    raw_accuracies = []
    for k in range(runs):
        policy_hist = policy_gradient([1/256]*256,iters,0.99,eta,T,samples)
        raw_accuracies.append(get_accuracies(policy_hist))

        converged_policy = policy_hist[-1]
#         for i in range(N):
#             for s in range(S):
#                 densities[s] += converged_policy[s,i]

#     densities = densities / runs

    print(raw_accuracies)

    #densities = densities / runs

    # max_length = 0
    # for j in range(runs):
    #     max_length = max(max_length, len(raw_accuracies[j]))

    # plot_accuracies = np.zeros((runs, max_length))

    # for j in range(runs):
    #     j_len = len(raw_accuracies[j])
    #     plot_accuracies[j][:j_len] = raw_accuracies[j]
    
    plot_accuracies = np.array(list(itertools.zip_longest(*raw_accuracies, fillvalue=np.nan))).T
    clrs = sns.color_palette("husl", 3)
    piters = list(range(plot_accuracies.shape[1]))

    fig2 = plt.figure(figsize=(6,4))
    for i in range(len(plot_accuracies)):
        plt.plot(piters, plot_accuracies[i])
    plt.grid(linewidth=0.6)
    plt.gca().set(xlabel='Iterations',ylabel='L1-accuracy', title='Policy Gradient: agents = {}, runs = {}, $\eta$ = {}'.format(N, runs,eta))
    plt.show()
    fig2.savefig('individual_runs_n{}.png'.format(N),bbox_inches='tight')
    #plt.close()
    
    plot_accuracies = np.nan_to_num(plot_accuracies)
    pmean = list(map(statistics.mean, zip(*plot_accuracies)))
    pstdv = list(map(statistics.stdev, zip(*plot_accuracies)))
    
    fig1 = plt.figure(figsize=(6,4))
    ax = sns.lineplot(piters, pmean, color = clrs[0],label= 'Mean L1-accuracy')
    ax.fill_between(piters, np.subtract(pmean,pstdv), np.add(pmean,pstdv), alpha=0.3, facecolor=clrs[0],label="1-standard deviation")
    ax.legend()
    plt.grid(linewidth=0.6)
    plt.gca().set(xlabel='Iterations',ylabel='L1-accuracy', title='Policy Gradient: agents = {}, runs = {}, $\eta$ = {}'.format(N, runs,eta))
    plt.show()
    fig1.savefig('avg_runs_n{}.png'.format(N),bbox_inches='tight')
    #plt.close()
    
    #print(densities)

    fig3, ax = plt.subplots()
    index = np.arange(D)
    bar_width = 0.35
    opacity = 1

    #print(len(index))
    #print(len(densities[0]))
#     rects1 = plt.bar(index, densities[0], bar_width,
#     alpha= .7 * opacity,
#     color='b',
#     label='Safe state')

#     rects2 = plt.bar(index + bar_width, densities[1], bar_width,
#     alpha= opacity,
#     color='r',
#     label='Distancing state')

    plt.gca().set(xlabel='Facility',ylabel='Average number of agents', title='Policy Gradient: agents = {}, runs = {}, $\eta$ = {}'.format(N,runs,eta))
    plt.xticks(index + bar_width/2, ('A', 'B', 'C', 'D'))
    plt.legend()
    fig3.savefig('facilities_n{}.png'.format(N),bbox_inches='tight')
   #plt.close()
    plt.show()

    return fig1, fig2, fig3

#full_experiment(10,1000,0.0001,20,10)
full_experiment(10,500,0.0001,20,10)

myp_end = process_time()
elapsed_time = myp_end - myp_start
print(elapsed_time)