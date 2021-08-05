import matplotlib.pyplot as plt
import itertools as it
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

class FishGame:
    #inputs: num players, num of fish,
    def __init__(self, N, F):
        self.N = N
        self.F = F
        self.num_actions = F+1
        if F > 0:
            self.actions = [i/F for i in range(self.F+1)]
        else:
            self.actions = [0]
            
def get_reward(fish_game, actions):
    rewards = np.log(np.dot(actions,fish_game.F))
    for i in range(len(rewards)):
        if rewards[i] < 0:
            rewards[i] = -10000
    return rewards

# Define the states and some necessary info
N = 3 #number of agents/countries
F = 10 #number of fish
S = F
M = F
alpha = 0.5 #alpha paramater for state transition

state_dic = {i:FishGame(N,i) for i in range(0,F+1)}

# Dictionary to store the action profiles and rewards to
selected_profiles = {}

def get_next_state(state, actions):
    acts_from_ints = actions
    new_state = max(0,np.ceil((state - np.dot(acts_from_ints, N*[state])) ** alpha))
    return new_state

def pick_action(prob_dist,N):
    # np.random.choice(range(len(prob_dist)), 1, p = prob_dist)[0]
    a = FishGame(N,len(prob_dist)-1)
    acts = a.actions
    action = np.random.choice(acts, 1, p = prob_dist)
    return action[0]

def value_function(policy,gamma,T,samples):
    value_fun = {(s,i): 0 for s in range(S) for i in range(N)}
    for k in range(samples):
        for state in range(S):
            curr_state = state
            for t in range(T):
                actions = [pick_action(policy[curr_state, i],N) for i in range(N)]
                q = tuple(actions+[curr_state])
                rewards = selected_profiles.setdefault(q,get_reward(state_dic[curr_state], actions))  
                for i in range(N):
                    value_fun[state,i] += (gamma**t)*rewards[i]
                curr_state = get_next_state(curr_state, actions)
    value_fun.update((x,v/samples) for (x,v) in value_fun.items())
    return value_fun

def Q_function(agent, state, action, policy, gamma, value_fun, samples):
    tot_reward = 0
    for i in range(samples):
        actions = [pick_action(policy[state, i],N) for i in range(N)]
        actions[agent] = action
        q = tuple(actions+[state])
        rewards = selected_profiles.setdefault(q,get_reward(state_dic[state], actions))
        tot_reward += rewards[agent] + gamma*value_fun[get_next_state(state, actions), agent]
    return (tot_reward / samples)

def policy_accuracy(policy_pi, policy_star):
    total_dif = N * [0]
    for agent in range(N):
        for state in range(S):
            total_dif[agent] += np.sum(np.abs((policy_pi[state, agent] - policy_star[state, agent])))
	  # total_dif[agent] += np.sqrt(np.sum((policy_pi[state, agent] - policy_star[state, agent])**2))
    return np.sum(total_dif) / N

def policy_gradient(mu, max_iters, gamma, eta, T, samples):

    policy = {(s,i): [1/(S)]*S for i in range(N) for s in range(S)}
    policy_hist = [copy.deepcopy(policy)]

    for t in range(max_iters):

        #print(t)
            
        grads = np.zeros((N, S, M))
        value_fun = value_function(policy, gamma, T, samples)

        for agent in range(N):
            for st in range(S):
                for act in range(M):
                    grads[agent, st, act] = Q_function(agent, st, act, policy, gamma, value_fun, samples)

        for agent in range(N):
            for st in range(S):
                print(np.add(policy[st, agent], eta * grads[agent,st]))                
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

def full_experiment(runs,iters,eta,T,samples):


    raw_accuracies = []
    for k in range(runs):

        policy_hist = policy_gradient(F * [1/F],iters,0.99,eta,T,samples)
        raw_accuracies.append(get_accuracies(policy_hist))

        converged_policy = policy_hist[-1]
 

    #densities = densities / runs

    # max_length = 0
    # for j in range(runs):
    #     max_length = max(max_length, len(raw_accuracies[j]))

    # plot_accuracies = np.zeros((runs, max_length))

    # for j in range(runs):
    #     j_len = len(raw_accuracies[j])
    #     plot_accuracies[j][:j_len] = raw_accuracies[j]
    
    plot_accuracies = np.array(list(it.zip_longest(*raw_accuracies, fillvalue=np.nan))).T
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

    plt.gca().set(xlabel='Facility',ylabel='Average number of agents', title='Policy Gradient: agents = {}, runs = {}, $\eta$ = {}'.format(N,runs,eta))
    plt.xticks(index + bar_width/2, ('A', 'B', 'C', 'D'))
    plt.legend()
    fig3.savefig('facilities_n{}.png'.format(N),bbox_inches='tight')
   #plt.close()
    plt.show()

    return fig1, fig2, fig3

#full_experiment(10,1000,0.0001,20,10)
full_experiment(10,10,0.0001,2,2)

myp_end = process_time()
elapsed_time = myp_end - myp_start
print(elapsed_time)
