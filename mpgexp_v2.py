import numpy as np

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

def get_reward(state, act1, act2):
	if state == 0:
		if act1 == 0 and act2 == 0:
			return [5,2]
		elif act1 == 0 and act2 == 1:
			return [-1,-2]
		elif act1 == 1 and act2 == 0:
			return [-5,-4]
		else:
			return [1,4]

	elif state == 1:
		if act1 == 0 and act2 == 0:
			return [32,29]
		elif act1 == 0 and act2 == 1:
			return [-28,-29]
		elif act1 == 1 and act2 == 0:
			return [-32,-31]
		else:
			return [28,31]
	else:
		return [0,0]

def get_next_state(state, a_act, b_act):
	if a_act != b_act:
		if state == 0:
			return 1
		return 0
	return state

def pick_action(prob_dist):
	acts = [i for i in range(len(prob_dist))]
	action = np.random.choice(acts, 1, p = prob_dist)
	return action

def visit_dist(state, policy, gamma, T):
    curr_state = state
    dist = {curr_state: 1, 1-curr_state:0}
    for t in range(1,T):
        action1 = pick_action(policy[curr_state,0])
        action2 = pick_action(policy[curr_state,1])
        curr_state = get_next_state(curr_state, action1, action2)
        dist[curr_state] += gamma**t
    #if gamma != 1:
    #    for state in dist:
    #        dist[state] /=(1-gamma**(T))/(1-gamma)
    return(dist)

   # *** 
   # This is the normalizing factor to make it into a distribution (a proxy of (1-gamma)).
   # But since, in the gradient step, we divide with this factor (1/1-gamma), we leave it out.
   # *** 
   #print(dist)
   #return dist

def Q_function1(state, action, policy, gamma, T):
	curr_state = state
	agent1totalreward = 0
	action1 = action
	action2 = pick_action(policy[curr_state,1])
	reward = get_reward(curr_state, action1, action2)
	agent1totalreward += reward[0]
	curr_state = get_next_state(curr_state, action1, action2)
	for t in range(1,T):
		action1 = pick_action(policy[curr_state,0])
		action2 = pick_action(policy[curr_state,1])
		reward = get_reward(curr_state, action1, action2)
		curr_state = get_next_state(curr_state, action1, action2)
		agent1totalreward += (gamma ** t) * reward[0]

	return agent1totalreward

def Q_function2(state, action, policy, gamma, T):
	curr_state = state
	agent2totalreward = 0
	action2 = action
	action1 = pick_action(policy[curr_state,0])
	reward = get_reward(curr_state, action1, action2)
	agent2totalreward += reward[1]
	curr_state = get_next_state(curr_state, action1, action2)
	for t in range(1,T):
		action1 = pick_action(policy[curr_state,0])
		action2 = pick_action(policy[curr_state,1])
		reward = get_reward(curr_state, action1, action2)
		curr_state = get_next_state(curr_state, action1, action2)

		agent2totalreward += (gamma ** t) * reward[1]

	return agent2totalreward

def policy_gradient(mu, max_iters, gamma, eta, T):
	joint_policy = {}
	for i in range(2):
			joint_policy[i, 0] = [.5, .5]
			joint_policy[i, 1] = [.5, .5]
	dist = {0:mu[0], 1:mu[1]}
            
	for i in range(max_iters):
		print(i)
		dist0 = visit_dist(0, joint_policy, gamma, T)
		dist1 = visit_dist(1, joint_policy, gamma, T)
		#print(dist0)		
		#print(dist1)		
		dist[0] = mu[0]*dist0[0]+mu[1]*dist1[0]
		dist[1] = mu[0]*dist0[1]+mu[1]*dist1[1]
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


print(policy_gradient([1, 0],60,0.99,0.001,30))
