import itertools as it
import numpy as np
from math import comb

class CongGame:
	#inputs: num players, max facilities per player, list of linear multiplier on utility for num of players
	def __init__(self, n, d, weights):
		self.n = n
		self.d = d
		self.weights = weights
		self.m = len(weights) #number of facilities
		self.num_actions = comb(self.n + self.d-1, self.d)
		self.facilities = [i for i in range(self.m)]
		self.actions = list(it.combinations_with_replacement(self.facilities,d))

	def get_counts(self, actions):
		count = dict.fromkeys(range(self.m),0)
		#print(actions)
		for action in actions:
			#print(action)
			for facility in action:
				#print(facility)
				#print(np.shape(facility))
				count[facility] += 1
		#print(list(count.values()))
		return list(count.values())

	def get_facilitiy_rewards(self, actions):
		density = self.get_counts(actions)
		#print(density)
		facility_rewards = self.m * [0]
		for j in range(self.m):
			#print(facility_rewards)
			facility_rewards[j] = density[j] * self.weights[j]
		return facility_rewards

def get_agent_reward(cong_game, actions, agent_action):
	agent_reward = 0
	facility_rewards = cong_game.get_facilitiy_rewards(actions)
	for facility in agent_action:
		agent_reward += facility_rewards[facility]
	return agent_reward

def get_reward(cong_game, actions):
	rewards = cong_game.n * [0]
	for i in range(cong_game.n):
		rewards[i] = get_agent_reward(cong_game, actions, actions[i])
	return rewards


A = CongGame(4,2,[1,2,3,4])

acts = [(0,1), (2,3), (1,3), (1,2)]

#print(A.actions)
# gives correct result of [7, 14, 14, 12] as can be verified by hand


