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
		self.num_actions = sum(comb(self.m,k) for k in range(1,self.d+1))
		self.facilities = [i for i in range(self.m)]
		self.actions = list(it.chain.from_iterable(it.combinations(self.facilities, r) for r in range(1,self.d+1)))

	def get_counts(self, actions):
		count = dict.fromkeys(range(self.m),0)
		for action in actions:
			for facility in action:
				count[facility] += 1
		return list(count.values())

	def get_facility_rewards(self, actions):
		density = self.get_counts(actions)
		facility_rewards = self.m * [0]
		for j in range(self.m):
			facility_rewards[j] = density[j] * self.weights[j][0] + self.weights[j][1]
		return facility_rewards

def get_agent_reward(cong_game, actions, agent_action):
	agent_reward = 0
	facility_rewards = cong_game.get_facility_rewards(actions)
	for facility in agent_action:
		agent_reward += facility_rewards[facility]
	return agent_reward

def get_reward(cong_game, actions):
	rewards = cong_game.n * [0]
	for i in range(cong_game.n):
		rewards[i] = get_agent_reward(cong_game, actions, actions[i])
	return rewards
