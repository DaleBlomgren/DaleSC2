import math
import random 

import numpy as np 
import pandas as pd 

from pysc2.agents import base_agent
from pysc2.lib import actions
from pysc2.lib import features

_NO_OP = actions.FUNCTIONS.no_op.id
_SELECT_POINT = actions.FUNCTIONS.select_point.id
_BUILD_SUPPLY_DEPOT = actions.FUNCTIONS.Build_SupplyDepot_screen.id
_BUILD_BARRACKS = actions.FUNCTIONS.Build_Barracks_screen.id
_TRAIN_MARINE = actions.FUNCTIONS.Train_Marine_quick.id
_TRAIN_SCV = actions.FUNCTIONS.Train_SCV_quick.id
_SELECT_ARMY = actions.FUNCTIONS.select_army.id
_SELECT_IDLE_SCV = actions.FUNCTIONS.select_idle_worker.id
_ATTACK_MINIMAP = actions.FUNCTIONS.Attack_minimap.id

_PLAYER_RELATIVE = features.SCREEN_FEATURES.player_relative.index
_UNIT_TYPE = features.SCREEN_FEATURES.unit_type.index
_PLAYER_ID = features.SCREEN_FEATURES.player_id.index

_PLAYER_SELF = 1

_TERRAN_COMMANDCENTER = 18
_TERRAN_SCV = 45 
_TERRAN_SUPPLY_DEPOT = 19
_TERRAN_BARRACKS = 21

_NOT_QUEUED = [0]
_QUEUED = [1]

_PLAYER_HOSTILE = 4

# Define actions

ACTION_DO_NOTHING = 'donothing'
ACTION_SELECT_SCV = 'selectscv'
ACTION_BUILD_SUPPLY_DEPOT = 'buildsupplydepot'
ACTION_BUILD_BARRACKS = 'buildbarracks'
ACTION_SELECT_BARRACKS = 'selectbarracks'
ACTION_BUILD_MARINE = 'buildmarine'
ACTION_SELECT_ARMY = 'selectarmy'
ACTION_ATTACK = 'attack'
ACTION_SELECT_IDLE_SCV = 'selectidlescv'
ACTION_BUILD_SCV = 'buildscv'


smart_actions = [
	ACTION_DO_NOTHING,
	ACTION_SELECT_SCV,
	ACTION_BUILD_SUPPLY_DEPOT,
	ACTION_BUILD_BARRACKS,
	ACTION_SELECT_BARRACKS,
	ACTION_BUILD_MARINE,
	ACTION_SELECT_ARMY,
	ACTION_SELECT_IDLE_SCV,
	ACTION_BUILD_SCV 
]

for mm_x in range(0, 64):
	for mm_y in range(0,64):
		if (mm_x + 1) % 16 == 0 and (mm_y + 1) % 16 == 0:
			smart_actions.append(ACTION_ATTACK + '_' + str(mm_x - 8) + '_' + str(mm_y - 8))



KILL_UNIT_REWARD = 0.2
KILL_BUILDING_REWARD = 0.5
BUILDING_DEATH_REWARD = -0.5





class AttackAgent(base_agent.BaseAgent):
	def __init__(self):
		super(AttackAgent, self).__init__()

		self.qlearn = QLearningTable(actions=list(range(len(smart_actions))))

		self.previous_killed_unit_score = 0
		self.previous_killed_building_score = 0

		self.previous_action = None
		self.previous_state = None
		self.barracks_built = 0

	# def fillActionArray(self):
		### REMEMBER TO EMPTY LIST EVERY STEP WITH THIS METHOD ###
		# If we are about to be supply blocked, add 'build supply depot' to list
		# If we have idle scv's, build scv's
		# If buildings are idle, build units
		# If we dont have gas, build refinery
		# If we have lots of minerals, build command center
		# If we have a command center, build a second barracks
		# If we have 2 barracks, build a factory

	#	smart_actions.append(ACTION_DO_NOTHING)
		#smart_actions.append(AC)

		# if food supply is within range of food cap

	#	if (obs.observation['player'][3] - 3) > (obs.observation['player'][4]):
	#		smart_actions.append(ACTION_BUILD_SUPPLY_DEPOT)

	#	if self.barracks_built < 1:
	#		smart_actions.append(ACTION_BUILD_BARRACKS)

	def transformDistance(self, x, x_distance, y, y_distance):
		if not self.base_top_left:
			return [x - x_distance, y - y_distance]
		
		return [x + x_distance, y + y_distance]
	
	def transformLocation(self, x, y):
		if not self.base_top_left:
			return [64 - x, 64 - y]
		
		return [x, y]

	def step(self, obs):
		super(AttackAgent, self).step(obs)

		player_y, player_x = (obs.observation['feature_minimap'][_PLAYER_RELATIVE] == _PLAYER_SELF).nonzero()
		self.base_top_left = 1 if player_y.any() and player_y.mean() <= 31 else 0

		unit_type = obs.observation['feature_screen'][_UNIT_TYPE]

		depot_y, depot_x = (unit_type == _TERRAN_SUPPLY_DEPOT).nonzero()
		supply_depot_count = 1 if depot_y.any() else 0

		barracks_y, barracks_x = (unit_type == _TERRAN_BARRACKS).nonzero()
		barracks_count = 1 if barracks_y.any() else 0

		supply_limit = obs.observation['player'][4]
		army_supply = obs.observation['player'][5]

		killed_unit_score = obs.observation['score_cumulative'][5]
		killed_building_score = obs.observation['score_cumulative'][6]
##This area will append the wrong amount of actions and attack routes, account for this!
		current_state = np.zeros(20)
		current_state[0] = supply_depot_count
		current_state[1] = barracks_count
		current_state[2] = supply_limit
		current_state[3] = army_supply

		hot_squares = np.zeros(16)
		enemy_y, enemy_x = (obs.observation['feature_minimap'][_PLAYER_RELATIVE] == _PLAYER_HOSTILE).nonzero()
		for i in range(0, len(enemy_y)):
			y = int(math.ceil((enemy_y[i] + 1) / 16))
			x = int(math.ceil((enemy_x[i] + 1) / 16))

			hot_squares[((y - 1) * 4) + (x - 1)] = 1

		if not self.base_top_left:
			hot_squares = hot_squares[::-1]

		for i in range(0, 16):
			current_state[i + 4] = hot_squares[i]

		if self.previous_action is not None:
			reward = 0

			if killed_unit_score > self.previous_killed_unit_score:
				reward += KILL_UNIT_REWARD

			if killed_building_score > self.previous_killed_building_score:
				reward += KILL_BUILDING_REWARD

			self.qlearn.learn(str(self.previous_state), self.previous_action, reward, str(current_state))

#		self.fillActionArray()

# Use Q table to choose action
		rl_action = self.qlearn.choose_action(str(current_state))
		smart_action = smart_actions[rl_action]

		self.previous_killed_unit_score = killed_unit_score
		self.previous_killed_building_score = killed_building_score
		self.previous_state = current_state
		self.previous_action = rl_action

		x = 0
		y = 0
		if '_' in smart_action:
			smart_action, x, y = smart_action.split('_')

		if smart_action == ACTION_DO_NOTHING:
			return actions.FunctionCall(_NO_OP, [])

		elif smart_action == ACTION_SELECT_SCV:
			unit_type = obs.observation['feature_screen'][_UNIT_TYPE]
			unit_y, unit_x = (unit_type == _TERRAN_SCV).nonzero()

			if unit_y.any():
				i = random.randint(0, len(unit_y) - 1)
				target = [unit_x[i], unit_y[i]]

				return actions.FunctionCall(_SELECT_POINT, [_NOT_QUEUED, target])
#	Needs supply depot to be built in a randomish position
		elif smart_action == ACTION_BUILD_SUPPLY_DEPOT:
			if _BUILD_SUPPLY_DEPOT in obs.observation['available_actions']:
				unit_type = obs.observation['feature_screen'][_UNIT_TYPE]
				unit_y, unit_x = (unit_type == _TERRAN_COMMANDCENTER).nonzero()

				if unit_y.any():
					target = self.transformDistance(int(unit_x.mean()), 0, int(unit_y.mean()), 20)

					return actions.FunctionCall(_BUILD_SUPPLY_DEPOT, [_NOT_QUEUED, target])

		elif smart_action == ACTION_BUILD_SCV:
			unit_type = obs.observation['feature_screen'][_UNIT_TYPE]
			unit_y, unit_x = (unit_type == _TERRAN_COMMANDCENTER).nonzero()
			target = [unit_x[0], unit_y[0]]
			actions.FunctionCall(_SELECT_POINT, [_NOT_QUEUED, target])

			if _TRAIN_SCV in obs.observation['available_actions']:
				return actions.FunctionCall(_TRAIN_SCV, [_QUEUED])

#			if unit_y.any():
#				target = self.transformDistance(int(unit_x.mean()), 0, int(unit_y.mean()), 20)

		elif smart_action == ACTION_BUILD_BARRACKS:
			if _BUILD_BARRACKS in obs.observation['available_actions']:
				unit_type = obs.observation['feature_screen'][_UNIT_TYPE]
				unit_y, unit_x = (unit_type == _TERRAN_COMMANDCENTER).nonzero()
				
				if unit_y.any():
					target = self.transformDistance(int(unit_x.mean()), 20, int(unit_y.mean()), 0)
					self.barracks_built = True
					return actions.FunctionCall(_BUILD_BARRACKS, [_NOT_QUEUED, target])
	
		elif smart_action == ACTION_SELECT_BARRACKS:
			unit_type = obs.observation['feature_screen'][_UNIT_TYPE]
			unit_y, unit_x = (unit_type == _TERRAN_BARRACKS).nonzero()
				
			if unit_y.any():
				target = [int(unit_x.mean()), int(unit_y.mean())]
		
				return actions.FunctionCall(_SELECT_POINT, [_NOT_QUEUED, target])
		

		elif smart_action == ACTION_BUILD_MARINE:
			unit_type = obs.observation['feature_screen'][_UNIT_TYPE]
			unit_y, unit_x = (unit_type == _TERRAN_BARRACKS).nonzero()
				
			if unit_y.any():
				target = [int(unit_x.mean()), int(unit_y.mean())]
		
				actions.FunctionCall(_SELECT_POINT, [_NOT_QUEUED, target])

			if _TRAIN_MARINE in obs.observation['available_actions']:
				return actions.FunctionCall(_TRAIN_MARINE, [_QUEUED])
		
		elif smart_action == ACTION_SELECT_ARMY:
			if _SELECT_ARMY in obs.observation['available_actions']:
				return actions.FunctionCall(_SELECT_ARMY, [_NOT_QUEUED])
		
		elif smart_action == ACTION_ATTACK:
			if obs.observation['single_select'][0][0] != _TERRAN_SCV and _ATTACK_MINIMAP in obs.observation['available_actions']:
				return actions.FunctionCall(_ATTACK_MINIMAP, [_NOT_QUEUED, self.transformLocation(int(x), int(y))])

		elif smart_action == ACTION_SELECT_IDLE_SCV:
			if _SELECT_IDLE_SCV in obs.observation['available_actions']:
				actions.FunctionCall(_SELECT_IDLE_SCV, [_NOT_QUEUED])


		return actions.FunctionCall(_NO_OP, [])

class QLearningTable:
	def __init__(self, actions, learning_rate=0.01, reward_decay=0.9, e_greedy=0.9):
		self.actions = actions
		self.lr = learning_rate
		self.gamma = reward_decay
		self.epsilon = e_greedy
		self.q_table = pd.DataFrame(columns=self.actions, dtype=np.float64)

	def choose_action(self, observation):
		self.check_state_exist(observation)
		
		if np.random.uniform() < self.epsilon:
			# choose best action
			state_action = self.q_table.ix[observation, :]
			
			# some actions have the same value
			state_action = state_action.reindex(np.random.permutation(state_action.index))
			
			action = state_action.idxmax()
		else:
			# choose random action
			action = np.random.choice(self.actions)
			
		return action

	def learn(self, s, a, r, s_):
		self.check_state_exist(s_)
		self.check_state_exist(s)
		
		q_predict = self.q_table.ix[s, a]
		q_target = r + self.gamma * self.q_table.ix[s_, :].max()
		
		# update
		self.q_table.ix[s, a] += self.lr * (q_target - q_predict)

	def check_state_exist(self, state):
		if state not in self.q_table.index:
			# append new state to q table
			self.q_table = self.q_table.append(pd.Series([0] * len(self.actions), index=self.q_table.columns, name=state))



