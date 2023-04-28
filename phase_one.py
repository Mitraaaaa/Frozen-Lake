#import library

import gymnasium as gym
import numpy as np
from math import sqrt
from gymnasium.envs.toy_text.frozen_lake import generate_random_map

#create Enviroment

env = gym.make("FrozenLake-v1", desc=generate_random_map(size=4), render_mode="human", is_slippery=True)

observation, info = env.reset(seed=42)
state_1 = env.reset()[0]
max_iter_number = 1000
holes = []
P_1 = {}
for key1, value1 in env.P.items():
    dict_of_list_of_list = {}
    for key2, value2 in value1.items():
        list_of_list = [list(t) for t in value2]
        dict_of_list_of_list[key2] = list_of_list
    P_1[key1] = dict_of_list_of_list

# print(P)
def initialize_prop():
    for state in range(env.observation_space.n):
            for action in range(env.action_space.n):
                for next_sr_idx in range(len(P_1[state][action])):
                    if P_1[state][action][next_sr_idx][3] == False and P_1[state][action][next_sr_idx][2] == 0:
                        P_1[state][action][next_sr_idx][2] = -0.001

def value_iteration(env, gamma):
    
    value_table = np.zeros(env.observation_space.n)  
    no_of_iterations = 100
    for i in range(no_of_iterations):
        for state in range(env.observation_space.n):
            Q_value = []
            for action in range(env.action_space.n):
                next_states_rewards = []
                for next_sr in P_1[state][action]: 
                    trans_prob, next_state, reward_prob, _ = next_sr 
                    next_states_rewards.append((trans_prob * (reward_prob + gamma * value_table[next_state]))) 
    
                Q_value.append(np.sum(next_states_rewards))
                
            value_table[state] = max(Q_value) 
    
    return value_table

def extract_policy(value_table, gamma):
    policy = np.zeros(env.observation_space.n) 
    for state in range(env.observation_space.n):
        Q_table = np.zeros(env.action_space.n)
        
        for action in range(env.action_space.n):
            for next_sr in P_1[state][action]: 
                trans_prob, next_state, reward_prob, _ = next_sr 
                Q_table[action] += (trans_prob * (reward_prob + gamma * value_table[next_state]))
        
        policy[state] = np.argmax(Q_table)
    return policy

def set_hole_value(new_hole_state : int):
    if len(holes)!=0 and new_hole_state in holes:
        return
    else :
        for state in range(env.observation_space.n):
            if state == new_hole_state - 1 or state == new_hole_state + 1 \
                  or state == new_hole_state + sqrt(env.observation_space.n) or state == new_hole_state - sqrt(env.observation_space.n) :
                for action in range(env.action_space.n):
                    for next_sr_idx in range(len(P_1[state][action])): 
                        _, next_state, reward , _ = P_1[state][action][next_sr_idx]
                        if reward == 1 : 
                            return
                        if next_state == new_hole_state:
                            P_1[state][action][next_sr_idx][2] = -1
        holes.append(new_hole_state)
                    
initialize_prop()
for _ in range(max_iter_number):

   ##################################
   # # TODO # #
   optimal_value_function = value_iteration(env=env,gamma=0.7)
   optimal_policy = extract_policy(optimal_value_function, gamma=0.6)
   action = int(optimal_policy[observation])

   observation, reward, terminated, truncated, info = env.step(action)

   state_1 = observation
   if truncated:
      observation, info = env.reset()

   if terminated : 
      observation, info = env.reset()
      set_hole_value(state_1)

env.close()