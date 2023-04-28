#import library

import gymnasium as gym
import numpy as np
from math import sqrt
from gymnasium.envs.toy_text.frozen_lake import generate_random_map

#create Enviroment

env = gym.make("FrozenLake-v1", desc=generate_random_map(size=4), render_mode="human", is_slippery=True)

observation, info = env.reset(seed=42)
state = env.reset()[0]
# print(state)

max_iter_number = 1000
# print(env.P)
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
                        P_1[state][action][next_sr_idx][2] = -0.01

def value_iteration(env, gamma = 1.0):
    
    value_table = np.zeros(env.observation_space.n)
    
    no_of_iterations = 100
    threshold = 1e-20
    
    for i in range(no_of_iterations):
        updated_value_table = np.copy(value_table) 
        for state in range(env.observation_space.n):
            Q_value = []
            for action in range(env.action_space.n):
                next_states_rewards = []
                for next_sr in P_1[state][action]: 
                    # print(next_sr)
                    trans_prob, next_state, reward_prob, _ = next_sr 
                    next_states_rewards.append((trans_prob * (reward_prob + gamma * updated_value_table[next_state]))) 
                
                Q_value.append(np.sum(next_states_rewards))
                
            value_table[state] = max(Q_value) 
    
        if (np.sum(np.fabs(updated_value_table - value_table)) <= threshold):
             break
    
    return value_table

def extract_policy(value_table, gamma = 1.0):

    policy = np.zeros(env.observation_space.n) 
    for state in range(env.observation_space.n):
        Q_table = np.zeros(env.action_space.n)
        
        for action in range(env.action_space.n):
            for next_sr in P_1[state][action]: 
                trans_prob, next_state, reward_prob, _ = next_sr 
                Q_table[action] += (trans_prob * (reward_prob + gamma * value_table[next_state]))
        
        policy[state] = np.argmax(Q_table)
    return policy

def is_in_range(x,y):
    if x>=0 and x<env.observation_space.n and y>=0 and y < env.observation_space.n:
        return True
    else : return False

def set_hole_value(new_hole_state : int):
    if len(holes)!=0 and new_hole_state in holes:
        return
    else :
        holes.append(new_hole_state)
        for state in range(env.observation_space.n):
            if state == new_hole_state - 1 or state == new_hole_state + 1 \
                  or state == new_hole_state + sqrt(env.observation_space.n) or state == new_hole_state - sqrt(env.observation_space.n) :
                for action in range(env.action_space.n):
                    for next_sr_idx in range(len(P_1[state][action])): 
                        _, next_state, reward , _ = P_1[state][action][next_sr_idx]
                        if reward == 1 : 
                            return
                        if next_state == new_hole_state:
                            print(new_hole_state)
                            print(P_1[state][action][next_sr_idx])
                            P_1[state][action][next_sr_idx][2] = -1 
                            print(P_1[state][action][next_sr_idx])
                    
initialize_prop()
for _ in range(max_iter_number):

   ##################################

   # # TODO # #
   optimal_value_function = value_iteration(env=env,gamma=1.0)

   optimal_policy = extract_policy(optimal_value_function, gamma=1.0)
   action = int(optimal_policy[observation])
   env.step(action)

   observation, reward, terminated, truncated, info = env.step(action)
   if truncated:
      observation, info = env.reset()

   if terminated : 
      observation, info = env.reset()
      set_hole_value(state)
    #   print(P_1[state][action])
    #   P[state][action][2]= -1
    #   print("=================")

   state = observation
      
env.close()