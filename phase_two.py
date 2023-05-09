import gymnasium as gym
import numpy as np
import math
from gymnasium.envs.toy_text.frozen_lake import generate_random_map

# create Environment
env = gym.make("FrozenLake-v1", desc=generate_random_map(size=16), is_slippery=False)
goal_reward = 5.0

def epsilon_greedy_policy(state,epsilon):
    if np.random.uniform() < epsilon:
        action = env.action_space.sample()
    else:
        action = np.argmax(Q[state, :])
    return action

def save_q_table(q_table):
    map_size = int(math.sqrt(env.observation_space.n))
    terminal_states = set()
    goal_state = (map_size - 1) * map_size + (map_size - 1)

    for state in env.P:
        if state == goal_state:
            continue
        for act in env.P[state]:
            for probability, nextState, reward, isTerminalState in env.P[state][act]:
                if (reward == 0) and isTerminalState:
                    terminal_states.add(nextState)

    with open('q_table.txt', 'w', encoding="utf-8") as inp:
        for state in range(map_size**2):
            if state in terminal_states:
                inp.write(u'☠️\t')
            elif state == goal_state:
                inp.write(u'🪙\t')
            
            else:
                if np.all(q_table[state] == 0):
                    inp.write(u'⬜️\t')
                else:
                    argm = np.argmax(q_table[state])
                    if argm == 0:
                        inp.write(u'←\t')
                    elif argm == 1:
                        inp.write(u'↓\t')
                    elif argm == 2:
                        inp.write(u'→\t')
                    elif argm == 3:
                        inp.write(u'↑\t')
            if (state + 1) % map_size == 0:
                inp.write('\n')

# initialize variables
observation, info = env.reset(seed=42)
max_iter_number = 1000
epsilon = 0.08
alpha = 0.1
gamma = 0.9
Q = np.zeros([env.observation_space.n, env.action_space.n])

# main loop
for episode in range(max_iter_number):
    state = env.reset()
    state = state[0]
    action = epsilon_greedy_policy(state, epsilon)
    done = False
    while not done:
        # next_state, reward, done, info = env.step(action)
        next_observation, reward, terminated, truncated, info = env.step(action)
        if terminated == True and reward!= 1.0:
            reward = -1.0
        elif reward!= 1.0:  reward =-0.001
        else : 
            reward = goal_reward
        # next_action = epsilon_greedy_policy(next_observation, epsilon)
        next_action = np.argmax(Q[next_observation, :])
        Q[state, action] += alpha * (reward + gamma * Q[next_observation, next_action] - Q[state, action])
        state = next_observation
        action = next_action
        done = truncated or terminated

    # reset the environment if terminated or truncated
    if terminated or truncated:
        observation, info = env.reset()
    else:
        observation = next_observation

save_q_table(Q)
print("done")
num_episodes = 100
rewards = []
for episode in range(num_episodes):
    state = env.reset()
    state = state[0]
    done = False
    episode_reward = 0
    while not done:
        action = np.argmax(Q[state, :])
        next_observation, reward, terminated, truncated, info = env.step(action)
        if terminated == True and reward!= 1.0:
            reward = -1.0
        elif reward!= 1.0:  reward =-0.001
        else : 
            reward = goal_reward
        episode_reward += reward
        done = truncated or terminated
    rewards.append(episode_reward)

# print("Average reward over {} episodes: {}".format(num_episodes, np.mean(rewards)))
env.close()