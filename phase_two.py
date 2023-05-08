import gymnasium as gym
import numpy as np
from gymnasium.envs.toy_text.frozen_lake import generate_random_map

def epsilon_greedy_policy(state, Q, epsilon):
    if np.random.uniform() < epsilon:
        action = env.action_space.sample()
    else:
        action = np.argmax(Q[state, :])
    return action

# create Environment
env = gym.make("FrozenLake-v1", desc=generate_random_map(size=4), render_mode="human", is_slippery=True)

# initialize variables
observation, info = env.reset(seed=42)
max_iter_number = 1000
epsilon = 0.1
alpha = 0.1
gamma = 0.99
Q = np.zeros([env.observation_space.n, env.action_space.n])

# main loop
for i in range(max_iter_number):
    action = epsilon_greedy_policy(observation, Q, epsilon, env.action_space.n)
    next_observation, reward, terminated, truncated, info = env.step(action)
    if terminated == True and reward!= 1.0:
        reward = -1.0
    elif reward!= 1.0:  reward =-0.001
    else : 
        reward = 3.0

    # SARSA algorithm
    next_action = epsilon_greedy_policy(next_observation, Q, epsilon, env.action_space.n)
    Q[observation, action] += alpha * (reward + gamma * Q[next_observation, next_action] - Q[observation, action])

    # reset the environment if terminated or truncated
    if terminated or truncated:
        observation, info = env.reset()
    else:
        observation = next_observation

env.close()