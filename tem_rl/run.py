import random
import os

import matplotlib

from model import *
from world import *
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from datetime import datetime
matplotlib.use('Agg')

env = Navigation(edge_length=5, num_objects=40)
baseline_agent = actor_critic_agent(
    input_dimensions=25,
    action_dimensions=5,
    batch_size=1,
    hidden_types=['linear', 'linear'],
    hidden_dimensions=[128, 128]
)
rnn_agent = actor_critic_agent(
    input_dimensions=25,
    action_dimensions=5,
    batch_size=1,
    hidden_types=['lstm', 'linear'],
    hidden_dimensions=[128, 128]
)
# breakpoint()
num_envs = 20
num_episodes_per_env = 1000
rewards = []

# ===== random policy =====
for i_block in tqdm(range(num_envs)):
    env.env_reset()
    for i_episode in tqdm(range(num_episodes_per_env)):
        done = False
        env.trial_reset()
        episode_reward = 0
        while not done:
            action = env.action_space.sample()
            observation, reward, done, info = env.step(action)
            episode_reward += reward
        rewards.append([episode_reward])

plt.figure()
plt.plot(np.arange(num_envs*num_episodes_per_env), np.array(rewards))
plt.title('Random Policy')
plt.savefig('Random Policy.svg', format='svg')
