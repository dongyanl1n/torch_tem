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


def bin_rewards(epi_rewards, window_size):
    """
    Average the epi_rewards with a moving window.
    """
    epi_rewards = epi_rewards.astype(np.float32)
    avg_rewards = np.zeros_like(epi_rewards)
    for i_episode in range(1, len(epi_rewards)+1):
        if 1 < i_episode < window_size:
            avg_rewards[i_episode-1] = np.mean(epi_rewards[:i_episode])
        elif window_size <= i_episode <= len(epi_rewards):
            avg_rewards[i_episode-1] = np.mean(epi_rewards[i_episode - window_size: i_episode])
    return avg_rewards


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
num_envs = 10
num_episodes_per_env = 100000
rewards = []

# ===== random policy =====
# for i_block in tqdm(range(num_envs)):
#     env.env_reset()
#     print(f'Env {i_block}, Goal location {env.goal_location}')  # TODO: write this into logger file
#     for i_episode in tqdm(range(num_episodes_per_env)):
#         done = False
#         env.trial_reset()
#         episode_reward = 0
#         while not done:
#             action = env.action_space.sample()
#             observation, reward, done, info = env.step(action)
#             episode_reward += reward
#         rewards.append([episode_reward])

# plt.figure()
# plt.plot(np.arange(num_envs*num_episodes_per_env), bin_rewards(np.array(rewards), window_size=1000))
# plt.vlines(x=np.arange(start=num_episodes_per_env, stop=num_envs*num_episodes_per_env, step=num_episodes_per_env),
#            ymin=min(bin_rewards(np.array(rewards), window_size=1000))-5,
#            ymax=max(bin_rewards(np.array(rewards), window_size=1000))+5, linestyles='dotted')
# plt.title('Random Policy')
# plt.savefig('random_policy.svg', format='svg')

# ======== RL agent ===========
optimizer = torch.optim.Adam(rnn_agent.parameters(), lr=1e-4)

for i_block in tqdm(range(num_envs)):
    env.env_reset()
    print(f'Env {i_block}, Goal location {env.goal_location}')  # TODO: write this into logger file
    for i_episode in tqdm(range(num_episodes_per_env)):
        done = False
        env.trial_reset()
        rnn_agent.reinit_hid()
        # episode_reward = 0
        while not done:
            breakpoint()
            # convert object identity to one-hot vector
            x_t = np.zeros(env.num_objects)
            x_t[env.observation] = 1
            pol, val = rnn_agent.forward(x_t)
            act, p, v = select_action(rnn_agent, pol, val)
            new_obs, reward, done, info = env.step(act)
            rnn_agent.rewards.append(reward)
        rewards.append(sum(rnn_agent.rewards))
        p_loss, v_loss = finish_trial(rnn_agent, 0.99, optimizer)

plt.figure()
plt.plot(np.arange(num_envs*num_episodes_per_env), bin_rewards(np.array(rewards), window_size=1000))
plt.vlines(x=np.arange(start=num_episodes_per_env, stop=num_envs*num_episodes_per_env, step=num_episodes_per_env),
           ymin=min(bin_rewards(np.array(rewards), window_size=1000))-5,
           ymax=max(bin_rewards(np.array(rewards), window_size=1000))+5, linestyles='dotted')
plt.title('RNN agent')
plt.savefig('rnn_agent.svg', format='svg')