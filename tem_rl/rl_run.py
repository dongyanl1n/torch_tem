import random
import os

import matplotlib

from rl_model import *
from rl_world import *
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from datetime import datetime
import argparse
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


def random_policy(env, num_envs, num_episodes_per_env):

    for i_block in tqdm(range(num_envs)):
        env.env_reset()
        print(f'Env {i_block}, Goal location {env.goal_location}')  # TODO: write this into logger file
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
    plt.plot(np.arange(num_envs*num_episodes_per_env), bin_rewards(np.array(rewards), window_size=1000))
    plt.vlines(x=np.arange(start=num_episodes_per_env, stop=num_envs*num_episodes_per_env, step=num_episodes_per_env),
               ymin=min(bin_rewards(np.array(rewards), window_size=1000))-5,
               ymax=max(bin_rewards(np.array(rewards), window_size=1000))+5, linestyles='dotted')
    plt.title('Random Policy')
    plt.savefig('random_policy.svg', format='svg')


def train_neural_net(env, agent, num_envs, num_episodes_per_env, lr, n_rollout):
    optimizer = torch.optim.Adam(agent.parameters(), lr=lr)
    rewards = np.zeros(num_envs*num_episodes_per_env, dtype=np.float16)

    for i_block in tqdm(range(num_envs)):
        env.env_reset()
        print(f'Env {i_block}, Goal location {env.goal_location}')  # TODO: write this into logger file
        for i_episode in tqdm(range(num_episodes_per_env)):
            done = False
            env.trial_reset()
            if not isinstance(agent, AC_MLP):
                agent.reinit_hid()
            while not done:
                pol, val = agent.forward(torch.unsqueeze(torch.unsqueeze(torch.as_tensor(env.observation), dim=0), dim=1).float())
                act, p, v = select_action(agent, pol, val)
                new_obs, reward, done, info = env.step(act)
                agent.rewards.append(reward)
            # print(f"This episode has {len(agent.rewards)} steps")
            rewards[i_block*num_episodes_per_env + i_episode] = sum(agent.rewards)
            if len(agent.rewards) <= n_rollout:
                p_loss, v_loss = finish_trial(agent, 0.99, optimizer)
            else:
                p_losses, v_losses = finish_trial_truncated_BPTT(agent, 0.99, optimizer, n_rollout)

    return rewards


def plot_results(num_envs, num_episodes_per_env, rewards, title):

    plt.figure()
    plt.plot(np.arange(num_envs*num_episodes_per_env), bin_rewards(np.array(rewards), window_size=1000))
    plt.vlines(x=np.arange(start=num_episodes_per_env, stop=num_envs*num_episodes_per_env, step=num_episodes_per_env),
               ymin=min(bin_rewards(np.array(rewards), window_size=1000))-5,
               ymax=max(bin_rewards(np.array(rewards), window_size=1000))+5, linestyles='dotted')
    plt.title(title)
    plt.savefig(f'{title}.svg', format='svg')


parser = argparse.ArgumentParser(description="Run neural networks on tem-rl")
parser.add_argument("--num_envs",type=int,default=10,help='Number of environments with different object-location pairings')
parser.add_argument("--num_episodes_per_env",type=int,default=10000,help="Number of episodes to train agent on each environment")
parser.add_argument("--lr",type=float,default=0.0001,help="learning rate")
parser.add_argument("--edge_length",type=int,default=5,help="Length of edge for the environment")
parser.add_argument("--num_objects",type=int,default=40,help="Number of object that could be associated with different locations of the environment")
parser.add_argument("--num_neurons", type=int, default=128, help="Number of units in each hidden layer")
parser.add_argument("--n_rollout", type=int, default=20, help="Number of timestep to unroll when performing truncated BPTT")
args = parser.parse_args()
argsdict = args.__dict__

num_envs = argsdict['num_envs']
num_episodes_per_env = argsdict['num_episodes_per_env']
lr = argsdict['lr']
edge_length = argsdict['edge_length']
num_objects = argsdict['num_objects']
num_neurons = argsdict['num_neurons']
n_rollout = argsdict["n_rollout"]

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu:0')

env = Navigation(edge_length, num_objects)
# baseline_agent = actor_critic_agent(
#     input_dimensions=num_objects,
#     action_dimensions=5,
#     batch_size=1,
#     hidden_types=['linear', 'linear'],
#     hidden_dimensions=[num_neurons, num_neurons]
# ).to(device)
# rnn_agent = actor_critic_agent(
#     input_dimensions=num_objects,
#     action_dimensions=5,
#     batch_size=1,
#     hidden_types=['lstm', 'linear'],
#     hidden_dimensions=[num_neurons, num_neurons]
# ).to(device)

baseline_agent = AC_MLP(
    input_size=num_objects,
    hidden_size=[num_neurons, num_neurons],  # linear, linear
    action_size=5
).to(device)
rnn_agent = AC_RNN(
    input_size=num_objects,
    hidden_size=[num_neurons,num_neurons],  # LSTM, linear
    batch_size=1,
    num_LSTM_layers=1,
    action_size=5
).to(device)


torch.autograd.set_detect_anomaly(True)
rewards = train_neural_net(env, rnn_agent, num_envs, num_episodes_per_env, lr, n_rollout)
plot_results(num_envs, num_episodes_per_env, rewards, 'rnn_agent')