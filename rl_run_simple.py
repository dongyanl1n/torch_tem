import matplotlib

import rl_world
from rl_model import *
from rl_world import *
from rl_run import *
import numpy as np
import torch
import matplotlib.pyplot as plt
import gym
from tqdm import tqdm
import argparse
import os
import datetime
matplotlib.use('Agg')


parser = argparse.ArgumentParser(description="Run neural networks on simple navigation environment")
parser.add_argument("--num_episodes",type=int,default=10000,help="Number of episodes to train agent on the environment")
parser.add_argument("--lr",type=float,default=0.0001,help="learning rate")
parser.add_argument("--size",type=int,default=5,help="Length of edge for the environment")
parser.add_argument("--num_neurons", type=int, default=128, help="Number of units in each hidden layer")
parser.add_argument("--window_size", type=int, default=1000, help="Size of rolling window for smoothing out performance plot")
parser.add_argument("--agent_type", type=str, default='rnn', help="type of agent to use. Either 'rnn' or 'mlp'.")
parser.add_argument("--save_dir", type=str, default='experiments/', help="path to save figures.")
parser.add_argument("--save_model_freq", type=int, default=1000, help="Frequency (# of episodes) of saving model checkpoint")
parser.add_argument("--load_existing_agent", type=str, default=None, help="path to existing agent to load")
args = parser.parse_args()
argsdict = args.__dict__

num_episodes = argsdict['num_episodes']
lr = argsdict['lr']
size = argsdict['size']
num_neurons = argsdict['num_neurons']
window_size = argsdict["window_size"]
agent_type = argsdict["agent_type"]
save_dir = argsdict["save_dir"]
save_model_freq = argsdict["save_model_freq"]
load_existing_agent = argsdict["load_existing_agent"]

print(argsdict)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu:0')

env = SimpleNavigation(size=size)

torch.autograd.set_detect_anomaly(True)

if agent_type == 'mlp':
    baseline_agent = AC_MLP(
        input_size=4,  # 2 for agent location, 2 for goal location
        hidden_size=[num_neurons, num_neurons],  # linear, linear
        action_size=4
    ).to(device)
    optimizer = torch.optim.Adam(baseline_agent.parameters(), lr=lr)
    add_input = None
    if load_existing_agent is not None:
        checkpoint = torch.load(load_existing_agent)
        baseline_agent.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        p_loss = checkpoint['p_loss']
        v_loss = checkpoint['v_loss']
        i_episode = checkpoint['i_episode']
        baseline_agent.train()
    steps_taken, init_loc, target_loc = train_neural_net_on_SimpleNavigation(env, baseline_agent, optimizer, num_episodes, save_model_freq, add_input, 'baseline', save_dir, agent_type)
    plot_results(1, num_episodes, steps_taken, window_size, save_dir, 'mlp_agent_steps_taken')
elif agent_type == 'rnn':
    rnn_agent = AC_RNN(
        input_size=4,  # 2 for agent location, 2 for goal location
        hidden_size=[num_neurons,num_neurons],  # LSTM, linear
        batch_size=1,
        num_LSTM_layers=1,
        action_size=4
    ).to(device)
    optimizer = torch.optim.Adam(rnn_agent.parameters(), lr=lr)
    add_input = None
    if load_existing_agent is not None:
        checkpoint = torch.load(load_existing_agent)
        rnn_agent.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        p_loss = checkpoint['p_loss']
        v_loss = checkpoint['v_loss']
        i_episode = checkpoint['i_episode']  # TODO: integrate this into train_neural_net_on_simpleNavigation
        rnn_agent.train()
    steps_taken, init_loc, target_loc = train_neural_net_on_SimpleNavigation(env, rnn_agent, optimizer, num_episodes, save_model_freq, add_input, 'baseline', save_dir, agent_type)
    plot_results(1, num_episodes, steps_taken, window_size, save_dir, 'rnn_agent_steps_taken')

np.save(os.path.join(save_dir, f"baseline_{agent_type}_init_locations.npy"), init_loc)
np.save(os.path.join(save_dir, f"baseline_{agent_type}_goal_locations.npy"), target_loc)