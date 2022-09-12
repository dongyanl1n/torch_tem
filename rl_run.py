import matplotlib

import rl_world
from rl_model import *
from rl_world import *
from rl_run_utils import *
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
parser.add_argument("--agent_type", type=str, default='rnn', help="type of agent to use. 'rnn' or 'mlp' or 'conv_rnn' or 'conv_mlp")
parser.add_argument("--save_dir", type=str, default='experiments/', help="path to save figures.")
parser.add_argument("--save_model_freq", type=int, default=1000, help="Frequency (# of episodes) of saving model checkpoint")
parser.add_argument("--load_existing_agent", type=str, default=None, help="path to existing agent to load")
parser.add_argument("--record_activity", type=str, default="False", help="whether to record behaviour and neural data or not")
parser.add_argument("--morris_water_maze", type=str, default="False", help="make this task a morris water maze")
parser.add_argument("--num_episodes_per_block", type=int, default=3, help="number of episodes in each block of morris water maze (i.e. how often the platform changes location")

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
record_activity = True if argsdict["record_activity"] == "True" or argsdict["record_activity"] == True else False
morris_water_maze = True if argsdict["morris_water_maze"] == "True" or argsdict["morris_water_maze"] == True else False
num_episodes_per_block = argsdict["num_episodes_per_block"]

print(argsdict)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu:0')

env = SimpleNavigation(size=size)

torch.autograd.set_detect_anomaly(True)

rfsize = 2
padding = 0
stride = 1
dilation = 1
layer_1_out_h, layer_1_out_w = conv_output(size, size, padding, dilation, rfsize, stride)
layer_2_out_h, layer_2_out_w = conv_output(layer_1_out_h, layer_1_out_w, padding, dilation, rfsize, stride)
layer_3_out_h, layer_3_out_w = conv_output(layer_2_out_h, layer_2_out_w, padding, dilation, rfsize, stride)
layer_4_out_h, layer_4_out_w = conv_output(layer_3_out_h, layer_3_out_w, padding, dilation, rfsize, stride)
conv_1_features = 16
conv_2_features = 32


if agent_type == 'mlp':
    agent = AC_Net(
            input_dimensions=4,
            action_dimensions=4,
            batch_size=1,
            hidden_types=['linear', 'linear'],
            hidden_dimensions=[num_neurons, num_neurons]
        ).to(device)
elif agent_type == 'rnn':
    agent = AC_Net(
            input_dimensions=4,
            action_dimensions=4,
            batch_size=1,
            hidden_types=['lstm', 'linear'],
            hidden_dimensions=[num_neurons, num_neurons]
        ).to(device)
elif agent_type == 'conv_mlp':
        agent = AC_Conv_Net(
            input_dimensions=(size, size, 3),
            action_dimensions=4,
            batch_size=1,
            hidden_types= ['conv', 'pool', 'conv', 'pool', 'linear', 'linear'],
            hidden_dimensions=[
                (layer_1_out_h, layer_1_out_w, conv_1_features),  # conv
                (layer_2_out_h, layer_2_out_w, conv_1_features),  # pool
                (layer_3_out_h, layer_3_out_w, conv_2_features),  # conv
                (layer_4_out_h, layer_4_out_w, conv_2_features),  # pool
                num_neurons,
                num_neurons],
            rfsize=rfsize,
            padding=padding,
            stride=stride
        ).to(device)
elif agent_type == 'conv_rnn':
    agent = AC_Conv_Net(
        input_dimensions=(size, size, 3),
        action_dimensions=4,
        batch_size=1,
        hidden_types= ['conv', 'pool', 'conv', 'pool', 'lstm', 'linear'],
        hidden_dimensions=[
            (layer_1_out_h, layer_1_out_w, conv_1_features),  # conv
            (layer_2_out_h, layer_2_out_w, conv_1_features),  # pool
            (layer_3_out_h, layer_3_out_w, conv_2_features),  # conv
            (layer_4_out_h, layer_4_out_w, conv_2_features),  # pool
            num_neurons,
            num_neurons],
        rfsize=rfsize,
        padding=padding,
        stride=stride
    ).to(device)


optimizer = torch.optim.Adam(agent.parameters(), lr=lr)
add_input = None
if load_existing_agent is not None:
    checkpoint = torch.load(load_existing_agent)
    agent.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    p_loss = checkpoint['p_loss']
    v_loss = checkpoint['v_loss']
    i_episode = checkpoint['i_episode']  # TODO: integrate this into train_neural_net_on_simpleNavigation
    agent.train()

if not morris_water_maze:
    steps_taken, data = train_neural_net_on_SimpleNavigation(env, agent, optimizer, num_episodes, save_model_freq, save_dir, agent_type, record_activity)
else:
    steps_taken, data = train_neural_net_on_MorrisWaterMaze(env, agent, optimizer, num_episodes, save_model_freq,  save_dir, agent_type, num_episodes_per_block=num_episodes_per_block, record_activity=record_activity)
plot_results(1, num_episodes, steps_taken, window_size, save_dir, agent_type+'steps_taken')

if not morris_water_maze:
    np.save(os.path.join(save_dir, f"{agent_type}_SimpleNav_data.npy"), data)
else:
    np.save(os.path.join(save_dir, f"{agent_type}_MorrisWaterMaze_data.npy"), data)
