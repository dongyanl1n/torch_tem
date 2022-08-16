# Standard library imports
import os
import numpy as np
import torch
import glob
import matplotlib.pyplot as plt
import importlib.util
# Own module imports. Note how model module is not imported, since we'll used the model from the training run
import world
import analyse
import plot
import sys
sys.path.append("./tem_rl")
from tem_rl.rl_world import Navigation
from tem_rl.rl_model import *
from tem_rl.rl_run import bin_rewards, train_neural_net, plot_results
import argparse

# Set random seeds for reproducibility
np.random.seed(0)
torch.manual_seed(0)

# Choose which trained model to load
date = '2022-07-31' # 2020-10-13 run 0 for successful node agent
run = '0'
index = '1000'  # TODO: argparser so that we can make a loop in sh script to loop through models of different indices
# breakpoint()
# Load the model: use import library to import module from specified path
model_spec = importlib.util.spec_from_file_location("model", './Summaries/' + date + '/run' + run + '/script/model.py')
model = importlib.util.module_from_spec(model_spec)
model_spec.loader.exec_module(model)

# Load the parameters of the model
params = torch.load('./Summaries/' + date + '/run' + run + '/model/params_' + index + '.pt')
# Create a new tem model with the loaded parameters
tem = model.Model(params)
# Load the model weights after training
model_weights = torch.load('./Summaries/' + date + '/run' + run + '/model/tem_' + index + '.pt')
# Set the model weights to the loaded trained model weights
tem.load_state_dict(model_weights)
# Make sure model is in evaluate mode (not crucial because it doesn't currently use dropout or batchnorm layers)
tem.eval()

# ====== Run Model on TEM environments to generate place representations ================

# Make list of all the environments that this model was trained on
envs = list(glob.iglob('./Summaries/' + date + '/run' + run + '/script/envs/*'))
# Set which environments will include shiny objects
shiny_envs = [False]
# Set the number of walks to execute in parallel (batch size)
n_walks = len(shiny_envs)  # 1
# Select environments from the environments included in training
environments = [world.World(graph, randomise_observations=True, shiny=(params['shiny'] if shiny_envs[env_i] else None))
                for env_i, graph in enumerate(np.random.choice(envs, n_walks))]  # 1
# Determine the length of each walk
walk_len = np.median([env.n_locations * 50 for env in environments]).astype(int)  # 1250
# And generate walks for each environment
walks = [env.generate_walks(walk_len, 1)[0] for env in environments]  # 1

# Generate model input from specified walk and environment: group steps from all environments together to feed to model in parallel
model_input = [[[[walks[i][j][k]][0] for i in range(len(walks))] for k in range(3)] for j in range(walk_len)]
# model_input has same length as walk_len. Each model_input element (i.e. each step) contains 4 inputs (for the 4 parallel envs), each in the form of {'id', 'shiny', 45-dim one-hot vector, action id}
for i_step, step in enumerate(model_input):
    model_input[i_step][1] = torch.stack(step[1], dim=0)  # reformat model_input

# Run a forward pass through the model using this data, without accumulating gradients
with torch.no_grad():
    forward = tem(model_input, prev_iter=None)

# Decide whether to include stay-still actions as valid occasions for inference
include_stay_still = True

# Generate rate maps
g, p = analyse.rate_map(forward, tem, environments)
# g: list of 1 lists (for 1 env) of 5 arrays (for 5 steams) of shape (25, n_g) (for 25 locations). Each element is the firing rate of that grid cell (in that stream) at that location in that env.
# p: list of 1 lists (for 1 env) of 5 arrays (for 5 steams) of shape (25, n_p). Each element is the firing rate of that place cell (in that stream) at that location in that env.

g_cat = np.hstack(g[0])  # NOTE: index 0 because currently there's only 1 env. TODO: what if we want to increase the number of environments on which TEM is pretrained?
p_cat = np.hstack(p[0])
# ======== MAKING IT RL ============
edge_length = 5  # TODO: is there a param in TEM that determines the edge length?
num_objects = params['n_x']
num_neurons = 128
num_envs = 10
num_episodes_per_env = 1000
lr = 0.0001
n_rollout = 20


rl_env = Navigation(edge_length, num_objects)

downstream_mlp_agent = AC_MLP(
    input_size=len(p_cat),
    hidden_size=[num_neurons, num_neurons],
    action_size=5
)
downstream_rnn_agent = AC_RNN(
    input_size=len(p_cat),
    hidden_size=[num_neurons, num_neurons],
    batch_size=1,
    num_LSTM_layers=1,
    action_size=5
)

torch.autograd.set_detect_anomaly(True)
rewards = train_neural_net(rl_env, downstream_mlp_agent, num_envs, num_episodes_per_env, lr, n_rollout)
plot_results(num_envs, num_episodes_per_env, rewards, 'mlp_readout_agent')