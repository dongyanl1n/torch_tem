# Standard library imports
import glob
import importlib.util
# Own module imports. Note how model module is not imported, since we'll used the model from the training run
import numpy as np
import gym
import world
import analyse
from rl_world import Navigation
from rl_model import *
from rl_run import train_neural_net, plot_results
from tqdm import tqdm
import argparse
import os

# Set random seeds for reproducibility
np.random.seed(0)
torch.manual_seed(0)

parser = argparse.ArgumentParser(description="Run neural networks on tem-rl environment")
parser.add_argument("--date",type=str,default='2022-07-31',help='Date to load model')
parser.add_argument("--run",type=str,default='0',help="Run to load model")
parser.add_argument("--index",type=str,default="1000",help="Model index (i.e. how many pretraining episodes in) to load model")

parser.add_argument("--num_envs",type=int,default=10,help='Number of environments with different object-location pairings')
parser.add_argument("--num_episodes_per_env",type=int,default=10000,help="Number of episodes to train agent on each environment")
parser.add_argument("--lr",type=float,default=0.0001,help="learning rate")
parser.add_argument("--edge_length",type=int,default=5,help="Length of edge for the environment")
# parser.add_argument("--num_objects",type=int,default=45,help="Number of object that could be associated with different locations of the environment")
parser.add_argument("--num_neurons", type=int, default=128, help="Number of units in each hidden layer")
parser.add_argument("--window_size", type=int, default=1000, help="Size of rolling window for smoothing out performance plot")
parser.add_argument("--agent_type", type=str, default='mlp', help="type of agent to use. Either 'rnn' or 'mlp'.")
parser.add_argument("--save_dir", type=str, default='experiments/', help="path to save figures.")
parser.add_argument("--save_model_freq", type=int, default=1000, help="Frequency (# of episodes) of saving model checkpoint")
args = parser.parse_args()
argsdict = args.__dict__

print(argsdict)

# Choose which trained model to load
date = argsdict["date"]
run = argsdict["run"]
index = argsdict["index"]
save_dir = argsdict["save_dir"]

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

g_cat = np.hstack(g[0]).flatten()  # NOTE: index 0 because currently there's only 1 env. TODO: what if we want to increase the number of environments on which TEM is pretrained?
p_cat = np.hstack(p[0]).flatten()
# ======== MAKING IT RL ============
edge_length = argsdict["edge_length"]  # TODO: is there a param in TEM that determines the edge length?
num_objects = params['n_x']
num_neurons = argsdict["num_neurons"]
num_envs = argsdict["num_envs"]
num_episodes_per_env = argsdict["num_episodes_per_env"]
lr = argsdict["lr"]
window_size = argsdict["window_size"]
agent_type = argsdict["agent_type"]
save_model_freq = argsdict["save_model_freq"]

torch.autograd.set_detect_anomaly(True)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu:0')

rl_env = Navigation(edge_length, num_objects)
# rl_env = gym.wrappers.FlattenObservation(rl_env)

if agent_type == 'mlp':
    downstream_mlp_agent = AC_MLP(
        input_size=p_cat.size+num_objects*2,
        hidden_size=[num_neurons, num_neurons],
        action_size=5
    ).to(device)
    add_input = p_cat
    rewards, goal_locations, node_visit_counter_list, steps_taken_list, init_locations_list = train_neural_net(rl_env, downstream_mlp_agent, num_envs, num_episodes_per_env, lr, save_model_freq, add_input, 'tem', save_dir, agent_type)
    plot_results(num_envs, num_episodes_per_env, rewards, window_size, save_dir, 'tem_mlp_readout_agent')

elif agent_type == 'rnn':
    downstream_rnn_agent = AC_RNN(
        input_size=p_cat.size+num_objects*2,
        hidden_size=[num_neurons, num_neurons],
        batch_size=1,
        num_LSTM_layers=1,
        action_size=5
    ).to(device)
    add_input = p_cat
    rewards, goal_locations, node_visit_counter_list, steps_taken_list, init_locations_list = train_neural_net(rl_env, downstream_rnn_agent, num_envs, num_episodes_per_env, lr, save_model_freq, add_input, 'tem', save_dir, agent_type)
    plot_results(num_envs, num_episodes_per_env, rewards, window_size, save_dir, 'tem_rnn_readout_agent')

np.save(os.path.join(save_dir, f"tem_{agent_type}_{date}_run{run}_{index}_goal_locations.npy"), goal_locations)
np.save(os.path.join(save_dir, f"tem_{agent_type}_{date}_run{run}_{index}_node_visit_counter.npy"), node_visit_counter_list)
np.save(os.path.join(save_dir, f"tem_{agent_type}_{date}_run{run}_{index}_steps_taken.npy"), steps_taken_list)
np.save(os.path.join(save_dir, f"tem_{agent_type}_{date}_run{run}_{index}_init_locations.npy"), init_locations_list)
