import numpy as np
import matplotlib.pyplot as plt
from rl_run import bin_rewards

# Assume we have:
# goal_locations.npy: (num_envs,)
# init_locations.npy: (num_envs, num_episodes_per_env)
# node_visit_counter.npy: (num_envs, num_locations)
# steps_taken.npy: (num_envs, num_episodes_per_env)

data_dir = 'experiments/2022-08-17/'
agent_type = 'baseline_rnn'

goal_locations = np.load(data_dir+agent_type+'_goal_locations.npy')
init_locations = np.load(data_dir+agent_type+'_init_locations.npy')
node_visit_counter = np.load(data_dir+agent_type+'_node_visit_counter.npy')
steps_taken = np.load(data_dir+agent_type+'_steps_taken.npy')

num_envs = goal_locations.shape[0]
num_episodes_per_env = init_locations.shape[1]
num_locations = node_visit_counter.shape[1]
edge_length = int(np.sqrt(num_locations))
grid = np.arange(num_locations).reshape((edge_length, edge_length))
window_size = 1000

# Calculate shortest distance
goal_locations = np.tile(goal_locations, (num_episodes_per_env, 1)).T
shortest_distance = []

for goal_location, init_location in zip(goal_locations.flatten().tolist(), init_locations.flatten().tolist()):
    shortest_distance.append(abs(
        int(np.where(grid == goal_location)[0] - np.where(grid == init_location)[0])) + abs(
        int(np.where(grid == goal_location)[1] - np.where(grid == init_location)[1])))

shortest_distance = np.reshape(np.array(shortest_distance), (num_envs, num_episodes_per_env))

# Steps taken vs shortest distance figure
plt.figure()
plt.plot(np.arange(num_envs*num_episodes_per_env), bin_rewards(shortest_distance.flatten(), window_size), label='shortest distance')
plt.plot(np.arange(num_envs*num_episodes_per_env), bin_rewards(steps_taken.flatten(), window_size), label='# steps')
plt.vlines(x=np.arange(start=num_episodes_per_env, stop=num_envs*num_episodes_per_env, step=num_episodes_per_env),
           ymin=min(bin_rewards(shortest_distance.flatten(), window_size))-5,
           ymax=max(bin_rewards(steps_taken.flatten(), window_size))+5, linestyles='dotted')
plt.legend()
plt.title(agent_type)
plt.show()
plt.savefig(data_dir+agent_type+"_steps_taken.svg", format='svg')

# occupancy figure
num_rows = 2
num_cols = num_envs // num_rows
fig, axs = plt.subplots(num_rows, num_cols)
for i_row in range(num_rows):
    for i_col in range(num_cols):
        axs[i_row, i_col].imshow(node_visit_counter[i_row*num_cols+i_col].reshape(edge_length, edge_length))
        # axs[i_row, i_col].set_title(f'Env {i_row*num_cols+i_col}, goal location {goal_locations[i_row*num_cols+i_col, 0]}')
fig.suptitle('Agent occupancy', fontsize=16)
plt.show()
plt.savefig(data_dir+agent_type+"_occupancy.svg", format='svg')


