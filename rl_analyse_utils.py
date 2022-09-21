import numpy as np
import matplotlib.pyplot as plt
import os
from rl_model import conv_output, AC_Net, AC_Conv_Net
import torch


def get_rnn_weights(network_path, agent_type, num_neurons, size,
                    rfsize = 2, padding = 0, stride = 1, dilation = 1, conv_1_features = 16, conv_2_features = 32):
    layer_1_out_h, layer_1_out_w = conv_output(size, size, padding, dilation, rfsize, stride)
    layer_2_out_h, layer_2_out_w = conv_output(layer_1_out_h, layer_1_out_w, padding, dilation, rfsize, stride)
    layer_3_out_h, layer_3_out_w = conv_output(layer_2_out_h, layer_2_out_w, padding, dilation, rfsize, stride)
    layer_4_out_h, layer_4_out_w = conv_output(layer_3_out_h, layer_3_out_w, padding, dilation, rfsize, stride)

    if agent_type == 'mlp':
        agent = AC_Net(
            input_dimensions=4,
            action_dimensions=4,
            batch_size=1,
            hidden_types=['linear', 'linear'],
            hidden_dimensions=[num_neurons, num_neurons]
        )
    elif agent_type == 'rnn':
        agent = AC_Net(
            input_dimensions=4,
            action_dimensions=4,
            batch_size=1,
            hidden_types=['lstm', 'linear'],
            hidden_dimensions=[num_neurons, num_neurons]
        )
    elif agent_type == 'conv_mlp':
        agent = AC_Conv_Net(
            input_dimensions=(size, size, 3),
            action_dimensions=4,
            batch_size=1,
            hidden_types=['conv', 'pool', 'conv', 'pool', 'linear', 'linear'],
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
        )
    elif agent_type == 'conv_rnn':
        agent = AC_Conv_Net(
            input_dimensions=(size, size, 3),
            action_dimensions=4,
            batch_size=1,
            hidden_types=['conv', 'pool', 'conv', 'pool', 'lstm', 'linear'],
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
        )
    checkpoint = torch.load(network_path,map_location=torch.device('cpu'))
    agent.load_state_dict(checkpoint['model_state_dict'])
    agent.eval()
    rnn_params = {}
    lstm_index = agent.hidden_types.index('lstm')
    for name, param in agent.named_parameters():
        if name == f'hidden.{lstm_index}.weight_ih':
            rnn_params["weight_ih"] = param
        if name == f'hidden.{lstm_index}.weight_hh':
            rnn_params["weight_hh"] = param
        if name == f'hidden.{lstm_index}.bias_ih':
            rnn_params["bias_ih"] = param
        if name == f'hidden.{lstm_index}.bias_hh':
            rnn_params["bias_hh"] = param
    return rnn_params


def get_cell_num_by_max_location(size, num_neurons, location, activity):
    _, mean_place_activity = place_cells(size, num_neurons, location, activity, save_dir=None, plot=False)
    cell_num_by_max_location = {}
    assert size == mean_place_activity.shape[0], "Wrong size"
    assert num_neurons == mean_place_activity.shape[-1], "Wrong num_neurons"
    max_loc = np.argmax(mean_place_activity.reshape(size**2, num_neurons), axis=0)
    for i_loc in range(5**2):
        cell_num_by_max_location[i_loc] = np.where(max_loc == i_loc)[0]
    return max_loc, cell_num_by_max_location


def get_cell_num_by_max_HD(action, activity):
    activity_by_action, mean_HD_activity = HD_cells(256,4,action, activity, save_dir=None, plot=False)
    actual_mean_HD_activity = mean_HD_activity[:4]
    cell_num_by_max_HD = {}
    max_HD = np.argmax(actual_mean_HD_activity, axis=0)
    for i_HD in range(4):
        cell_num_by_max_HD[i_HD] = np.where(max_HD == i_HD)[0]
    return max_HD, cell_num_by_max_HD


def sort_weight_matrix(weight_matrix, cell_nums_wanted):
    new_weight_matrix = np.empty_like(weight_matrix)
    new_idx = np.concatenate(list(cell_nums_wanted.values()))
    for i, i_new_idx in enumerate(new_idx):
        new_weight_matrix[i] = weight_matrix[i_new_idx]
    for i, i_new_idx in enumerate(new_idx):
        new_weight_matrix[:, i] = weight_matrix[:, i_new_idx]
    return new_weight_matrix


def neighbours_and_connections(size, num_neurons, location, activity, weight_matrix):
    neighbour_dict = {}
    connection_dict = {}
    HD_loc_mapping = {
        'Right': 1,
        'Up': -size,
        'Left': -1,
        'Down': size
    }
    max_loc, cell_num_by_max_location = get_cell_num_by_max_location(size, num_neurons, location, activity)
    for i_neuron in range(num_neurons):
        neuron_neighbour_dict = {}
        neuron_connection_dict = {}
        neuron_place_field = max_loc[i_neuron]
        for HD in ['Right', 'Up', 'Left', 'Down']:
            next_loc = neuron_place_field + HD_loc_mapping[HD]
            if 0 <= next_loc <= size**2-1:
                neuron_neighbour_dict[HD] = np.where(max_loc == next_loc)[0]
                neuron_connection_dict[HD] = weight_matrix[i_neuron, neuron_neighbour_dict[HD]]
        neighbour_dict[i_neuron] = neuron_neighbour_dict
        connection_dict[i_neuron] = neuron_connection_dict
    return neighbour_dict, connection_dict


def calculate_extra_steps(steps_taken, init_loc, target_loc):
    shortest_path = np.linalg.norm(
        init_loc - target_loc, ord=1, axis=1
    )
    extra_steps = steps_taken - shortest_path
    return extra_steps

def place_cells(size, num_neurons,location, activity, save_dir, plot=True):
    occupancy = np.zeros((size, size))
    mean_activity = np.zeros((size, size, num_neurons))
    for x in range(size):
        for y in range(size):
            occupancy[x, y] = np.sum(np.all(location == [x,y], axis=1))  # How many times agent visited this location
            mean_activity[x, y] = np.sum(activity[np.where(np.all(location == [x,y], axis=1))], axis=0)  # each neuron's summation of activity at this location
            mean_activity[x, y] /= occupancy[x, y]  # I really mean "mean activity"
    for i_neuron in range(num_neurons):  # for each neuron, normalize its activity across locations
        mean_activity[:, :, i_neuron] = (mean_activity[:, :, i_neuron] - np.min(mean_activity[:, :, i_neuron])) / np.ptp(mean_activity[:, :, i_neuron])
    if plot:
        num_cols = 10
        num_rows = num_neurons // num_cols
        fig, axs = plt.subplots(num_rows, num_cols, figsize=(6, 14)) #figsize=(num_cols*4,num_rows*4)
        for i_row in range(num_rows):
            for i_col in range(num_cols):
                axs[i_row, i_col].imshow(mean_activity[:, :, i_row*num_cols+i_col])
                # axs[i_row, i_col].set_title(f"Neuron {i_row*num_cols+i_col}", fontsize=8)
                axs[i_row, i_col].get_xaxis().set_visible(False)
                axs[i_row, i_col].get_yaxis().set_visible(False)
        fig.suptitle('Mean activity by agent location', fontsize=16)
        plt.tight_layout()
        # plt.show()
        plt.savefig(os.path.join(save_dir, "place_cells.svg"), format='svg')
    return occupancy, mean_activity


def HD_cells(num_neurons, num_actions, action, activity, save_dir, plot=True):
    action_count = np.zeros(num_actions)
    mean_activity = np.zeros((num_actions+1, num_neurons))
    activity_by_action = []
    for i_action in range(num_actions):
        activity_by_action.append(activity[action==i_action])
        action_count[i_action] = np.sum(action == i_action)
        mean_activity[i_action] = np.sum(activity[np.where(action == i_action)], axis=0)
        mean_activity[i_action] /= action_count[i_action]
    mean_activity[-1] = mean_activity[0]  # repeat the first row to close the polar curve
    # for i_neuron in range(num_neurons):  # for each neuron, normalize its activity across actions
    #     mean_activity[:, i_neuron] = (mean_activity[:, i_neuron] - np.min(mean_activity[:, i_neuron])) / np.ptp(mean_activity[:, i_neuron])
    if plot:
        fig_save_dir = os.path.join(save_dir, 'hd_cells')
        if not os.path.exists(fig_save_dir):
            os.mkdir(fig_save_dir)
        for i_neuron in range(num_neurons):
            data = []
            for i_action in range(num_actions):
                data.append(activity_by_action[i_action][:, i_neuron])
            fig = plt.figure(figsize=(10, 5))
            ax1 = plt.subplot(1,2,1)
            ax1.set_title('All occurrences')
            ax1.set_ylabel('Activation')
            ax1.set_xlabel('Action')
            ax1.set_xticklabels(['a', 'Right', 'Up', 'Left', 'Down'])  # I don't know why but I have to put a placeholder label in the first position for the labels to look normal
            ax1.violinplot(data, [1, 2, 3, 4], widths=1, showmeans=True, showextrema=False)

            ax2 = plt.subplot(1,2,2, projection='polar')
            ax2.plot(np.arange(0, 450, 90)*(np.pi/180), mean_activity[:, i_neuron])
            ax2.set_title('Mean activity (not normalized)')
            # plt.show()
            plt.savefig(os.path.join(fig_save_dir, f"{i_neuron}.svg"), format='svg')
    return activity_by_action, mean_activity


def expand_loc(loc, trial_counter):
    '''
    Expand the loc array of length n_episodes to an array of length n_timesteps and perserve the location info
    '''
    expanded_loc = np.zeros((len(trial_counter), 2))
    for i_trial in range(len(loc)):
        expanded_loc[trial_counter == i_trial] = loc[i_trial]
    return expanded_loc


def goal_direction_cell(num_neurons, trial_counter, target_loc, location, action, activity, save_dir, plot=True):
    _action_to_direction = {
        0: np.array([1, 0]),
        1: np.array([0, 1]),
        2: np.array([-1, 0]),
        3: np.array([0, -1]),
        }
    valid_trial_idx = np.arange(len(trial_counter))[np.where(np.any(target_loc != location, axis=1))]
    # all idx of trial_counter should be valid (because as soon as agent moves to target, the next trial starts and agent gets a new obs
    goal_direction = target_loc[valid_trial_idx] - location[valid_trial_idx]
    action_direction = np.zeros((len(valid_trial_idx), 2))
    for i, idx in enumerate(valid_trial_idx):
        action_direction[i] = _action_to_direction[action[idx]]

    # let a, b be N x 2 arrays. Then d will be a N x 1 array containing the dot products of each entry in a and b
    # No I don't have any idea why this works, but https://medium.com/analytics-vidhya/tensordot-explained-6673cfa5697f
    # has a good explanation (specifically when axes = 1).
    c = np.tensordot(action_direction.T, goal_direction, axes = ((0), (1)))
    dot_products = np.diag(c)
    norm_products = np.linalg.norm(action_direction, axis=1) * np.linalg.norm(goal_direction, axis=1)
    action_goal_angles = np.arccos(dot_products / norm_products) # a list of thetas between the action direction, i.e. (t, )
    unique_angles = np.unique(action_goal_angles)
    # and the goal direction
    if plot:
        fig_save_dir = os.path.join(save_dir, 'goal_dir_cells')
        if not os.path.exists:
            os.mkdir(fig_save_dir)
        for i_neuron in range(num_neurons):
            responses = []
            for unique_angle in unique_angles:
                responses.append(activity[action_goal_angles==unique_angle][:, i_neuron])
            fig = plt.figure(figsize=(5, 5))
            ax1 = plt.subplot(1,1,1)
            ax1.set_title('All occurrences')
            ax1.set_ylabel('Activation')
            ax1.set_xlabel('Goal direction')
            # ax1.set_xticklabels(['a', 'Right', 'Up', 'Left', 'Down'])  # I don't know why but I have to put a placeholder label in the first position for the labels to look normal
            ax1.violinplot(responses, widths=1, showmeans=True, showextrema=False)
            # plt.show()
            plt.savefig(os.path.join(fig_save_dir, f"{i_neuron}.svg"), format='svg')
    return action_goal_angles


if __name__ == "__main__":
    data = np.load('/Users/dongyanlin/Desktop/TEM_RL/SimpleNav_data/3/baseline_og_rnn_data.npy', allow_pickle=True).item()  # TODO：collect new rnn data corresponding to the weights
    init_loc = data["init_loc"]
    target_loc = data["target_loc"]
    hx_activity = data["hx_activity"]
    cx_activity = data["cx_activity"]
    # lin_1_activity = data["lin_1_activity"]
    # lin_2_activity = data["lin_2_activity"]
    trial_counter = data["trial_counter"]
    location = data["location"]
    action = data["action"]
    init_loc = expand_loc(init_loc, trial_counter)
    target_loc = expand_loc(target_loc, trial_counter)

    occupancy, mean_place_activity = place_cells(5, 256, location, hx_activity, '/Users/dongyanlin/Desktop/TEM_RL/SimpleNav_data/3/rnn/hx', plot=False)
    activity_by_action, mean_HD_activity = HD_cells(256,4,action, hx_activity, '/Users/dongyanlin/Desktop/TEM_RL/SimpleNav_data/3/rnn/hx', plot=False)

    # action_goal_angles = goal_direction_cell(256, trial_counter, target_loc, location, action, hx_activity, '/Users/dongyanlin/Desktop/TEM_RL/SimpleNav_data/rnn/hx', plot=True)

    rnn_params = get_rnn_weights('/Users/dongyanlin/Desktop/TEM_RL/SimpleNav_data/trained_agent/conv_rnn_Epi49999.pt', agent_type='conv_rnn', num_neurons=256,size=5)
    weight_matrix = rnn_params['weight_hh'][:256].detach().numpy()  # TODO: play with this
    max_loc, cell_num_by_max_location = get_cell_num_by_max_location(5, 256, location, hx_activity)
    max_HD, cell_num_by_max_HD = get_cell_num_by_max_HD(action, hx_activity)

    neighbour_dict, connection_dict = neighbours_and_connections(5, 256, location, hx_activity, weight_matrix)

    # TODO: think about what to do from here
    i_neuron = 251
    print(max_HD[i_neuron])
    for HD in ['Right', 'Up', 'Left', 'Down']:
        if HD in list(connection_dict[i_neuron].keys()):
            print(HD, np.mean(connection_dict[i_neuron]['Left']))
    # if you want to check sorted weights. Otherwise comment these lines out
    # plt.figure()
    # plt.imshow(sort_weight_matrix(weight_matrix, cell_num_by_max_HD))
    # plt.imshow()