import numpy as np
import matplotlib.pyplot as plt
import os


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
        plt.show()
        # plt.savefig(os.path.join(save_dir, "place_cells.svg"), format='svg')
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
        # fig_save_dir = os.path.join(save_dir, 'hd_cells')
        # if not os.path.exists:
        #     os.mkdir(fig_save_dir)
        for i_neuron in range(num_neurons)[:2]:
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
            plt.show()
            # plt.savefig(os.path.join(fig_save_dir, f"{i_neuron}.svg"), format='svg')
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
    goal_direction = target_loc[valid_trial_idx] - location[valid_trial_idx]
    action_direction = np.zeros((len(valid_trial_idx), 2))
    breakpoint()
    for i, idx in enumerate(valid_trial_idx):
        action_direction[i] = action[idx][_action_to_direction]

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
        for i_neuron in range(num_neurons)[:2]:
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
            plt.show()
    return action_goal_angles


if __name__ == "__main__":
    data = np.load('/Users/dongyanlin/Desktop/TEM_RL/SimpleNav_data/3/baseline_og_rnn_data.npy', allow_pickle=True).item()
    init_loc = data["init_loc"]
    target_loc = data["target_loc"]
    hx_activity = data["hx_activity"]
    cx_activity = data["cx_activity"]
    lin_activity = data["lin_activity"]
    trial_counter = data["trial_counter"]
    location = data["location"]
    action = data["action"]
    init_loc = expand_loc(init_loc, trial_counter)
    target_loc = expand_loc(target_loc, trial_counter)

    # occupancy, mean_place_activity = place_cells(5, 256, location, hx_activity, 'do not save for now', plot=True)
    # activity_by_action, mean_HD_activity = HD_cells(256,4,action,hx_activity, 'do not save for now', plot=True)
    action_goal_angles = goal_direction_cell(256, trial_counter, target_loc, location, action, hx_activity, 'do not save for now', plot=True)



# TODO: generate some random data of shape (t, 256) and run place/HD tuning analysis
# TODO: repeat analysis for hx, ccx, lin_activity, lin_1_activity, lin_2_activity