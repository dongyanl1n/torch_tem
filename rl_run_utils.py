import matplotlib

import rl_world
from rl_model import *
from rl_world import *
import numpy as np
import torch
import matplotlib.pyplot as plt
import gym
from tqdm import tqdm
import argparse
import os
import datetime
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


def train_neural_net_on_SimpleNavigation(env, agent, optimizer, num_episodes, save_model_freq, save_dir, agent_type, record_activity=False):
    steps_taken = []
    if record_activity:
        trial_counter = []
        location = []
        action = []
        if agent_type == "mlp" or agent_type == 'conv_mlp':
            lin_1_activity = []
            lin_2_activity = []
        elif agent_type == "rnn" or agent_type == 'conv_rnn':
            hx_activity = []
            cx_activity = []
            lin_activity = []
    init_loc = np.zeros((num_episodes, 2), dtype=np.int8)
    target_loc = np.zeros((num_episodes, 2), dtype=np.int8)
    for i_episode in tqdm(range(num_episodes)):
        done = False
        observation = env.reset()
        init_loc[i_episode] = observation["agent"]
        target_loc[i_episode] = observation["target"]
        agent.reinit_hid()
        while not done: # act, step
            if agent_type == "rnn" or agent_type == "mlp":
                input_to_model = torch.unsqueeze(torch.unsqueeze(torch.as_tensor(np.concatenate(list(observation.values()))), dim=0), dim=1).float()[0]
            elif agent_type == "conv_rnn" or agent_type == "conv_mlp":
                input_to_model = torch.unsqueeze(torch.Tensor(np.reshape(render_observation_as_image(
                    env.size, observation["agent"], observation["target"], show_target=True), (3, env.size, env.size))), dim=0).float()
            pol, val = agent.forward(input_to_model)
            act, p, v = select_action(agent, pol, val)
            if record_activity:
                action.append(act)
                trial_counter.append(i_episode)
                location.append(observation["agent"])
                if agent_type == "mlp" or agent_type == "conv_mlp":
                    lin_1_activity.append(agent.cell_out[agent.hidden_types.index("linear")].clone().detach().cpu().numpy().squeeze())
                    lin_2_activity.append(agent.cell_out[agent.hidden_types.index("linear")+1].clone().detach().cpu().numpy().squeeze())
                elif agent_type == "rnn" or agent_type == "conv_rnn":
                    hx_activity.append(agent.hx[agent.hidden_types.index("lstm")].clone().detach().cpu().numpy().squeeze())
                    cx_activity.append(agent.cx[agent.hidden_types.index("lstm")].clone().detach().cpu().numpy().squeeze())
                    lin_activity.append(agent.cell_out[agent.hidden_types.index("lstm")+1].clone().detach().cpu().numpy().squeeze())
            observation, reward, done, info = env.step(act)
            agent.rewards.append(reward)
        steps_taken.append(len(agent.rewards))
        p_loss, v_loss = finish_trial(agent, 0.99, optimizer)
        if (i_episode + 1) % save_model_freq == 0:
            torch.save({
                'i_episode': i_episode,
                'model_state_dict': agent.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'p_loss': p_loss,
                'v_loss': v_loss
            }, os.path.join(save_dir, f'{agent_type}_Epi{i_episode}.pt'))
    if record_activity:  # zip recorded data
        if agent_type == "mlp":
            data = {"init_loc": init_loc,
                    "target_loc": target_loc,
                    "action": np.array(action),
                    "trial_counter": np.array(trial_counter),
                    "location": np.array(location),
                    "lin_1_activity": np.array(lin_1_activity),
                    "lin_2_activity": np.array(lin_2_activity)}
        elif agent_type == "rnn":
            data = {"init_loc": init_loc,
                    "target_loc": target_loc,
                    "action": np.array(action),
                    "trial_counter": np.array(trial_counter),
                    "location": np.array(location),
                    "hx_activity": np.array(hx_activity),
                    "cx_activity": np.array(cx_activity),
                    "lin_activity": np.array(lin_activity)}
    else:
        data = {"init_loc": init_loc,
                "target_loc": target_loc}
    return steps_taken, data


def train_neural_net_on_MorrisWaterMaze(env, agent, optimizer, num_episodes, save_model_freq, save_dir, agent_type, num_episodes_per_block=2, record_activity=False):
    assert isinstance(agent, AC_Conv_Net), "agent has to have convolutional component because images are used as inputs here"
    steps_taken = []
    if record_activity:
        trial_counter = []
        location = []
        action = []
        if agent_type == "conv_mlp":
            lin_1_activity = []
            lin_2_activity = []
        elif agent_type == "conv_rnn":
            hx_activity = []
            cx_activity = []
            lin_activity = []
    init_loc = np.zeros((num_episodes, 2), dtype=np.int8)
    target_loc = np.zeros((num_episodes, 2), dtype=np.int8)
    # breakpoint()
    for i_episode in tqdm(range(num_episodes)):
        step_counter = 0
        done = False
        observation = env.reset()
        init_loc[i_episode] = observation["agent"]
        target_loc[i_episode] = observation["target"]
        if i_episode % num_episodes_per_block == 0:
            agent.reinit_hid()
        while not done: # act, step
            if i_episode % num_episodes_per_block == 0:
                input_to_model = torch.unsqueeze(torch.Tensor(np.reshape(render_observation_as_image(
                    env.size, observation["agent"], observation["target"], show_target=True), (3, env.size, env.size))), dim=0).float()
            else:
                input_to_model = torch.unsqueeze(torch.Tensor(np.reshape(render_observation_as_image(
                    env.size, observation["agent"], observation["target"], show_target=False), (3, env.size, env.size))), dim=0).float()
            pol, val = agent.forward(input_to_model)
            act, p, v = select_action(agent, pol, val)
            if record_activity:
                action.append(act)
                trial_counter.append(i_episode)
                location.append(observation["agent"])
                if agent_type == "mlp":
                    lin_1_activity.append(agent.cell_out[0].clone().detach().cpu().numpy().squeeze())
                    lin_2_activity.append(agent.cell_out[1].clone().detach().cpu().numpy().squeeze())
                elif agent_type == "rnn":
                    hx_activity.append(agent.hx[0].clone().detach().cpu().numpy().squeeze())
                    cx_activity.append(agent.cx[0].clone().detach().cpu().numpy().squeeze())
                    lin_activity.append(agent.cell_out[1].clone().detach().cpu().numpy().squeeze())
            observation, reward, done, info = env.step(act)
            agent.rewards.append(reward)
            step_counter += 1
        steps_taken.append(step_counter)  # no not really
        if i_episode % num_episodes_per_block == num_episodes_per_block - 1:  # last trial of the block
            p_loss, v_loss = finish_trial(agent, 0.99, optimizer)
        if (i_episode + 1) % save_model_freq == 0:
            torch.save({
                'i_episode': i_episode,
                'model_state_dict': agent.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'p_loss': p_loss,
                'v_loss': v_loss
            }, os.path.join(save_dir, f'{agent_type}_Epi{i_episode}.pt'))
    if record_activity:  # zip recorded data
        if agent_type == "mlp":
            data = {"init_loc": init_loc,
                    "target_loc": target_loc,
                    "action": np.array(action),
                    "trial_counter": np.array(trial_counter),
                    "location": np.array(location),
                    "lin_1_activity": np.array(lin_1_activity),
                    "lin_2_activity": np.array(lin_2_activity)}
        elif agent_type == "rnn":
            data = {"init_loc": init_loc,
                    "target_loc": target_loc,
                    "action": np.array(action),
                    "trial_counter": np.array(trial_counter),
                    "location": np.array(location),
                    "hx_activity": np.array(hx_activity),
                    "cx_activity": np.array(cx_activity),
                    "lin_activity": np.array(lin_activity)}
    else:
        data = {"init_loc": init_loc,
                "target_loc": target_loc}
    return steps_taken, data



def plot_results(num_envs, num_episodes_per_env, rewards, window_size, save_dir, title):

    plt.figure()
    plt.plot(np.arange(num_envs*num_episodes_per_env), bin_rewards(np.array(rewards), window_size=window_size))
    plt.vlines(x=np.arange(start=num_episodes_per_env, stop=num_envs*num_episodes_per_env, step=num_episodes_per_env),
               ymin=min(bin_rewards(np.array(rewards), window_size=window_size))-5,
               ymax=max(bin_rewards(np.array(rewards), window_size=window_size))+5, linestyles='dotted')
    plt.title(title)
    plt.savefig(os.path.join(save_dir, f'{title}_behavioural_results.svg'), format='svg')

