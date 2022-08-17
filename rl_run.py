import matplotlib

from rl_model import *
from rl_world import *
import numpy as np
import torch
import matplotlib.pyplot as plt
import gym
from tqdm import tqdm
import argparse
import os
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

    rewards = []

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

    return rewards


def train_neural_net(env, agent, num_envs, num_episodes_per_env, lr, save_model_freq):
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
                input_to_model = np.concatenate(list(env.observation.values()))
                pol, val = agent.forward(torch.unsqueeze(torch.unsqueeze(torch.as_tensor(input_to_model), dim=0), dim=1).float())  # TODO: differentiate between inputs for AC_MLP, AC_RNN, and actor_critic_agent
                act, p, v = select_action(agent, pol, val)
                new_obs, reward, done, info = env.step(act)
                agent.rewards.append(reward)
            # print(f"This episode has {len(agent.rewards)} steps")
            rewards[i_block*num_episodes_per_env + i_episode] = sum(agent.rewards)
            p_loss, v_loss = finish_trial(agent, 0.99, optimizer)
            if i_block*num_episodes_per_env + i_episode % save_model_freq == 0:
                torch.save({
                    'i_env': i_block,
                    'i_episode': i_episode,
                    'model_state_dict': agent.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'p_loss': p_loss,
                    'v_loss': v_loss
                }, os.path.join(save_dir, f'{agent_type}_Env{i_block}_Epi{i_episode}.pt'))

    return rewards


def plot_results(num_envs, num_episodes_per_env, rewards, window_size, save_dir, title):

    plt.figure()
    plt.plot(np.arange(num_envs*num_episodes_per_env), bin_rewards(np.array(rewards), window_size=window_size))
    plt.vlines(x=np.arange(start=num_episodes_per_env, stop=num_envs*num_episodes_per_env, step=num_episodes_per_env),
               ymin=min(bin_rewards(np.array(rewards), window_size=window_size))-5,
               ymax=max(bin_rewards(np.array(rewards), window_size=window_size))+5, linestyles='dotted')
    plt.title(title)
    plt.savefig(os.path.join(save_dir, f'{title}.svg'), format='svg')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run neural networks on tem-rl environment")
    parser.add_argument("--num_envs",type=int,default=10,help='Number of environments with different object-location pairings')
    parser.add_argument("--num_episodes_per_env",type=int,default=10000,help="Number of episodes to train agent on each environment")
    parser.add_argument("--lr",type=float,default=0.0001,help="learning rate")
    parser.add_argument("--edge_length",type=int,default=5,help="Length of edge for the environment")
    parser.add_argument("--num_objects",type=int,default=45,help="Number of object that could be associated with different locations of the environment")
    parser.add_argument("--num_neurons", type=int, default=128, help="Number of units in each hidden layer")
    parser.add_argument("--window_size", type=int, default=1000, help="Size of rolling window for smoothing out performance plot")
    parser.add_argument("--agent_type", type=str, default='rnn', help="type of agent to use. Either 'rnn' or 'mlp'.")
    parser.add_argument("--save_dir", type=str, default='experiments/', help="path to save figures.")
    parser.add_argument("--save_model_freq", type=int, default=1000, help="Frequency (# of episodes) of saving model checkpoint")
    args = parser.parse_args()
    argsdict = args.__dict__

    num_envs = argsdict['num_envs']
    num_episodes_per_env = argsdict['num_episodes_per_env']
    lr = argsdict['lr']
    edge_length = argsdict['edge_length']
    num_objects = argsdict['num_objects']
    num_neurons = argsdict['num_neurons']
    window_size = argsdict["window_size"]
    agent_type = argsdict["agent_type"]
    save_dir = argsdict["save_dir"]
    save_model_freq = argsdict["save_model_freq"]

    print(argsdict)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu:0')

    env = Navigation(edge_length, num_objects)
    # env = gym.wrappers.FlattenObservation(env)
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

    torch.autograd.set_detect_anomaly(True)

    if agent_type == 'mlp':
        baseline_agent = AC_MLP(
            input_size=num_objects*2,
            hidden_size=[num_neurons, num_neurons],  # linear, linear
            action_size=5
        ).to(device)
        rewards = train_neural_net(env, baseline_agent, num_envs, num_episodes_per_env, lr, save_model_freq)
        plot_results(num_envs, num_episodes_per_env, rewards, window_size, save_dir, 'mlp_agent')
    elif agent_type == 'rnn':
        rnn_agent = AC_RNN(
            input_size=num_objects*2,
            hidden_size=[num_neurons,num_neurons],  # LSTM, linear
            batch_size=1,
            num_LSTM_layers=1,
            action_size=5
        ).to(device)
        rewards = train_neural_net(env, rnn_agent, num_envs, num_episodes_per_env, lr, save_model_freq)
        plot_results(num_envs, num_episodes_per_env, rewards, window_size, save_dir, 'rnn_agent')




