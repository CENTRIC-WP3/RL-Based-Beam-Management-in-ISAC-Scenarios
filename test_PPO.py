import os
import glob
import time
from datetime import datetime

import torch
import numpy as np

import gymnasium as gym
from beam_management_env_letter import BeamManagementEnv
# import roboschool

from PPO import PPO


#################################### Testing ###################################
def test():
    print("============================================================================================")

    ################## hyperparameters ##################

    # env_name = "CartPole-v1"
    # has_continuous_action_space = False
    # max_ep_len = 400
    # action_std = None

    # env_name = "LunarLander-v2"
    # has_continuous_action_space = False
    # max_ep_len = 300
    # action_std = None

    # env_name = "BipedalWalker-v2"
    # has_continuous_action_space = True
    # max_ep_len = 1500           # max timesteps in one episode
    # action_std = 0.1            # set same std for action distribution which was used while saving

    env_name = "BeamManagementEnv-v1"
    has_continuous_action_space = False
    max_ep_len = 1000           # max timesteps in one episode
    action_std = 0.1            # set same std for action distribution which was used while saving

    render = False              # render environment on screen
    frame_delay = 0             # if required; add delay b/w frames

    total_test_episodes = 10000   # total num of testing episodes

    K_epochs = 80               # update policy for K epochs
    eps_clip = 0.2              # clip parameter for PPO
    gamma = 0.99                # discount factor

    lr_actor = 0.0003           # learning rate for actor
    lr_critic = 0.001           # learning rate for critic

    #####################################################

    env = gym.make(env_name, N_f=1000)

    # state space dimension
    state_dim = env.observation_space.shape[0]

    # action space dimension
    if has_continuous_action_space:
        action_dim = env.action_space.shape[0]
    else:
        action_dim = env.action_space.n

    # initialize a PPO agent
    ppo_agent = PPO(state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, has_continuous_action_space, action_std)

    # preTrained weights directory

    random_seed = 0             #### set this to load a particular checkpoint trained on random seed
    run_num_pretrained = 0      #### set this to load a particular checkpoint num

    directory = "PPO_preTrained" + '/' + env_name + '/'
    checkpoint_path = directory + "PPO_{}_{}_{}.pth".format(env_name, random_seed, run_num_pretrained)
    # checkpoint_path = directory + 'PPO_BeamManagementEnv-v1_WINNER.pth'
    print("loading network from : " + checkpoint_path)

    ppo_agent.load(checkpoint_path)

    print("--------------------------------------------------------------------------------------------")

    test_running_reward = 0
    mean_packet_rate_episode = []
    th_episode = []
    reward_episode = []
    action_list = []

    for ep in range(1, total_test_episodes+1):
        ep_reward = 0
        state = env.reset()
        counter = 0
        reward_list = []

        for t in range(1, max_ep_len+1):
            action = ppo_agent.select_action(state)
            if action == 0 or action == 3 or action == 6:
                # print(action)
                counter += 1
            # print(action)
            action_list.append(action)
            state, reward, done, info = env.step(action)
            ep_reward += reward
            reward_list.append(reward)

            if render:
                env.render()
                time.sleep(frame_delay)

            if done:
                mean_packet_rate_episode.append(np.mean(info['lost_traffic_episode'] / info['total_traffic_episode']))
                reward_episode.append(np.mean(reward_list))
                th_episode.append(sum(info['tx_packets_episode']) / 1000)
                break

        # clear buffer
        ppo_agent.buffer.clear()

        test_running_reward +=  ep_reward
        print('Episode: {} \t\t Reward: {}'.format(ep, round(ep_reward, 2)))
        ep_reward = 0

    env.close()

    print("============================================================================================")
    print("Total lost traffic DL and UL: " + str(sum(info['lost_traffic'])))
    print("Total traffic DL and UL: " + str(sum(info['total_traffic'])))
    print("Total lost traffic DL and UL for 3 users:\n" + str(info['lost_traffic'] / info['total_traffic']))
    print("Average PER: " + str(np.mean(info['lost_traffic'] / info['total_traffic'])))
    print("Total Throughput: " + str(1 - sum(info['failure_type'] / sum(sum(info['total_traffic'])))))
    print("(%) of failure types (Buffer, Beam, Blockage) over failed traffic: " + str(
        info['failure_type'] / sum(info['failure_type'])))
    print("(%) of failure types (Buffer, Beam, Blockage) over total traffic: " + str(
        info['failure_type'] / sum(sum(info['total_traffic']))))
    print("(%) of drops over the total traffic: " + str(sum(info['failure_type'] / sum(sum(info['total_traffic'])))))
    print("Lost packets by type: " + str(info['failure_type']))
    print("============================================================================================")

    # np.savetxt('results_beam_tracking/reward_episode_PPO_cttc', reward_episode)
    # np.savetxt('results_beam_tracking/mean_packet_rate_episode_PPO_cttc', mean_packet_rate_episode)
    # np.savetxt('results_beam_tracking/action_list_PPO_cttc', action_list)
    # np.savetxt('results_beam_tracking/failure_type_PPO_cttc', info['failure_type'])
    # np.savetxt('results_beam_tracking/th_PPO_cttc', th_episode)

if __name__ == '__main__':

    test()