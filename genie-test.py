import gymnasium as gym
import math
import random
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count
import numpy as np
import pandas as pd
import pickle

from beam_management_env_letter import BeamManagementEnv

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

torch.set_default_dtype(torch.float32)
experiment_number = random.randint(0, 1000000000)

env = gym.make('BeamManagementEnv-v1', N_f=1000)
num_episodes = 10000
episode_durations = []
reward_episode = []
lost_packet_rate_episode = []
mean_packet_rate_episode = []
action_list = []
action = 0
counter = 0
sum_reward = []
th_episode = []
for i_episode in range(num_episodes):
    # Initialize the environment and get it's state
    print("Episode: " + str(i_episode), flush=True)
    action = 1
    state = env.reset()
    state = torch.tensor(state, dtype=torch.float32)
    position_ue_list = []
    reward_list = []
    l_a = 0
    for t in count():
        action_list.append(action)
        observation, reward, terminated, info = env.step(action)
        # print(observation)
        position_ue_list.append(info['ue_position'])
        reward_list.append(reward)
        reward = torch.tensor([reward])
        done = terminated

        action += 1
        if action == 3 or action == 6:
            action += 1
        if action == 9:
            action = 1

        if done:
            episode_durations.append(t + 1)
            mean_packet_rate_episode.append(np.mean(info['lost_traffic_episode'] / info['total_traffic_episode']))
            reward_episode.append(np.mean(reward_list))
            th_episode.append(sum(info['tx_packets_episode']) / 1000)
            break

print('Complete')
# print(np.mean(mean_packet_rate_episode))
# print(np.mean(sum_reward))
# print(np.mean(th_episode))
print("Total lost traffic DL and UL: " + str(sum(info['lost_traffic'])))
print("Total traffic DL and UL: " + str(sum(info['total_traffic'])))
print("Total lost traffic DL and UL for 3 users:\n" + str(info['lost_traffic'] / info['total_traffic']))
print("Average PER: " + str(np.mean(info['lost_traffic'] / info['total_traffic'])))
print("Total Throughput: " + str(1-sum(info['failure_type']/sum(sum(info['total_traffic'])))))
print("(%) of failure types (Buffer, Beam, Blockage) over failed traffic: " + str(info['failure_type']/sum(info['failure_type'])))
print("(%) of failure types (Buffer, Beam, Blockage) over total traffic: " + str(info['failure_type']/sum(sum(info['total_traffic']))))
print("(%) of drops over the total traffic: " + str(sum(info['failure_type']/sum(sum(info['total_traffic'])))))
print("Lost packets by type: " + str(info['failure_type']))


np.savetxt('results_beam_tracking/reward_episode_genie_cttc', reward_episode)
np.savetxt('results_beam_tracking/mean_packet_rate_episode_genie_cttc', mean_packet_rate_episode)
np.savetxt('results_beam_tracking/action_list_genie_cttc', action_list)
np.savetxt('results_beam_tracking/failure_type_genie_cttc', info['failure_type'])
# np.savetxt('results_beam_tracking/total_traffic_cttc', info['total_traffic'])
np.savetxt('results_beam_tracking/th_genie_cttc', th_episode)

