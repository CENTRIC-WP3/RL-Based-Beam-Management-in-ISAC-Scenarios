import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm

def extract_slot_sequence(action_list):

    slots_statistics = np.zeros(3)
    sequence = []
    for a in action_list:
        if a == 0:
            sequence.append('S1')
            slots_statistics[0] += 1
        elif a == 1:
            sequence.append('DL1')
            slots_statistics[1] += 1
        elif a == 2:
            sequence.append('UL1')
            slots_statistics[2] += 1
        elif a == 3:
            sequence.append('S2')
            slots_statistics[0] += 1
        elif a == 4:
            sequence.append('DL2')
            slots_statistics[1] += 1
        elif a == 5:
            sequence.append('UL2')
            slots_statistics[2] += 1
        elif a == 6:
            sequence.append('S3')
            slots_statistics[0] += 1
        elif a == 7:
            sequence.append('DL3')
            slots_statistics[1] += 1
        elif a == 8:
            sequence.append('UL3')
            slots_statistics[2] += 1
    return slots_statistics, sequence

reward_episode_genie = np.loadtxt("results_beam_tracking/reward_episode_genie_cttc")
mean_packet_rate_episode_genie = np.loadtxt('results_beam_tracking/mean_packet_rate_episode_genie_cttc')
action_list_genie = np.loadtxt('results_beam_tracking/action_list_genie_cttc')
failure_type_genie = np.loadtxt('results_beam_tracking/failure_type_genie_cttc')

reward_episode = np.loadtxt("results_beam_tracking/reward_episode_PPO_cttc")
mean_packet_rate_episode = np.loadtxt('results_beam_tracking/mean_packet_rate_episode_PPO_cttc')
action_list = np.loadtxt('results_beam_tracking/action_list_PPO_cttc')
failure_type = np.loadtxt('results_beam_tracking/failure_type_PPO_cttc')

reward_episode_random = np.loadtxt("results_beam_tracking/reward_episode_random_cttc")
mean_packet_rate_episode_random = np.loadtxt('results_beam_tracking/mean_packet_rate_episode_random_cttc')
action_list_random = np.loadtxt('results_beam_tracking/action_list_random_cttc')
failure_type_random = np.loadtxt('results_beam_tracking/failure_type_random_cttc')

reward_episode_3tdma = np.loadtxt("results_beam_tracking/reward_episode_3TDMA_cttc")
mean_packet_rate_episode_3tdma = np.loadtxt('results_beam_tracking/mean_packet_rate_episode_3TDMA_cttc')
sequence = np.array([0, 1, 2, 1, 2, 1, 2, 3, 4, 5, 4, 5, 4, 5, 6, 7, 8, 7, 8, 7, 8])
action_list_3tdma = np.tile(sequence, 100000 // len(sequence) + 1)[:100000]
failure_type_3tdma = np.loadtxt('results_beam_tracking/failure_type_3TDMA_cttc')

reward_episode_tdma = np.loadtxt("results_beam_tracking/reward_episode_1TDMA_cttc")
mean_packet_rate_episode_tdma = np.loadtxt('results_beam_tracking/mean_packet_rate_episode_1TDMA_cttc')
sequence = np.arange(10)
action_list_tdma = np.tile(sequence, 10000)
failure_type_tdma = np.loadtxt('results_beam_tracking/failure_type_1TDMA_cttc')

reward_episode_6tdma = np.loadtxt("results_beam_tracking/reward_episode_6TDMA_cttc")
mean_packet_rate_episode_6tdma = np.loadtxt('results_beam_tracking/mean_packet_rate_episode_6TDMA_cttc')
sequence = np.array([0, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 3, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 6, 7, 8, 7, 8, 7, 8, 7, 8, 7, 8, 7, 8])
action_list_6tdma = np.tile(sequence, 100000 // len(sequence) + 1)[:100000]
failure_type_6tdma = np.loadtxt('results_beam_tracking/failure_type_6TDMA_cttc')

total_traffic = np.loadtxt("results_beam_tracking/total_traffic_cttc")

reward_episode_df = pd.DataFrame(reward_episode, columns=['value'])
mean_packet_rate_episode_df = pd.DataFrame(mean_packet_rate_episode, columns=['value'])


slots_statistics_tdma, sequence_tdma = extract_slot_sequence(action_list_tdma)
slots_statistics_random, sequence_random = extract_slot_sequence(action_list_random)
slots_statistics_PPO, sequence_PPO = extract_slot_sequence(action_list)
slots_statistics_3tdma, sequence_3tdma = extract_slot_sequence(action_list_3tdma)
slots_statistics_6tdma, sequence_6tdma = extract_slot_sequence(action_list_6tdma)

plt.figure(1)

data = np.array(mean_packet_rate_episode)
data2 = np.array(mean_packet_rate_episode_random)
data3 = np.array(mean_packet_rate_episode_tdma)
data4 = np.array(mean_packet_rate_episode_6tdma)
data5 = np.array(mean_packet_rate_episode_3tdma)
data6 = np.array(mean_packet_rate_episode_genie)
# Create a figure and axis
fig, ax = plt.subplots()

# Create a sorted array of unique values for each dataset
x1 = np.sort(np.unique(data))
x2 = np.sort(np.unique(data2))
x3 = np.sort(np.unique(data3))
x4 = np.sort(np.unique(data4))
x5 = np.sort(np.unique(data5))
x6 = np.sort(np.unique(data6))

# Compute the empirical CDF for each dataset
y1 = np.arange(1, len(x1)+1) / len(x1)
y2 = np.arange(1, len(x2)+1) / len(x2)
y3 = np.arange(1, len(x3)+1) / len(x3)
y4 = np.arange(1, len(x4)+1) / len(x4)
y5 = np.arange(1, len(x5)+1) / len(x5)
y6 = np.arange(1, len(x6)+1) / len(x6)

# Plot the ECDFs
ax.plot(x6, y6, label='Genie-aided')
ax.plot(x1, y1, label='PPO (ours)', color='b')
ax.plot(x2, y2, label='Random', color='r')
ax.plot(x3, y3, label='1-TDMA', color='g')
ax.plot(x5, y5, label='3-TDMA', color='y')
ax.plot(x4, y4, label='6-TDMA', color='m')
# Set the axis labels and legend
ax.set_xlabel('% Packet Error Rate (PER)')
ax.set_ylabel('Probability')
ax.legend()
plt.grid()

fig = plt.figure(2)
# Creating a figure and axis object
fig, ax = plt.subplots()


data1 = [slots_statistics_random[0] / sum(slots_statistics_random),
         slots_statistics_random[1] / sum(slots_statistics_random),
         slots_statistics_random[2] / sum(slots_statistics_random)]
data2 = [slots_statistics_tdma[0] / sum(slots_statistics_tdma),
         slots_statistics_tdma[1] / sum(slots_statistics_tdma),
         slots_statistics_tdma[2] / sum(slots_statistics_tdma)]
data3 = [slots_statistics_PPO[0] / sum(slots_statistics_PPO),
         slots_statistics_PPO[1] / sum(slots_statistics_PPO),
         slots_statistics_PPO[2] / sum(slots_statistics_PPO)]
data4 = [slots_statistics_3tdma[0] / sum(slots_statistics_3tdma),
         slots_statistics_3tdma[1] / sum(slots_statistics_3tdma),
         slots_statistics_3tdma[2] / sum(slots_statistics_3tdma)]
data5 = [slots_statistics_6tdma[0] / sum(slots_statistics_6tdma),
         slots_statistics_6tdma[1] / sum(slots_statistics_6tdma),
         slots_statistics_6tdma[2] / sum(slots_statistics_6tdma)]

# Setting the x-axis labels and their positions
labels = ['Sensing', 'Downlink', 'Uplink']
x_pos = np.arange(len(labels))

# Width of each bar
bar_width = 0.15

# Creating the barplot
ax.bar(x_pos - 2 * bar_width, data3, bar_width, label='PPO (ours)', color='b')
ax.bar(x_pos - bar_width, data1, bar_width, label='Random', color='r')
ax.bar(x_pos, data2, bar_width, label='1-TDMA', color='g')
ax.bar(x_pos + bar_width, data4, bar_width, label='3-TDMA', color='y')
ax.bar(x_pos + 2 * bar_width, data5, bar_width, label='6-TDMA', color='m')

# Adding a legend and axis labels
ax.legend()
ax.set_ylabel('(%)')

# Setting the ticks and labels for the x-axis
ax.set_xticks(x_pos)
ax.set_xticklabels(labels)

fig = plt.figure(3)
# Creating a figure and axis object
fig, ax = plt.subplots()

data1 = [failure_type_random[0] / sum(sum(total_traffic)),
         failure_type_random[1] / sum(sum(total_traffic)),
         failure_type_random[2] / sum(sum(total_traffic))]
data2 = [failure_type_tdma[0] / sum(sum(total_traffic)),
         failure_type_tdma[1] / sum(sum(total_traffic)),
         failure_type_tdma[2] / sum(sum(total_traffic))]
data3 = [failure_type[0] / sum(sum(total_traffic)),
         failure_type[1] / sum(sum(total_traffic)),
         failure_type[2] / sum(sum(total_traffic))]
data4 = [failure_type_3tdma[0] / sum(sum(total_traffic)),
         failure_type_3tdma[1] / sum(sum(total_traffic)),
         failure_type_3tdma[2] / sum(sum(total_traffic))]
data5 = [failure_type_6tdma[0] / sum(sum(total_traffic)),
         failure_type_6tdma[1] / sum(sum(total_traffic)),
         failure_type_6tdma[2] / sum(sum(total_traffic))]

# Setting the x-axis labels and their positions
labels = ['Buffer drop', 'Beam drop', 'Blockage drop']
x_pos = np.arange(len(labels))

# Width of each bar
bar_width = 0.15

# Creating the barplot
ax.bar(x_pos - 2 * bar_width, data3, bar_width, label='PPO (ours)', color='b')
ax.bar(x_pos - bar_width, data1, bar_width, label='Random', color='r')
ax.bar(x_pos, data2, bar_width, label='1-TDMA', color='g')
ax.bar(x_pos + bar_width, data4, bar_width, label='3-TDMA', color='y')
ax.bar(x_pos + 2 * bar_width, data5, bar_width, label='6-TDMA', color='m')

# Adding a legend and axis labels
ax.legend()
ax.set_ylabel('(%)')

# Setting the ticks and labels for the x-axis
ax.set_xticks(x_pos)
ax.set_xticklabels(labels)

fig = plt.figure(4)
fig, ax = plt.subplots()


data1 = [failure_type_random[0] / sum(sum(total_traffic)), (failure_type_random[1] + failure_type_random[2])/ sum(sum(total_traffic))]
data2 = [failure_type_tdma[0] / sum(sum(total_traffic)), (failure_type_tdma[1] + failure_type_tdma[2]) / sum(sum(total_traffic))]
data3 = [failure_type_3tdma[0] /sum(sum(total_traffic)), (failure_type_3tdma[1] + failure_type_3tdma[2]) / sum(sum(total_traffic))]
data4 = [failure_type[0] / sum(sum(total_traffic)), (failure_type[1] + failure_type[2]) / sum(sum(total_traffic))]
data5 = [failure_type_6tdma[0] / sum(sum(total_traffic)), (failure_type_6tdma[1] + failure_type_6tdma[2]) / sum(sum(total_traffic))]

labels = ['Communication drop', 'Sensing drop']
x_pos = np.arange(len(labels))

bar_width = 0.15

ax.bar(x_pos - 2.5 * bar_width, data4, bar_width, label='PPO (ours)', color='b')
ax.bar(x_pos - 1.5 * bar_width, data1, bar_width, label='Random', color='r')
ax.bar(x_pos - 0.5 * bar_width, data2, bar_width, label='1-TDMA', color='g')
ax.bar(x_pos + 0.5 * bar_width, data3, bar_width, label='3-TDMA', color='y')
ax.bar(x_pos + 1.5 * bar_width, data5, bar_width, label='6-TDMA', color='m')

ax.legend()
ax.set_ylabel('(%)')

ax.set_xticks(x_pos)
ax.set_xticklabels(labels)


fig = plt.figure(5)
# Creating a figure and axis object
fig, ax = plt.subplots()

data1 = [np.min(mean_packet_rate_episode_random), np.mean(mean_packet_rate_episode_random), np.max(mean_packet_rate_episode_random)]
data2 = [np.min(mean_packet_rate_episode_tdma), np.mean(mean_packet_rate_episode_tdma), np.max(mean_packet_rate_episode_tdma)]
data3 = [np.min(mean_packet_rate_episode_3tdma), np.mean(mean_packet_rate_episode_3tdma), np.max(mean_packet_rate_episode_3tdma)]
data4 = [np.min(mean_packet_rate_episode), np.mean(mean_packet_rate_episode), np.max(mean_packet_rate_episode)]
data5 = [np.min(mean_packet_rate_episode_6tdma), np.mean(mean_packet_rate_episode_6tdma), np.max(mean_packet_rate_episode_6tdma)]


# Setting the x-axis labels and their positions
labels = ['Min Avg PER', 'Mean Avg PER', 'Max Avg PER']
x_pos = np.arange(len(labels))

# Setting the width of each bar and the spacing between the bars
width = 0.1
spacing = 0.05

# Creating the barplot
ax.bar(x_pos - 2 * width - 2 * spacing, data4, width, label='PPO (ours)', color='b')
ax.bar(x_pos - width - spacing, data1, width, label='Random', color='r')
ax.bar(x_pos, data2, width, label='1-TDMA', color='g')
ax.bar(x_pos + width + spacing, data3, width, label='3-TDMA', color='y')
ax.bar(x_pos + 2 * width + 2 * spacing, data5, width, label='6TDMA', color='m')


# Adding a legend and axis labels
ax.legend()
ax.set_ylabel('(%)')

# Setting the ticks and labels for the x-axis
ax.set_xticks(x_pos)
ax.set_xticklabels(labels)

# Show the plot
plt.show()
