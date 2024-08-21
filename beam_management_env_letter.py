from channel_functions import *
import gymnasium as gym
from gymnasium.envs import register
from gymnasium import spaces
from motion import *
import random
random.seed(1)
np.random.seed(1)

def bernoulli(p):
    return 1 if random.random() <= p else 0
    # return 1


class BeamManagementEnv(gym.Env):
    def __init__(self, N_f):
        self.N_f = N_f  # total number of channel uses
        self.N_UEs = 3
        self.r_h = []
        self.r_a = []
        self.r_p = []
        self.th = np.zeros((1000,))
        self.th_episode = np.zeros((1000,))
        self.config = np.zeros((self.N_UEs,))
        self.action_space = spaces.Discrete(9)
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(12,))
        self.number_of_antennas = 32  # number of antennas at the ULA BS
        self.fc = 28e9  # carrier frequency (Hz)
        self.c = 3e8  # speed of light (m/s)
        self.lambda_ = self.c / self.fc  # wavelength (m)
        self.Pt = 5  # W
        self.d_ULA = np.arange(self.number_of_antennas) * self.lambda_ / 2  # ULA with half-wavelength spacing
        self.codebook = dft_codebook(self.number_of_antennas, self.number_of_antennas)  # set of beamforming directions
        self.current_slot = 0  # current slot/channel use
        self.current_beam = np.zeros((self.N_UEs,)).astype(int)  # current beam direction
        self.current_traffic = np.zeros(
            (self.N_UEs, 2))  # current downlink and uplink traffic #DL1/UL1/DL2/UL2/DL3/UL3
        self.total_traffic = np.zeros(
            (self.N_UEs, 2))  # total downlink and uplink traffic
        self.lost_traffic = np.zeros(
            (self.N_UEs, 2))  # Traffic lost
        self.total_traffic_episode = np.zeros(
            (self.N_UEs, 2))  # total downlink and uplink traffic
        self.lost_traffic_episode = np.zeros(
            (self.N_UEs, 2))  # Traffic lost
        self.failure_counter = np.zeros(
            (self.N_UEs, 2))  # Consecutive failures (packet lost due to buffer full or bad beam)
        self.previous_beam = np.zeros((5, self.N_UEs,)).astype(int)  # current beam direction
        self.previous_traffic = np.zeros(
            (5, self.N_UEs, 2))  # current downlink and uplink traffic #DL1/UL1/DL2/UL2/DL3/UL3
        self.previous_power = np.zeros((5, self.N_UEs))
        self.counter_slots_blocked = np.zeros((self.N_UEs, )).astype(int) #Counter for slots blockages
        self.previous_counter = 0
        self.failure_type = np.zeros(3)
        self.dl_buffer_size = 100
        self.ul_buffer_size = 100
        self.done = False  # flag indicating whether the episode is over
        self.base_station_position = np.array([50, 50])  # [x_bs, y_bs, z_bs]
        self.scenario_dimensions = np.array([0, 0, 100, 100])  # [x_min, y_min, x_max, y_max]
        self.user_positions = np.zeros((self.N_UEs, 2))

        self.d = np.zeros((self.N_UEs,))
        self.AoD = np.zeros((self.N_UEs,))
        self.a_BS = np.zeros((self.N_UEs, self.number_of_antennas)) + 1j
        self.mmWave_LoS_NLoS_channel = np.zeros((self.N_UEs, 1, self.number_of_antennas)) + 1j
        self.R = np.zeros((self.N_UEs,))
        for i in range(self.N_UEs):
            self.user_positions[i][0] = 0
            self.user_positions[i][1] = 80 / (i + 1)
            self.d[i] = np.linalg.norm(self.base_station_position - self.user_positions[
                i])  # euclidean distance between the BS and the UEs

            self.AoD[i] = compute_AoD_at_BS(self.base_station_position, self.user_positions[i])  # current AoD

            self.a_BS[i] = compute_ULA_response(self.number_of_antennas,
                                                self.AoD[i])  # current array response

            self.mmWave_LoS_NLoS_channel[i] = compute_mmWave_LoS_NLoS_channel(self.number_of_antennas, self.a_BS[i],
                                                                    self.d[i])  # current LoS channel

            self.R[i] = compute_power_given_beam(self.Pt, self.mmWave_LoS_NLoS_channel[i],
                                                 self.codebook[:, 0])  # current received power
        self.power_threshold = 1
        self.seed = 1

    def reset(self):
        self.r_h = []
        self.r_a = []
        self.r_p = []
        # self.th = np.zeros((1000,))
        self.th_episode = np.zeros((1000,))
        self.current_slot = 0
        self.done = False
        self.current_beam = np.zeros((self.N_UEs,)).astype(int)
        self.current_traffic = np.zeros(
            (self.N_UEs, 2))  # current downlink and uplink traffic #DL1/UL1/DL2/UL2/DL3/UL3
        # self.total_traffic = np.zeros(
        #     (self.N_UEs, 2))  # total downlink and uplink traffic
        # self.lost_traffic = np.zeros(
        #     (self.N_UEs, 2))  # Traffic lost
        self.total_traffic_episode = np.zeros((self.N_UEs, 2))  # total downlink and uplink traffic per episode
        self.lost_traffic_episode = np.zeros((self.N_UEs, 2))  # traffic lost per episode
        self.failure_counter = np.zeros(
            (self.N_UEs, 2))  # Consecutive failures (packet lost due to buffer full or bad beam)
        # self.failure_type = np.zeros(3)
        self.previous_beam = np.zeros((5, self.N_UEs,)).astype(int)  # current beam direction
        self.previous_traffic = np.zeros(
            (5, self.N_UEs, 2))  # current downlink and uplink traffic #DL1/UL1/DL2/UL2/DL3/UL3
        self.previous_power = np.zeros((5, self.N_UEs))
        self.previous_counter = 0

        for i in range(self.N_UEs):
            self.config[i] = np.random.randint(4)
            if self.config[i] == 0:
                self.user_positions[i][0] = 0
                self.user_positions[i][1] = 80 / (i + 1)
            elif self.config[i] == 1:
                self.user_positions[i][0] = 100
                self.user_positions[i][1] = 80 / (i + 1)
            elif self.config[i] == 2:
                self.user_positions[i][0] = 80 / (i + 1)
                self.user_positions[i][1] = 0
            else:
                self.user_positions[i][0] = 80 / (i + 1)
                self.user_positions[i][1] = 100

            self.d[i] = np.linalg.norm(self.base_station_position - self.user_positions[
                i])  # euclidean distance between the BS and the UEs

            self.AoD[i] = compute_AoD_at_BS(self.base_station_position, self.user_positions[i])  # current AoD

            self.a_BS[i] = compute_ULA_response(self.number_of_antennas,
                                                self.AoD[i])  # current array response

            self.mmWave_LoS_NLoS_channel[i] = compute_mmWave_LoS_NLoS_channel(self.number_of_antennas, self.a_BS[i],
                                                                    self.d[i])  # current LoS channel
            self.current_beam[i] = np.argmax(np.abs(np.sqrt(self.Pt) * self.mmWave_LoS_NLoS_channel[i] @ self.codebook))

            self.R[i] = compute_power_given_beam(self.Pt, self.mmWave_LoS_NLoS_channel[i],
                                                 self.codebook[:, self.current_beam[i]])  # current received power
            self.counter_slots_blocked[i] = 0
        return self._get_state()

    def step(self, action):
        # print("action: " + str(action))
        reward = 0
        done = False
        best_power_index = np.zeros((self.N_UEs,)).astype(int)
        best_power = np.zeros((self.N_UEs,))
        packet = np.zeros((self.N_UEs,)).astype(float)
        was_dropped = np.zeros((self.N_UEs, 2))
        drop = 0
        for i in range(self.N_UEs):
            for j in range(2):
                self.previous_traffic[self.previous_counter][i][j] = self.current_traffic[i][
                                                                         j].copy() / self.dl_buffer_size
            self.previous_beam[self.previous_counter][i] = self.current_beam[i].copy()

        for i in range(self.N_UEs):

            # In every slot, a blockage might occur
            # When genie-aided comment this
            if self.counter_slots_blocked[i] == 0:
                if bernoulli(0.3):
                    self.counter_slots_blocked[i] = np.random.randint(1, 6)

            velocity = sample_velocity()
            self.user_positions[i] = update_position(self.user_positions[i], velocity, 0.1, self.config[i])

            self.d[i] = np.linalg.norm(self.base_station_position - self.user_positions[
                i])  # euclidean distance between the BS and the UEs

            self.AoD[i] = compute_AoD_at_BS(self.base_station_position, self.user_positions[i])  # current AoD

            self.a_BS[i] = compute_ULA_response(self.number_of_antennas,
                                                self.AoD[i])  # current array response
            if self.counter_slots_blocked[i] != 0:
                self.mmWave_LoS_NLoS_channel[i] = compute_mmWave_LoS_NLoS_channel(self.number_of_antennas, self.a_BS[i],
                                                                        self.d[i], 0.0032)  # current NLoS channel
            else:
                self.mmWave_LoS_NLoS_channel[i] = compute_mmWave_LoS_NLoS_channel(self.number_of_antennas, self.a_BS[i],
                                                                        self.d[i]) # current LoS channel
            self.R[i] = compute_power_given_beam(self.Pt, self.mmWave_LoS_NLoS_channel[i],
                                                 self.codebook[:, self.current_beam[i]])
            self.previous_power[self.previous_counter][i] = self.R[i].copy()
            best_power_index[i] = np.argmax(np.abs(np.sqrt(self.Pt) * self.mmWave_LoS_NLoS_channel[i] @ self.codebook))
            best_power[i] = compute_power_given_beam(self.Pt, self.mmWave_LoS_NLoS_channel[i],
                                                     self.codebook[:, best_power_index[i]])

            # In every slot, a new packet might arrive for DL or UL traffic according to a Bernouilli random variable
            if i == 0:
                packet[i] = bernoulli(0.6)
            elif i == 1:
                packet[i] = bernoulli(0.4)
            else:
                packet[i] = bernoulli(0.3)

            if packet[i]:
                if bernoulli(0.5):  # If 1, downlink packet arrived
                    self.current_traffic[i][0] += 1  # Donwlink packet arrived
                    self.total_traffic[i][0] += 1
                    self.total_traffic_episode[i][0] += 1
                else:
                    self.current_traffic[i][1] += 1  # Uplink packet arrived
                    self.total_traffic[i][1] += 1
                    self.total_traffic_episode[i][1] += 1
                    # Define weights for each component

        already_checked = False
        for i in range(self.N_UEs):
            # When genie-aided uncomment
            # self.R[i] = best_power[i]
            if not already_checked and (action == 0 or action == 3 or action == 6):
                already_checked = True
                if action == 0:
                    UE = 0
                elif action == 3:
                    UE = 1
                elif action == 6:
                    UE = 2
                selected = self.current_beam[UE]
                for j in range(3):
                    if j < 1:
                        neigh_beam = (self.current_beam[UE] - (j + 1)) % len(self.codebook)
                        if compute_power_given_beam(self.Pt, self.mmWave_LoS_NLoS_channel[UE],
                                                    self.codebook[:, neigh_beam]) > compute_power_given_beam(self.Pt,
                                                                                                             self.mmWave_LoS_NLoS_channel[
                                                                                                                 UE],
                                                                                                             self.codebook[
                                                                                                             :,
                                                                                                             selected]):
                            selected = neigh_beam
                    elif j > 1:
                        neigh_beam = (self.current_beam[UE] + (j - 1)) % len(self.codebook)
                        if compute_power_given_beam(self.Pt, self.mmWave_LoS_NLoS_channel[UE],
                                                    self.codebook[:, neigh_beam]) > compute_power_given_beam(self.Pt,
                                                                                                             self.mmWave_LoS_NLoS_channel[
                                                                                                                 UE],
                                                                                                             self.codebook[
                                                                                                             :,
                                                                                                             selected]):
                            selected = neigh_beam
                if compute_power_given_beam(self.Pt, self.mmWave_LoS_NLoS_channel[UE],
                                            self.codebook[:, selected]) > compute_power_given_beam(self.Pt,
                                                                                                   self.mmWave_LoS_NLoS_channel[
                                                                                                       UE],
                                                                                                   self.codebook[:,
                                                                                                   self.current_beam[
                                                                                                       UE]]):
                    self.current_beam[UE] = selected
                    self.R[UE] = compute_power_given_beam(self.Pt, self.mmWave_LoS_NLoS_channel[UE],
                                                          self.codebook[:, self.current_beam[UE]])
                    reward += 3
                else:
                    reward -= 1

            if i == 0 and action == 1 and self.R[0] == best_power[0] and self.current_traffic[i][0] != 0 and self.counter_slots_blocked[i] == 0:
                self.current_traffic[i][0] -= 1
                self.th[self.current_slot] += 1
                self.th_episode[self.current_slot] += 1
            elif i == 0 and action == 1 and (self.R[0] != best_power[0] or self.counter_slots_blocked[0] != 0) and self.current_traffic[i][0] != 0:
                self.lost_traffic[i][0] += 1
                self.lost_traffic_episode[i][0] += 1
                if self.R[0] != best_power[0]:
                    self.failure_type[1] += 1
                elif self.counter_slots_blocked[0] != 0:
                    self.failure_type[2] += 1

                self.current_traffic[i][0] -= 1
                drop += 1
                if self.counter_slots_blocked[0] != 0:
                    reward -= 1
            if i == 0 and action == 2 and self.R[0] == best_power[0] and self.current_traffic[i][1] != 0 and self.counter_slots_blocked[i] == 0:
                self.current_traffic[i][1] -= 1
                self.th[self.current_slot] += 1
                self.th_episode[self.current_slot] += 1
            elif i == 0 and action == 2 and (self.R[0] != best_power[0] or self.counter_slots_blocked[0] != 0) and self.current_traffic[i][1] != 0:
                self.lost_traffic[i][1] += 1
                self.lost_traffic_episode[i][1] += 1
                if self.R[0] != best_power[0]:
                    self.failure_type[1] += 1
                elif self.counter_slots_blocked[0] != 0:
                    self.failure_type[2] += 1
                self.current_traffic[i][1] -= 1
                drop += 1
                if self.counter_slots_blocked[0] != 0:
                    reward -= 1

            if i == 1 and action == 4 and self.R[1] == best_power[1] and self.current_traffic[i][0] != 0 and self.counter_slots_blocked[i] == 0:
                self.current_traffic[i][0] -= 1
                self.th[self.current_slot] += 1
                self.th_episode[self.current_slot] += 1
            elif i == 1 and action == 4 and (self.R[1] != best_power[1] or self.counter_slots_blocked[1] != 0) and self.current_traffic[i][0] != 0:
                self.lost_traffic[i][0] += 1
                self.lost_traffic_episode[i][0] += 1
                if self.R[1] != best_power[1]:
                    self.failure_type[1] += 1
                elif self.counter_slots_blocked[1] != 0:
                    self.failure_type[2] += 1
                self.current_traffic[i][0] -= 1
                drop += 1
                if self.counter_slots_blocked[1] != 0:
                    reward -= 1

            if i == 1 and action == 5 and self.R[1] == best_power[1] and self.current_traffic[i][1] != 0 and self.counter_slots_blocked[i] == 0:
                self.current_traffic[i][1] -= 1
                self.th[self.current_slot] += 1
                self.th_episode[self.current_slot] += 1
            elif i == 1 and action == 5 and (self.R[1] != best_power[1] or self.counter_slots_blocked[1] != 0) and self.current_traffic[i][1] != 0:
                self.lost_traffic[i][1] += 1
                self.lost_traffic_episode[i][1] += 1
                if self.R[1] != best_power[1]:
                    self.failure_type[1] += 1
                elif self.counter_slots_blocked[1] != 0:
                    self.failure_type[2] += 1
                self.current_traffic[i][1] -= 1
                drop += 1
                if self.counter_slots_blocked[1] != 0:
                    reward -= 1

            if i == 2 and action == 7 and self.R[2] == best_power[2] and self.current_traffic[i][0] != 0 and self.counter_slots_blocked[i] == 0:
                self.current_traffic[i][0] -= 1
                self.th[self.current_slot] += 1
                self.th_episode[self.current_slot] += 1
            elif i == 2 and action == 7 and (self.R[2] != best_power[2] or self.counter_slots_blocked[2] != 0) and self.current_traffic[i][0] != 0:
                self.lost_traffic[i][0] += 1
                self.lost_traffic_episode[i][0] += 1
                if self.R[2] != best_power[2]:
                    self.failure_type[1] += 1
                elif self.counter_slots_blocked[2] != 0:
                    self.failure_type[2] += 1
                self.current_traffic[i][0] -= 1
                drop += 1
                if self.counter_slots_blocked[2] != 0:
                    reward -= 1

            if i == 2 and action == 8 and self.R[2] == best_power[2] and self.current_traffic[i][1] != 0 and self.counter_slots_blocked[i] == 0:
                self.current_traffic[i][1] -= 1
                self.th[self.current_slot] += 1
                self.th_episode[self.current_slot] += 1
            elif i == 2 and action == 8 and (self.R[2] != best_power[2] or self.counter_slots_blocked[2] != 0) and self.current_traffic[i][1] != 0:
                self.lost_traffic[i][1] += 1
                self.lost_traffic_episode[i][1] += 1
                if self.R[2] != best_power[2]:
                    self.failure_type[1] += 1
                elif self.counter_slots_blocked[2] != 0:
                    self.failure_type[2] += 1
                self.current_traffic[i][1] -= 1
                drop += 1
                if self.counter_slots_blocked[2] != 0:
                    reward -= 1

            if self.current_traffic[i][0] > self.dl_buffer_size:
                self.lost_traffic[i][0] += 1
                self.lost_traffic_episode[i][0] += 1
                self.failure_type[0] += 1
                self.current_traffic[i][0] = self.dl_buffer_size
                drop += 1
                was_dropped[i][0] += 1
            if self.current_traffic[i][1] > self.ul_buffer_size:
                self.lost_traffic[i][1] += 1
                self.lost_traffic_episode[i][1] += 1
                self.failure_type[0] += 1
                self.current_traffic[i][1] = self.ul_buffer_size
                drop += 1
                was_dropped[i][1] += 1

        for l in range(self.N_UEs):
            if self.counter_slots_blocked[l] != 0:
                self.counter_slots_blocked[l] -= 1
            for m in range(2):
                if self.current_traffic[l][m] < self.dl_buffer_size and was_dropped[l][m] == 0:
                    reward += 1/6
        reward -= drop/4
        self.r_p.append(drop)
        self.r_h.append(reward)
        # print("Reward:" + str(reward))
        # print("Actions: " + str(action))
        self.current_slot += 1
        self.previous_counter += 1
        
        if self.previous_counter == 5:
            self.previous_counter = 0
        if self.current_slot == self.N_f:
            done = True
            print(self.current_traffic)
        return self._get_state(), reward, done, {'ue_position': self.user_positions.copy(),
                                                 'bs_position': self.base_station_position.copy(),
                                                 'lost_traffic': self.lost_traffic.copy(),
                                                 'lost_traffic_episode': self.lost_traffic_episode.copy(),
                                                 'total_traffic': self.total_traffic.copy(),
                                                 'total_traffic_episode': self.total_traffic_episode.copy(),
                                                 'failure_type': self.failure_type.copy(),
                                                 'tx_packets': self.th.copy(),
                                                 'tx_packets_episode': self.th_episode.copy()}

    def _get_state(self):
        return (self.current_beam[0] / (self.number_of_antennas - 1),
                self.current_beam[1] / (self.number_of_antennas - 1),
                self.current_beam[2] / (self.number_of_antennas - 1), self.current_traffic[0][0] / self.dl_buffer_size,
                self.current_traffic[0][1] / self.dl_buffer_size, self.current_traffic[1][0] / self.dl_buffer_size,
                self.current_traffic[1][1] / self.dl_buffer_size,
                self.current_traffic[2][0] / self.dl_buffer_size,
                self.current_traffic[2][1] / self.dl_buffer_size, self.R[0], self.R[1], self.R[2])

register(id='BeamManagementEnv-v1', entry_point=BeamManagementEnv)
