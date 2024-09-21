import os, sys, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
import numpy as np
import torch
# from avg_corr.main import collect_dataset
import gym
import matplotlib.pyplot as plt
import itertools

"""
Compute this per-step error under a fixed setting : 

random_weight, buffer, discount_factor = 0.5, 80, 0.95

1. To use the code, first resample two large datasets:

buf_td = collect_dataset(env, gamma, buffer_size=2000, max_len=50, path=path,
                      random_weight=random_weight, fold=1)
second_buf_td = collect_dataset(env, gamma, buffer_size=2000, max_len=50,
                           path=path, random_weight=random_weight, fold=1)

2. Create the TD_computation

TD_err = TD_computation(buf_td,second_buf_td,gamma)

3. Call it during training

td_err = TD_err.compute(weight)

"""

# def plot_state_range():
#     env = gym.make('CartPole-v1')
#     buf = collect_dataset(env, 0.95, buffer_size=80, max_len=50, path='./exper/cartpole.pth',
#                           random_weight=0.5, fold=1)
#     states = buf.obs_buf
#     interval = [0.09/5,0.6/5,0.08/5,0.55/5]
#     interval = [0.12 / 10, 0.8 / 10, 0.1 / 10, 0.65 / 10]
#     interval = np.tile(np.array(interval),(states.shape[0],1))
#     # bottom = [-2.4,-10.0,-0.2095,-10.0]
#     # bottom = np.tile(np.array(bottom),(1,states.shape[0]))
#     index = (states / interval).astype(np.int32)
#     index = np.where(index < -10, -11 * np.ones(index.shape), index)
#     index = np.where(index > 9, 9 * np.ones(index.shape), index)
#
#     plt.subplot(411)
#     unique, counts = np.unique(index[:,0], return_counts=True, axis=0)
#     plt.bar(unique,counts)
#     plt.subplot(412)
#     unique, counts = np.unique(index[:, 1], return_counts=True, axis=0)
#     plt.bar(unique, counts)
#     plt.subplot(413)
#     unique, counts = np.unique(index[:, 2], return_counts=True, axis=0)
#     plt.bar(unique, counts)
#     plt.subplot(414)
#     unique, counts = np.unique(index[:, 3], return_counts=True, axis=0)
#     plt.bar(unique, counts)
#     plt.show()

class TD_computation():
    def __init__(self,buf,second_buf,gamma):
        self.buf, self.second.buf, self.gamma = buf,second_buf,gamma

        self.next_states = buf.next_obs_buf
        # interval = [0.09/5,0.6/5,0.08/5,0.55/5]
        interval = [0.12 / 10, 0.8 / 10, 0.1 / 10, 0.65 / 10]
        interval_tile = np.tile(np.array(interval), (self.next_states.shape[0], 1))
        # bottom = [-0.09,-0.6,-0.08,-0.55]
        bottom = [-0.12, -0.8, -0.1, -0.65]
        bottom_tile = np.tile(np.array(bottom), (self.next_states.shape[0], 1))
        index = ((self.next_states - bottom_tile) / interval_tile).astype(np.int32)
        index = np.where(index < 0, -1 * np.ones(index.shape), index)
        # index = np.where(index > 9, 10 * np.ones(index.shape), index).astype(np.int32)
        index = np.where(index > 19, 20 * np.ones(index.shape), index).astype(np.int32)
        index += 1
        index = tuple([index[:, i] for i in range(4)])
        # index = np.ravel_multi_index(index, (12,12,12,12),order='C')
        self.index = np.ravel_multi_index(index, (22, 22, 22, 22), order='C')

        self.unique, self.counts = np.unique(self.index, return_counts=True)

        # print(np.mean(counts))
        # print(unique.shape, "::", np.max(counts), ":::", np.sort(counts)[-15:-7], ":::", np.mean(counts))

        # CartPole has observations of dimension four
        # states are initialized randomly among the interval -0.05 to 0.05
        initial_states = [np.arange(-0.05, 0.05, interval[i]).tolist() for i in range(4)]

        prob = []
        for i in range(4):
            if len(initial_states[i]) == 1:
                pos = [1]
            else:
                pos = [(-0.05 + interval[i] - initial_states[i][0]) / 0.1]
                for _ in range(len(initial_states[i]) - 2):
                    pos.append(interval[i] / 0.1)
                pos.append((0.05 - initial_states[i][-1]) / 0.1)
            prob.append(pos)
        initial_states = np.array([elm for elm in itertools.product(*initial_states)])
        prob = np.array([elm for elm in itertools.product(*prob)])
        self.prob = np.prod(prob, axis=1)

        interval_tile = np.tile(np.array(interval), (initial_states.shape[0], 1))
        bottom_tile = np.tile(np.array(bottom), (initial_states.shape[0], 1))
        index = ((initial_states - bottom_tile) / interval_tile).astype(np.int32)
        # index = np.where(index > 9, 10 * np.ones(index.shape), index).astype(np.int32)
        index = np.where(index > 19, 20 * np.ones(index.shape), index).astype(np.int32)
        index += 1
        index = tuple([index[:, i] for i in range(4)])
        # index = np.ravel_multi_index(index, (12,12,12,12),order='C')
        self.initial_index = np.ravel_multi_index(index, (22, 22, 22, 22), order='C')

        # NEXT step is to use a second buffer to average over all values
        self.sec_next_states = second_buf.next_obs_buf
        interval_tile = np.tile(np.array(interval), (self.sec_next_states.shape[0], 1))
        bottom_tile = np.tile(np.array(bottom), (self.sec_next_states.shape[0], 1))
        index = ((self.sec_next_states - bottom_tile) / interval_tile).astype(np.int32)
        index = np.where(index < -0, -1 * np.ones(index.shape), index)
        # index = np.where(index > 9, 10 * np.ones(index.shape), index).astype(np.int32)
        index = np.where(index > 19, 20 * np.ones(index.shape), index).astype(np.int32)
        index += 1
        index = tuple([index[:, i] for i in range(4)])

        # index = np.ravel_multi_index(index, (12,12,12,12),order='C')
        self.sec_index = np.ravel_multi_index(index, (22, 22, 22, 22), order='C')
        self.sec_unique, self.sec_counts = np.unique(self.sec_index, return_counts=True)

    def compute(self,weight=None):
        # current dataset error
        # discretize next state to 20**4
        # record the info of next states
        # [4.8 inf 0.42 inf]

        values = np.zeros(22 ** 4)

        ratio = weight(torch.as_tensor(self.buf.obs_buf, dtype=torch.float32)).detach().numpy()
        next_ratio = weight(torch.as_tensor(self.next_states, dtype=torch.float32)).detach().numpy()
        # ratio = np.concatenate((0.3*np.ones(next_states.shape[0]//2),5*np.ones(next_states.shape[0]//2)))
        # next_ratio = np.concatenate((0.5 * np.ones(next_states.shape[0] //2), 2* np.ones(next_states.shape[0] // 2)))
        IS = np.exp(self.buf.logtarg_buf - self.buf.logbev_buf)

        for i in range(self.next_states.shape[0]):
            values[self.index[i]] += self.gamma * ratio[i] * IS[i]-next_ratio[i]
        values[self.unique] = values[self.unique]/self.counts

        values[self.initial_index] = (1-self.gamma) * self.prob
        err = np.sum((values[self.sec_unique]**2)*self.sec_counts) / self.sec_next_states.shape[0]

        return err


# plot_state_range()

# env = gym.make('CartPole-v1')
# buf = collect_dataset(env, 0.95, buffer_size=4000, max_len=50, path='./exper/cartpole.pth',
#                       random_weight=0.5, fold=1)
# second_buf = collect_dataset(env, 0.95, buffer_size=4000, max_len=50, path='./exper/cartpole.pth',
#                       random_weight=0.5, fold=1)
# print(temporal_error(buf,second_buf,0.95))