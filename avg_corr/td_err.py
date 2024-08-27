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

def temporal_error(buf,second_buf,gamma,weight=None):
    # current dataset error
    # discretize next state to 20**4
    # record the info of next states
    # [4.8 inf 0.42 inf]
    values = np.zeros(22**4)

    next_states = buf.next_obs_buf
    # interval = [0.09/5,0.6/5,0.08/5,0.55/5]
    interval = [0.12 / 10, 0.8 / 10, 0.1 / 10, 0.65 / 10]
    interval_tile = np.tile(np.array(interval), (next_states.shape[0], 1))
    # bottom = [-0.09,-0.6,-0.08,-0.55]
    bottom = [-0.12,-0.8,-0.1,-0.65]
    bottom_tile = np.tile(np.array(bottom),(next_states.shape[0],1))
    index = ((next_states -bottom_tile) / interval_tile).astype(np.int32)
    index = np.where(index < 0, -1 * np.ones(index.shape), index)
    # index = np.where(index > 9, 10 * np.ones(index.shape), index).astype(np.int32)
    index = np.where(index > 19, 20 * np.ones(index.shape), index).astype(np.int32)
    index += 1
    index = tuple([index[:,i] for i in range(4)])

    # index = np.ravel_multi_index(index, (12,12,12,12),order='C')
    index = np.ravel_multi_index(index, (22,22,22,22), order='C')

    ratio = weight(torch.as_tensor(buf.obs_buf, dtype=torch.float32)).detach().numpy()
    next_ratio = weight(torch.as_tensor(buf.next_obs_buf, dtype=torch.float32)).detach().numpy()
    # ratio = np.concatenate((0.3*np.ones(next_states.shape[0]//2),5*np.ones(next_states.shape[0]//2)))
    # next_ratio = np.concatenate((0.5 * np.ones(next_states.shape[0] //2), 2* np.ones(next_states.shape[0] // 2)))
    IS = np.exp(buf.logtarg_buf - buf.logbev_buf)

    unique, counts = np.unique(index, return_counts=True)

    print(np.mean(counts))
    print(unique.shape, "::", np.max(counts), ":::", np.sort(counts)[-15:-7], ":::", np.mean(counts))

    for i in range(next_states.shape[0]):
        values[index[i]] += gamma * ratio[i] * IS[i]-next_ratio[i]
    values[unique] = values[unique]/counts

    # CartPole has observations of dimension four
    # states are initialized randomly among the interval -0.05 to 0.05
    initial_states = [np.arange(-0.05,0.05,interval[i]).tolist() for i in range(4)]

    prob = []
    for i in range(4):
        if len(initial_states[i])==1:
            pos = [1]
        else:
            pos = [(-0.05+interval[i]-initial_states[i][0])/0.1]
            for _ in range(len(initial_states[i])-2):
                pos.append(interval[i]/0.1)
            pos.append((0.05 - initial_states[i][-1])/0.1)
        prob.append(pos)
    initial_states = np.array([elm for elm in itertools.product(*initial_states)])
    prob = np.array([elm for elm in itertools.product(*prob)])
    prob = np.prod(prob,axis=1)

    interval_tile = np.tile(np.array(interval), (initial_states.shape[0], 1))
    bottom_tile = np.tile(np.array(bottom), (initial_states.shape[0], 1))
    index = ((initial_states - bottom_tile) / interval_tile).astype(np.int32)
    # index = np.where(index > 9, 10 * np.ones(index.shape), index).astype(np.int32)
    index = np.where(index > 19, 20 * np.ones(index.shape), index).astype(np.int32)
    index += 1
    index = tuple([index[:, i] for i in range(4)])
    # index = np.ravel_multi_index(index, (12,12,12,12),order='C')
    initial_index = np.ravel_multi_index(index, (22, 22, 22, 22), order='C')

    values[initial_index] = (1-gamma) * prob

    # NEXT step is to use a second buffer to average over all values
    next_states = second_buf.next_obs_buf
    interval_tile = np.tile(np.array(interval), (next_states.shape[0], 1))
    bottom_tile = np.tile(np.array(bottom), (next_states.shape[0], 1))
    index = ((next_states - bottom_tile) / interval_tile).astype(np.int32)
    index = np.where(index < -0, -1 * np.ones(index.shape), index)
    # index = np.where(index > 9, 10 * np.ones(index.shape), index).astype(np.int32)
    index = np.where(index > 19, 20 * np.ones(index.shape), index).astype(np.int32)
    index += 1
    index = tuple([index[:, i] for i in range(4)])

    # index = np.ravel_multi_index(index, (12,12,12,12),order='C')
    index = np.ravel_multi_index(index, (22, 22, 22, 22), order='C')
    unique, counts = np.unique(index, return_counts=True)
    err = np.sum((values[unique]**2)*counts) / next_states.shape[0]

    return err


# plot_state_range()

# env = gym.make('CartPole-v1')
# buf = collect_dataset(env, 0.95, buffer_size=4000, max_len=50, path='./exper/cartpole.pth',
#                       random_weight=0.5, fold=1)
# second_buf = collect_dataset(env, 0.95, buffer_size=4000, max_len=50, path='./exper/cartpole.pth',
#                       random_weight=0.5, fold=1)
# print(temporal_error(buf,second_buf,0.95))