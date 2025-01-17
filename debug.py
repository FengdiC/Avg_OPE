import numpy as np
import pickle
from avg_corr.main import train as train_mse, PPOBuffer

for idx in range(0,18):

    discount_factor_lists = [0.8, 0.9, 0.95, 0.99, 0.995]
    size_lists = [2000, 4000, 8000, 16000]

    weight_lists = [1.4, 1.8, 2.0, 2.4, 2.8]
    length_lists = [20, 50, 100, 200]
    random_weight, length, discount_factor, size = (
        2.0,
        100,
        0.95,
        4000,
    )
    env = ['MountainCarContinuous-v0', 'Hopper-v4',
           'HalfCheetah-v4', 'Ant-v4',
           'Walker2d-v4']
    path_lists = {
        'CartPole-v1': './exper/cartpole.pth',
        'Acrobot-v1': './exper/acrobot.pth',
        'MountainCarContinuous-v0': './exper/mountaincar.pth',
        'Hopper-v4': './exper/hopper.pth',
        'HalfCheetah-v4': './exper/halfcheetah_0.pth',
        'Ant-v4': './exper/ant.pth',
        'Swimmer-v4': './exper/swimmer.pth',
        'Walker2d-v4': './exper/walker.pth',
    }

    if idx < 5:
        discount_factor = discount_factor_lists[idx]
    elif idx < 9:
        size = size_lists[idx - 5]
    elif idx < 14:
        random_weight = weight_lists[idx - 9]
    else:
        length = length_lists[idx - 14]
    print(idx,":: ", random_weight, length, discount_factor, size)
