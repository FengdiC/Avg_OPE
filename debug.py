import numpy as np
import pickle
from avg_corr.main import train as train_mse, PPOBuffer

env_lists = [
    'CartPole-v1', 'Hopper-v4',
    'HalfCheetah-v4', 'Ant-v4',
    'Walker2d-v4',
    'Acrobot-v1',
]

biased_obj = {}
for env_id in env_lists:
    biased_obj[env_id] = {'train':{}, 'test':{}}


env_lists = ['CartPole-v1', 'Acrobot-v1']
seeds = range(10)

data_dir = './dataset/'

discount_factor_lists = [0.8, 0.9,0.95, 0.99, 0.995]
size_lists = [2000, 4000, 8000, 16000]

weight_lists = [0.1, 0.2, 0.3, 0.4, 0.5]
length_lists = [20, 40, 80, 100]

for array in range(18):
    random_weight, length, discount_factor, buffer_size = (
        0.3,
        40,
        0.95,
        4000,
    )
    if array < 5:
        discount_factor = discount_factor_lists[array]
    elif array < 9:
        buffer_size = size_lists[array - 5]
        if buffer_size == 4000:
            continue
    elif array < 14:
        random_weight = weight_lists[array - 9]
        if random_weight == 0.3:
            continue
    else:
        length = length_lists[array - 14]
        if length == 40:
            continue

    for seed in seeds:
        for env in env_lists:
            name = ['discount_factor', 0.8, 'random_weight', random_weight, 'max_length', length,
                        'buffer_size', 16000, 'seed', seed, 'env', env]
            name = '-'.join(str(x) for x in name)

            with open(data_dir +'/dataset/'+ name + '.pkl', 'rb') as outp:
                print(data_dir +'/dataset/'+ name + '.pkl')
                buf =  pickle.load(outp)
            name = ['discount_factor', 0.8, 'random_weight', random_weight, 'max_length', length,
                    'buffer_size', 16000, 'seed', seed+1314, 'env', env]
            name = '-'.join(str(x) for x in name)

            with open(data_dir+'/dataset_test/' + name + '.pkl', 'rb') as outp:
                buf_test = pickle.load(outp)

            name = ['discount_factor', discount_factor, 'random_weight', random_weight, 'max_length', length,
                    'buffer_size',buffer_size,'seed',seed]
            name = '-'.join(str(x) for x in name)
            obj = (1-discount_factor) *np.mean(buf.rew_buf[:buf.ptr])
            obj_test = (1 - discount_factor) * np.mean(buf_test.rew_buf[:buf_test.ptr])

            biased_obj[env]['train'][name] = obj
            biased_obj[env]['test'][name] = obj_test

####################
"""
MuJoCo
"""
env_lists = [
    'Hopper-v4',
    'HalfCheetah-v4', 'Ant-v4',
    'Walker2d-v4',
]
discount_factor_lists = [0.8, 0.9, 0.95, 0.99, 0.995]
size_lists = [2000, 4000, 8000, 16000]

weight_lists = [1.4, 1.8, 2.0, 2.4, 2.8]
length_lists = [20, 50, 100, 200]

for array in range(18):
    random_weight, length, discount_factor, buffer_size = (
        2.0,
        100,
        0.95,
        4000,
    )
    if array < 5:
        discount_factor = discount_factor_lists[array]
    elif array < 9:
        buffer_size = size_lists[array - 5]
        if buffer_size == 4000:
            continue
    elif array < 14:
        random_weight = weight_lists[array - 9]
        if random_weight == 2.0:
            continue
    else:
        length = length_lists[array - 14]
        if length == 100:
            continue
    for seed in seeds:
        for env in env_lists:
            name = ['discount_factor', 0.8, 'random_weight', random_weight, 'max_length', length,
                    'buffer_size', 16000, 'seed', seed, 'env', env]
            name = '-'.join(str(x) for x in name)

            with open(data_dir + '/dataset/' + name + '.pkl', 'rb') as outp:
                print(data_dir + '/dataset/' + name + '.pkl')
                buf = pickle.load(outp)
            name = ['discount_factor', 0.8, 'random_weight', random_weight, 'max_length', length,
                    'buffer_size', 16000, 'seed', seed + 1314, 'env', env]
            name = '-'.join(str(x) for x in name)

            with open(data_dir + '/dataset_test/' + name + '.pkl', 'rb') as outp:
                buf_test = pickle.load(outp)

            name = ['discount_factor', discount_factor, 'random_weight', random_weight, 'max_length', length,
                    'buffer_size', buffer_size, 'seed', seed]
            name = '-'.join(str(x) for x in name)
            obj = (1 - discount_factor) * np.mean(buf.rew_buf[:buf.ptr])
            obj_test = (1 - discount_factor) * np.mean(buf_test.rew_buf[:buf_test.ptr])

            biased_obj[env]['train'][name] = obj
            biased_obj[env]['test'][name] = obj_test

with open(data_dir+'biased_obj.pkl', 'wb') as outp:
    pickle.dump(biased_obj, outp, pickle.HIGHEST_PROTOCOL)