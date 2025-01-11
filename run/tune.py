import os, sys, inspect, itertools

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
import numpy as np

import gymnasium as gym

import time, csv
import torch
from tqdm import tqdm

from avg_corr.main import train, argsparser
import pickle

path_lists = {
    'CartPole-v1':'./exper/cartpole.pth',
    'Acrobot-v1':'./exper/acrobot.pth',
    'MountainCarContinuous-v0':'./exper/mountaincar.pth',
    'Hopper-v4':'./exper/hopper.pth',
    'HalfCheetah-v4':'./exper/halfcheetah_0.pth',
    'Ant-v4':'./exper/ant.pth',
    'Swimmer-v4':'./exper/swimmer.pth',
    'Walker2d-v4':'./exper/walker.pth',
}

# train weight net
def tune_mse():

    env_lists = ['CartPole-v1', 'Acrobot-v1',
                 'MountainCarContinuous-v0', 'Hopper-v4',
                 'HalfCheetah-v4', 'Ant-v4',
                 'Walker2d-v4'
                 ]
    random_weight_lists = [0.3,0.3,
                           2.0,2.0,2.0,2.0,2.0]
    mujoco_lists = [False, False,
                    True, True, True, True, True]

    alpha = [0, 0.001, 0.01, 0.1]
    batch_size = 512
    link = 'identity'
    lr = [0.00005, 0.0001, 0.0005, 0.001, 0.005]
    reg_lambda = [0.5, 2, 10, 20 ,40]

    args = argsparser()
    seeds = range(5)
    idx = np.unravel_index(args.array, (4, 5, 5))
    buffer_size = 40
    discount_factor, max_len = 0.95, 100


    alpha, lr, reg_lambda = alpha[idx[0]], lr[idx[1]], reg_lambda[idx[2]]
    filename = args.log_dir + 'mse-tune--alpha-' + str(alpha) + '-lr-' \
               + str(lr) + '-lambda-' + str(reg_lambda) + '.csv'
    os.makedirs(args.log_dir, exist_ok=True)
    mylist = [str(i) for i in range(0, args.epoch * args.steps, args.steps)] + ['hyperparam']
    with open(filename, 'w+', newline='') as file:
        # Step 4: Using csv.writer to write the list to the CSV file
        writer = csv.writer(file)
        writer.writerow(mylist)  # Use writerow for single list

    for seed in tqdm(seeds, desc="Seeds"):
        for i in range(len(env_lists)):
            env_name, path, mujoco, random_weight = (
                env_lists[i],
                path_lists[env_lists[i]],
                mujoco_lists[i],
                random_weight_lists[i],
            )
            if mujoco:
                with open(args.data_dir+'mujoco_obj.pkl', 'rb') as file:
                    obj = pickle.load(file)
                true_obj = obj[env_name][2]
            else:
                with open(args.data_dir+'classic_obj.pkl', 'rb') as file:
                    obj = pickle.load(file)
                true_obj = obj[env_name][2]

            result = []
            result_val = []
            print("Finish one combination of hyperparameters!")
            cv, cv_val = train(lr=lr, env=env_name, seed=seed, path=path, hyper_choice=args.seed,
                               link=link, random_weight=random_weight, l1_lambda=alpha,
                               reg_lambda=reg_lambda, discount=discount_factor,
                               checkpoint=args.steps, epoch=args.epoch, cv_fold=5,
                               batch_size=batch_size, buffer_size=buffer_size,
                               max_len=max_len, mujoco=mujoco)

            print("Return result shape: ", len(cv), ":::", args.steps)
            result.append((cv-true_obj)**2)
            result_val.append((cv_val-true_obj)**2)
            name = ['seed', seed, 'env',env_name, 'train']
            name = [str(s) for s in name]
            cv = np.around(cv, decimals=4)
            mylist = [str(i) for i in list(cv)] + ['-'.join(name)]
            with open(filename, 'a', newline='') as file:
                # Step 4: Using csv.writer to write the list to the CSV file
                writer = csv.writer(file)
                writer.writerow(mylist)  # Use writerow for single list

            name = ['seed', seed, 'env',env_name, 'val']
            name = [str(s) for s in name]
            cv_val = np.around(cv_val, decimals=4)
            mylist = [str(i) for i in list(cv_val)] + ['-'.join(name)]
            with open(filename, 'a', newline='') as file:
                # Step 4: Using csv.writer to write the list to the CSV file
                writer = csv.writer(file)
                writer.writerow(mylist)  # Use writerow for single list

            torch.cuda.empty_cache()


if __name__ == "__main__":
    tune_mse()