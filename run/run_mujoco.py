import os, sys, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
import numpy as np
import csv
from avg_corr.main import train as train_mse, PPOBuffer, WeightNet
from avg_corr.gamma import train as train_gamma
import pickle
import torch

def argsparser():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default='./')
    parser.add_argument('--log_dir', type=str, default='./')
    parser.add_argument('--data_dir', type=str, default='/scratch/fengdic/')
    parser.add_argument('--env', type=str, default='Hopper-v4')
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--steps', type=int, default=5)
    parser.add_argument('--epoch', type=int, default=250)
    parser.add_argument('--array', type=int, default=1)

    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--buffer_size', type=int, default=20)
    parser.add_argument('--max_len', type=int, default=50)
    parser.add_argument('--link', type=str, default='log')
    parser.add_argument('--l1_lambda', type=float, default=1.0)
    parser.add_argument('--random_weight', type=float, default=0.3)
    parser.add_argument('--true_value', type=float, default=1.0)
    args = parser.parse_args()
    return args

def run_mujoco():
    args = argsparser()
    seeds = range(10)
    seed = args.seed

    if args.array < 36:
        return -1

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

    idx = np.unravel_index(args.array - 36, (18, 5))
    if idx[0] < 5:
        discount_factor = discount_factor_lists[idx[0]]
    elif idx[0] < 9:
        size = size_lists[idx[0] - 5]
    elif idx[0] < 14:
        random_weight = weight_lists[idx[0] - 9]
    else:
        length = length_lists[idx[0] - 14]
    env, path = env[idx[1]], path_lists[env[idx[1]]]

    batch, link, alpha, lr, loss, reg_lambda = 512,'identity',0.001,0.0005,'mse', 0.5

    filename = args.log_dir + 'final-mujoco-' + str(env) +'-discount-'+str(discount_factor)\
               +'-length-'+str(length)+'-random-'+str(random_weight)+'-size-'+str(size) +'-seed-'+str(seed)+'.csv'
    os.makedirs(args.log_dir, exist_ok=True)
    mylist = [str(i) for i in range(0, args.epoch * args.steps, args.steps)] + ['hyperparam']
    with open(filename, 'w+', newline='') as file:
        # Step 4: Using csv.writer to write the list to the CSV file
        writer = csv.writer(file)
        writer.writerow(mylist)  # Use writerow for single list

    result_train, result_test = [], []
    name = ['discount_factor', 0.8, 'random_weight', random_weight, 'max_length', length,
            'buffer_size', 16000, 'seed', seed, 'env', env]
    name = '-'.join(str(x) for x in name)

    with open(args.data_dir+'/dataset/' + name + '.pkl', 'rb') as outp:
        buf = pickle.load(outp)
    name = ['discount_factor', 0.8, 'random_weight', random_weight, 'max_length', length,
            'buffer_size', 16000, 'seed', seed + 1314, 'env', env]
    name = '-'.join(str(x) for x in name)

    with open(args.data_dir +'/dataset_test/'+ name + '.pkl', 'rb') as outp:
        buf_test = pickle.load(outp)
    buf.ptr, buf.max_size = size, size
    buf_test.ptr, buf_test.max_size = size, size
    if loss == 'mse':
        print("loss: mse!")
        train, test, weight = train_mse(lr=lr, env=env, seed=seed, path=path, hyper_choice=args.seed,
                                link=link, random_weight=random_weight, l1_lambda=alpha,
                                buf=buf, buf_test=buf_test, reg_lambda=reg_lambda,
                                discount=discount_factor,
                                checkpoint=args.steps, epoch=args.epoch, cv_fold=1,
                                batch_size=batch, buffer_size=size // length,
                                max_len=length,mujoco=True)
    elif loss == 'gamma':
        print("loss: gamma!")
        train, test = train_gamma(lr=lr, env=env, seed=seed, path=args.path, hyper_choice=args.seed,
                                  link=link, random_weight=random_weight, buf=buf, buf_test=buf_test,
                                  l1_lambda=alpha, discount=discount_factor,
                                  checkpoint=args.steps, epoch=args.epoch, cv_fold=1,
                                  batch_size=batch, buffer_size=size // length,
                                  max_len=length,mujoco=True)
    train, test = np.around(train, decimals=4), np.around(test, decimals=4)
    result_train.append(train)
    result_test.append(test)
    mylist = [str(i) for i in list(train)] + ['-'.join(['train', 'seed', str(seed)])]
    with open(filename, 'a', newline='') as file:
        # Step 4: Using csv.writer to write the list to the CSV file
        writer = csv.writer(file)
        writer.writerow(mylist)  # Use writerow for single list
    mylist = [str(i) for i in list(test)] + ['-'.join(['test',  'seed', str(seed)])]
    with open(filename, 'a', newline='') as file:
        # Step 4: Using csv.writer to write the list to the CSV file
        writer = csv.writer(file)
        writer.writerow(mylist)  # Use writerow for single list

    torch.save(weight.state_dict(), args.log_dir + '/model-' + str(env) + '-discount-' + str(discount_factor) \
               + '-length-' + str(length) + '-length-' + str(random_weight) + '-size-' + str(size) + '-seed-' + str(
        seed) + '.pth')


run_mujoco()
