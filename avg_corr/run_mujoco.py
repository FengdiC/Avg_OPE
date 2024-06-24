import os, sys, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
import numpy as np
import csv
from avg_corr.main import train as train_mse
from avg_corr.gamma import train as train_gamma
from arguments import classic

def argsparser():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default='./')
    parser.add_argument('--log_dir', type=str, default='./')
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

    discount_factor = [0.8, 0.9,0.95, 0.99, 0.995]
    buffer = [40, 80, 200]
    random_weight = [0.3, 0.5, 0.7]
    env = ['MountainCarContinuous-v0','Hopper-v4','HalfCheetah-v4','HalfCheetah-v4','Ant-v4',
           'Swimmer-v4','Walker2d-v4']
    path = ['./exper/mountaincar.pth','./exper/hopper.pth','./exper/halfcheetah_0.pth',
            './exper/halfcheetah_1.pth','./exper/ant.pth','./exper/swimmer.pth','./exper/walker.pth']
    idx = np.unravel_index(args.array, (3, 3, 5, 3))
    random_weight, buffer, discount_factor = random_weight[idx[0]], buffer[idx[1]], discount_factor[idx[2]]
    env, path = env[idx[3]], path[idx[3]]
    batch, link, alpha, lr, loss = 512,'inverse',0.0005,0.0001,'gamma'

    filename = args.log_dir + 'final-mujoco-' + str(env) +'-discount-'+str(discount_factor)\
               +'-buffer-'+str(buffer)+'-random-'+str(random_weight)+'.csv'
    os.makedirs(args.log_dir, exist_ok=True)
    mylist = [str(i) for i in range(0, args.epoch * args.steps, args.steps)] + ['hyperparam']
    with open(filename, 'w+', newline='') as file:
        # Step 4: Using csv.writer to write the list to the CSV file
        writer = csv.writer(file)
        writer.writerow(mylist)  # Use writerow for single list

    result_train, result_test = [], []
    for seed in seeds:
        if loss=='mse':
            train, test = train_mse(lr=lr, env=env, seed=seed, path=path, hyper_choice=args.seed,
                   link=link, random_weight=random_weight, l1_lambda=alpha, discount = discount_factor,
                   checkpoint=args.steps, epoch=args.epoch, cv_fold=1,
                   batch_size=batch, buffer_size=buffer,
                   max_len=args.max_len)
        elif loss=='gamma':
            train, test = train_gamma(lr=lr, env=env, seed=seed, path=args.path, hyper_choice=args.seed,
                   link=link, random_weight=random_weight, l1_lambda=alpha, discount = discount_factor,
                   checkpoint=args.steps, epoch=args.epoch, cv_fold=1,
                   batch_size=batch, buffer_size=buffer,
                   max_len=args.max_len)
        result_train.append(train)
        result_test.append(test)
        mylist = [str(i) for i in list(train)] + ['-'.join(['train','seed',str(seed)])]
        with open(filename, 'a', newline='') as file:
            # Step 4: Using csv.writer to write the list to the CSV file
            writer = csv.writer(file)
            writer.writerow(mylist)  # Use writerow for single list
        mylist = [str(i) for i in list(test)] + ['-'.join(['test', 'seed', str(seed)])]
        with open(filename, 'a', newline='') as file:
            # Step 4: Using csv.writer to write the list to the CSV file
            writer = csv.writer(file)
            writer.writerow(mylist)  # Use writerow for single list

run_mujoco()
