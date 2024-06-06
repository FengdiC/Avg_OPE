import os, sys, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
import numpy as np
import csv
import main.train as train_mse
import gamma.train as train_gamma
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

def run_classic():
    args = argsparser()
    seeds = range(10)

    discount_factor = [0.8, 0.99, 0.995]
    buffer = [40, 80, 200]
    random_weight = [0.3, 0.5, 0.7]
    env = ['CartPole-v1','Acrobot-v1','MountainCarContinuous-v0']
    path = ['./exper/cartpole.pth','./exper/acrobot.pth', './exper/mountaincar.pth']
    idx = np.unravel_index(args.array, (3, 3, 3, 3))
    random_weight, buffer, discount_factor = random_weight[idx[0]], buffer[idx[1]], discount_factor[idx[2]]
    env, path = env[idx[3]], path[idx[3]]
    batch, link, alpha, lr, loss = classic(buffer,random_weight)

    filename = args.log_dir + 'final-result-' + str(env) +'-array-'+str(args.array) + '.csv'
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

    # result = np.array(result)
    # ret = np.around(np.mean(result, axis=0), decimals=4)
    # var = np.around(np.var(result, axis=0), decimals=4)
    # print("Mean shape: ", ret.shape, ":::", var.shape)
    # name = ['lr', lr, 'alpha', alpha]
    # name = [str(s) for s in name]
    # name_1 = name + ['mean']
    # name_2 = name + ['var']
    # mylist = [str(i) for i in list(ret)] + ['-'.join(name_1)]
    # with open(filename, 'a', newline='') as file:
    #     # Step 4: Using csv.writer to write the list to the CSV file
    #     writer = csv.writer(file)
    #     writer.writerow(mylist)  # Use writerow for single list
    # print('-'.join(name_1))

run_classic()