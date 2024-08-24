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

def run_classic():
    args = argsparser()
    seeds = range(10)

    random_weight, buffer, discount_factor = 0.5, 80, 0.95
    env,path = 'CartPole-v1','./exper/cartpole.pth'

    batch, link, alpha, lr, loss, reg_lambda = 512,'identity',0.001,0.0001,'mse', 2

    filename = args.log_dir + 'cartpole-mse-td-err.csv'
    os.makedirs(args.log_dir, exist_ok=True)
    mylist = [str(i) for i in range(0, args.epoch * args.steps, args.steps)] + ['hyperparam']
    with open(filename, 'w+', newline='') as file:
        # Step 4: Using csv.writer to write the list to the CSV file
        writer = csv.writer(file)
        writer.writerow(mylist)  # Use writerow for single list

    result_train, result_test, result_err = [], [], []
    for seed in seeds:
        if loss=='mse':
            print("loss: mse!")
            train, test,err = train_mse(lr=lr, env=env, seed=seed, path=path, hyper_choice=args.seed,
                   link=link, random_weight=random_weight, l1_lambda=alpha,reg_lambda=reg_lambda,
                   discount = discount_factor,
                   checkpoint=args.steps, epoch=args.epoch, cv_fold=1,
                   batch_size=batch, buffer_size=buffer,
                   max_len=args.max_len)
        elif loss=='gamma':
            print("ERROR: NOT MSE LOSS!")
        train,test, err = np.around(train,decimals=4),np.around(test,decimals=4), np.around(err,decimals=4)
        result_train.append(train)
        result_test.append(test)
        result_err.append(err)
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
        mylist = [str(i) for i in list(err)] + ['-'.join(['err', 'seed', str(seed)])]
        with open(filename, 'a', newline='') as file:
            # Step 4: Using csv.writer to write the list to the CSV file
            writer = csv.writer(file)
            writer.writerow(mylist)  # Use writerow for single list

    train = np.around(np.mean(np.array(result_train),axis=0),decimals=4)
    test = np.around(np.mean(np.array(result_test),axis=0),decimals=4)
    err = np.around(np.mean(np.array(result_err), axis=0), decimals=4)
    mylist = [str(i) for i in list(train)] + ['-'.join(['train', 'mean'])]
    with open(filename, 'a', newline='') as file:
        # Step 4: Using csv.writer to write the list to the CSV file
        writer = csv.writer(file)
        writer.writerow(mylist)  # Use writerow for single list
    mylist = [str(i) for i in list(test)] + ['-'.join(['test', 'mean'])]
    with open(filename, 'a', newline='') as file:
        # Step 4: Using csv.writer to write the list to the CSV file
        writer = csv.writer(file)
        writer.writerow(mylist)  # Use writerow for single list
    mylist = [str(i) for i in list(test)] + ['-'.join(['err', 'mean'])]
    with open(filename, 'a', newline='') as file:
        # Step 4: Using csv.writer to write the list to the CSV file
        writer = csv.writer(file)
        writer.writerow(mylist)  # Use writerow for single list


run_classic()
