import os, sys, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from avg_corr.main import eval_policy
import _pickle as pickle


def plot_algo(gamma,buffer,random_weight,env,train='train'):
    result = []
    # plot the result for our avg algorithm
    for filename in os.listdir('./tune_log/classic'):
        f = os.path.join('./tune_log/classic/', filename)
        # checking if it is a file
        if not f.endswith('.csv'):
            continue
        if '-'+str(gamma)+'-' in filename and str(buffer) in filename and str(random_weight) in filename \
                and str(env) in filename and 'mse' in filename:
            data = pd.read_csv(f, header=0, index_col='hyperparam')
            data.columns = data.columns.astype(int)
            data = data.sort_index(axis=1, ascending=True)
            for name in data.index.to_list():
                if train in name and 'mean' in name:
                    print(filename)
                    result.append(data.loc[name].to_list())
    mean_avg_mse = np.mean(result,axis=0)
    # var_avg = np.var(result, axis=0)

    for filename in os.listdir('./tune_log/classic'):
        f = os.path.join('./tune_log/classic/', filename)
        # checking if it is a file
        if not f.endswith('.csv'):
            continue
        if '-'+str(gamma)+'-' in filename and str(buffer) in filename and str(random_weight) in filename \
                and str(env) in filename and 'gamma' in filename:
            data = pd.read_csv(f, header=0, index_col='hyperparam')
            data.columns = data.columns.astype(int)
            data = data.sort_index(axis=1, ascending=True)
            for name in data.index.to_list():
                if train in name and 'mean' in name:
                    print(filename)
                    result.append(data.loc[name].to_list())
    mean_avg_gamma = np.mean(result,axis=0)
    # var_avg = np.var(result, axis=0)

    # plot best dice
    result = []
    for filename in os.listdir('./tune_log/bestdice_cartpole'):
        f = os.path.join('./tune_log/bestdice_cartpole', filename)
        # checking if it is a file
        if not f.endswith('.csv'):
            continue
        if '_'+"%.2f" % gamma+'_' in filename and 'numtraj_'+str(buffer) in filename \
                and str(random_weight) in filename and train in filename:
            data = pd.read_csv(f, header=0)
            mean_dice = data.loc[:,'MSE']
            result.append(mean_dice.to_list())
        mean_dice = np.mean(result,axis=0)

    plt.figure()
    plt.plot(range(mean_avg_mse.shape[0]), mean_avg_mse, label='avg_corr_mse')
    plt.plot(range(mean_avg_gamma.shape[0]), mean_avg_gamma, label='avg_corr_gamma')
    plt.plot(range(len(mean_dice)), mean_dice, label='best_dice')
    plt.plot(range(len(mean_dice)), 0.99951*np.ones(len(mean_dice)), label='true_value')
    plt.legend()
    plt.title('last')
    plt.show()

def plot_err():
    # plot the result for our avg algorithm
    for filename in os.listdir('./tune_log/cartpole_err'):
        f = os.path.join('./tune_log/cartpole_err/', filename)
        # checking if it is a file
        if not f.endswith('.csv'):
            continue
        data = pd.read_csv(f, header=0, index_col='hyperparam')
        data.columns = data.columns.astype(int)
        data = data.sort_index(axis=1, ascending=True)
        result = []
        for name in data.index.to_list():
            if 'err' in name and 'mean' not in name:
                result.append(data.loc[name].to_list())
        mean = np.mean(result,axis=0)
        std = np.std(result,axis=0)
        plt.subplot(311)
        plt.title('err')
        plt.fill_between(range(mean.shape[0]), mean + std, mean-std, alpha=0.1)
        plt.plot(range(mean.shape[0]),mean)

        result = []
        for name in data.index.to_list():
            if 'train' in name and 'mean' not in name:
                result.append(data.loc[name].to_list())
        mean = np.mean(result, axis=0)
        std = np.std(result, axis=0)
        plt.subplot(312)
        plt.title('train')
        plt.fill_between(range(mean.shape[0]), mean + std, mean - std, alpha=0.1)
        plt.plot(range(mean.shape[0]), mean)

        result = []
        for name in data.index.to_list():
            if 'test' in name and 'mean' not in name:
                result.append(data.loc[name].to_list())
        mean = np.mean(result, axis=0)
        std = np.std(result, axis=0)
        plt.subplot(313)
        plt.title('test')
        plt.fill_between(range(mean.shape[0]), mean + std, mean - std, alpha=0.1)
        plt.plot(range(mean.shape[0]), mean)

    plt.show()

plot_err()