import os, sys, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from avg_corr.main import eval_policy
import _pickle as pickle


def plot_algo(gamma,size,random_weight,length,true_obj,
              mujoco=True,env_name='Hopper-v4',env_id='hopper',train='train'):
    # plot the result for our avg algorithm

    if mujoco:
        log_dir = '../avg_tune_log/mujoco/'
        filename = f"final-mujoco-{env_name}-discount-{gamma}-length-{length}-random-{random_weight}.csv"
    else:
        log_dir = '../avg_tune_log/classic/'
        filename = f"final-classic-mse-{env_name}-discount-{gamma}-length-{length}-length-{random_weight}.csv"

    f = os.path.join(log_dir, filename)

    data = pd.read_csv(f, header=0, index_col='hyperparam')
    data.columns = data.columns.astype(int)
    data = data.sort_index(axis=1, ascending=True)
    result = []
    for name in data.index.to_list():
        if train in name and str(size) in name:
            result.append(data.loc[name].to_list())
    result = np.array(result)
    mean_avg_mse = np.mean(result,axis=0)
    var_avg = np.var(result, axis=0)

    # plot COP-TD
    result = []
    log_dir = "../avg_tune_log/COP-TD/results/results/" + env_id + "/"
    if mujoco:
        filename = f"mse-tune-random_weight_{random_weight}-discount_factor_{gamma}-max_ep_{size // length}-max_len_{length}-link_default-batch_size_512-bootstrap_target_target_network-lr_0.005-alpha_0.0.pkl"
    else:
        filename = f"mse-tune-random_weight_{random_weight}-discount_factor_{gamma}-max_ep_{size//length}-max_len_{length}-link_default-batch_size_512-bootstrap_target_target_network-lr_0.005-alpha_0.01.pkl"

    f = os.path.join(log_dir, filename)
    run_data = pickle.load(open(f, "rb"))
    for seed in run_data["seeds"]:
        result.append(np.array(run_data["results"][seed][1]))
    mean_cop_mse = np.mean(result, axis=0)
    var_cop = np.var(result, axis=0)

    # # plot best dice
    # result = []
    # log_dir = "../avg_tune_log/dice/" +  env_id + "/"
    # if mujoco:
    #     print("Not Implemented")
    # else:
    #     filename = f"dice-classic-['CartPole-v1', 'Acrobot-v1']-discount-{gamma}-length-{length}-random-{random_weight}.csv"
    # f = os.path.join(log_dir, filename)
    #
    # data = pd.read_csv(f, header=0, index_col='hyperparam')
    # data.columns = data.columns.astype(int)
    # data = data.sort_index(axis=1, ascending=True)
    # result = []
    # for name in data.index.to_list():
    #     if train in name and str(size) in name:
    #         result.append(data.loc[name].to_list())
    # result = np.array(result)
    # mean_dice_mse = np.mean(result, axis=0)
    # var_dice = np.var(result, axis=0)

    plt.figure()
    plt.plot(range(mean_avg_mse.shape[0]), mean_avg_mse, label='avg_corr_mse')
    plt.plot(range(mean_cop_mse.shape[0]), mean_cop_mse, label='cop_td')
    # plt.plot(range(mean_dice_mse.shape[0]), mean_dice_mse, label='best_dice')
    plt.plot(range(mean_avg_mse.shape[0]), true_obj*np.ones(mean_avg_mse.shape[0]), label='true_value')
    plt.legend()
    plt.title(env_name)
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

# plot_err()

if __name__ == "__main__":
    env_lists=['Hopper-v4',
           'HalfCheetah-v4','Ant-v4',
           'Walker2d-v4']
    env_id_lists = ['hopper',

                    'halfcheetah',
                    'ant',
                    'walker',
                    ]
    env_lists = ['CartPole-v1','Acrobot-v1']
    env_id_lists = ['cartpole','acrobot']
    discount_factor_lists = [0.8, 0.9, 0.95, 0.99, 0.995]
    size_lists = [2000, 4000, 8000, 16000]

    random_weight_lists = [1.4, 1.8, 2.0, 2.4, 2.8]
    length_lists = [20, 50, 100, 200]
    train = 'test'

    avg_mse, dice, cop = [], [], []
    for i in range(len(env_lists)):
        env_name = env_lists[i]
        env_id= env_id_lists[i]
        gamma = 0.8
        size, random_weight, length = 2000, 0.5,20
        with open('./dataset/classic_obj.pkl', 'rb') as file:
            obj = pickle.load(file)
        true_obj = obj[env_name][2]
        plot_algo(gamma = gamma,
                  size = size,
                  random_weight = random_weight,
                  length = length,
                  true_obj = true_obj,
                  mujoco=False,
                  env_name=env_name,
                  env_id=env_id,
                  train=train)