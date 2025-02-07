import os, sys, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import _pickle as pickle
import csv

def setsizes():
    plt.rcParams['axes.linewidth'] = 1.0
    plt.rcParams['lines.markeredgewidth'] = 1.0
    plt.rcParams['lines.markersize'] = 3

    plt.rcParams['xtick.labelsize'] = 9.0
    plt.rcParams['ytick.labelsize'] = 9.0
    plt.rcParams['xtick.direction'] = "out"
    plt.rcParams['ytick.direction'] = "in"
    plt.rcParams['lines.linewidth'] = 2.0
    plt.rcParams['ytick.minor.pad'] = 50.0

def setaxes():
    plt.gcf().subplots_adjust(bottom=0.2)
    plt.gcf().subplots_adjust(left=0.2)
    ax = plt.gca()
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    # ax.spines['left'].set_color('none')
    # ax.xaxis.set_ticks_position('bottom')
    # ax.yaxis.set_ticks_position('left')
    # ax.tick_params(axis='both', direction='out', which='minor', width=2, length=3,
    #                labelsize=8, pad=8)
    # ax.tick_params(axis='both', direction='out', which='major', width=2, length=8,
    #                labelsize=8, pad=8)
    ax.spines['left'].set_linewidth(2)
    ax.spines['bottom'].set_linewidth(2)
    ax.spines['right'].set_linewidth(2)
    ax.spines['top'].set_linewidth(2)

def plot_algo(gamma,size,random_weight,length,true_obj,
              mujoco=True,env_name='Hopper-v4',env_id='hopper',train='train'):
    # load the biased
    with open('./dataset/biased_obj.pkl', 'rb') as file:
        biased_obj = pickle.load(file)
    result = []
    for seed in range(10):
        name = ['discount_factor', gamma, 'random_weight', random_weight, 'max_length', length,
                'buffer_size', size, 'seed', seed]
        name = '-'.join(str(x) for x in name)
        result.append([biased_obj[env_name][train][name]])
    result = np.array(result)
    mean_biased = np.mean((result - true_obj) ** 2)
    var_biased = np.std((result - true_obj) ** 2) / result.shape[0]

    # plot the result for our avg algorithm
    log_dir = '../avg_tune_log/avg_corr/'
    result = []
    for seed in range(10):
        if mujoco:
            filename = f"final-mujoco-{env_name}-discount-{gamma}-length-{length}-random-{random_weight}-size-{size}-seed-{seed}.csv"
        else:
            filename = f"final-classic-mse-{env_name}-discount-{gamma}-length-{length}-length-{random_weight}-size-{size}-seed-{seed}.csv"

        f = os.path.join(log_dir, filename)

        data = pd.read_csv(f, header=0, index_col='hyperparam')
        data.columns = data.columns.astype(int)
        data = data.sort_index(axis=1, ascending=True)
        for name in data.index.to_list():
            if train in name:
                if len(data.loc[name].shape) == 1:
                    result.append(data.loc[name].to_list())
                else:
                    result.append((data.loc[name]).mean(axis=0).tolist())
    result = np.array(result)
    mean_avg_mse = np.mean((result-true_obj)**2,axis=0)
    var_avg = np.std((result - true_obj) ** 2, axis=0) / result.shape[0]

    # plot bestDICE
    result = []
    log_dir = "../avg_tune_log/dice/" + env_id +'/'

    for seed in range(10):
        if mujoco:
            filename =  'dice-mujoco-' + str(env_name) + '-discount-' + str(gamma) \
                       + '-length-' + str(length) + '-random-' + str(random_weight) + \
                       '-size-' + str(size) + 'seed' + str(seed) + '.csv'
        else:
            seed_classic = seed+1
            filename = 'dice-classic-' + str(env_name) + '-discount-' + str(gamma) \
                       + '-length-' + str(length) + '-random-' + str(random_weight) + \
                       '-size-' + str(size) + 'seed' + str(seed_classic) + '.csv'
        f = log_dir + filename

        with open(f, "r") as scraped:
            reader = csv.reader(scraped, delimiter=',')
            row_index = 0
            for row in reader:
                if row:  # avoid blank lines
                    row_index += 1
                    # assign the last element as the name, containing see and hyperparameter choices
                    name = row[-1]
                    if train in name:
                        result.append([float(i) for i in row[:-1]])
                        break
    result = np.array(result)
    mean_dice = np.mean((result - true_obj) ** 2, axis=0)
    var_dice = np.std((result - true_obj) ** 2, axis=0) / result.shape[0]

    # plot COP-TD
    result = []
    if not mujoco:
        log_dir = "../avg_tune_log/COP-TD/" + env_id + "/"
        filename = f"mse-tune-random_weight_{random_weight}-discount_factor_{gamma}-max_ep_{size // length}-max_len_{length}-link_default-batch_size_512-bootstrap_target_target_network-lr_0.001-alpha_0.0.pkl"
    else:
        log_dir = "../avg_tune_log/COP-TD/" + env_id + "/"
        filename = f"mse-tune-random_weight_{random_weight}-discount_factor_{gamma}-max_ep_{size//length}-max_len_{length}-link_default-batch_size_512-bootstrap_target_target_network-lr_0.005-alpha_0.0.pkl"

    f = os.path.join(log_dir, filename)
    run_data = pickle.load(open(f, "rb"))
    for seed in run_data["seeds"]:
        result.append(np.array(run_data["results"][seed][0]))
    result=np.array(result)
    mean_cop_mse = np.mean((result-true_obj)**2, axis=0)
    var_cop = np.std((result - true_obj) ** 2, axis=0) / result.shape[0]

    # plot SR-DICE
    seeds = range(1,10)
    result = []
    log_dir = '../avg_tune_log/MIS/'
    for seed in seeds:
        if mujoco:
            filename = f"{env_name}SR_DICE-mujoco-{env_name}-discount-{gamma}-length-{length}-random-{random_weight}-size-{size}-seed-{seed}.csv"
        else:
            filename = f"{env_name}SR_DICE-classic-{env_name}-discount-{gamma}-length-{length}-random-{random_weight}-size-{size}-seed-{seed}.csv"

        f = os.path.join(log_dir, filename)
        data = pd.read_csv(f, header=0, index_col='hyperparam')
        data.columns = data.columns.astype(int)
        data = data.sort_index(axis=1, ascending=True)
        for name in data.index.to_list():
            if train in name:
                result.append(data.loc[name].to_list())
    result = np.array(result)
    mean_mis = np.mean((result-true_obj)**2, axis=0)
    var_mis = np.std((result - true_obj) ** 2, axis=0) / result.shape[0]

    # plot Deep_TD
    seeds = range(1, 10)
    result = []
    log_dir = '../avg_tune_log/TD/'
    for seed in seeds:
        if mujoco:
            filename = f"{env_name}Deep_TD-mujoco-{env_name}-discount-{gamma}-length-{length}-random-{random_weight}-size-{size}-seed-{seed}.csv"
        else:
            filename = f"{env_name}Deep_TD-classic-{env_name}-discount-{gamma}-length-{length}-random-{random_weight}-size-{size}-seed-{seed}.csv"

        f = os.path.join(log_dir, filename)
        data = pd.read_csv(f, header=0, index_col='hyperparam')
        data.columns = data.columns.astype(int)
        data = data.sort_index(axis=1, ascending=True)
        for name in data.index.to_list():
            if train in name:
                if len(data.loc[name].shape) == 1:
                    result.append(data.loc[name].to_list())
                else:
                    result.append((data.loc[name]).mean(axis=0).tolist())

    result = np.array(result)
    mean_td = np.mean((result-true_obj)**2, axis=0)
    var_td = np.std((result - true_obj) ** 2, axis=0) / result.shape[0]

    # plt.figure(figsize=(10,6))
    setsizes()
    setaxes()
    mean_avg_mse,var_avg = mean_avg_mse[:5000],var_avg[:5000]
    mean_cop_mse,var_cop = mean_cop_mse[:5000],var_cop[:5000]
    mean_mis, var_mis = mean_mis[:5000],var_mis[:5000]
    mean_td, var_td = mean_td[:5000],var_td[:5000]
    plt.plot(range(0,mean_avg_mse.shape[0]*5,5), mean_biased* np.ones(mean_avg_mse.shape[0]),
             '--',color = 'tab:brown',label='behaviour policy')

    plt.plot(range(0,mean_avg_mse.shape[0]*5,5), mean_avg_mse,color = 'tab:red', label='our correction')
    plt.fill_between(range(0,mean_avg_mse.shape[0]*5,5),mean_avg_mse-var_avg,mean_avg_mse+var_avg,
                     color = 'tab:red',alpha=0.2)

    plt.plot(range(0,mean_cop_mse.shape[0]*5,5), mean_cop_mse,color = 'tab:green', label='COP-TD')
    plt.fill_between(range(0,mean_cop_mse.shape[0]*5,5),mean_cop_mse-var_cop,mean_cop_mse+var_cop,
                     color = 'tab:green',alpha=0.2)

    plt.plot(range(0,mean_avg_mse.shape[0]*5,5), mean_mis,color = 'tab:orange', label='SR-DICE')
    plt.fill_between(range(0,mean_avg_mse.shape[0]*5,5),mean_mis-var_mis, mean_mis+var_mis,
                     color = 'tab:orange', alpha=0.2)

    plt.plot(range(0,mean_avg_mse.shape[0]*5,5), mean_td,color = 'tab:blue', label='Deep TD')
    plt.fill_between(range(0,mean_avg_mse.shape[0]*5,5),mean_td-var_td, mean_td+var_td,
                     color = 'tab:blue', alpha=0.2)

    plt.plot(range(0,mean_dice.shape[0]*5,5), mean_dice,color = 'tab:purple', label='bestDICE')
    plt.fill_between(range(0,mean_dice.shape[0]*5,5), mean_dice-var_dice, mean_dice+var_dice,
                     color = 'tab:purple',alpha=0.2)

    plt.yscale('log')
    plt.ylabel('Log MSE', fontsize=10)
    plt.ylim(0.001,1000)
    plt.xlabel("Training Steps", fontsize=10)
    # plt.xscale('log')
    plt.xticks(rotation=25, fontsize=9)
    plt.xticks(np.arange(0,5000*6, 5000))
    plt.title(env_name, fontsize=12)
    # plt.show()

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

# plot_err()

def setsizes():
    plt.rcParams['axes.linewidth'] = 1.0
    plt.rcParams['lines.markeredgewidth'] = 1.0
    plt.rcParams['lines.markersize'] = 3

    plt.rcParams['xtick.labelsize'] = 12.0
    plt.rcParams['ytick.labelsize'] = 12.0
    plt.rcParams['xtick.direction'] = "out"
    plt.rcParams['ytick.direction'] = "in"
    plt.rcParams['lines.linewidth'] = 2.0
    plt.rcParams['ytick.minor.pad'] = 50.0

def setaxes():
    plt.gcf().subplots_adjust(bottom=0.2)
    plt.gcf().subplots_adjust(left=0.2)
    ax = plt.gca()
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    # ax.spines['left'].set_color('none')
    # ax.xaxis.set_ticks_position('bottom')
    # ax.yaxis.set_ticks_position('left')
    # ax.tick_params(axis='both', direction='out', which='minor', width=2, length=3,
    #                labelsize=8, pad=8)
    # ax.tick_params(axis='both', direction='out', which='major', width=2, length=8,
    #                labelsize=8, pad=8)
    ax.spines['left'].set_linewidth(2)
    ax.spines['bottom'].set_linewidth(2)
    ax.spines['right'].set_linewidth(2)
    ax.spines['top'].set_linewidth(2)
    # for tick in ax.xaxis.get_major_ticks():
    #     tick.label.set_fontsize(getxticklabelsize())
    # for tick in ax.yaxis.get_major_ticks():
    #     tick.label.set_fontsize(getxticklabelsize())



if __name__ == "__main__":
    fig,axs = plt.subplots(2,3,figsize=(13,6))
    # plt.title('Log MSE of the Estimated Objective', fontsize=13)

    env_lists=[
           'Hopper-v4',
           'HalfCheetah-v4',
           'Ant-v4',
           'Walker2d-v4',
    ]
    env_id_lists = [
                    'hopper',
                    'halfcheetah',
                    'ant',
                    'walker'
                    ]

    train = 'test'

    avg_mse, dice, cop = [], [], []
    for i in range(len(env_lists)):
        plt.subplot(2,3, i + 1)

        env_name = env_lists[i]
        env_id= env_id_lists[i]
        gamma = 0.95
        size, random_weight, length = 4000, 2.0,100
        with open('./dataset/mujoco_obj.pkl', 'rb') as file:
            obj = pickle.load(file)
        true_obj = obj[env_name][2]
        plot_algo(gamma = gamma,
                  size = size,
                  random_weight = random_weight,
                  length = length,
                  true_obj = true_obj,
                  mujoco=True,
                  env_name=env_name,
                  env_id=env_id,
                  train=train)

    env_lists = ['CartPole-v1','Acrobot-v1']
    env_id_lists = ['cartpole','acrobot']

    train = 'test'

    avg_mse, dice, cop = [], [], []
    for i in range(len(env_lists)):
        plt.subplot(2,3, i + 5)
        env_name = env_lists[i]
        env_id = env_id_lists[i]
        gamma = 0.95
        size, random_weight, length = 4000, 0.3, 100
        with open('./dataset/classic_obj.pkl', 'rb') as file:
            obj = pickle.load(file)
        true_obj = obj[env_name][2]
        plot_algo(gamma=gamma,
                  size=size,
                  random_weight=random_weight,
                  length=length,
                  true_obj=true_obj,
                  mujoco=False,
                  env_name=env_name,
                  env_id=env_id,
                  train=train)

    plt.legend(loc='upper center', bbox_to_anchor=(1.4, 1.2),
               fancybox=True, shadow=True, ncol=2)
    plt.subplots_adjust(bottom=0.2,left=0.1,right=0.8, top=0.9, wspace=0.4, hspace=0.4)
    plt.tight_layout()
    plt.show()