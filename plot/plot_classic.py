import os, sys, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from avg_corr.main import eval_policy
import _pickle as pickle
import csv

def setsizes():
    plt.rcParams['axes.linewidth'] = 1.0
    plt.rcParams['lines.markeredgewidth'] = 1.0
    plt.rcParams['lines.markersize'] = 3

    plt.rcParams['xtick.labelsize'] = 10.0
    plt.rcParams['ytick.labelsize'] = 10.0
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

def compute_points(gamma,size,random_weight,length,env_name,train,true_obj,env_id,mujoco=False):
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
    biased = [mean_biased,var_biased]

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
    result = np.array(result[:5000])
    mean_avg_mse = np.mean((result-true_obj)**2,axis=0)
    var_avg = np.std((result - true_obj) ** 2, axis=0) / result.shape[0]
    avg = [np.mean(mean_avg_mse),np.mean(var_avg)]

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
    result = np.array(result[:5000])
    mean_dice = np.mean((result - true_obj) ** 2, axis=0)
    var_dice = np.std((result - true_obj) ** 2, axis=0) / result.shape[0]
    dice = [np.mean(mean_dice),np.mean(var_dice)]

    # plot COP-TD
    result = []
    log_dir = "../avg_tune_log/COP-TD//" + env_id + "/"

    if not mujoco:
        filename = f"mse-tune-random_weight_{random_weight}-discount_factor_{gamma}-max_ep_{size // length}-max_len_{length}-link_default-batch_size_512-bootstrap_target_target_network-lr_0.001-alpha_0.0.pkl"
    else:
        filename = f"mse-tune-random_weight_{random_weight}-discount_factor_{gamma}-max_ep_{size//length}-max_len_{length}-link_default-batch_size_512-bootstrap_target_target_network-lr_0.005-alpha_0.0.pkl"

    f = os.path.join(log_dir, filename)
    run_data = pickle.load(open(f, "rb"))
    for seed in run_data["seeds"]:
        result.append(np.array(run_data["results"][seed][0]))
    result=np.array(result[:5000])
    mean_cop_mse = np.mean((result-true_obj)**2, axis=0)
    var_cop = np.std((result - true_obj) ** 2, axis=0) / result.shape[0]
    cop = [np.mean(mean_cop_mse),np.mean(var_cop)]

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
    result = np.array(result[:5000])
    mean_mis = np.mean((result-true_obj)**2, axis=0)
    var_mis = np.std((result - true_obj) ** 2, axis=0) / result.shape[0]
    mis = [np.mean(mean_mis),np.mean(var_mis)]

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

    result = np.array(result[:5000])
    mean_td = np.mean((result-true_obj)**2, axis=0)
    var_td = np.std((result - true_obj) ** 2, axis=0) / result.shape[0]
    td = [np.mean(mean_td),np.mean(var_td)]

    return biased, avg, cop, td, mis, dice


def plot_classic(env_file='./exper/cartpole.pth',env='CartPole-v1',env_name='cartpole',mujoco=False):
    if mujoco:
        discount_factor_lists = [0.8, 0.9, 0.95, 0.99, 0.995]
        size_lists = [2000, 4000, 8000, 16000]

        random_weight_lists = [1.4, 1.8, 2.0, 2.4, 2.8]
        length_lists = [20, 50, 100, 200]
    else:
        discount_factor_lists = [0.8, 0.9, 0.95, 0.99, 0.995]
        size_lists = [2000, 4000, 8000, 16000]

        random_weight_lists = [0.1, 0.2, 0.3, 0.4, 0.5]
        length_lists = [20, 40, 80, 100]
    train = 'test'

    biased_list, avg_list, cop_list, td_list, mis_list, dice_list = [],[],[],[],[],[]
    for i in range(len(discount_factor_lists)):
        gamma = discount_factor_lists[i]
        size, random_weight, length = 4000,2.0,100
        if mujoco:
            with open('./dataset/mujoco_obj.pkl', 'rb') as file:
                obj = pickle.load(file)
            true_obj = obj[env][i]
        else:
            with open('./dataset/classic_obj.pkl', 'rb') as file:
                obj = pickle.load(file)
            true_obj = obj[env][i]
        biased, avg, cop, td, mis, dice = compute_points(gamma, size, random_weight, length,
                                                         env, train, true_obj, env_name,mujoco=mujoco)
        biased_list.append(biased)
        avg_list.append(avg)
        cop_list.append(cop)
        td_list.append(td)
        mis_list.append(mis)
        dice_list.append(dice)

    biased_list, avg_list, cop_list, td_list, mis_list, dice_list = (np.array(biased_list),
                                                                     np.array(avg_list),
                                                                     np.array(cop_list),
                                                                     np.array(td_list),
                                                                     np.array(mis_list),
                                                                     np.array(dice_list),
                                                                     )

    if mujoco:
        with open('./dataset/mujoco_obj.pkl', 'rb') as file:
            obj = pickle.load(file)
        true_obj = obj[env][2]
    else:
        with open('./dataset/classic_obj.pkl', 'rb') as file:
            obj = pickle.load(file)
        true_obj = obj[env][2]

    fig = plt.figure(figsize=(9,6))
    plt.title(env_id, fontsize=16)
    plt.axis('off')
    plt.subplot(221)
    plt.errorbar(range(len(discount_factor_lists)), biased_list[:, 0], yerr=biased_list[:, 1],
                 color='tab:brown', label='behaviour policy')
    plt.errorbar(range(len(discount_factor_lists)), avg_list[:, 0], yerr=avg_list[:, 1],
                 color='tab:red', label='our correction')
    plt.errorbar(range(len(discount_factor_lists)), cop_list[:, 0], yerr=cop_list[:, 1],
                 color='tab:green', label='COP-TD')
    plt.errorbar(range(len(discount_factor_lists)), mis_list[:, 0], yerr=mis_list[:, 1],
                 color='tab:orange', label='SR-DICE')
    plt.errorbar(range(len(discount_factor_lists)), td_list[:, 0], yerr=td_list[:, 1],
                 color='tab:blue', label='Deep TD')
    plt.errorbar(range(len(discount_factor_lists)), dice_list[:, 0], yerr=dice_list[:, 1],
                 color='tab:purple', label='best_dice')
    setsizes()
    setaxes()
    plt.xticks(ticks=range(len(discount_factor_lists)), labels=discount_factor_lists )
    plt.yscale('log')
    plt.ylabel('Log MSE', fontsize=10)
    plt.ylim(0.001, 100)
    # plt.xscale('log')
    plt.xlabel('Discount Factor', fontsize=10)

    biased_list, avg_list, cop_list, td_list, mis_list, dice_list = [],[],[],[],[],[]
    for size in size_lists:
        gamma, random_weight, length= 0.95,2.0,100
        biased, avg, cop, td, mis, dice = compute_points(gamma, size, random_weight, length,
                                                         env, train, true_obj, env_name,mujoco=mujoco)
        biased_list.append(biased)
        avg_list.append(avg)
        cop_list.append(cop)
        td_list.append(td)
        mis_list.append(mis)
        dice_list.append(dice)

    biased_list, avg_list, cop_list, td_list, mis_list, dice_list = (np.array(biased_list),
                                                                     np.array(avg_list),
                                                                     np.array(cop_list),
                                                                     np.array(td_list),
                                                                     np.array(mis_list),
                                                                     np.array(dice_list),
                                                                     )
    plt.subplot(222)
    plt.errorbar(range(len(size_lists)), biased_list[:, 0], yerr=biased_list[:, 1],
                 color='tab:brown', label='behaviour policy')
    plt.errorbar(range(len(size_lists)), avg_list[:, 0], yerr=avg_list[:, 1],
                 color='tab:red', label='our correction')
    plt.errorbar(range(len(size_lists)), cop_list[:, 0], yerr=cop_list[:, 1],
                 color='tab:green', label='COP-TD')
    plt.errorbar(range(len(size_lists)), mis_list[:, 0], yerr=mis_list[:, 1],
                 color='tab:orange', label='SR-DICE')
    plt.errorbar(range(len(size_lists)), td_list[:, 0], yerr=td_list[:, 1],
                 color='tab:blue', label='Deep TD')
    plt.errorbar(range(len(size_lists)), dice_list[:, 0], yerr=dice_list[:, 1],
                 color='tab:purple', label='best_dice')
    setsizes()
    setaxes()
    plt.xticks(ticks=range(len(size_lists)), labels=size_lists)
    plt.yscale('log')
    plt.ylabel('Log MSE', fontsize=10)
    plt.ylim(0.001, 100)
    # plt.xscale('log')
    plt.xlabel('Buffer Size', fontsize=10)

    biased_list, avg_list, cop_list, td_list, mis_list, dice_list = [],[],[],[],[],[]
    for random_weight in random_weight_lists:
        size, gamma, length = 4000,0.95,100
        biased, avg, cop, td, mis, dice = compute_points(gamma, size, random_weight, length,
                                                         env, train, true_obj, env_name,mujoco=mujoco)
        biased_list.append(biased)
        avg_list.append(avg)
        cop_list.append(cop)
        td_list.append(td)
        mis_list.append(mis)
        dice_list.append(dice)

    biased_list, avg_list, cop_list, td_list, mis_list, dice_list = (np.array(biased_list),
                                                                     np.array(avg_list),
                                                                     np.array(cop_list),
                                                                     np.array(td_list),
                                                                     np.array(mis_list),
                                                                     np.array(dice_list),
                                                                     )
    plt.subplot(223)
    plt.errorbar(range(len(random_weight_lists)), biased_list[:, 0], yerr=biased_list[:, 1],
                 color='tab:brown', label='behaviour policy')
    plt.errorbar(range(len(random_weight_lists)), avg_list[:, 0], yerr=avg_list[:, 1],
                 color='tab:red', label='our correction')
    plt.errorbar(range(len(random_weight_lists)), cop_list[:, 0], yerr=cop_list[:, 1],
                 color='tab:green', label='COP-TD')
    plt.errorbar(range(len(random_weight_lists)), mis_list[:, 0], yerr=mis_list[:, 1],
                 color='tab:orange', label='SR-DICE')
    plt.errorbar(range(len(random_weight_lists)), td_list[:, 0], yerr=td_list[:, 1],
                 color='tab:blue', label='Deep TD')
    plt.errorbar(range(len(random_weight_lists)), dice_list[:, 0], yerr=dice_list[:, 1],
                 color='tab:purple', label='best_dice')
    setsizes()
    setaxes()
    plt.xticks(ticks=range(5), labels=random_weight_lists)
    plt.yscale('log')
    plt.ylabel('Log MSE', fontsize=10)
    plt.xlabel('Distance Between Behavior & Target Policies', fontsize=10)
    plt.ylim(0.001, 100)
    # plt.xscale('log')

    biased_list, avg_list, cop_list, td_list, mis_list, dice_list = [],[],[],[],[],[]
    for length in length_lists:
        size, gamma, random_weight = 4000, 0.95, 2.0
        biased, avg, cop, td, mis, dice = compute_points(gamma, size, random_weight, length,
                                                        env, train, true_obj, env_name,mujoco=mujoco)
        biased_list.append(biased)
        avg_list.append(avg)
        cop_list.append(cop)
        td_list.append(td)
        mis_list.append(mis)
        dice_list.append(dice)

    biased_list, avg_list, cop_list, td_list, mis_list, dice_list = (np.array(biased_list),
                                                                     np.array(avg_list),
                                                                     np.array(cop_list),
                                                                     np.array(td_list),
                                                                     np.array(mis_list),
                                                                     np.array(dice_list),
                                                                     )

    plt.subplot(224)
    plt.errorbar(range(len(length_lists)), biased_list[:, 0], yerr=biased_list[:, 1],
                 color='tab:brown', label='behaviour policy')
    plt.errorbar(range(len(length_lists)), avg_list[:, 0], yerr=avg_list[:, 1],
                 color='tab:red',label='our correction')
    plt.errorbar(range(len(length_lists)), cop_list[:, 0], yerr=cop_list[:, 1],
                 color='tab:green',label='COP-TD')
    plt.errorbar(range(len(length_lists)), mis_list[:, 0], yerr=mis_list[:, 1],
                 color='tab:orange',label='SR-DICE')
    plt.errorbar(range(len(length_lists)), td_list[:, 0], yerr=td_list[:, 1],
                 color='tab:blue',label='Deep TD')
    plt.errorbar(range(len(length_lists)), dice_list[:, 0], yerr=dice_list[:, 1],
                 color='tab:purple',label='best_dice')
    setsizes()
    setaxes()

    plt.xticks(ticks=range(len(length_lists)), labels=length_lists)
    plt.yscale('log')
    plt.ylabel('Log MSE', fontsize=10)
    plt.ylim(0.001, 100)
    # plt.xscale('log')
    plt.xlabel('Trajectory Length', fontsize=10)

    plt.tight_layout(rect=[0, 0, 0.82, 1])
    plt.legend(bbox_to_anchor=(1.02, 1.1), loc="upper left", ncol=1)

    plt.show()


if __name__ == "__main__":
    env_lists = [
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

    # env_lists = ['CartPole-v1', 'Acrobot-v1']
    # env_id_lists = ['cartpole', 'acrobot']

    for i in range(len(env_lists)):
        env_name = env_lists[i]
        env_id = env_id_lists[i]
        plot_classic(None,env_name,env_id,mujoco=True)

