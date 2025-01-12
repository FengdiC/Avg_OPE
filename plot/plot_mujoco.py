import os, sys, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from avg_corr.main import eval_policy
import _pickle as pickle


def compute_points(gamma,size,random_weight,length,env,train,true_obj,env_name,mujoco=True):
    result = []
    # plot the result for our avg algorithm
    for filename in os.listdir('../avg_tune_log/mujoco'):
        f = os.path.join('../avg_tune_log/mujoco/', filename)
        # checking if it is a file
        if not f.endswith('.csv'):
            continue
        if '-'+str(gamma)+'-' in filename and str(random_weight) in filename \
                and str(env) in filename and '-'+str(length)+'-' in filename:
            data = pd.read_csv(f, header=0, index_col='hyperparam')
            data.columns = data.columns.astype(int)
            data = data.sort_index(axis=1, ascending=True)
            for name in data.index.to_list():
                if train in name and str(size) in name:
                    result.append(data.loc[name].to_list())
    result = np.array(result)
    print(result.shape)
    first_est = result[:,:50]
    last_est = result[:,:-50]
    first_logmse = (first_est-true_obj)**2
    last_logmse = (last_est-true_obj)**2
    values_avg_mse = [np.mean(first_est),np.mean(np.var(first_est,axis=0)),
              np.mean(last_est),np.mean(np.var(last_est,axis=0)),
              np.mean(first_logmse),np.mean(np.var(first_logmse,axis=0)),
              np.mean(last_logmse),np.mean(np.var(last_logmse,axis=0))]

    result = []
    result_dir = "../avg_tune_log/COP-TD/results/results/"+env_name+"/"
    # plot the result for our avg algorithm
    for filename in os.listdir(result_dir):
        f = os.path.join(result_dir, filename)
        # checking if it is a file
        if not f.endswith('.pkl'):
            continue
        if filename.startswith('.'):
            continue
        if str(gamma) + '-' in filename and str(random_weight) in filename \
                and str(size//length)+'-' in filename and str(length) + '-' in filename:
            print(f)
            run_data = pickle.load(open(f, "rb"))
            for seed in run_data["seeds"]:
                result.append(np.array(run_data["results"][seed][1]))
    result = np.array(result)
    print(result.shape)
    first_est = result[:, :50]
    last_est = result[:, :-50]
    first_logmse = (first_est - true_obj) ** 2
    last_logmse = (last_est - true_obj) ** 2
    values_cop_td = [np.mean(first_est), np.mean(np.var(first_est, axis=0)),
                      np.mean(last_est), np.mean(np.var(last_est, axis=0)),
                      np.mean(first_logmse), np.mean(np.var(first_logmse, axis=0)),
                      np.mean(last_logmse), np.mean(np.var(last_logmse, axis=0))]
    # values_cop_td = [0,0,0,0,0,0,0,0]

    ### SR-DICE
    result = []
    seeds = range(1,10,1)
    for seed in seeds:
        log_dir = '../avg_tune_log/MIS/'
        if mujoco:
            filename = f"{env}SR_DICE-mujoco-{env}-discount-{gamma}-length-{length}-random-{random_weight}-size-{size}-seed-{seed}.csv"
        else:
            filename = f"{env}SR_DICE-classic-{env}-discount-{gamma}-length-{length}-random-{random_weight}-size-{size}-seed-{seed}.csv"

        f = os.path.join(log_dir, filename)

        data = pd.read_csv(f, header=0, index_col='hyperparam')
        data.columns = data.columns.astype(int)
        data = data.sort_index(axis=1, ascending=True)
        for name in data.index.to_list():
            if train in name:
                result.append(data.loc[name].to_list())

    result = np.array(result)
    print(result.shape)
    first_est = result[:, :50]
    last_est = result[:, :-50]
    first_logmse = (first_est - true_obj) ** 2
    last_logmse = (last_est - true_obj) ** 2
    values_mis = [np.mean(first_est), np.mean(np.var(first_est, axis=0)),
                     np.mean(last_est), np.mean(np.var(last_est, axis=0)),
                     np.mean(first_logmse), np.mean(np.var(first_logmse, axis=0)),
                     np.mean(last_logmse), np.mean(np.var(last_logmse, axis=0))]

    ### Deep TD
    result = []
    seeds = range(1, 10, 1)
    for seed in seeds:
        log_dir = '../avg_tune_log/TD/'
        if mujoco:
            filename = f"{env}Deep_TD-mujoco-{env}-discount-{gamma}-length-{length}-random-{random_weight}-size-{size}-seed-{seed}.csv"
        else:
            filename = f"{env}Deep_TD-classic-{env}-discount-{gamma}-length-{length}-random-{random_weight}-size-{size}-seed-{seed}.csv"

        f = os.path.join(log_dir, filename)

        data = pd.read_csv(f, header=0, index_col='hyperparam')
        data.columns = data.columns.astype(int)
        data = data.sort_index(axis=1, ascending=True)
        for name in data.index.to_list():
            if train in name:
                if len(data.loc[name].shape) == 1:
                    result.append(data.loc[name].to_list())
                else:
                    print(f)
                    print(len(data.loc[name].shape))
                    result.append((data.loc[name]).mean(axis=0).tolist())

    result = np.array(result)
    print(result.shape)
    first_est = result[:, :50]
    last_est = result[:, :-50]
    first_logmse = (first_est - true_obj) ** 2
    last_logmse = (last_est - true_obj) ** 2
    values_td = [np.mean(first_est), np.mean(np.var(first_est, axis=0)),
                  np.mean(last_est), np.mean(np.var(last_est, axis=0)),
                  np.mean(first_logmse), np.mean(np.var(first_logmse, axis=0)),
                  np.mean(last_logmse), np.mean(np.var(last_logmse, axis=0))]



    values_dice = [0, 0, 0, 0, 0, 0, 0, 0]

    return values_avg_mse,values_dice,values_cop_td, values_mis, values_td


def plot_classic(env_file='./exper/ant.pth',env='Ant-v4',env_name='ant'):
    discount_factor_lists = [0.8, 0.9, 0.95, 0.99, 0.995]
    size_lists = [2000, 4000, 8000, 16000]

    random_weight_lists = [1.4, 1.8, 2.0, 2.4, 2.8]
    length_lists = [20, 50, 100, 200]
    train = 'train'

    avg_mse, dice,cop, mis, td = [], [], [], [], []
    for i in range(len(discount_factor_lists)):
        gamma = discount_factor_lists[i]
        size, random_weight, length = 4000,2.0,100
        with open('./dataset/mujoco_obj.pkl', 'rb') as file:
            obj = pickle.load(file)
        true_obj = obj[env][i]
        values_avg_mse, values_dice,values_cop,values_mis, values_td = compute_points(gamma,size,random_weight,length,
                                                                       env,train,true_obj,env_name)
        avg_mse.append([values_avg_mse[6],values_avg_mse[7]])
        cop.append([values_cop[6],values_cop[7]])
        dice.append([values_dice[6], values_dice[7]])
        mis.append([values_mis[6], values_mis[7]])
        td.append([values_td[6],values_td[7]])

    avg_mse, dice, cop, mis, td = np.array(avg_mse), np.array(dice), np.array(cop), np.array(mis), np.array(td)
    plt.subplot(411)
    plt.errorbar(range(len(discount_factor_lists)), avg_mse[:,0], yerr=avg_mse[:,1],label='avg_mse')
    plt.plot(range(len(discount_factor_lists)), cop[:, 0], label='cop_td')
    plt.plot(range(len(discount_factor_lists)), mis[:, 0], label='sr_dice')
    plt.plot(range(len(discount_factor_lists)), td[:, 0], label='deep_td')
    # plt.errorbar(range(len(discount_factor_lists)), cop[:, 0], yerr=cop[:, 1], label='cop_td')
    # plt.errorbar(range(len(discount_factor_lists)), dice[:, 0], yerr=dice[:, 1], label='best_dice')
    plt.xticks(ticks=range(len(discount_factor_lists)), labels=discount_factor_lists )
    plt.yscale('log')
    plt.legend()

    with open('./dataset/mujoco_obj.pkl', 'rb') as file:
        obj = pickle.load(file)
    true_obj = obj[env][2]
    avg_mse, dice,cop, mis, td = [], [], [], [], []
    for size in size_lists:
        gamma, random_weight, length= 0.95,2.0,100
        values_avg_mse, values_dice, values_cop,values_mis, values_td = compute_points(gamma, size, random_weight,length,
                                                     env, train, true_obj,env_name)
        avg_mse.append([values_avg_mse[6], values_avg_mse[7]])
        dice.append([values_dice[6], values_dice[7]])
        cop.append([values_cop[6], values_cop[7]])
        mis.append([values_mis[6], values_mis[7]])
        td.append([values_td[6], values_td[7]])

    avg_mse, dice, cop, mis, td = np.array(avg_mse), np.array(dice), np.array(cop), np.array(mis), np.array(td)
    plt.subplot(412)
    plt.errorbar(range(len(size_lists)), avg_mse[:, 0], yerr=avg_mse[:, 1], label='avg_mse')
    plt.plot(range(len(size_lists)), cop[:, 0], label='cop_td')
    plt.plot(range(len(size_lists)), mis[:, 0], label='sr_dice')
    plt.plot(range(len(size_lists)), td[:, 0], label='deep_td')
    # plt.errorbar(range(len(size_lists)), cop[:, 0], yerr=cop[:, 1], label='cop_td')
    # plt.errorbar(range(len(size_lists)), dice[:, 0], yerr=dice[:, 1], label='best_dice')
    plt.xticks(ticks=range(len(size_lists)), labels=size_lists)
    plt.yscale('log')
    plt.legend()

    avg_mse, dice,cop, mis, td = [], [], [], [], []
    for random_weight in random_weight_lists:
        size, gamma, length = 4000,0.95,100
        values_avg_mse, values_dice, values_cop ,values_mis, values_td = compute_points(gamma, size, random_weight,length,
                                                     env, train, true_obj,env_name)
        avg_mse.append([values_avg_mse[6], values_avg_mse[7]])
        dice.append([values_dice[6], values_dice[7]])
        cop.append([values_cop[6], values_cop[7]])
        mis.append([values_mis[6], values_mis[7]])
        td.append([values_td[6], values_td[7]])

    avg_mse, dice, cop, mis, td = np.array(avg_mse), np.array(dice), np.array(cop), np.array(mis), np.array(td)
    plt.subplot(413)
    plt.errorbar(range(len(random_weight_lists)), avg_mse[:, 0], yerr=avg_mse[:, 1], label='avg_mse')
    plt.plot(range(len(random_weight_lists)), cop[:, 0], label='cop_td')
    plt.plot(range(len(random_weight_lists)), mis[:, 0], label='sr_dice')
    plt.plot(range(len(random_weight_lists)), td[:, 0], label='deep_td')
    # plt.errorbar(range(len(random_weight_lists)), cop[:, 0], yerr=cop[:, 1], label='cop_td')
    # plt.errorbar(range(len(random_weight_lists)), dice[:, 0], yerr=dice[:, 1], label='best_dice')
    plt.xticks(ticks=range(len(random_weight_lists)), labels=random_weight_lists)
    plt.yscale('log')
    plt.legend()

    avg_mse, dice, cop, mis, td = [], [], [], [], []
    for length in length_lists:
        size, gamma, random_weight = 4000, 0.95, 2.0
        values_avg_mse, values_dice, values_cop, values_mis, values_td = compute_points(gamma, size, random_weight,
                                                                                        length,
                                                                                        env, train, true_obj, env_name)
        avg_mse.append([values_avg_mse[6], values_avg_mse[7]])
        dice.append([values_dice[6], values_dice[7]])
        cop.append([values_cop[6], values_cop[7]])
        mis.append([values_mis[6], values_mis[7]])
        td.append([values_td[6], values_td[7]])

    avg_mse, dice, cop, mis, td = np.array(avg_mse), np.array(dice), np.array(cop), np.array(mis), np.array(td)
    plt.subplot(414)
    plt.errorbar(range(len(length_lists)), avg_mse[:, 0], yerr=avg_mse[:, 1], label='avg_mse')
    plt.plot(range(len(length_lists)), cop[:, 0], label='cop_td')
    plt.plot(range(len(length_lists)), mis[:, 0], label='sr_dice')
    plt.plot(range(len(length_lists)), td[:, 0], label='deep_td')
    # plt.errorbar(range(len(random_weight_lists)), cop[:, 0], yerr=cop[:, 1], label='cop_td')
    # plt.errorbar(range(len(random_weight_lists)), dice[:, 0], yerr=dice[:, 1], label='best_dice')
    plt.xticks(ticks=range(len(length_lists)), labels=length_lists)
    plt.yscale('log')
    plt.legend()

    plt.show()


plot_classic('./exper/cartpole.pth','Hopper-v4','hopper')
# plot(0.8,80,0.5,'CartPole-v1','train')