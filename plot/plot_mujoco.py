import os, sys, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from avg_corr.main import eval_policy
import _pickle as pickle


def compute_points(gamma,size,random_weight,length,env,train,true_obj,env_name):
    result = []
    # plot the result for our avg algorithm
    for filename in os.listdir('./tune_log/mujoco'):
        f = os.path.join('./tune_log/mujoco/', filename)
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
    result_dir = "tune_log/COP-TD/results/results/"+env_name+"/"
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

    # result = []
    # for filename in os.listdir('./tune_log/bestdice_cartpole'):
    #     f = os.path.join('./tune_log/bestdice_cartpole', filename)
    #     # checking if it is a file
    #     if not f.endswith('.csv'):
    #         continue
    #     if '_'+"%.2f" % gamma+'_' in filename and 'numtraj_'+str(buffer) in filename and str(random_weight) in filename \
    #             and train in filename:
    #         data = pd.read_csv(f, header=0)
    #         mean_dice = data.loc[:, 'MSE']
    #         result.append(mean_dice.to_list()[:-1])
    # result = np.array(result)
    # print(result.shape)
    # first_est = result[:,:50]
    # last_est = result[:,:-50]
    # first_logmse = (first_est-true_obj)**2
    # last_logmse = (last_est-true_obj)**2
    # values_dice = [np.mean(first_est),np.mean(np.var(first_est,axis=0)),
    #           np.mean(last_est),np.mean(np.var(last_est,axis=0)),
    #           np.mean(first_logmse),np.mean(np.var(first_logmse,axis=0)),
    #           np.mean(last_logmse),np.mean(np.var(last_logmse,axis=0))]
    values_dice = [0, 0, 0, 0, 0, 0, 0, 0]

    return values_avg_mse,values_dice,values_cop_td


def plot_classic(env_file='./exper/ant.pth',env='Ant-v4',env_name='ant'):
    discount_factor_lists = [0.8, 0.9, 0.95, 0.99, 0.995]
    size_lists = [2000, 4000, 8000, 16000]

    random_weight_lists = [1.4, 1.8, 2.0, 2.4, 2.8]
    length_lists = [20, 50, 100, 200]
    train = 'test'

    avg_mse, dice,cop = [], [], []
    for i in range(len(discount_factor_lists)):
        gamma = discount_factor_lists[i]
        size, random_weight, length = 4000,2.0,50
        with open('./dataset/mujoco_obj.pkl', 'rb') as file:
            obj = pickle.load(file)
        true_obj = obj[env][i]
        values_avg_mse, values_dice,values_cop = compute_points(gamma,size,random_weight,length,
                                                                       env,train,true_obj,env_name)
        avg_mse.append([values_avg_mse[6],values_avg_mse[7]])
        cop.append([values_cop[6],values_cop[7]])
        dice.append([values_dice[6], values_dice[7]])

    avg_mse, dice, cop = np.array(avg_mse), np.array(dice), np.array(cop)
    plt.subplot(311)
    plt.errorbar(range(len(discount_factor_lists)), avg_mse[:,0], yerr=avg_mse[:,1],label='avg_mse')
    plt.plot(range(len(discount_factor_lists)), cop[:, 0], label='cop_td')
    # plt.errorbar(range(len(discount_factor_lists)), cop[:, 0], yerr=cop[:, 1], label='cop_td')
    plt.errorbar(range(len(discount_factor_lists)), dice[:, 0], yerr=dice[:, 1], label='best_dice')
    plt.xticks(ticks=range(len(discount_factor_lists)), labels=discount_factor_lists )
    plt.legend()

    with open('./dataset/mujoco_obj.pkl', 'rb') as file:
        obj = pickle.load(file)
    true_obj = obj[env][2]
    avg_mse, dice,cop = [], [], []
    for size in size_lists:
        gamma, random_weight, length= 0.95,2.0,50
        values_avg_mse, values_dice, values_cop = compute_points(gamma, size, random_weight,length,
                                                     env, train, true_obj,env_name)
        avg_mse.append([values_avg_mse[6], values_avg_mse[7]])
        dice.append([values_dice[6], values_dice[7]])
        cop.append([values_cop[6], values_cop[7]])

    avg_mse, dice, cop = np.array(avg_mse), np.array(dice), np.array(cop)
    plt.subplot(312)
    plt.errorbar(range(len(size_lists)), avg_mse[:, 0], yerr=avg_mse[:, 1], label='avg_mse')
    plt.plot(range(len(size_lists)), cop[:, 0], label='cop_td')
    # plt.errorbar(range(len(size_lists)), cop[:, 0], yerr=cop[:, 1], label='cop_td')
    plt.errorbar(range(len(size_lists)), dice[:, 0], yerr=dice[:, 1], label='best_dice')
    plt.xticks(ticks=range(len(size_lists)), labels=size_lists)
    plt.legend()

    avg_mse, dice,cop = [], [], []
    for random_weight in random_weight_lists:
        size, gamma, length = 4000,0.95,50
        values_avg_mse, values_dice, _ = compute_points(gamma, size, random_weight,length,
                                                     env, train, true_obj,env_name)
        avg_mse.append([values_avg_mse[6], values_avg_mse[7]])
        dice.append([values_dice[6], values_dice[7]])
        cop.append([values_cop[6], values_cop[7]])

    avg_mse, dice, cop = np.array(avg_mse), np.array(dice), np.array(cop)
    plt.subplot(313)
    plt.errorbar(range(len(random_weight_lists)), avg_mse[:, 0], yerr=avg_mse[:, 1], label='avg_mse')
    plt.plot(range(len(random_weight_lists)), cop[:, 0], label='cop_td')
    # plt.errorbar(range(len(random_weight_lists)), cop[:, 0], yerr=cop[:, 1], label='cop_td')
    plt.errorbar(range(len(random_weight_lists)), dice[:, 0], yerr=dice[:, 1], label='best_dice')
    plt.xticks(ticks=range(len(random_weight_lists)), labels=random_weight_lists)
    plt.legend()

    plt.show()


plot_classic()
# plot(0.8,80,0.5,'CartPole-v1','train')