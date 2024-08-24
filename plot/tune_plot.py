import os, sys, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from avg_corr.main import eval_policy
import _pickle as pickle

def tune_result(env):
    result = {}
    result_var = {}
    result_train = {}
    for filename in os.listdir('./tune_log/'+env+'/'):
        f = os.path.join('./tune_log/'+env+'/', filename)
        # checking if it is a file
        if not f.endswith('.csv'):
            continue
        if 'gamma' not in filename:
            continue
        data = pd.read_csv(f, header=0,index_col='hyperparam')
        data.columns = data.columns.astype(int)
        data = data.sort_index(axis=1, ascending=True)
        for name in data.index.to_list():
            if 'mean' in name and 'val' in name:
                result[filename] = data.loc[name].to_list()
            if 'mean' in name and 'train' in name:
                result_train[filename] = data.loc[name].to_list()
        data = data.loc[['seed-0-val','seed-1-val','seed-2-val','seed-3-val','seed-4-val']]
        result_var[filename] = data.var(axis=0).to_list()

    # for filename in os.listdir('./tune_log/'+env):
    #     f = os.path.join('./tune_log/'+env, filename)
    #     # checking if it is a file
    #     if not f.endswith('.csv'):
    #         continue
    #     if '0.3-20' in filename:
    #         data = pd.read_csv(f, header=0,index_col='hyperparam')
    #         data.columns = data.columns.astype(int)
    #         data = data.sort_index(axis=1, ascending=True)
    #         for name in data.index.to_list():
    #             if 'mean' in name:
    #                 print(name)
    #                 result[filename + '-'+ name] = data[name].to_list()
    data = pd.DataFrame.from_dict(result, orient='index')
    var = pd.DataFrame.from_dict(result_var, orient='index')
    data_train = pd.DataFrame.from_dict(result_train, orient='index')
    return data,var,data_train

def top_five(data,var,data_train,best_value):
    # # average top five
    # avg = (data-best_value).abs()
    # avg = avg.mean(axis=1)
    # top_five = avg.nsmallest(5)
    # top_five = top_five.index
    # results = data.loc[top_five].to_numpy()
    # top_five = list(top_five)
    # plt.figure()
    # for i in range(results.shape[0]):
    #     plt.plot(range(results.shape[1]), results[i, :], label=top_five[i])
    # plt.legend()
    # plt.title('avg')
    # plt.show()

    # last few steps top five
    n = data.shape[1]
    last = data.iloc[:, n-800:n]
    last = (last - best_value).abs()
    avg = last.mean(axis=1)
    top_five = avg.nsmallest(10)
    top_five = top_five.index

    var = var.loc[top_five]
    var = var.iloc[:, n-200:n]
    var = var.mean(axis=1)
    top_five = var.nsmallest(5)
    top_five = top_five.index

    top_five = list(top_five)
    results = data.loc[top_five].to_numpy()
    results_train = data_train.loc[top_five].to_numpy()

    print('hyper: ',top_five)
    plt.subplot(211)
    for i in range(results.shape[0]):
        plt.plot(range(results.shape[1]), results[i, :], label=top_five[i])

    plt.plot(range(results.shape[1]), np.ones(results.shape[1]), label='one')

    plt.subplot(212)
    for i in range(results_train.shape[0]):
        plt.plot(range(results_train.shape[1]), results_train[i, :], label=top_five[i])

    plt.plot(range(results_train.shape[1]), np.ones(results_train.shape[1]), label='one')
    plt.legend()
    plt.title('train')
    plt.show()

def plot(gamma,buffer,random_weight,env,train='train'):
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

def compute_points(gamma,buffer,random_weight,env,train,true_obj,env_name):
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
                if train in name:
                    if 'mean' in name:
                        continue
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

    # result = []
    # result_dir = "tune_log/results/disc_cop"
    # # plot the result for our avg algorithm
    # for run_file in os.listdir(os.path.join(result_dir, env_name)):
    #     if 'batch_size_256-bootstrap_target_target_network-lr_0.005' in run_file:
    #         if str(gamma) in run_file and str(buffer) in run_file and str(random_weight) in run_file:
    #             print(run_file)
    #             run_data = pickle.load(open(os.path.join(result_dir, env_name, run_file), "rb"))
    #
    #             for seed in run_data["seeds"]:
    #                 result.append(np.array(run_data["results"][seed][1]))
    # result = np.array(result)
    # print(result.shape)
    # first_est = result[:, :50]
    # last_est = result[:, :-50]
    # first_logmse = (first_est - true_obj) ** 2
    # last_logmse = (last_est - true_obj) ** 2
    # values_cop_td = [np.mean(first_est), np.mean(np.var(first_est, axis=0)),
    #                   np.mean(last_est), np.mean(np.var(last_est, axis=0)),
    #                   np.mean(first_logmse), np.mean(np.var(first_logmse, axis=0)),
    #                   np.mean(last_logmse), np.mean(np.var(last_logmse, axis=0))]
    values_cop_td = [0,0,0,0,0,0,0,0]

    result = []
    for filename in os.listdir('./tune_log/bestdice_cartpole'):
        f = os.path.join('./tune_log/bestdice_cartpole', filename)
        # checking if it is a file
        if not f.endswith('.csv'):
            continue
        if '_'+"%.2f" % gamma+'_' in filename and 'numtraj_'+str(buffer) in filename and str(random_weight) in filename \
                and train in filename:
            data = pd.read_csv(f, header=0)
            mean_dice = data.loc[:, 'MSE']
            result.append(mean_dice.to_list()[:-1])
    result = np.array(result)
    print(result.shape)
    first_est = result[:,:50]
    last_est = result[:,:-50]
    first_logmse = (first_est-true_obj)**2
    last_logmse = (last_est-true_obj)**2
    values_dice = [np.mean(first_est),np.mean(np.var(first_est,axis=0)),
              np.mean(last_est),np.mean(np.var(last_est,axis=0)),
              np.mean(first_logmse),np.mean(np.var(first_logmse,axis=0)),
              np.mean(last_logmse),np.mean(np.var(last_logmse,axis=0))]

    return values_avg_mse,values_dice,values_cop_td


def plot_classic(env_file='./exper/cartpole.pth',env='CartPole-v1',env_name='cartpole'):
    discount_factor = [0.8, 0.9,0.95,0.99]
    buffer_num = [40, 80, 200]
    random_weight_val = [0.3, 0.5, 0.7]
    train = 'test'

    avg_mse, dice,cop = [], [], []
    for gamma in discount_factor:
        buffer, random_weight = 80,0.5
        true_obj,_,_ = eval_policy(path=env_file, env=env, gamma=gamma)
        values_avg_mse, values_dice,values_cop = compute_points(gamma,buffer,random_weight,
                                                                       env,train,true_obj,env_name)
        avg_mse.append([values_avg_mse[6],values_avg_mse[7]])
        cop.append([values_cop[6],values_cop[7]])
        dice.append([values_dice[6], values_dice[7]])

    avg_mse, dice, cop = np.array(avg_mse), np.array(dice), np.array(cop)
    plt.subplot(311)
    plt.errorbar(range(len(discount_factor)), avg_mse[:,0], yerr=avg_mse[:,1],label='avg_mse')
    # plt.errorbar(range(len(discount_factor)), cop[:, 0], yerr=cop[:, 1], label='cop_td')
    plt.errorbar(range(len(discount_factor)), dice[:, 0], yerr=dice[:, 1], label='best_dice')
    plt.xticks(ticks=range(len(discount_factor)), labels=discount_factor )
    plt.legend()

    avg_mse, dice,cop = [], [], []
    for buffer in buffer_num:
        gamma, random_weight = 0.8, 0.5
        true_obj, _, _ = eval_policy(path=env_file, env=env, gamma=gamma)
        values_avg_mse, values_dice, values_cop = compute_points(gamma, buffer, random_weight,
                                                     env, train, true_obj,env_name)
        avg_mse.append([values_avg_mse[6], values_avg_mse[7]])
        dice.append([values_dice[6], values_dice[7]])
        cop.append([values_cop[6], values_cop[7]])

    avg_mse, dice, cop = np.array(avg_mse), np.array(dice), np.array(cop)
    plt.subplot(312)
    plt.errorbar(range(len(buffer_num)), avg_mse[:, 0], yerr=avg_mse[:, 1], label='avg_mse')
    # plt.errorbar(range(len(buffer_num)), cop[:, 0], yerr=cop[:, 1], label='cop_td')
    plt.errorbar(range(len(buffer_num)), dice[:, 0], yerr=dice[:, 1], label='best_dice')
    plt.xticks(ticks=range(len(buffer_num)), labels=buffer_num)
    plt.legend()

    avg_mse, dice,cop = [], [], []
    for random_weight in random_weight_val:
        buffer, gamma = 80, 0.8
        true_obj, _, _ = eval_policy(path=env_file, env=env, gamma=gamma)
        values_avg_mse, values_dice, _ = compute_points(gamma, buffer, random_weight,
                                                     env, train, true_obj,env_name)
        avg_mse.append([values_avg_mse[6], values_avg_mse[7]])
        dice.append([values_dice[6], values_dice[7]])
        cop.append([values_cop[6], values_cop[7]])

    avg_mse, dice, cop = np.array(avg_mse), np.array(dice), np.array(cop)
    plt.subplot(313)
    plt.errorbar(range(len(random_weight_val)), avg_mse[:, 0], yerr=avg_mse[:, 1], label='avg_mse')
    # plt.errorbar(range(len(random_weight_val)), cop[:, 0], yerr=cop[:, 1], label='cop_td')
    plt.errorbar(range(len(random_weight_val)), dice[:, 0], yerr=dice[:, 1], label='best_dice')
    plt.xticks(ticks=range(len(random_weight_val)), labels=random_weight_val)
    plt.legend()

    plt.show()


def plot_hopper():
    result = []
    for filename in os.listdir('./tune_log/first_result'):
        f = os.path.join('./tune_log/first_result/', filename)
        # checking if it is a file
        if not f.endswith('.csv'):
            continue
        if '0.3' in filename and '0.99' in filename and '200' in filename and 'Hopper' in filename:
            data = pd.read_csv(f, header=0, index_col='hyperparam')
            data.columns = data.columns.astype(int)
            data = data.sort_index(axis=1, ascending=True)
            for name in data.index.to_list():
                if 'train' in name:
                    print(name)
                    result.append(data.loc[name].to_list())
    mean_avg = np.mean(result,axis=0)
    var_avg = np.var(result, axis=0)

    # result = []
    # for filename in os.listdir('./tune_log/dice_cartpole'):
    #     f = os.path.join('./tune_log/dice_cartpole', filename)
    #     # checking if it is a file
    #     if not f.endswith('.csv'):
    #         continue
    #     if '0.3' in filename and '0.9' in filename and '200' in filename and 'train' in filename:
    #         data = pd.read_csv(f, header=0)
    #         mean_dice = data.loc[:,'Mean MSE']

    plt.figure()
    plt.plot(range(mean_avg.shape[0]), mean_avg, label='avg_corr')
    # plt.plot(range(len(mean_dice)), mean_dice, label='best_dice')
    plt.plot(range(mean_avg.shape[0]), 2.65*np.ones(mean_avg.shape[0]), label='true_value')
    plt.legend()
    plt.title('last')
    plt.show()

def compare_data_procedure():
    filename = 'final-classic-CartPole-v1-discount-0.95-buffer-200-random-0.7.csv'
    f = os.path.join('./tune_log/', filename)
    result = []
    result_extend = []

    data = pd.read_csv(f, header=0, index_col='hyperparam')
    data.columns = data.columns.astype(int)
    data = data.sort_index(axis=1, ascending=True)
    for name in data.index.to_list():
        if 'train' and 'extend' in name:
            print(name)
            result_extend.append(data.loc[name].to_list())
        elif 'train' in name:
            print(name)
            result.append(data.loc[name].to_list())
    mean_resample = np.mean(result, axis=0)
    mean_restart = np.mean(result_extend, axis=0)
    plt.figure()
    plt.plot(range(mean_resample.shape[0]), mean_resample, label='natural')
    # plt.plot(range(len(mean_dice)), mean_dice, label='best_dice')
    plt.plot(range(mean_restart.shape[0]), mean_restart, label='restart')
    plt.legend()
    plt.title('last')
    plt.show()

# compare hyperparamters
# data,var,data_train = tune_result('hopper')
# top_five(data,var,data_train,2.7)

# plot out training curves
# plot_hopper()

plot_classic()
# plot(0.8,80,0.5,'CartPole-v1','train')