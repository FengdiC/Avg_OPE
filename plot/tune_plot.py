import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def tune_result(env):
    result = {}
    result_var = {}
    for filename in os.listdir('./tune_log/'+env+'/'):
        f = os.path.join('./tune_log/'+env+'/', filename)
        # checking if it is a file
        if not f.endswith('.csv'):
            continue
        data = pd.read_csv(f, header=0,index_col='hyperparam')
        data.columns = data.columns.astype(int)
        data = data.sort_index(axis=1, ascending=True)
        for name in data.index.to_list():
            if 'mean' in name:
                result[filename + '-'+ name] = data.loc[name].to_list()
        data = data.loc[['seed-0','seed-1','seed-2','seed-3','seed-4']]
        result_var[filename + '-' + name] = data.var(axis=0).to_list()

    # for filename in os.listdir('./tune_log/'+env+'/gamma/'):
    #     f = os.path.join('./tune_log/'+env+'/gamma/', filename)
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
    return data,var

def top_five(data,var,best_value):
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
    last = data.iloc[:, n-100:n]
    last = (last - best_value).abs()
    avg = last.mean(axis=1)
    top_five = avg.nsmallest(30)
    top_five = top_five.index

    var = var.loc[top_five]
    var = var.iloc[:, n-100:n]
    var = var.mean(axis=1)
    top_five = var.nsmallest(5)
    top_five = top_five.index

    top_five = list(top_five)
    results = data.loc[top_five].to_numpy()

    print('hyper: ',top_five)
    plt.figure()
    for i in range(results.shape[0]):
        plt.plot(range(results.shape[1]), results[i, :], label=top_five[i])

    plt.plot(range(results.shape[1]), np.ones(results.shape[1]), label=top_five[i])
    plt.legend()
    plt.title('last')
    plt.show()

def plot_cartpole():
    result = []
    for filename in os.listdir('./tune_log/cartpole'):
        f = os.path.join('./tune_log/cartpole/', filename)
        # checking if it is a file
        if not f.endswith('.csv'):
            continue
        # if '0.7' in filename and '0.95' in filename and '40' in filename and 'CartPole' in filename:
        if '0.7' in filename and '40' in filename and '512' in filename and 'identity' in filename:
            data = pd.read_csv(f, header=0, index_col='hyperparam')
            data.columns = data.columns.astype(int)
            data = data.sort_index(axis=1, ascending=True)
            for name in data.index.to_list():
                if 'lr-0.005-alpha-0.005-mean' in name:
                    print(name)
                    result.append(data.loc[name].to_list())
    mean_avg = np.mean(result,axis=0)
    var_avg = np.var(result, axis=0)

    result = []
    for filename in os.listdir('./tune_log/dice_cartpole'):
        f = os.path.join('./tune_log/dice_cartpole', filename)
        # checking if it is a file
        if not f.endswith('.csv'):
            continue
        if '0.7' in filename and '0.95' in filename and '40' in filename and 'test' in filename:
            data = pd.read_csv(f, header=0)
            mean_dice = data.loc[:,'Mean MSE']

    plt.figure()
    plt.plot(range(mean_avg.shape[0]), mean_avg, label='avg_corr')
    plt.plot(range(len(mean_dice)), mean_dice, label='best_dice')
    plt.plot(range(len(mean_dice)), 0.99951*np.ones(len(mean_dice)), label='true_value')
    plt.legend()
    plt.title('last')
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
    #
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

# data,var = tune_result('cartpole')
# top_five(data,var,0.998)
# plot_cartpole()
# plot_hopper()

compare_data_procedure()