import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def tune_result(env):
    result={}
    for filename in os.listdir('./tune_log/'+env+'/'):
        f = os.path.join('./tune_log/'+env+'/', filename)
        # checking if it is a file
        if not f.endswith('.csv'):
            continue
        if '0.7-40' in filename:
            data = pd.read_csv(f, header=0,index_col='hyperparam')
            data.columns = data.columns.astype(int)
            data = data.sort_index(axis=1, ascending=True)
            for name in data.index.to_list():
                if 'mean' in name:
                    print(name)
                    result[filename + '-'+ name] = data.loc[name].to_list()
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
    return data

def top_five(data,best_value):
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
    top_five = avg.nsmallest(5)
    top_five = top_five.index
    results = data.loc[top_five].to_numpy()
    top_five = list(top_five)
    print('hyper: ',top_five)
    plt.figure()
    for i in range(results.shape[0]):
        plt.plot(range(results.shape[1]), results[i, :], label=top_five[i])
    plt.legend()
    plt.title('last')
    plt.show()

def plot_cartpole():
    result = []
    for filename in os.listdir('./tune_log/cartpole_cv'):
        f = os.path.join('./tune_log/cartpole_cv/', filename)
        # checking if it is a file
        if not f.endswith('.csv'):
            continue
        if '0.7' in filename and '0.95' in filename and '40' in filename and 'CartPole' in filename:
            data = pd.read_csv(f, header=0, index_col='hyperparam')
            data.columns = data.columns.astype(int)
            data = data.sort_index(axis=1, ascending=True)
            for name in data.index.to_list():
                if 'train' in name:
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

# data = tune_result('cartpole')
# top_five(data,1)
plot_cartpole()
# plot_hopper()