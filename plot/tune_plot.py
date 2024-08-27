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

    n = data.shape[1]
    # check the training error
    last = data_train.iloc[:, n - 800:n]
    last = (last - best_value).abs()
    avg = last.mean(axis=1)
    top_five = avg.nsmallest(20)
    top_five = top_five.index

    # last few steps top five
    last = data.loc[top_five]
    last = last.iloc[:, n-800:n]
    last = (last - best_value).abs()
    avg = last.mean(axis=1)
    top_five = avg.nsmallest(5)
    top_five = top_five.index

    # var = var.loc[top_five]
    # var = var.iloc[:, n-200:n]
    # var = var.mean(axis=1)
    # top_five = var.nsmallest(5)
    # top_five = top_five.index

    top_five = list(top_five)
    results = data.loc[top_five].to_numpy()
    results_train = data_train.loc[top_five].to_numpy()

    print('hyper: ',top_five)
    plt.subplot(211)
    for i in range(results.shape[0]):
        plt.plot(range(results.shape[1]), results[i, :], label=top_five[i])

    plt.plot(range(results.shape[1]), best_value*np.ones(results.shape[1]), label='one')

    plt.subplot(212)
    for i in range(results_train.shape[0]):
        plt.plot(range(results_train.shape[1]), results_train[i, :], label=top_five[i])

    plt.plot(range(results_train.shape[1]), best_value*np.ones(results_train.shape[1]), label='one')
    plt.legend()
    plt.title('train')
    plt.show()


# compare hyperparamters
data,var,data_train = tune_result('hopper')
top_five(data,var,data_train,1.6973)
