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

data = tune_result('hopper')
top_five(data,2.651)

