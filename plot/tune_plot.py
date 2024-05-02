import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

logger.configure(args.log_dir, ['csv'], log_suffix='mse-tune-' + str(args.env)+'-'+
                                            str(args.link)+'-'+str(args.batch_size)+'-'+
                                                       str(args.buffer_size))
def hyper_choice():
    result={}
    for filename in os.listdir('./tune_log'):
        f = os.path.join('./tune_log', filename)
        # checking if it is a file
        if not f.endswith('.csv'):
            continue
        if '0.3-20' in filename:
            data = pd.read_csv(f, header=0,parse_dates={'timestamp': ['hyperparam']},
                               index_col='timestamp')
            data.columns = data.columns.astype(int)
            data = data.sort_index(axis=1, ascending=True)
            for name in data.index.to_list():
                if 'mean' in name:
                    print(name)
                    result[filename + '-'+ name] = data[name].to_list()



    for i in range(1,301,1):
        filename = 'cartpole'+str(i)+'/progressbiased-ppo-tune-CartPole-v1.csv'
        file = os.path.join('./exper', filename)
        if not f.endswith('.csv'):
            continue
        # checking if it is a file
        dummyname = 'cartpole'+str(i)+'/biased-ppo-tune-CartPole-v1.csv'
        dummy = os.path.join('./exper', dummyname)
        with open(file, 'r') as read_obj, open(dummy, 'w') as write_obj:
            # Iterate over the given list of strings and write them to dummy file as lines
            Lines = read_obj.readlines()
            Lines[0] = Lines[0].replace('\n',',hyperparam2\n')
            for line in Lines:
                write_obj.write(line)
        data = pd.read_csv(dummy, header=0,
                           parse_dates={'timestamp': ['hyperparam', 'hyperparam2']},
                           index_col='timestamp')
        data.columns = data.columns.astype(int)
        data = data.sort_index(axis=1, ascending=True)
        data = data.iloc[:, 15].to_numpy()
        rets = np.mean(data)
        if rets > best:
            print("Best idx: ",i,":::",rets)
            best=rets