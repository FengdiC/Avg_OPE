import _pickle as pickle
import matplotlib.pyplot as plt
import numpy as np
import os, sys, inspect
import pyarrow.feather as feather
import pandas as pd
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
result_dir = "tune_log/COP-TD/processed_best.feather"
env_name = "cartpole"

all_runs = []
all_errors = []
# only compare cartpole and hopper
# then find the hyperparameter choice for all the other environments

# cartpole:batch_size_256-bootstrap_target_target_network-lr_0.005
for run_file in os.listdir(os.path.join(result_dir, env_name)):
    if '0.7' in run_file and '40' in run_file:
        run_data = pickle.load(open(os.path.join(result_dir, env_name, run_file), "rb"))

        all_runs.append(run_file.split(".pkl")[0])
        all_errors.append([])

        for seed in run_data["seeds"]:
            all_errors[-1].append((np.array(run_data["results"][seed][1]) - baseline[seed][run_data["hyperparameters"]["discount"]][0]) ** 2)
all_runs = np.array(all_runs)
all_errors = np.array(all_errors)
mse_per_run = np.mean(all_errors, axis=(1, 2))
sort_idxes = np.argsort(mse_per_run)
mse_per_run[sort_idxes]
top_k = 5

fig, ax = plt.subplots(1, 1, figsize=(10, 10))
for run_name, errors in zip(all_runs[sort_idxes][:top_k], all_errors[sort_idxes][:top_k]):
    print(run_name)

    # log_errors = np.log10(errors)
    log_errors = (errors)
    mean_errors = np.mean(log_errors, axis=0)
    std_errors = np.std(log_errors, axis=0) / np.sqrt(len(log_errors))
    ax.plot(np.arange(errors.shape[1]), mean_errors)
    ax.fill_between(np.arange(errors.shape[1]), mean_errors + std_errors, mean_errors - std_errors, alpha=0.2)
fig.tight_layout()
plt.show()