import os
import csv
import numpy as np
from collections import defaultdict



def create_bash_script(command, bash_file_prefix, bash_file_index):
    bash_file_name = f"{bash_file_prefix}_{bash_file_index}.sh"
    file_exists = os.path.isfile(bash_file_name)
    with open(bash_file_name, 'a' if file_exists else 'w') as bash_file:
        if not file_exists:
            bash_file.write("#!/bin/bash\n")
            os.chmod(bash_file_name, 0o755)
        bash_file.write(command + "\n")

def extract_float_from_tensor(tensor_str):
    try:
        return float(tensor_str.split('(')[1].split(',')[0])
    except ValueError:
        raise ValueError(f"Unable to extract float from tensor string: {tensor_str}")

def read_csv_and_compute_mse(csv_file, true_value):
    mse_train_per_step = defaultdict(list)
    mse_test_per_step = defaultdict(list)
    with open(csv_file, mode='r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            step = int(row['Step'])
            eval_obj = extract_float_from_tensor(row['Eval_Obj'])
            eval_obj2 = extract_float_from_tensor(row['Eval_Obj2'])
            mse_train = (eval_obj - true_value) ** 2
            mse_test = (eval_obj2 - true_value) ** 2
            mse_train_per_step[step].append(mse_train)
            mse_test_per_step[step].append(mse_test)
    return mse_train_per_step, mse_test_per_step

#################
from absl import flags

FLAGS = flags.FLAGS

flags.DEFINE_float('nu_learning_rate', 0.0001, 'Learning rate for nu.')
flags.DEFINE_float('zeta_learning_rate', 0.0001, 'Learning rate for zeta.')
flags.DEFINE_float('nu_regularizer', 0.0, 'Ortho regularization on nu.')
flags.DEFINE_float('zeta_regularizer', 0.0, 'Ortho regularization on zeta.')

flags.DEFINE_float('f_exponent', 2, 'Exponent for f function.')
flags.DEFINE_bool('primal_form', False, 'Whether to use primal form of loss for nu.')

flags.DEFINE_float('primal_regularizer', 0., 'LP regularizer of primal variables.')
flags.DEFINE_float('dual_regularizer', 1., 'LP regularizer of dual variables.')
flags.DEFINE_bool('zero_reward', False, 'Whether to ignore reward in optimization.')
flags.DEFINE_float('norm_regularizer', 1., 'Weight of normalization constraint.')
flags.DEFINE_bool('zeta_pos', True, 'Whether to enforce positivity constraint.')

flags.DEFINE_float('scale_reward', 1., 'Reward scaling factor.')
flags.DEFINE_float('shift_reward', 0., 'Reward shift factor.')
flags.DEFINE_string('transform_reward', None, 'Non-linear reward transformation. One of [exp, cuberoot, None]')

# Ensure flags are parsed
FLAGS([''])

# Define train_hparam_str with default values from FLAGS
train_hparam_str = (
    'nlr{NLR}_zlr{ZLR}_zeror{ZEROR}_preg{PREG}_dreg{DREG}_nreg{NREG}_'
    'pform{PFORM}_fexp{FEXP}_zpos{ZPOS}_'
    'scaler{SCALER}_shiftr{SHIFTR}_transr{TRANSR}'
).format(
    NLR=FLAGS.nu_learning_rate,
    ZLR=FLAGS.zeta_learning_rate,
    ZEROR=FLAGS.zero_reward,
    PREG=FLAGS.primal_regularizer,
    DREG=FLAGS.dual_regularizer,
    NREG=FLAGS.norm_regularizer,
    PFORM=FLAGS.primal_form,
    FEXP=FLAGS.f_exponent,
    ZPOS=FLAGS.zeta_pos,
    SCALER=FLAGS.scale_reward,
    SHIFTR=FLAGS.shift_reward,
    TRANSR=FLAGS.transform_reward
)
#################

# Define hyperparameters to sweep over
alpha_values = [(1-i) for i in [0.1,0.2, 0.3,0.4,0.5]]
# seed_values = [100, 200]
seed_values = [5,6,7,8,9 ]
gamma_values = [0.8, 0.9, 0.95, 0.99 ,0.995]  # Example gamma values
num_trajectory_values = [2000,4000,8000,16000]  # Example num_trajectory values
max_len_values = [20,50,100,200,400]
max_num_bash = 5
num_bash = 0

true_values = { 0.8: 0.99992, 0.9: 1.0, 0.95: 0.99951, 0.99: 0.94339,}  # Replace with the actual true values for each gamma
num_steps = 10000

save_dir_prefix = './results10081001_r/cartpole/'
load_dir_prefix = './results10021944/cartpole/'
bash_file_prefix = './bash10021944/cartpole/run_dice_results10021944'
bash_file_index = 0

# Create the directory for bash scripts if it doesn't exist
os.makedirs(os.path.dirname(bash_file_prefix), exist_ok=True)

for gamma in gamma_values:
    # true_value = true_values[gamma]
    for alpha in alpha_values:
        for num_trajectory in num_trajectory_values:
            for max_trajectory_length in max_len_values:
                new_num_trajectory = int(num_trajectory / max_trajectory_length)
                mse_summary_train = defaultdict(list)
                mse_summary_test = defaultdict(list)
                for seed in seed_values:
                    hparam_str = ('{ENV_NAME}_tabular{TAB}_alpha{ALPHA}_seed{SEED}_'
                                  'numtraj{NUM_TRAJ}_maxtraj{MAX_TRAJ}_gamma{GAMMA}_random{RANDOM_WEIGHT}').format(
                        ENV_NAME="CartPole-v1",
                        TAB=False,
                        ALPHA=alpha,
                        SEED=seed,
                        NUM_TRAJ=new_num_trajectory,
                        MAX_TRAJ=max_trajectory_length,
                        GAMMA=gamma,
                        RANDOM_WEIGHT=0.2)

                    save_dir = os.path.join(save_dir_prefix, hparam_str, train_hparam_str)
                    csv_file = os.path.join(save_dir, 'evaluation_results.csv')

                    # Generate bash script for experiment
                    params = {
                        "save_dir": save_dir_prefix,
                        "load_dir": load_dir_prefix,
                        "load_dir_policy": "./exper/cartpole.pth",
                        "env_name": "CartPole-v1",
                        "num_trajectory": new_num_trajectory,
                        "max_trajectory_length": max_trajectory_length,
                        "alpha": alpha,
                        "tabular_obs": 0,
                        "seed": seed,
                        "primal_regularizer": 0.0,
                        "dual_regularizer": 1.0,
                        "zero_reward": 0,
                        "norm_regularizer": 1.0,
                        "zeta_pos": 1,
                        "num_steps": num_steps,
                        "gamma": gamma,
                    }
                    command = (
                        f"python scripts/run_neural_dice.py "
                        f"--save_dir={params['save_dir']} "
                        f"--load_dir={params['load_dir']} "
                        f"--load_dir_policy={params['load_dir_policy']} "
                        f"--env_name={params['env_name']} "
                        f"--num_trajectory={params['num_trajectory']} "
                        f"--max_trajectory_length={params['max_trajectory_length']} "
                        f"--alpha={params['alpha']} "
                        f"--tabular_obs={params['tabular_obs']} "
                        f"--seed={params['seed']} "
                        f"--primal_regularizer={params['primal_regularizer']} "
                        f"--dual_regularizer={params['dual_regularizer']} "
                        f"--zero_reward={params['zero_reward']} "
                        f"--norm_regularizer={params['norm_regularizer']} "
                        f"--zeta_pos={params['zeta_pos']} "
                        f"--num_steps={params['num_steps']} "
                        f"--gamma={params['gamma']}"
                    )
                    create_bash_script(command, bash_file_prefix, bash_file_index)
                    num_bash += 1
                    if num_bash == max_num_bash:
                        bash_file_index += 1
                        num_bash = 0

                #     # Compute MSE
                #     print("read", csv_file)
                #     if os.path.exists(csv_file):
                #         mse_train_per_step, mse_test_per_step = read_csv_and_compute_mse(csv_file, true_value)
                #         print("mse_train_per_step", mse_train_per_step)
                #         print("mse_test_per_step", mse_test_per_step)
                #         for step, mse_list in mse_train_per_step.items():
                #             mse_summary_train[step].extend(mse_list)
                #         for step, mse_list in mse_test_per_step.items():
                #             mse_summary_test[step].extend(mse_list)
                #
                # mean_mse_per_step_train = {
                #     step: np.mean(mse_values) for step, mse_values in mse_summary_train.items()
                # }
                # mean_mse_per_step_test = {
                #     step: np.mean(mse_values) for step, mse_values in mse_summary_test.items()
                # }
                #
                # # Save the summary
                # summary_dir = os.path.join(save_dir_prefix, f"{gamma}_{alpha}_{new_num_trajectory}", 'summary')
                # os.makedirs(summary_dir, exist_ok=True)
                # summary_file_train = os.path.join(summary_dir, f'summary_train_gamma_{gamma}_alpha_{1-alpha}_numtraj_{new_num_trajectory}.csv')
                # summary_file_test = os.path.join(summary_dir, f'summary_test_gamma_{gamma}_alpha_{1-alpha}_numtraj_{new_num_trajectory}.csv')
                #
                # with open(summary_file_train, mode='w', newline='') as file:
                #     writer = csv.writer(file)
                #     writer.writerow(['Step', 'Mean MSE'])
                #     for step, mean_mse in sorted(mean_mse_per_step_train.items()):
                #         print("train step, mean_mse", step, mean_mse)
                #         writer.writerow([step, mean_mse])
                #
                # with open(summary_file_test, mode='w', newline='') as file:
                #     writer = csv.writer(file)
                #     writer.writerow(['Step', 'Mean MSE'])
                #     for step, mean_mse in sorted(mean_mse_per_step_test.items()):
                #         print("test step, mean_mse", step, mean_mse)
                #         writer.writerow([step, mean_mse])

print(f"Generated {bash_file_index} bash scripts.")
