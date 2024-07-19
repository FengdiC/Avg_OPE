import os
import csv
import numpy as np
import subprocess
from collections import defaultdict

def run_experiment(params):
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
        f"--num_steps={params['num_steps']}"
    )
    subprocess.run(command, shell=True)

def extract_float_from_tensor(tensor_str):
    try:
        return float(tensor_str.split('(')[1].split(',')[0])
    except ValueError:
        raise ValueError(f"Unable to extract float from tensor string: {tensor_str}")

def read_csv_and_compute_mse(csv_file, true_value):
    mse_train_per_step = defaultdict(list)
    mse_test_per_step = defaultdict(list)
    try:
        with open(csv_file, mode='r') as file:
            reader = csv.DictReader(file)
            for row in reader:
                if not row['Step'] or not row['Eval_Obj'] or not row['Eval_Obj2']:
                    raise ValueError("Invalid data in CSV file")
                step = int(row['Step'])
                eval_obj = extract_float_from_tensor(row['Eval_Obj'])
                eval_obj2 = extract_float_from_tensor(row['Eval_Obj2'])
                # mse_train = (eval_obj - true_value) ** 2
                # mse_test = (eval_obj2 - true_value) ** 2
                mse_train = eval_obj
                mse_test = eval_obj2
                mse_train_per_step[step].append(mse_train)
                mse_test_per_step[step].append(mse_test)
    except Exception as e:
        print(f"Skipping file {csv_file} due to error: {e}")
        return None, None  # Return None to indicate an error
    return mse_train_per_step, mse_test_per_step

alpha_values = [0.3, 0.5, 0.7]
# seed_values = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
seed_values = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
gamma_values = [0.8, 0.9, 0.95, 0.99]  # Example gamma values
num_trajectory_values = [40, 80, 200]  # Example num_trajectory values
true_values = { 0.8: 0.99992, 0.9: 1.0, 0.95: 0.99951, 0.99: 0.94339,}  # Replace with the actual true values for each gamma
num_steps = 10000
save_dir_prefix = './results06271701/cartpole/'



results = []

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


for gamma in gamma_values:
    true_value = true_values[gamma]
    for alpha in alpha_values:
        for num_trajectory in num_trajectory_values:
            for seed in seed_values:
                hparam_str = ('{ENV_NAME}_tabular{TAB}_alpha{ALPHA}_seed{SEED}_'
                              'numtraj{NUM_TRAJ}_maxtraj{MAX_TRAJ}_gamma{GAMMA}_random{RANDOM_WEIGHT}').format(
                    ENV_NAME="CartPole-v1",
                    TAB=False,
                    ALPHA=alpha,
                    SEED=seed,
                    NUM_TRAJ=num_trajectory,
                    MAX_TRAJ=50,
                    GAMMA=gamma,
                    RANDOM_WEIGHT=0.2)

                save_dir = os.path.join(save_dir_prefix, hparam_str, train_hparam_str)
                csv_file = os.path.join(save_dir, 'evaluation_results.csv')

                # Run experiment
                params = {
                    "save_dir": save_dir_prefix,
                    "load_dir": save_dir_prefix,
                    "load_dir_policy": "./exper/cartpole.pth",
                    "env_name": "CartPole-v1",
                    "num_trajectory": num_trajectory,
                    "max_trajectory_length": 50,
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
                # run_experiment(params)

                # Compute MSE
                if os.path.exists(csv_file):
                    mse_train_per_step, mse_test_per_step = read_csv_and_compute_mse(csv_file, true_value)
                    if mse_train_per_step is None or mse_test_per_step is None:  # Skip if the file had an error
                        continue

                    # Save the MSE for each seed
                    summary_dir = os.path.join(save_dir_prefix, 'summary')
                    os.makedirs(summary_dir, exist_ok=True)
                    summary_file_train = os.path.join(summary_dir,
                                                      f'summary_train_gamma_{gamma:.2f}_alpha_{(1 - alpha):.1f}_numtraj_{num_trajectory}_seed_{seed}.csv')
                    summary_file_test = os.path.join(summary_dir,
                                                     f'summary_test_gamma_{gamma:.2f}_alpha_{(1 - alpha):.1f}_numtraj_{num_trajectory}_seed_{seed}.csv')

                    with open(summary_file_train, mode='w', newline='') as file:
                        writer = csv.writer(file)
                        writer.writerow(['Step', 'MSE'])
                        for step, mse_list in sorted(mse_train_per_step.items()):
                            for mse in mse_list:
                                writer.writerow([step, mse])

                    with open(summary_file_test, mode='w', newline='') as file:
                        writer = csv.writer(file)
                        writer.writerow(['Step', 'MSE'])
                        for step, mse_list in sorted(mse_test_per_step.items()):
                            for mse in mse_list:
                                writer.writerow([step, mse])

print("Summary saved successfully.")
