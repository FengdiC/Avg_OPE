import os, sys, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
import numpy as np
import csv
from avg_corr.main import train as PPOBuffer
import pickle

import utils
import Deep_TD
import Deep_SR
import SR_DICE
import TD3
import torch
from gymnasium.spaces import Box, Discrete
import gymnasium as gym

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def argsparser():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default='./')
    parser.add_argument('--log_dir', type=str, default='./')
    parser.add_argument('--data_dir', type=str, default='/scratch/fengdic/')
    parser.add_argument('--env', type=str, default='Hopper-v4')
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--steps', type=int, default=5)
    parser.add_argument('--epoch', type=int, default=250)
    parser.add_argument('--array', type=int, default=1)

    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--buffer_size', type=int, default=20)
    parser.add_argument('--max_len', type=int, default=50)
    parser.add_argument('--link', type=str, default='log')
    parser.add_argument('--l1_lambda', type=float, default=1.0)
    parser.add_argument('--random_weight', type=float, default=0.3)
    parser.add_argument('--true_value', type=float, default=1.0)
    args = parser.parse_args()
    return args

def load_dataset(log_dir,name,buffer_size,max_len,state_dim, action_dim):
    replay_buffer = utils.ReplayBuffer(state_dim, action_dim)
    with open(log_dir + name + '.pkl', 'rb') as outp:
        buf = pickle.load(outp)
    i = 0
    while i < buffer_size:
        if i % max_len == 0:
            replay_buffer.add_start(buf.obs_buf[i])
        done=False
        if i % max_len == max_len-1:
            done = True
        replay_buffer.add(buf.obs_buf[i], buf.act_buf[i], buf.next_obs_buf[i], buf.rew_buf[i], done)
        i += 1
    return replay_buffer

def run(size,length,random_weight,discount_factor,seed,num_steps,checkpoint):
    args = argsparser()

    file_name = "%s_%s_%s_%s" % (args.policy, args.env, str(args.seed), str(args.random))
    print("---------------------------------------")
    print(f"Settings: {file_name}")
    print("---------------------------------------")

    if not os.path.exists("./results"):
        os.makedirs("./results")

    env = gym.make(args.env)

    # Set seeds
    env.seed(args.seed)
    env.action_space.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    kwargs = {
        "state_dim": state_dim,
        "action_dim": action_dim,
        "max_action": max_action,
        "discount": discount_factor,
        "tau": args.tau,
    }

    # Initialize policy
    if args.policy == "SR_DICE":
        ope = SR_DICE.SR_DICE(**kwargs)
    if args.policy == "Deep_SR":
        ope = Deep_SR.Deep_SR(**kwargs)
    if args.policy == "Deep_TD":
        ope = Deep_TD.Deep_TD(**kwargs)

    kwargs["policy_noise"] = 0.2 * max_action
    kwargs["noise_clip"] = 0.5 * max_action
    kwargs["policy_freq"] = 2
    policy = TD3.TD3(**kwargs)

    name = ['discount_factor', 0.8, 'random_weight', random_weight, 'max_length', length,
            'buffer_size', 16000, 'seed', seed, 'env', env]
    name = '-'.join(str(x) for x in name)

    replay_buffer = load_dataset(args.data_dir, '/dataset/' + name, size, length, state_dim, action_dim)
    name = ['discount_factor', 0.8, 'random_weight', random_weight, 'max_length', length,
            'buffer_size', 16000, 'seed', seed + 1314, 'env', env]
    name = '-'.join(str(x) for x in name)

    replay_buffer_test = load_dataset(args.data_dir, '/dataset_test/' + name, size, length,
                                      state_dim, action_dim)

    # Train and evaluate OPE
    train_results, test_results = [], []
    if args.policy == "SR_DICE" or args.policy == "Deep_SR":

        print("Train Encoder-Decoder")

        for k in range(int(3e4)):
            if k % 1e3 == 0:
                print("k", k)
            ope.train_encoder_decoder(replay_buffer)

        print("Train SR")

        for k in range(int(1e5)):
            if k % 1e3 == 0:
                print("k", k)
            ope.train_SR(replay_buffer, policy.actor)

    print("Train MIS")

    for k in range(int(num_steps)):
        ope.train_OPE(replay_buffer, policy.actor)

        if k % checkpoint == 0:
            print("k", k)
            train_results.append(ope.eval_policy(replay_buffer, policy.actor))
            test_results.append(ope.eval_policy(replay_buffer_test, policy.actor))

def run_mujoco():
    args = argsparser()
    seeds = range(10)

    discount_factor_lists = [0.8, 0.9, 0.95, 0.99, 0.995]
    size_lists = [2000, 4000, 8000, 16000]

    weight_lists = [1.4, 1.8, 2.0, 2.4, 2.8]
    length_lists = [20, 50, 100, 200]
    env = ['MountainCarContinuous-v0','Hopper-v4','HalfCheetah-v4','Ant-v4',
           'Swimmer-v4','Walker2d-v4']
    path = ['./exper/mountaincar.pth','./exper/hopper.pth','./exper/halfcheetah_0.pth',
            './exper/ant.pth','./exper/swimmer.pth','./exper/walker.pth']
    idx = np.unravel_index(args.array, (5, 4, 5, 6))
    random_weight, length, discount_factor = (
        weight_lists[idx[0]],
        length_lists[idx[1]],
        discount_factor_lists[idx[2]],
    )
    env, path = env[idx[3]], path[idx[3]]

    filename = args.log_dir + +str(args.policy)+'-mujoco-' + str(env) +'-discount-'+str(discount_factor)\
               +'-length-'+str(length)+'-random-'+str(random_weight)+'.csv'
    os.makedirs(args.log_dir, exist_ok=True)
    mylist = [str(i) for i in range(0, args.epoch * args.steps, args.steps)] + ['hyperparam']
    with open(filename, 'w+', newline='') as file:
        # Step 4: Using csv.writer to write the list to the CSV file
        writer = csv.writer(file)
        writer.writerow(mylist)  # Use writerow for single list

    result_train, result_test = [], []
    for seed in seeds:
        for size in size_lists:
            train,test = run(size,length,random_weight,discount_factor,seed,args.steps*args.epoch,args.steps)
            train, test = np.around(train, decimals=4), np.around(test, decimals=4)
            result_train.append(train)
            result_test.append(test)
            mylist = [str(i) for i in list(train)] + ['-'.join(['train', 'size', str(size), 'seed', str(seed)])]
            with open(filename, 'a', newline='') as file:
                # Step 4: Using csv.writer to write the list to the CSV file
                writer = csv.writer(file)
                writer.writerow(mylist)  # Use writerow for single list
            mylist = [str(i) for i in list(test)] + ['-'.join(['test', 'size', str(size), 'seed', str(seed)])]
            with open(filename, 'a', newline='') as file:
                # Step 4: Using csv.writer to write the list to the CSV file
                writer = csv.writer(file)
                writer.writerow(mylist)  # Use writerow for single list

run_mujoco()
