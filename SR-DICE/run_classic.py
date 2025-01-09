import os, sys, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
import numpy as np
import csv
from dataset.generation import PPOBuffer
import pickle
from tqdm import tqdm

import utils
import Deep_TD
import Deep_SR
import SR_DICE
import torch
from gymnasium.spaces import Box, Discrete
import gymnasium as gym
from main import path_lists

from ppo.algo import core

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def argsparser():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default='./')
    parser.add_argument('--log_dir', type=str, default='./')
    parser.add_argument('--data_dir', type=str, default='/scratch/fengdic/')
    parser.add_argument('--policy', type=str, default='Deep_TD')

    parser.add_argument('--env', type=str, default='Hopper-v4')
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--steps', type=int, default=5)
    parser.add_argument('--epoch', type=int, default=250)
    parser.add_argument('--array', type=int, default=1)

    parser.add_argument("--tau", default=0.005)  # Target network update rate

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

# load target policy
def load(path,env):
    ac_kwargs = dict(hidden_sizes=[64,32])

    ac = core.MLPActorCritic(env.observation_space, env.action_space, **ac_kwargs)
    checkpoint = torch.load(path)
    ac.load_state_dict(checkpoint['model_state_dict'])
    return ac

def run(args,env_name,seed,size,length,random_weight,discount_factor,num_steps,checkpoint):
    env_name = env_name
    env = gym.make(env_name)
    path = path_lists[env_name]

    # Set seeds
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    env.reset(seed=seed)
    env.action_space.seed(seed)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space
    max_action = float(1)

    kwargs = {
        "state_dim": state_dim,
        "action_dim": action_dim.n,
        "max_action": max_action,
        "discount": discount_factor,
        "tau": args.tau,
        "mujoco":False,
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
    policy = load(path,env)

    name = ['discount_factor', 0.8, 'random_weight', random_weight, 'max_length', length,
            'buffer_size', 16000, 'seed', seed, 'env', env_name]
    name = '-'.join(str(x) for x in name)

    replay_buffer = load_dataset(args.data_dir, '/dataset/' + name, size, length, state_dim, action_dim.shape)
    name = ['discount_factor', 0.8, 'random_weight', random_weight, 'max_length', length,
            'buffer_size', 16000, 'seed', seed + 1314, 'env', env_name]
    name = '-'.join(str(x) for x in name)

    replay_buffer_test = load_dataset(args.data_dir, '/dataset_test/' + name, size, length,
                                      state_dim, action_dim.shape)

    # Train and evaluate OPE
    train_results, test_results = [], []
    if args.policy == "SR_DICE" or args.policy == "Deep_SR":

        print("Train Encoder-Decoder")

        for k in range(int(1e3)):
            # for k in range(int(3e4)):
            if k % 1e3 == 0:
                print("k", k)
            ope.train_encoder_decoder(replay_buffer)

        print("Train SR")

        # for k in range(int(1e5)):
        for k in range(int(1e3)):
            if k % 1e3 == 0:
                print("k", k)
            ope.train_SR(replay_buffer, policy)

    print("Train MIS")

    for k in range(int(num_steps)):
        ope.train_OPE(replay_buffer, policy)

        if k % checkpoint == 0:
            print("k", k)
            train_results.append(ope.eval_policy(replay_buffer, policy))
            test_results.append(ope.eval_policy(replay_buffer_test, policy))
    if args.policy == "SR_DICE":
        dir = os.path.join(args.log_dir, str(env_name))
        torch.save(
            {'encoder_dict': ope.encoder_decoder.state_dict(),
             'weight_dict': ope.W.state_dict()},
            dir
        )
        print("model saved")
    return train_results, test_results

def run_mujoco():
    args = argsparser()
    seed = args.seed

    if args.array >= 36:
        return -1

    discount_factor_lists = [0.8, 0.9, 0.95, 0.99, 0.995]
    size_lists = [2000, 4000, 8000, 16000]

    weight_lists = [0.1, 0.2, 0.3, 0.4, 0.5]
    length_lists = [20, 40, 80, 100]
    env = ['CartPole-v1', 'Acrobot-v1']
    path = ['./exper/cartpole.pth', './exper/acrobot.pth']
    random_weight, length, discount_factor, size = (
        0.3,
        40,
        0.95,
        4000,
    )
    idx = np.unravel_index(args.array, (18, 2))
    if idx[0] < 5:
        discount_factor = discount_factor_lists[idx[0]]
    elif idx[0] < 9:
        size = size_lists[idx[0] - 5]
    elif idx[0] < 14:
        random_weight = weight_lists[idx[0] - 9]
    else:
        length = length_lists[idx[0] - 14]
    env = env[idx[1]]

    os.makedirs(args.log_dir, exist_ok=True)
    dir = os.path.join(args.log_dir, str(env))
    os.makedirs(dir, exist_ok=True)

    filename = dir + f"{args.policy}-classic-{env}-discount-{discount_factor}-length-{length}-random-{random_weight}-size-{size}-seed-{seed}.csv"

    mylist = [str(i) for i in range(0, args.epoch * args.steps, args.steps)] + ['hyperparam']
    with open(filename, 'w+', newline='') as file:
        # Step 4: Using csv.writer to write the list to the CSV file
        writer = csv.writer(file)
        writer.writerow(mylist)  # Use writerow for single list

    result_train, result_test = [], []
    train,test = run(
        args=args,
        env_name=env,
        seed=seed,
        size=size,
        length=length,
        random_weight=random_weight,
        discount_factor=discount_factor,
        num_steps = args.steps*args.epoch,
        checkpoint=args.steps,
    )
    train, test = np.around(train, decimals=4), np.around(test, decimals=4)
    result_train.append(train)
    result_test.append(test)
    mylist = [str(i) for i in list(train)] + ['-'.join(['train', 'seed', str(seed)])]
    with open(filename, 'a', newline='') as file:
        # Step 4: Using csv.writer to write the list to the CSV file
        writer = csv.writer(file)
        writer.writerow(mylist)  # Use writerow for single list
    mylist = [str(i) for i in list(test)] + ['-'.join(['test', 'seed', str(seed)])]
    with open(filename, 'a', newline='') as file:
        # Step 4: Using csv.writer to write the list to the CSV file
        writer = csv.writer(file)
        writer.writerow(mylist)  # Use writerow for single list
    print("done")


run_mujoco()
