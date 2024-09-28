import os, sys, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
import numpy as np
import csv,pickle
from avg_corr.main import load
from gymnasium.spaces import Box, Discrete
import gymnasium as gym
import torch
from torch.distributions.normal import Normal

def argsparser():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default='./')
    parser.add_argument('--log_dir', type=str, default='./')
    parser.add_argument('--env', type=str, default='Hopper-v4')
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--steps', type=int, default=5)
    parser.add_argument('--epoch', type=int, default=250)
    parser.add_argument('--array', type=int, default=1)
    args = parser.parse_args()
    return args

def eval_policy(path='./exper/cartpole.pth',env='CartPole-v1',gamma=0.8):
    env = gym.make(env)
    ac = load(path, env)

    o, _ = env.reset()
    ep_len, ep_ret, ep_avg_ret = 0, 0, 0
    num_traj=0
    rets = []
    avg_rets = []

    while num_traj<100:
        a, _,logtarg = ac.step(torch.as_tensor(o, dtype=torch.float32))
        next_o, r, d, truncated, _ = env.step(a)
        ep_ret += r * gamma ** ep_len
        ep_avg_ret += r
        ep_len += 1
        # Update obs (critical!)
        o = next_o

        terminal = d or truncated

        if terminal:
            num_traj += 1
            rets.append(ep_ret)
            avg_rets.append(ep_avg_ret)
            o, _ = env.reset()
            ep_len, ep_ret, ep_avg_ret = 0, 0, 0
    return (1-gamma)*np.mean(rets),np.var(rets),np.mean(avg_rets)

def eval_behaviour_policy(path='./exper/cartpole.pth',env='CartPole-v1',
                          gamma=0.8,mujoco=False,random_weight=0.0):
    env = gym.make(env)
    ac = load(path, env)
    act_dim = env.action_space.shape

    o, _ = env.reset()
    ep_len, ep_ret, ep_avg_ret = 0, 0, 0
    num_traj=0
    rets = []
    avg_rets = []

    while num_traj<100:
        # pi = ac.pi._distribution(torch.as_tensor(o, dtype=torch.float32))
        if mujoco:
            std = torch.exp(ac.pi.log_std)
            mu = ac.pi.mu_net(torch.as_tensor(o, dtype=torch.float32))
            beh_pi = Normal(mu, std * random_weight)
            a = beh_pi.sample().numpy()
        else:
            targ_a, _, _ = ac.step(torch.as_tensor(o, dtype=torch.float32))
            if np.random.random() < random_weight:
                # random behaviour policy
                a = env.action_space.sample()
            else:
                a = targ_a
        next_o, r, d, truncated, _ = env.step(a)
        ep_ret += r * gamma ** ep_len
        ep_avg_ret += r
        ep_len += 1
        # Update obs (critical!)
        o = next_o

        terminal = d or truncated

        if terminal:
            num_traj += 1
            rets.append(ep_ret)
            avg_rets.append(ep_avg_ret)
            o, _ = env.reset()
            ep_len, ep_ret, ep_avg_ret = 0, 0, 0
    return (1-gamma)*np.mean(rets),np.var(rets),np.mean(avg_rets)


def eval_classic():
    args = argsparser()
    discount_factor = [0.8, 0.9,0.95, 0.99, 0.995]
    random_weight = [0.1, 0.2, 0.3, 0.4,0.5,0.6]
    env = ['CartPole-v1', 'Acrobot-v1', 'MountainCarContinuous-v0']
    path = ['./exper/cartpole.pth', './exper/acrobot.pth', './exper/mountaincar.pth']
    obj  ={}
    tar_rets, beh_rets = {}, {}
    for gamma in discount_factor:
        for random in random_weight:
            for i in range(len(env)):
                target_obj,_,target_ret = eval_policy(path[i],env[i],gamma)
                _,_,beh_ret = eval_behaviour_policy(path[i],env[i],gamma,mujoco=False,random_weight=random)
                name = [str(env[i]),str(random),str(gamma)]
                obj['-'.join(name)+'-obj'] = target_obj
                tar_rets['-'.join(name)+'-tar_ret'] = target_ret
                beh_rets['-'.join(name)+'-beh_ret'] = beh_ret
    filename = args.log_dir + 'classic_values.csv'
    with open(filename, 'w') as csv_file:
        writer = csv.writer(csv_file)
        for key, value in obj.items():
            writer.writerow([key, value])
        for key, value in tar_rets.items():
            writer.writerow([key, value])
        for key, value in beh_rets.items():
            writer.writerow([key, value])

def eval_mujoco():
    args = argsparser()
    discount_factor = [0.8, 0.9,0.95, 0.99, 0.995]
    # discount_factor= [0.95]
    random_weight = [1.4,1.8,2.0,2.4,2.8,3.2]
    # random_weight = [0.25]
    env = ['MountainCarContinuous-v0','Hopper-v4','HalfCheetah-v4','Ant-v4','Walker2d-v4']
    path = ['./exper/mountaincar.pth','./exper/hopper.pth','./exper/halfcheetah_1.pth',
            './exper/ant.pth','./exper/walker.pth']
    # env = ['Hopper-v4']
    # path = ['./exper/hopper.pth']

    obj = {}
    tar_rets, beh_rets = {}, {}
    for gamma in discount_factor:
        for random in random_weight:
            for i in range(len(env)):
                target_obj, _, target_ret = eval_policy(path[i], env[i], gamma)
                _, _, beh_ret = eval_behaviour_policy(path[i], env[i], gamma, mujoco=True, random_weight=random)
                name = [str(env[i]), str(random), str(gamma)]
                obj['-'.join(name) + '-obj'] = target_obj
                tar_rets['-'.join(name) + '-tar_ret'] = target_ret
                beh_rets['-'.join(name) + '-beh_ret'] = beh_ret
                print('-'.join(name), ":::",beh_ret)
    filename = args.log_dir + 'mujoco_values.csv'
    with open(filename, 'w') as csv_file:
        writer = csv.writer(csv_file)
        for key, value in obj.items():
            writer.writerow([key, value])
        for key, value in tar_rets.items():
            writer.writerow([key, value])
        for key, value in beh_rets.items():
            writer.writerow([key, value])

eval_classic()
eval_mujoco()