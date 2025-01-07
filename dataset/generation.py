import os, sys, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
import numpy as np
from gymnasium.spaces import Box, Discrete
import gymnasium as gym
import time
import torch
import torch.nn as nn
from torch.optim import Adam
import ppo.algo.core as core
from ppo.algo.random_search import random_search
from torch.distributions.normal import Normal
from avg_corr.td_err import TD_computation
import matplotlib.pyplot as plt
import pickle


class PPOBuffer:
    """
    A buffer for storing trajectories experienced by a PPO agent interacting
    with the environment, and using Generalized Advantage Estimation (GAE-Lambda)
    for calculating the advantages of state-action pairs.
    """

    def __init__(self, obs_dim, act_dim, size, fold, gamma=0.99):
        self.obs_buf = np.zeros(core.combined_shape(size, obs_dim), dtype=np.float32)
        self.next_obs_buf = np.zeros(core.combined_shape(size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(core.combined_shape(size, act_dim), dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.tim_buf = np.zeros(size, dtype=np.int32)
        self.logtarg_buf = np.zeros(size, dtype=np.float32)
        self.prod_buf = np.zeros(size, dtype=np.float32)
        self.logbev_buf = np.zeros(size, dtype=np.float32)
        self.gamma = gamma
        self.fold = fold
        self.ptr, self.path_start_idx, self.max_size = 0, 0, size

    def store(self, obs, next_obs,act, rew, tim, logbev, logtarg):
        """
        Append one timestep of agent-environment interaction to the buffer.
        """
        assert self.ptr < self.max_size  # buffer has to have room so you can store
        self.obs_buf[self.ptr] = obs
        self.next_obs_buf[self.ptr] = next_obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.tim_buf[self.ptr] = tim
        self.logbev_buf[self.ptr] = logbev
        self.logtarg_buf[self.ptr] = logtarg
        self.ptr += 1

    def finish_path(self):
        """
        Call this at the end of a trajectory, or when one gets cut off
        by an epoch ending. This looks back in the buffer to where the
        trajectory started, and uses rewards and value estimates from
        the whole trajectory to compute advantage estimates with GAE-Lambda,
        as well as compute the rewards-to-go for each state, to use as
        the targets for the value function.

        The "last_val" argument should be 0 if the trajectory ended
        because the agent reached a terminal state (died), and otherwise
        should be V(s_T), the value function estimated for the last state.
        This allows us to bootstrap the reward-to-go calculation to account
        for timesteps beyond the arbitrary episode horizon (or epoch cutoff).
        """

        path_slice = slice(self.path_start_idx, self.ptr)

        # the next two lines implement GAE-Lambda advantage calculation
        deltas = self.logtarg_buf - self.logbev_buf
        self.prod_buf[path_slice] = np.append(0,core.discount_cumsum(deltas[path_slice], 1)[:-1])

        self.path_start_idx = self.ptr

    def sample(self,batch_size,fold_num):
        """
        Call this at the end of an epoch to get all of the data from
        the buffer, with advantages appropriately normalized (shifted to have
        mean zero and std one). Also, resets some pointers in the buffer.
        """
        interval = int(self.ptr / self.fold)
        if self.fold>1:
            ind = np.random.randint(self.ptr-interval, size=batch_size)
            ind = ind + np.where(ind >= fold_num * interval, 1, 0) * interval
        else:
            ind = np.random.randint(self.ptr, size=batch_size)

        data = dict(obs=self.obs_buf[ind], act=self.act_buf[ind],
                    prod=self.prod_buf[ind],tim=self.tim_buf[ind],
                    logbev=self.logbev_buf[ind], logtarg=self.logtarg_buf[ind])
        return {k: torch.as_tensor(v, dtype=torch.float32) for k, v in data.items()}

    def delete_last_traj(self):
        self.ptr =self.path_start_idx

# load target policy
def load(path,env):
    ac_kwargs = dict(hidden_sizes=[64,32])

    ac = core.MLPActorCritic(env.observation_space, env.action_space, **ac_kwargs)
    checkpoint = torch.load(path)
    ac.load_state_dict(checkpoint['model_state_dict'])
    return ac

def eval_policy(path='./exper/cartpole.pth',env='CartPole-v1',gamma=0.8):
    env = gym.make(env)
    ac = load(path, env)

    o , _ = env.reset()
    ep_len, ep_ret, ep_avg_ret = 0 ,0, 0
    num_traj=0
    rets = []
    avg_rets = []

    while num_traj<100:
        a, _,logtarg = ac.step(torch.as_tensor(o, dtype=torch.float32))
        next_o, r, d,truncated, _ = env.step(a)
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
            ep_ret, ep_len, ep_avg_ret = 0, 0, 0
    return (1-gamma)*np.mean(rets),np.var(rets),np.mean(avg_rets)

# sample behaviour dataset
# behaviour policy = (1- random_weight) * target_policy + random_weight * random_policy
# behaviour policy = Normal(target_mu, torch.exp(random_weight))
def collect_dataset(env,gamma,buffer_size=20,max_len=200,
                    path='./exper/cartpole_998.pth', random_weight=0.2,fold=10,continuous=False):
    ac = load(path,env)
    obs_dim = env.observation_space.shape
    act_dim = env.action_space.shape

    buf=PPOBuffer(obs_dim, act_dim, buffer_size*max_len,fold, gamma)

    o, _ = env.reset()
    num_traj, ep_len = 0, 0

    if isinstance(env.action_space, Box):
        action_range = env.action_space.high - env.action_space.low
        assert np.any(action_range > 0)
        unif = 1 / np.prod(action_range)
    elif isinstance(env.action_space, Discrete):
        unif = 1 / env.action_space.n

    while num_traj < buffer_size:
        pi = ac.pi._distribution(torch.as_tensor(o, dtype=torch.float32))
        if continuous:
            std = torch.exp(ac.pi.log_std)
            mu = ac.pi.mu_net(torch.as_tensor(o, dtype=torch.float32))
            beh_pi = Normal(mu, std * random_weight)
            a = beh_pi.sample().numpy()
            logtarg = pi.log_prob(torch.as_tensor(a)).sum(axis=-1).detach().numpy()
            logbev = beh_pi.log_prob(torch.as_tensor(a)).sum(axis=-1).detach().numpy()
        else:
            targ_a, _, _ = ac.step(torch.as_tensor(o, dtype=torch.float32))
            if np.random.random() < random_weight:
                # random behaviour policy
                a = env.action_space.sample()
            else:
                a = targ_a
            logtarg = ac.pi._log_prob_from_distribution(pi, torch.as_tensor(a)).detach().numpy()
            logbev = np.log(random_weight * unif + (1 - random_weight) * np.exp(logtarg))

        next_o, r, d, truncated, _ = env.step(a)
        ep_len += 1

        # save and log
        buf.store(o, next_o,a, r, ep_len - 1, logbev, logtarg)

        # Update obs (critical!)
        o = next_o

        terminal = d or truncated
        epoch_ended = ep_len == max_len

        if terminal or epoch_ended:
            # if terminal and not (epoch_ended):
            #     o = env.reset()
            # else:
            #     buf.finish_path()
            #     o, ep_ret, ep_len = env.reset(), 0, 0
            #     num_traj += 1
            if terminal and not (epoch_ended):
                # print('Warning: trajectory ends early at %d steps.' % ep_len, flush=True)
                buf.delete_last_traj()
                o, _ = env.reset()
                ep_len, ep_ret, ep_avg_ret = 0, 0, 0
                continue
            o, _ = env.reset()
            ep_len, ep_ret, ep_avg_ret = 0, 0, 0
            num_traj += 1
            buf.finish_path()
    return buf

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

def data(log_dir,seed,gamma,weight,length,size,env,path,continuous=False):
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

    name = ['discount_factor', gamma, 'random_weight', weight, 'max_length', length,
            'buffer_size', size, 'seed', seed,'env',env]
    name = '-'.join(str(x) for x in name)
    print(name)

    env = gym.make(env)
    env.reset(seed=seed)

    num_traj = size//length
    buf = collect_dataset(env, gamma, buffer_size=num_traj, max_len=length, path=path,
                          random_weight=weight, fold=1, continuous=continuous)

    with open(log_dir+name+'.pkl', 'wb') as outp:
        pickle.dump(buf, outp, pickle.HIGHEST_PROTOCOL)


# print(eval_policy(path='./exper/hopper.pth',env='Hopper-v4',gamma=0.95))

def main():
    log_dir = './dataset_test/'
    discounts = [0.8, 0.9, 0.95, 0.99 ,0.995]
    discounts=[0.8]
    sizes = [2000,4000,8000,16000]
    sizes=[16000]
    seeds = range(1314,1324,1)

    weights= [0.1,0.2, 0.3,0.4,0.5]
    # weights=[0.5]
    lengths = [20,40,80,100]
    envs = ['CartPole-v1', 'Acrobot-v1']
    paths = ['./exper/cartpole.pth', './exper/acrobot.pth']
    continuous = False

    for gamma in discounts:
        for size in sizes:
            for weight in weights:
                for length in lengths:
                    for seed in seeds:
                        for i in range(len(envs)):
                            env, path = envs[i], paths[i]
                            data(log_dir,seed,gamma,weight,length,size,env,path,continuous)


    weights = [1.4,1.8,2.0,2.4,2.8]
    lengths = [20,50,100,200]
    envs = ['MountainCarContinuous-v0', 'Hopper-v4', 'HalfCheetah-v4', 'Ant-v4', 'Walker2d-v4']
    paths = ['./exper/mountaincar.pth', './exper/hopper.pth', './exper/halfcheetah_1.pth',
            './exper/ant.pth', './exper/walker.pth']
    continuous = True

    for gamma in discounts:
        for size in sizes:
            for weight in weights:
                for length in lengths:
                    for seed in seeds:
                        for i in range(len(envs)):
                            env, path = envs[i], paths[i]
                            data(log_dir,seed,gamma,weight,length,size,env,path,continuous)

swimmer = []
for gamma in [0.99]:
    value,_,_ = eval_policy(path='./exper/halfcheetah_0.pth',env='HalfCheetah-v4',gamma=gamma)
    print(gamma,":::",value)
    swimmer.append(value)
