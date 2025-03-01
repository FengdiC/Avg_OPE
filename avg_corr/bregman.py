import os, sys, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
import numpy as np
from gym.spaces import Box, Discrete
import gym
import time
import torch
import torch.nn as nn
from torch.optim import Adam, SGD
import ppo.algo.core as core
from ppo.algo.random_search import random_search
import ppo.utils.logger as logger
import matplotlib.pyplot as plt
import csv


class PPOBuffer:
    """
    A buffer for storing trajectories experienced by a PPO agent interacting
    with the environment, and using Generalized Advantage Estimation (GAE-Lambda)
    for calculating the advantages of state-action pairs.
    """

    def __init__(self, obs_dim, act_dim, size, fold, gamma=0.99):
        self.obs_buf = np.zeros(core.combined_shape(size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(core.combined_shape(size, act_dim), dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.tim_buf = np.zeros(size, dtype=np.int32)
        self.logtarg_buf = np.zeros(size, dtype=np.float32)
        self.prod_buf = np.zeros(size, dtype=np.float32)
        self.logbev_buf = np.zeros(size, dtype=np.float32)
        self.gamma = gamma
        self.fold = fold
        self.ptr, self.path_start_idx, self.max_size = 0, 0, size

    def store(self, obs, act, rew, tim, logbev, logtarg):
        """
        Append one timestep of agent-environment interaction to the buffer.
        """
        assert self.ptr < self.max_size  # buffer has to have room so you can store
        self.obs_buf[self.ptr] = obs
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
        if self.fold > 1:
            ind = np.random.randint(self.ptr - interval, size=batch_size)
            ind = ind + np.where(ind >= fold_num * interval, 1, 0) * interval
        else:
            ind = np.random.randint(self.ptr, size=batch_size)

        data = dict(obs=self.obs_buf[ind], act=self.act_buf[ind], prod=self.prod_buf[ind],
                    tim=self.tim_buf[ind],
                    logbev=self.logbev_buf[ind], logtarg=self.logtarg_buf[ind])
        return {k: torch.as_tensor(v, dtype=torch.float32) for k, v in data.items()}

    def delete_last_traj(self):
        self.ptr =self.path_start_idx

class WeightNet(nn.Module):
    def __init__(self, o_dim, hidden_sizes,activation):
        super(WeightNet, self).__init__()
        sizes = [o_dim] + list(hidden_sizes)
        print(sizes)
        layers = []
        for j in range(len(sizes) - 1):
            layers += [nn.Linear(sizes[j], sizes[j + 1]),nn.Tanh()]
        self.body = nn.Sequential(*layers)
        self.weight = nn.Sequential(nn.Linear(sizes[-1], 1),activation())  #nn.Identity()

    def forward(self, obs):
        obs = obs.float()
        body = self.body(obs)
        weight = self.weight(body)
        return torch.squeeze(weight)

# load target policy
def load(path,env):
    ac_kwargs = dict(hidden_sizes=[64,32])

    ac = core.MLPActorCritic(env.observation_space, env.action_space, **ac_kwargs)
    checkpoint = torch.load(path)
    ac.load_state_dict(checkpoint['model_state_dict'])
    return ac

def eval_policy(path='./exper/cartpole_998.pth'):
    env = gym.make('MountainCarContinuous-v0')
    ac = load(path, env)
    hyperparam = random_search(196)
    gamma = hyperparam['gamma']

    o, ep_len, ep_ret, ep_avg_ret = env.reset(), 0 ,0, 0
    num_traj=0
    rets = []
    avg_rets = []

    while num_traj<100:
        a, _,logtarg = ac.step(torch.as_tensor(o, dtype=torch.float32))
        next_o, r, d, _ = env.step(a)
        ep_ret += r * gamma ** ep_len
        ep_avg_ret += r
        ep_len += 1
        # Update obs (critical!)
        o = next_o

        terminal = d

        if terminal:
            num_traj+=1
            rets.append(ep_ret)
            avg_rets.append(ep_avg_ret)
            o, ep_ret, ep_len, ep_avg_ret = env.reset(), 0, 0, 0
    return (1-gamma)*np.mean(rets),np.var(rets),np.mean(avg_rets)

# sample behaviour dataset
# behaviour policy = (1- random_weight) * target_policy + random_weight * random_policy
def collect_dataset(env,gamma,buffer_size=20,max_len=200,
                    path='./exper/cartpole_998.pth', random_weight=0.2,fold=10):
    ac = load(path,env)
    obs_dim = env.observation_space.shape
    act_dim = env.action_space.shape

    buf=PPOBuffer(obs_dim, act_dim, buffer_size*max_len,fold, gamma)

    o, ep_len = env.reset(), 0
    num_traj = 0

    if isinstance(env.action_space, Box):
        action_range = env.action_space.high - env.action_space.low
        assert np.any(action_range > 0)
        unif = 1 / np.prod(action_range)
    elif isinstance(env.action_space, Discrete):
        unif = 1 / env.action_space.n

    while num_traj < buffer_size:
        targ_a, _, _ = ac.step(torch.as_tensor(o, dtype=torch.float32))
        if np.random.random() < random_weight:
            # random behaviour policy
            a = env.action_space.sample()
        else:
            a = targ_a
        pi = ac.pi._distribution(torch.as_tensor(o, dtype=torch.float32))
        logtarg = ac.pi._log_prob_from_distribution(pi, torch.as_tensor(a)).detach().numpy()
        logbev = np.log(random_weight * unif + (1 - random_weight) * np.exp(logtarg))
        next_o, r, d, _ = env.step(a)
        ep_len += 1

        # save and log
        buf.store(o, a, r, ep_len - 1, logbev, logtarg)

        # Update obs (critical!)
        o = next_o

        terminal = d
        epoch_ended = ep_len == max_len

        if terminal or epoch_ended:
            if terminal and not (epoch_ended):
                o = env.reset()
            else:
                buf.finish_path()
                o, ep_ret, ep_len = env.reset(), 0, 0
                num_traj += 1
    return buf

# train weight net
def train(lr, env,seed,path,hyper_choice,link,random_weight,l1_lambda,reg_lambda=0,discount=0.95,
          checkpoint=5,epoch=1000,cv_fold=10,batch_size=256,buffer_size=20,max_len=50):
    hyperparam = random_search(hyper_choice)
    gamma = discount
    env = gym.make(env)

    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    env.seed(seed)

    buf = collect_dataset(env,gamma,buffer_size=buffer_size,max_len=max_len,path=path,
                            random_weight=random_weight,fold = cv_fold)
    buf_test = collect_dataset(env, gamma,buffer_size=buffer_size,max_len=max_len,
                                      path=path,random_weight=random_weight,fold=cv_fold)
    if link=='inverse' or link=='identity':
        weight = WeightNet(env.observation_space.shape[0], hidden_sizes=(256,256),activation=nn.ReLU)
    else:
        weight = WeightNet(env.observation_space.shape[0], hidden_sizes=(256,256), activation=nn.Identity)

    start_time = time.time()

    # Set up optimizers for policy and value function
    optimizer = Adam(weight.parameters(), lr)

    def update(fold_num):
        #sample minibatches
        data = buf.sample(batch_size,fold_num)

        obs, act = data['obs'], data['act']
        tim, prod = data['tim'], data['prod']
        logtarg, logbev = data['logtarg'], data['logbev']

        f = weight(obs)
        label = np.exp(np.log(gamma) * tim + prod)
        if link == "inverse":
            loss = (-torch.log(f) + f * label).mean()
            ratio = 1/(f+0.0001)
        elif link == "identity":
            loss = (torch.log(weight(obs)) + label/f).mean()
            ratio = f
        elif link == 'loglog':
            loss = (torch.log(torch.exp(torch.exp(f))-1)+label/torch.exp(torch.exp(f))-1).mean()
            ratio = torch.exp(torch.exp(f))
        else:
            loss = (f + label/ torch.exp(f)).mean()
            ratio = torch.exp(f)

        regularizer = torch.mean((ratio * torch.exp(logtarg - logbev) * max_len * (1 - gamma) - 1)**2)

        l1_norm = sum(torch.linalg.norm(p, 1) for p in weight.parameters())
        loss = loss + l1_lambda * l1_norm + reg_lambda * regularizer

        optimizer.zero_grad()
        loss.backward()
        # for p in weight.parameters():
        #     print(p.grad.norm())
        optimizer.step()

    def eval(buffer):
        ratio = weight(torch.as_tensor(buffer.obs_buf[:buffer.ptr],dtype=torch.float32)).detach().numpy()
        if link == "inverse":
            ratio = 1/(ratio+0.001)
        elif link =='identity':
            pass
        elif link=='loglog':
            ratio = np.exp(np.exp(ratio))
        else:
            ratio = np.exp(ratio)
        obj = np.mean(ratio * np.exp(buffer.logtarg_buf[:buffer.ptr]
                                     - buffer.logbev_buf[:buffer.ptr])*buffer.rew_buf[:buffer.ptr])
        return obj*max_len*(1-gamma)

    def eval_cv(buffer,fold_num):
        interval = int(buffer.ptr/buffer.fold)
        ind = range(fold_num* interval,(fold_num+1)* interval,1)
        other = np.ones(buffer.ptr)
        other[ind] = 0
        ratio = weight(torch.as_tensor(buffer.obs_buf,dtype=torch.float32)).detach().numpy()
        if link == "inverse":
            ratio = 1 / (ratio + 0.001)
        elif link == 'identity':
            pass
        elif link == 'loglog':
            ratio = np.exp(np.exp(ratio))-1
        else:
            ratio = np.exp(ratio)
        # print(ratio)
        obj = np.mean(ratio[ind] * np.exp(buffer.logtarg_buf[ind]
                                     - buffer.logbev_buf[ind])*buffer.rew_buf[ind])
        obj_other = np.mean(ratio[other] * np.exp(buffer.logtarg_buf[other]
                                     - buffer.logbev_buf[other]) * buffer.rew_buf[other])
        return obj*max_len*(1-gamma),obj_other*max_len*(1-gamma)

    objs_mean = []
    objs_val_mean = []
    for fold_num in range(cv_fold):
        objs, objs_test,objs_val = [], [], []
        for steps in range(epoch * checkpoint):
            update(fold_num)
            if steps % checkpoint == 0:
                obj_val, obj = eval_cv(buf, fold_num)
                # obj, obj_test = eval(buf), eval(buf_test)
                objs.append(obj)
                objs_val.append(obj_val)
                # objs_cv.append(np.around(obj_cv,decimals=4))
        objs_mean.append(objs)
        objs_val_mean.append(objs_val)
    # return objs, objs_test
    return np.around(np.mean(np.array(objs_mean),axis=0),decimals=4),\
           np.around(np.mean(np.array(objs_val_mean),axis=0),decimals=4)

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

def tune():
    alpha = [0, 0.0005, 0.001, 0.002, 0.005, 0.01]
    batch_size = [256, 512]
    link = ['inverse', 'identity']
    lr = [0.0001, 0.0005, 0.001, 0.005]
    reg_lambda = [0.1,0.5,1,2,5]

    args = argsparser()
    seeds = range(5)
    idx = np.unravel_index(args.array, (6, 2, 2, 4, 5))

    random_weight, buffer_size = 0.7, 40
    discount_factor = 0.95
    alpha, lr, reg_lambda = alpha[idx[0]], lr[idx[3]], reg_lambda[idx[4]]
    link, batch_size = link[idx[1]], batch_size[idx[2]]

    filename = args.log_dir + 'gamma-tune-square-reg-alpha-' + str(alpha) + '-lr-' \
               + str(lr) + '-lambda-' + str(reg_lambda) + '-' + str(link) + '-' + str(batch_size) + '.csv'
    os.makedirs(args.log_dir, exist_ok=True)
    mylist = [str(i) for i in range(0, args.epoch * args.steps, args.steps)] + ['hyperparam']
    with open(filename, 'w+', newline='') as file:
        # Step 4: Using csv.writer to write the list to the CSV file
        writer = csv.writer(file)
        writer.writerow(mylist)  # Use writerow for single list

    result = []
    result_val = []
    print("Finish one combination of hyperparameters!")
    for seed in seeds:
        cv,cv_val = train(lr=lr,env=args.env,seed=seed,path=args.path,hyper_choice=args.seed,
                       link=link,random_weight=random_weight,l1_lambda=alpha,
                       reg_lambda=reg_lambda,discount=discount_factor,
                       checkpoint=args.steps,epoch=args.epoch, cv_fold=10,
                       batch_size=batch_size,buffer_size=buffer_size,
                       max_len=args.max_len)
        print("Return result shape: ", len(cv), ":::", args.steps)
        result.append(cv)
        result_val.append(cv_val)
        name = ['seed', seed, 'train']
        name = [str(s) for s in name]
        cv = np.around(cv, decimals=4)
        mylist = [str(i) for i in list(cv)] + ['-'.join(name)]
        with open(filename, 'a', newline='') as file:
            # Step 4: Using csv.writer to write the list to the CSV file
            writer = csv.writer(file)
            writer.writerow(mylist)  # Use writerow for single list

        name = ['seed', seed, 'val']
        name = [str(s) for s in name]
        cv_val = np.around(cv_val, decimals=4)
        mylist = [str(i) for i in list(cv_val)] + ['-'.join(name)]
        with open(filename, 'a', newline='') as file:
            # Step 4: Using csv.writer to write the list to the CSV file
            writer = csv.writer(file)
            writer.writerow(mylist)  # Use writerow for single list
    result = np.array(result)
    ret = np.around(np.mean(result, axis=0), decimals=4)
    mylist = [str(i) for i in list(ret)] + ['mean-train']
    with open(filename, 'a', newline='') as file:
        # Step 4: Using csv.writer to write the list to the CSV file
        writer = csv.writer(file)
        writer.writerow(mylist)  # Use writerow for single list
    result_val = np.array(result_val)
    ret = np.around(np.mean(result_val, axis=0), decimals=4)
    mylist = [str(i) for i in list(ret)] + ['mean-val']
    with open(filename, 'a', newline='') as file:
        # Step 4: Using csv.writer to write the list to the CSV file
        writer = csv.writer(file)
        writer.writerow(mylist)  # Use writerow for single list

# print(eval_policy('/scratch/fengdic/avg_discount/mountaincar/model-1epoch-30.pth'))
# objs = train(0.001,env='CartPole-v1',seed=2,
#              path='./exper/cartpole.pth',hyper_choice=32,
#              link='identity',random_weight=0.7,l1_lambda=0.05,checkpoint=5,
#              epoch=100, cv_fold=10,batch_size=256,buffer_size=80,max_len=50)
# print('identity')
# plt.plot(range(len(objs)),objs)
# plt.plot(range(len(objs)),0.998*np.ones(len(objs)))
# plt.ylim((0,5))
# plt.show()

tune()
