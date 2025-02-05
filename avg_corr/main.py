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
import csv

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

class WeightNet(nn.Module):
    def __init__(self, o_dim, hidden_sizes,activation):
        super(WeightNet, self).__init__()
        sizes = [o_dim] + list(hidden_sizes)
        print(sizes)
        layers = []
        for j in range(len(sizes) - 1):
            layers += [nn.Linear(sizes[j], sizes[j + 1]),nn.ReLU()]
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

def eval_policy(env, gamma, path="./exper/cartpole.pth", random_weight=0.0,mujoco=False):
    """
    Evaluates a policy loaded from a file on a given environment.

    This function runs 100 trajectories using the loaded policy and optionally includes random actions
    based on the random_weight parameter. It computes discounted returns and average returns.

    Args:
        env: Gym environment object
        gamma (float): Discount factor for computing returns
        path (str, optional): Path to the saved policy file
        random_weight (float, optional, default 0.0): Probability of taking random actions.

    Returns:
        tuple: Contains:
            - float: Mean return (discounted) across all trajectories
            - float: Variance of the returns
            - float: Mean sum of reward (undiscounted) across all trajectories

    Example:
        >>> mean_ret, var_ret, mean_sum_rew = eval_policy(env, 0.99)
    """
    ac = load(path, env)

    o, _ = env.reset()
    ep_len, ep_ret, ep_sum_rew = 0, 0, 0
    num_traj, max_num_traj = 0, 100
    rets = []
    sum_rew = []

    while num_traj < max_num_traj:
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
        terminal = d or truncated
        ep_ret += r * gamma**ep_len
        ep_sum_rew += r
        ep_len += 1
        # Update obs (critical!)
        o = next_o

        if terminal:
            num_traj += 1
            rets.append(ep_ret)
            sum_rew.append(ep_sum_rew)
            o, _ = env.reset()
            ep_len, ep_ret, ep_sum_rew = 0, 0, 0
    return (1 - gamma) * np.mean(rets), np.var(rets), np.mean(sum_rew)

# sample behaviour dataset
# behaviour policy = (1- random_weight) * target_policy + random_weight * random_policy
# behaviour policy = Normal(target_mu, torch.exp(random_weight))
def collect_dataset(env,gamma,buffer_size=20,max_len=200,
                    path='./exper/cartpole_998.pth', random_weight=0.2,fold=10,mujoco=False):
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
        if mujoco:
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

# train weight net
def train(lr, env,seed,path,hyper_choice,link,random_weight,l1_lambda,buf=None,buf_test=None,
          reg_lambda=0, discount=0.95,
          checkpoint=5,epoch=1000,cv_fold=1,batch_size=256,buffer_size=20,max_len=50,mujoco=False):
    hyperparam = random_search(hyper_choice)
    # gamma = hyperparam['gamma']
    gamma = discount
    env = gym.make(env)

    true_obj, _, target_ret = eval_policy(env, gamma, path, random_weight=0,mujoco=False)
    _, _, behaviour_ret = eval_policy(env, gamma, path, random_weight=random_weight,mujoco=mujoco)
    print("target: ", target_ret, " and behaviour: ", behaviour_ret)

    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    env.reset(seed=seed)

    if buf==None:
        buf = collect_dataset(env,gamma,buffer_size=buffer_size,max_len=max_len,path=path,
                                random_weight=random_weight,fold =cv_fold,mujoco=mujoco)
        buf_test = collect_dataset(env, gamma,buffer_size=buffer_size,max_len=max_len,
                                          path=path,random_weight=random_weight,fold=cv_fold,mujoco=mujoco)

    # buf_td = collect_dataset(env, gamma, buffer_size=2000, max_len=50, path=path,
    #                       random_weight=random_weight, fold=1)
    # second_buf_td = collect_dataset(env, gamma, buffer_size=2000, max_len=50,
    #                            path=path, random_weight=random_weight, fold=1)
    # TD_err = TD_computation(buf_td,second_buf_td,gamma)
    if link=='inverse' or link=='identity':
        weight = WeightNet(env.observation_space.shape[0], hidden_sizes=(256,256),activation=nn.ReLU)
    else:
        weight = WeightNet(env.observation_space.shape[0], hidden_sizes=(256,256), activation=nn.Identity)

    weight = weight.to(device)

    start_time = time.time()

    # Set up optimizers for policy and value function
    optimizer = Adam(weight.parameters(), lr)

    def update(fold_num):
        #sample minibatches
        data = buf.sample(batch_size,fold_num)

        obs, act = data['obs'].to(device), data['act'].to(device)
        tim, prod = data['tim'].to(device), data['prod'].to(device)
        logtarg, logbev = data['logtarg'].to(device), data['logbev'].to(device)

        discount = torch.as_tensor(gamma,dtype=torch.float32).to(device)
        label = torch.exp(torch.log(discount) * tim + prod)
        if link=="inverse":
            loss = ((1/weight(obs) - label) ** 2).mean()
            ratio = 1 / (weight(obs) + 0.0001)
        elif link=='identity':
            loss = ((weight(obs) - label) ** 2).mean()
            ratio = weight(obs)
        elif link =='loglog':
            loss = ((torch.exp(torch.exp(weight(obs)))-1 - label) ** 2).mean()
            ratio = torch.exp(torch.exp(weight(obs)))-1
        else:
            # loss = ((weight(obs) - (torch.log(discount) * tim + prod)) ** 2).mean()
            loss = ((torch.exp(weight(obs)) - label) ** 2).mean()
            ratio = torch.exp(weight(obs))


        with torch.no_grad():
            eta = torch.mean(ratio*max_len*(1-gamma)-1)
        regularizer = torch.mean(eta*ratio*max_len*(1-gamma)-eta)

        l1_norm = sum(torch.linalg.norm(p, 1) for p in weight.parameters())
        loss = loss + l1_lambda * l1_norm+ reg_lambda * regularizer

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Let's record the estimated objective value = 1/n \sum_{i=1}^n est_ratio(s_i,a_i) r(s_i,a_i)
    # classic control: train 5k steps and checkpoint =5

    def eval(buffer):
        ratio = weight(torch.as_tensor(buffer.obs_buf[:buffer.ptr],dtype=torch.float32).to(device),).detach().cpu().numpy()
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
        other = np.ones(buffer.ptr,dtype=np.int32)
        other[ind] = 0
        ratio = weight(torch.as_tensor(buffer.obs_buf,dtype=torch.float32).to(device),).detach().cpu().numpy()
        if link == "inverse":
            ratio = 1 / (ratio + 0.001)
        elif link == 'identity':
            pass
        elif link == 'loglog':
            ratio = np.exp(np.exp(ratio))-1
        else:
            ratio = np.exp(ratio)
        obj = np.mean(ratio[ind] * np.exp(buffer.logtarg_buf[ind]
                                          - buffer.logbev_buf[ind]) * buffer.rew_buf[ind])
        obj_other = np.mean(ratio[other] * np.exp(buffer.logtarg_buf[other]
                                                  - buffer.logbev_buf[other]) * buffer.rew_buf[other])
        return obj * max_len * (1 - gamma), obj_other * max_len * (1 - gamma)

    objs_mean = []
    objs_val_mean = []
    for fold_num in range(cv_fold):
        objs, objs_test, objs_val = [], [], []
        # err = []
        for steps in range(epoch * checkpoint):
            update(fold_num)
            if steps % checkpoint == 0:
                obj_val, obj = eval_cv(buf, fold_num)
                # obj, obj_test = eval(buf), eval(buf_test)
                # td_err = TD_err.compute(weight)
                objs.append(obj)
                objs_val.append(obj_val)
                # objs_test.append(obj_test)
                # err.append(td_err)
        objs_mean.append(objs)
        objs_val_mean.append(objs_val)
        # print("One fold done: ",fold_num)
    # return objs, objs_test, weight #, err
    return np.around(np.mean(np.array(objs_mean), axis=0), decimals=4), \
           np.around(np.mean(np.array(objs_val_mean), axis=0), decimals=4)

def argsparser():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default='./')
    parser.add_argument('--log_dir', type=str, default='./')
    parser.add_argument('--data_dir', type=str, default='./')
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
    alpha= [0, 0.001, 0.01, 0.1]
    batch_size= 512
    link = 'identity'
    lr = [0.00005,0.0001,0.0005,0.001,0.005]
    reg_lambda = [0.5,2,10,20]

    # env = ['MountainCarContinuous-v0', 'Hopper-v4', 'HalfCheetah-v4', 'Ant-v4','Walker2d-v4']
    # path = ['./exper/mountaincar.pth',
    #         './exper/hopper.pth',
    #         './exper/halfcheetah_1.pth',
    #         './exper/ant.pth',
    #         './exper/walker.pth']

    args = argsparser()
    # seeds = range(5)
    seed= args.seed
    idx = np.unravel_index(args.array, (4,5,4))
    random_weight, buffer_size = 2.0, 40
    discount_factor, max_len = 0.95, 100
    alpha, lr, reg_lambda = alpha[idx[0]], lr[idx[1]], reg_lambda[idx[2]]
    filename = args.log_dir+'mse-tune-square-reg-alpha-' + str(alpha)+'-lr-'\
               +str(lr)+'-lambda-'+str(reg_lambda)+'.csv'
    os.makedirs(args.log_dir, exist_ok=True)
    mylist = [str(i) for i in range(0,args.epoch*args.steps,args.steps)] + ['hyperparam']
    with open(filename, 'w+', newline='') as file:
        # Step 4: Using csv.writer to write the list to the CSV file
        writer = csv.writer(file)
        writer.writerow(mylist)  # Use writerow for single list

    result = []
    result_val = []
    print("Finish one combination of hyperparameters!")
    cv, cv_val = train(lr=lr,env=args.env,seed=seed,path=args.path,hyper_choice=args.seed,
                   link=link,random_weight=random_weight,l1_lambda=alpha,
                   reg_lambda=reg_lambda,discount = discount_factor,
                   checkpoint=args.steps,epoch=args.epoch, cv_fold=10,
                   batch_size=batch_size,buffer_size=buffer_size,
                   max_len=max_len,mujoco=True)
    print("Return result shape: ",len(cv),":::", args.steps)
    result.append(cv)
    result_val.append(cv_val)
    name = ['seed',seed,'train']
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
    # result = np.array(result)
    # ret = np.around(np.mean(result,axis=0),decimals=4)
    # mylist = [str(i) for i in list(ret)] + ['mean-train']
    # with open(filename, 'a', newline='') as file:
    #     # Step 4: Using csv.writer to write the list to the CSV file
    #     writer = csv.writer(file)
    #     writer.writerow(mylist)  # Use writerow for single list
    # result_val = np.array(result_val)
    # ret = np.around(np.mean(result_val, axis=0), decimals=4)
    # mylist = [str(i) for i in list(ret)] + ['mean-val']
    # with open(filename, 'a', newline='') as file:
    #     # Step 4: Using csv.writer to write the list to the CSV file
    #     writer = csv.writer(file)
    #     writer.writerow(mylist)  # Use writerow for single list

if __name__ == "__main__" :
    # print(eval_policy('/scratch/fengdic/avg_discount/mountaincar/model-1epoch-30.pth'))
    batch_size, link, alpha, lr, loss, reg_lambda = 512,'identity',0.001,0.0001,'mse', 20
    objs,objs_test = train(lr=lr,env='HalfCheetah-v4',seed=9,path='./exper/halfcheetah_1.pth',hyper_choice=274,
                           link=link,random_weight=2.0,l1_lambda=alpha,
                           reg_lambda=reg_lambda,discount = 0.95,
                           checkpoint=5,epoch=10000, cv_fold=10,
                           batch_size=batch_size,buffer_size=40,
                           max_len=100,mujoco=True)
    plt.plot(range(len(objs)),objs_test)
    plt.show()
    # plt.plot(range(len(objs)),0.327*np.ones(len(objs)))
    # plt.savefig('hopper.png')

    # tune()

# print(eval_policy(path='./exper/hopper.pth',env='Hopper-v4',gamma=0.95))
