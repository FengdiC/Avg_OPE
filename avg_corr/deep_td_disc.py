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
from torch.optim import Adam
import ppo.algo.core as core
from ppo.algo.random_search import random_search
import ppo.utils.logger as logger
import matplotlib.pyplot as plt
import csv
import torch.nn.functional as F
import copy


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.l1 = nn.Linear(state_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, action_dim)


    def forward(self, state):
        q1 = F.relu(self.l1(state))
        q1 = F.relu(self.l2(q1))
        return self.l3(q1)
    

class PPOBuffer:
    """
    A buffer for storing trajectories experienced by a PPO agent interacting
    with the environment, and using Generalized Advantage Estimation (GAE-Lambda)
    for calculating the advantages of state-action pairs.
    """

    def __init__(self, obs_dim, act_dim, size, fold, gamma=0.99):
        self.obs_buf = np.zeros(core.combined_shape(size, obs_dim), dtype=np.float32)
        self.next_obs_buf = np.zeros(core.combined_shape(size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(core.combined_shape(size, act_dim), dtype=np.int32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.not_done_buf = np.zeros(size, dtype=np.float32)
        self.tim_buf = np.zeros(size, dtype=np.int32)
        self.logtarg_buf = np.zeros(size, dtype=np.float32)
        self.prod_buf = np.zeros(size, dtype=np.float32)
        self.logbev_buf = np.zeros(size, dtype=np.float32)
        self.gamma = gamma
        self.fold = fold
        self.ptr, self.path_start_idx, self.max_size = 0, 0, size

    def store(self, obs, act, rew, tim, logbev, logtarg, next_obs, not_done):
        """
        Append one timestep of agent-environment interaction to the buffer.
        """
        assert self.ptr < self.max_size  # buffer has to have room so you can store
        self.obs_buf[self.ptr] = obs
        self.next_obs_buf[self.ptr] = next_obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.not_done_buf[self.ptr] = not_done
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

        data = dict(obs=self.obs_buf[ind], act=self.act_buf[ind], prod=self.prod_buf[ind],
                    tim=self.tim_buf[ind], rew=self.rew_buf[ind],
                    logbev=self.logbev_buf[ind], logtarg=self.logtarg_buf[ind],
                    next_obs=self.next_obs_buf[ind], not_done=self.not_done_buf[ind])
        
        return {k: torch.as_tensor(v) for k, v in data.items()}

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


# sample behaviour dataset
# behaviour policy = (1- random_weight) * target_policy + random_weight * random_policy
# random_weight = 0.3  0.5  0.7
# classic control max_len=50 number_traj = 40 80 200
def collect_dataset(env,gamma,buffer_size=20,max_len=200,
                    path='./exper/cartpole.pth', random_weight=0.2,fold=10):
    ac = load(path, env)
    obs_dim = env.observation_space.shape
    act_dim = (env.action_space.n, )

    buf=PPOBuffer(obs_dim, 1, buffer_size*max_len,fold, gamma)

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
        if isinstance(a, np.ndarray):
            a = a.item()
        pi = ac.pi._distribution(torch.as_tensor(o, dtype=torch.float32))
        logtarg = ac.pi._log_prob_from_distribution(pi, torch.as_tensor(a)).detach().numpy()
        logbev = np.log(random_weight * unif + (1 - random_weight) * np.exp(logtarg))

        next_o, r, d, _ = env.step(a)
        ep_len += 1

        done_bool = float(d) if ep_len < max_len else 0.0

        # save and log
        buf.store(o, a, r, ep_len - 1, logbev, logtarg, next_o, done_bool)

        # Update obs (critical!)
        o = next_o

        terminal = d
        epoch_ended = ep_len == max_len - 1

        if terminal or epoch_ended:
            if terminal and not (epoch_ended):
                # print('Warning: trajectory ends early at %d steps.' % ep_len, flush=True)
                buf.delete_last_traj()
                o, ep_ret, ep_len = env.reset(), 0, 0
                continue
            o, ep_ret, ep_len = env.reset(), 0, 0
            num_traj += 1
            buf.finish_path()
    return buf

# train weight net
def train(lr, env,seed,path,hyper_choice,link,random_weight,l1_lambda, discount=0.95,
          checkpoint=5,epoch=1000,cv_fold=1,batch_size=256,buffer_size=20,max_len=50):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    hyperparam = random_search(hyper_choice)
    # gamma = hyperparam['gamma']
    gamma = discount
    env = gym.make(env)

    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    env.reset(seed=seed)

    buf = collect_dataset(env,gamma,buffer_size=buffer_size,max_len=max_len,path=path,
                            random_weight=random_weight,fold = 1)

    critic = Critic(env.observation_space.shape[0], env.action_space.n).to(device)
    critic_target = copy.deepcopy(critic).to(device)
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=3e-4)
    tau=0.005
    
    start_time = time.time()
 
    ac = load(path, env)

    def update(fold_num):
        data = buf.sample(batch_size,fold_num)
        state = data['obs']
        action = data['act'].to(torch.int64)
        next_state = data['next_obs']
        reward = data['rew']
        not_done = data['not_done']

        with torch.no_grad(): 
            targ_a, _, _ = ac.step(torch.as_tensor(next_state, dtype=torch.float32)) 
            next_state = next_state.to(device)
            reward = reward.to(device).unsqueeze(-1)
            not_done = not_done.to(device).unsqueeze(-1)
            targ_a = torch.from_numpy(targ_a).to(device)
            next_q = critic_target(next_state).gather(1, targ_a.unsqueeze(-1)).reshape(-1,1)
            target_Q = reward + discount * not_done * next_q
            

        state = state.to(device)
        action = action.to(device) 
        current_Q = critic(state).gather(1, action).reshape(-1,1)
        # print(next_q.shape, target_Q.shape, current_Q.shape)
        # print(state.shape, action.shape, current_Q.shape, target_Q.shape, reward.shape, not_done.shape)
        critic_loss = F.mse_loss(current_Q, target_Q)

        critic_optimizer.zero_grad()
        critic_loss.backward()
        critic_optimizer.step()

        for param, target_param in zip(critic.parameters(), critic_target.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

    def eval():
        num= 0
        initial = []
        while num<200:
            num+=1
            o, ep_ret, ep_len = env.reset(), 0, 0
            a, _, _ = ac.step(torch.as_tensor(o, dtype=torch.float32))
            o = torch.as_tensor(o, dtype=torch.float32).to(device)
            a = torch.from_numpy(a).to(device)
            q_value = critic(o.unsqueeze(0)).gather(1, a.reshape(1, 1)).reshape(-1,1)
            initial.append((1-gamma)*q_value.detach().cpu().numpy())
        return np.mean(initial)

    for fold_num in range(cv_fold):
        objs = [] 
        for steps in range(epoch*checkpoint):
            update(fold_num)
            if steps%checkpoint==0:
                obj  = eval()
                objs.append(obj)
    return objs


def argsparser():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default='../exper/cartpole.pth')
    parser.add_argument('--log_dir', type=str, default='./')
    parser.add_argument('--env', type=str, default='CartPole-v1')
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
    random_weight = [0.3, 0.5, 0.7]
    batch_size= [256,512]
    buffer = [40,80,200]

    args = argsparser()
    seeds = range(3)
    idx = np.unravel_index(args.array, (3,2,2,3))
    random_weight,batch_size = random_weight[idx[0]],batch_size[idx[1]]
    buffer_size = buffer[idx[3]]
    filename = args.log_dir+'mse-tune-' + str(random_weight)+\
               '-'+str(buffer_size)+'-'+str(batch_size)+'.csv'
    os.makedirs(args.log_dir, exist_ok=True)
    mylist = [str(i) for i in range(0,args.epoch*args.steps,args.steps)] + ['hyperparam']
    with open(filename, 'w+', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(mylist)  # Use writerow for single list

    for alpha in [0.0005,0.001,0.002,0.005,0.01]:
        for lr in [0.0001,0.0005,0.001,0.005]:
            result = []
            print("Finish one combination of hyperparameters!")
            for seed in seeds:
                t1 = time.time()
                cv = train(lr=lr,env=args.env,seed=seed,path=args.path,hyper_choice=args.seed,
                               link=None,random_weight=random_weight,l1_lambda=alpha,
                               checkpoint=args.steps,epoch=args.epoch, cv_fold=10,
                               batch_size=batch_size,buffer_size=buffer_size,
                               max_len=args.max_len)
                # print("Return result shape: ",cv.shape,":::", args.steps,":::",seeds)
                t2 = time.time()
                print(f'Time: {t2 - t1}')
                result.append(cv)

                break

            break

            result = np.array(result)
            ret = np.around(np.mean(result,axis=0),decimals=4)
            var = np.around(np.var(result,axis=0),decimals=4)
            # print("Mean shape: ",ret.shape,":::",var.shape)
            name = ['lr',lr,'alpha',alpha]
            name = [str(s) for s in name]
            name_1 = name +['mean']
            name_2 = name+ ['var']
            mylist = [str(i) for i in list(ret)] + ['-'.join(name_1)]
            with open(filename, 'a', newline='') as file:
                # Step 4: Using csv.writer to write the list to the CSV file
                writer = csv.writer(file)
                writer.writerow(mylist)  # Use writerow for single list
            print('-'.join(name_1))
        
        break
    

tune()