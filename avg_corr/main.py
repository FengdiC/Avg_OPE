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
import matplotlib.pyplot as plt


class PPOBuffer:
    """
    A buffer for storing trajectories experienced by a PPO agent interacting
    with the environment, and using Generalized Advantage Estimation (GAE-Lambda)
    for calculating the advantages of state-action pairs.
    """

    def __init__(self, obs_dim, act_dim, size, gamma=0.99):
        self.obs_buf = np.zeros(core.combined_shape(size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(core.combined_shape(size, act_dim), dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.tim_buf = np.zeros(size, dtype=np.int32)
        self.logtarg_buf = np.zeros(size, dtype=np.float32)
        self.prod_buf = np.zeros(size, dtype=np.float32)
        self.logbev_buf = np.zeros(size, dtype=np.float32)
        self.gamma = gamma
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

    def sample(self,batch_size):
        """
        Call this at the end of an epoch to get all of the data from
        the buffer, with advantages appropriately normalized (shifted to have
        mean zero and std one). Also, resets some pointers in the buffer.
        """

        ind = np.random.randint(self.ptr, size=batch_size)

        data = dict(obs=self.obs_buf[ind], act=self.act_buf[ind], prod=self.prod_buf[ind],
                    tim=self.tim_buf[ind],
                    logbev=self.logbev_buf[ind], logtarg=self.logtarg_buf[ind])
        return {k: torch.as_tensor(v, dtype=torch.float32) for k, v in data.items()}

class WeightNet(nn.Module):
    def __init__(self, o_dim, hidden_sizes,activation):
        super(WeightNet, self).__init__()
        sizes = [o_dim] + list(hidden_sizes)
        print(sizes)
        layers = []
        for j in range(len(sizes) - 1):
            layers += [nn.Linear(sizes[j], sizes[j + 1]),activation()]
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
    env = gym.make('Walker2d-v4')
    ac = load(path, env)
    hyperparam = random_search(39)
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
                    path='./exper/cartpole_998.pth', random_weight=0.2):
    ac = load(path,env)
    obs_dim = env.observation_space.shape
    act_dim = env.action_space.shape

    buf=PPOBuffer(obs_dim, act_dim, buffer_size*max_len, gamma)

    o, ep_len = env.reset(), 0
    num_traj = 0

    if isinstance(env.action_space, Box):
        action_range = env.action_space.high - env.action_space.low
        assert action_range > 0
        unif = 1 / np.prod(action_range)
    elif isinstance(env.action_space, Discrete):
        unif = 1 / env.action_space.n

    while num_traj < buffer_size:
        targ_a, _, logtarg = ac.step(torch.as_tensor(o, dtype=torch.float32))
        if np.random.random() < random_weight:
            # random behaviour policy
            a = env.action_space.sample()
            pi = ac.pi._distribution(torch.as_tensor(o, dtype=torch.float32))
            logp = ac.pi._log_prob_from_distribution(pi, torch.as_tensor(a)).detach().numpy()
            logbev = np.log(random_weight * unif + (1 - random_weight) * np.exp(logp))
        else:
            a = targ_a
            logbev = np.log(random_weight * unif + (1 - random_weight) * np.exp(logtarg))
        next_o, r, d, _ = env.step(a)
        ep_len += 1

        # save and log
        buf.store(o, a, r, ep_len - 1, logbev, logtarg)

        # Update obs (critical!)
        o = next_o

        terminal = d
        epoch_ended = ep_len == max_len - 1

        if terminal or epoch_ended:
            num_traj +=1
            if terminal and not (epoch_ended):
                print('Warning: trajectory ends early at %d steps.' % ep_len, flush=True)
            buf.finish_path()
            o, ep_ret, ep_len = env.reset(), 0, 0
    return buf,num_traj

# train weight net
def train(lr, batch_size=256):
    hyperparam = random_search(32)
    gamma = hyperparam['gamma']
    env = gym.make('CartPole-v1')
    true_value = 0.998
    T = 50
    buf,k = collect_dataset(env,gamma,buffer_size=20,max_len=T)
    buf_test,k_test = collect_dataset(env, gamma,buffer_size=20,max_len=T)
    weight = WeightNet(env.observation_space.shape[0], hidden_sizes=[256,256],activation=nn.ReLU)

    start_time = time.time()

    # Set up optimizers for policy and value function
    optimizer = Adam(weight.parameters(), lr)

    def update():
        #sample minibatches
        data = buf.sample(batch_size)

        obs, act = data['obs'], data['act']
        tim, prod = data['tim'], data['prod']

        loss = ((weight(obs) - 1/np.exp(np.log(gamma) * tim + prod)) ** 2).mean()
        l1_lambda = 0.001
        l1_norm = sum(torch.linalg.norm(p, 1) for p in weight.parameters())
        loss = loss + l1_lambda * l1_norm

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    def eval(buffer):
        ratio = weight(torch.as_tensor(buffer.obs_buf[:buffer.ptr],dtype=torch.float32)).detach().numpy()
        obj = np.mean(1/(ratio+0.001) * np.exp(buffer.logtarg_buf[:buffer.ptr]
                                     - buffer.logbev_buf[:buffer.ptr])*buffer.rew_buf[:buffer.ptr])
        return obj*T*(1-gamma)

    objs, objs_test = [], []
    err, terr_test = 100, 100
    for steps in range(200* 10):
        update()
        if steps>0 and steps%10==0:
            obj, obj_test  = eval(buf), eval(buf_test)
            objs.append(obj)
            objs_test.append(obj_test)
    return objs

print(eval_policy('/scratch/fengdic/avg_discount/walker/model-1epoch-249.pth'))
# objs = train(0.0001)
# plt.plot(range(len(objs)),objs)
# plt.plot(range(len(objs)),0.998*np.ones(len(objs)))
# plt.show()
