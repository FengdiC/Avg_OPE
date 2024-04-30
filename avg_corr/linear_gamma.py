import os, sys, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
from gym.spaces import Box, Discrete
import gym
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import ppo.algo.core as core
from ppo.algo.random_search import random_search
import matplotlib.pyplot as plt

import statsmodels.api as sm
import pandas as pd

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Encoder_Decoder(nn.Module):
	def __init__(self, state_dim, action_dim):
		super(Encoder_Decoder, self).__init__()

		self.e1 = nn.Linear(state_dim + action_dim, 256)
		self.e2 = nn.Linear(256, 256)

		self.r1 = nn.Linear(256, 1, bias=False)

		self.a1 = nn.Linear(256, 256)
		self.a2 = nn.Linear(256, action_dim)

		self.d1 = nn.Linear(256, 256)
		self.d2 = nn.Linear(256, state_dim)

	def forward(self, state, action):
		l = F.relu(self.e1(torch.cat([state, action], 1)))
		l = F.relu(self.e2(l))

		r = self.r1(l)

		d = F.relu(self.d1(l))
		ns = self.d2(d)

		d = F.relu(self.a1(l))
		a = self.a2(d)

		return ns, r, a, l

	def latent(self, state, action):
		l = F.relu(self.e1(torch.cat([state, action], 1)))
		l = F.relu(self.e2(l))
		return l

class PPOBuffer:
    """
    A buffer for storing trajectories experienced by a PPO agent interacting
    with the environment, and using Generalized Advantage Estimation (GAE-Lambda)
    for calculating the advantages of state-action pairs.
    """

    def __init__(self, obs_dim, act_dim, size, gamma=0.99):
        self.obs_buf = np.zeros(core.combined_shape(size, obs_dim), dtype=np.float32)
        self.next_obs_buf = np.zeros(core.combined_shape(size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(core.combined_shape(size, act_dim), dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.tim_buf = np.zeros(size, dtype=np.int32)
        self.logtarg_buf = np.zeros(size, dtype=np.float32)
        self.prod_buf = np.zeros(size, dtype=np.float32)
        self.logbev_buf = np.zeros(size, dtype=np.float32)
        self.gamma = gamma
        self.ptr, self.path_start_idx, self.max_size = 0, 0, size

    def store(self, obs, act, rew, next_obs, tim, logbev, logtarg):
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

    def sample(self,batch_size):
        """
        Call this at the end of an epoch to get all of the data from
        the buffer, with advantages appropriately normalized (shifted to have
        mean zero and std one). Also, resets some pointers in the buffer.
        """

        ind = np.random.randint(self.ptr, size=batch_size)

        data = dict(obs=self.obs_buf[ind], act=self.act_buf[ind], prod=self.prod_buf[ind],
                    tim=self.tim_buf[ind], next_obs = self.next_obs_buf[ind],rew = self.rew_buf[ind],
                    logbev=self.logbev_buf[ind], logtarg=self.logtarg_buf[ind])
        return {k: torch.as_tensor(v, dtype=torch.float32) for k, v in data.items()}

    def delete_last_traj(self):
        self.ptr = self.path_start_idx

# load target policy
def load(path,env):
    ac_kwargs = dict(hidden_sizes=[64,32])

    ac = core.MLPActorCritic(env.observation_space, env.action_space, **ac_kwargs)
    checkpoint = torch.load(path)
    ac.load_state_dict(checkpoint['model_state_dict'])
    return ac

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
        assert np.all(action_range < np.inf)
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
        buf.store(o, a, r,next_o, ep_len - 1, logbev, logtarg)

        # Update obs (critical!)
        o = next_o

        terminal = d
        epoch_ended = ep_len == max_len - 1

        if terminal or epoch_ended:
            if terminal and not (epoch_ended):
                print('Warning: trajectory ends early at %d steps.' % ep_len, flush=True)
                buf.delete_last_traj()
                o, ep_ret, ep_len = env.reset(), 0, 0
                continue
            o, ep_ret, ep_len = env.reset(), 0, 0
            num_traj += 1
            buf.finish_path()

    return buf,num_traj

class AvgOPE():
    def __init__(self,lr, env,seed,path,link=sm.families.links.InversePower(),batch_size=256):
        super(AvgOPE, self).__init__()
        self.lr, self.seed, self.path,self.link,self.batch_size = lr, seed, path, link, batch_size
        self.env = gym.make(env)
        obs_dim = self.env.observation_space.shape[0]
        if isinstance(self.env.action_space, Box):
            act_dim = self.env.action_space.shape
        elif isinstance(self.env.action_space, Discrete):
            act_dim = self.env.action_space.n
        self.encoder_decoder = Encoder_Decoder(obs_dim, act_dim).to(device)
        self.ed_optimizer = torch.optim.Adam(self.encoder_decoder.parameters(), lr=3e-4)

    def train_encoder_decoder(self, data):
        state, action, next_state, reward = data['obs'], data['act'], data['next_obs'], data['rew']
        if isinstance(self.env.action_space, Discrete):
            action = F.one_hot(action.to(torch.int64), num_classes=-1).to(torch.float32)

        recons_next, recons_reward, recons_action, lat = self.encoder_decoder(state, action)
        ed_loss = F.mse_loss(recons_next, next_state) + 0.1 * F.mse_loss(recons_reward, reward) + F.mse_loss(
            recons_action, action)

        self.ed_optimizer.zero_grad()
        ed_loss.backward()
        self.ed_optimizer.step()

    # train weight net
    def train(self):
        hyperparam = random_search(self.seed)
        gamma = hyperparam['gamma']

        T = 100
        buf,k = collect_dataset(self.env,gamma,buffer_size=20,max_len=T,path=self.path)
        # buf_test,k_test = collect_dataset(self.env, gamma,buffer_size=20,max_len=T,path=self.path)

        start_time = time.time()
        for _ in range(20000):
            data = buf.sample(self.batch_size)
            self.train_encoder_decoder(data)

        action=torch.as_tensor(buf.act_buf[:buf.ptr], dtype=torch.float32)
        if isinstance(self.env.action_space, Discrete):
            action = F.one_hot(action.to(torch.int64), num_classes=-1).to(torch.float32)
        latent = self.encoder_decoder.latent(torch.as_tensor(buf.obs_buf[:buf.ptr], dtype=torch.float32),
             action).detach().numpy()
        print(latent.shape)
        print(np.sum(np.isnan(latent)))
        feature = pd.DataFrame(latent)
        label_log = np.expand_dims(np.log(gamma) * buf.tim_buf[:buf.ptr] + buf.prod_buf[:buf.ptr],axis=1)
        label = pd.DataFrame(np.exp(label_log))
        print(label.shape)

        gamma_model = sm.GLM(label, feature, family=sm.families.Gamma(self.link))
        gamma_results = gamma_model.fit()
        ratio = gamma_results.predict(feature)

        obj = np.mean(ratio *buf.rew_buf[:buf.ptr])
        return obj*T*(1-gamma)

# print(eval_policy('/scratch/fengdic/avg_discount/mountaincar/model-1epoch-30.pth'))
algo = AvgOPE(0.0001,env='CartPole-v1',seed=32,path='./exper/cartpole.pth')
objs = algo.train()
plt.plot(range(10),objs * np.ones(10),'r')
plt.plot(range(10),0.998* np.ones(10))
plt.show()