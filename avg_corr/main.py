import numpy as np
from gym.spaces import Box, Discrete
import gym
import time
import torch
import torch.nn as nn
from torch.optim import Adam
import ppo.algo.core as core
from ppo.algo.random_search import random_search


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
        self.prod_buf[path_slice] = core.discount_cumsum(deltas, 1)

        self.path_start_idx = self.ptr

    def get(self):
        """
        Call this at the end of an epoch to get all of the data from
        the buffer, with advantages appropriately normalized (shifted to have
        mean zero and std one). Also, resets some pointers in the buffer.
        """
        assert self.ptr == self.max_size  # buffer has to be full before you can get
        self.ptr, self.path_start_idx = 0, 0

        data = dict(obs=self.obs_buf, act=self.act_buf, prod=self.prod_buf, tim=self.tim_buf,
                    logbev=self.logbev_buf, logtarg=self.logtarg_buf)
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
        self.weight = nn.Sequential(nn.Linear(sizes[-1], 1),activation())

    def forward(self, obs):
        obs = obs.float()
        body = self.body(obs)
        weight = self.weight(body)
        return torch.squeeze(weight)

# load target policy
def load(seed,path,env):
    hyperparam = random_search(seed)
    pi_lr = hyperparam["pi_lr"],
    vf_lr = hyperparam['vf_lr']

    ac_kwargs = dict(hidden_sizes=args.hid)

    ac = core.MLPActorCritic(env.observation_space, env.action_space, **ac_kwargs)
    pi_optimizer = Adam(ac.pi.parameters(), lr=pi_lr)
    vf_optimizer = Adam(ac.v.parameters(), lr=vf_lr)

    checkpoint = torch.load(path)
    ac.load_state_dict(checkpoint['model_state_dict'])
    vf_optimizer.load_state_dict(checkpoint['vf_optimizer_state_dict'])
    pi_optimizer.load_state_dict(checkpoint['pi_optimizer_state_dict'])
    epoch = checkpoint['epoch']
    return ac

# sample behaviour dataset randomly
def collect_dataset(buffer_size,random=True):
    env = gym.make('CartPole-v0')
    ac = load(seed,path,env)
    obs_dim = env.observation_space.shape
    act_dim = env.action_space.shape
    buf=PPOBuffer(obs_dim, act_dim, buffer_size, gamma, lam)

    o, ep_len = env.reset(), 0

    if isinstance(env.action_space, Box):
        action_range=env.action_space.high-env.action_space.low
        assert action_range>0
        logbev = -np.log(np.prod(action_range))
    elif isinstance(env.action_space, Discrete):
        logbev = -np.log(env.action_space.n)

    for t in range(buffer_size):
        _, _, logtarg = ac.step(torch.as_tensor(o, dtype=torch.float32))
        if random:
            # random behaviour policy
            a = env.action_space.sample()
        else:
            _,a,logbev = ac_bev.step(torch.as_tensor(o, dtype=torch.float32))
        next_o, r, d, _ = env.step(a)
        ep_len += 1

        # save and log
        buf.store(o, a, r, ep_len - 1, logbev, logtarg)

        # Update obs (critical!)
        o = next_o

        terminal = d
        epoch_ended = t == buffer_size - 1

        if terminal or epoch_ended:
            if epoch_ended and not (terminal):
                print('Warning: trajectory cut off by epoch at %d steps.' % ep_len, flush=True)
            buf.finish_path()
            o, ep_ret, ep_len = env.reset(), 0, 0
    return buf

# train weight net
def train():
    buf = collect_dataset(buffer_size=4000,random=True)
    data = buf.get()

    start_time = time.time()
    def compute_loss_v(data):
        obs, act = data['obs'], data['act']
        tim, adv, logp_old = data['tim'], data['adv'], data['logp']

        return ((ac.v(obs) - ret)**2).mean()

    # Set up optimizers for policy and value function
    pi_optimizer = Adam(ac.pi.parameters(), lr=pi_lr)
    vf_optimizer = Adam(ac.v.parameters(), lr=vf_lr)

    # Set up model saving
    logger.setup_pytorch_saver(ac)

    def update():
        data = buf.get()

        pi_l_old, pi_info_old = compute_loss_pi(data)
        pi_l_old = pi_l_old.item()
        v_l_old = compute_loss_v(data).item()

        # Train policy with multiple steps of gradient descent
        for i in range(train_pi_iters):
            pi_optimizer.zero_grad()
            loss_pi, pi_info = compute_loss_pi(data)
            kl = mpi_avg(pi_info['kl'])
            if kl > 1.5 * target_kl:
                logger.log('Early stopping at step %d due to reaching max kl.'%i)
                break
            loss_pi.backward()
            mpi_avg_grads(ac.pi)    # average grads across MPI processes
            pi_optimizer.step()

        logger.store(StopIter=i)

        # Value function learning
        for i in range(train_v_iters):
            vf_optimizer.zero_grad()
            loss_v = compute_loss_v(data)
            loss_v.backward()
            mpi_avg_grads(ac.v)    # average grads across MPI processes
            vf_optimizer.step()

        # Log changes from update
        kl, ent, cf = pi_info['kl'], pi_info_old['ent'], pi_info['cf']
        logger.store(LossPi=pi_l_old, LossV=v_l_old,
                     KL=kl, Entropy=ent, ClipFrac=cf,
                     DeltaLossPi=(loss_pi.item() - pi_l_old),
                     DeltaLossV=(loss_v.item() - v_l_old))