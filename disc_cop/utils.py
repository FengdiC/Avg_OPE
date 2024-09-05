import _pickle as pickle
import gym
import inspect
import numpy as np
import os
import sys
import torch

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from gym.spaces import Box, Discrete

import ppo.algo.core as core


def set_seed(seed):
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


class PPOBuffer:
    """
    A buffer for storing trajectories experienced by a PPO agent interacting
    with the environment, and using Generalized Advantage Estimation (GAE-Lambda)
    for calculating the advantages of state-action pairs.
    """

    def __init__(self, obs_dim, act_dim, size, fold, gamma=0.99):
        self.obs_buf = np.zeros(core.combined_shape(size, obs_dim), dtype=np.float32)
        self.next_obs_buf = np.zeros(
            core.combined_shape(size, obs_dim), dtype=np.float32
        )
        self.act_buf = np.zeros(core.combined_shape(size, act_dim), dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.tim_buf = np.zeros(size, dtype=np.int32)
        self.logtarg_buf = np.zeros(size, dtype=np.float32)
        self.prod_buf = np.zeros(size, dtype=np.float32)
        self.logbev_buf = np.zeros(size, dtype=np.float32)
        self.gamma = gamma
        self.fold = fold
        self.ptr, self.path_start_idx, self.max_size = 0, 0, size
        self.ep_start_inds = [0]

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
        self.prod_buf[path_slice] = np.append(
            0, core.discount_cumsum(deltas[path_slice], 1)[:-1]
        )

        self.path_start_idx = self.ptr
        self.ep_start_inds.append(self.ptr)

    def set_size(self, size):
        assert len(self.ep_start_inds) >= size > 0
        if size < len(self.ep_start_inds):
            self.ptr = self.ep_start_inds[size]

    def sample(self, batch_size, fold_num):
        interval = int(self.ptr / self.fold)
        if self.fold > 1:
            ind = np.random.randint(self.ptr - interval, size=batch_size)
            ind = ind + np.where(ind >= fold_num * interval, 1, 0) * interval
        else:
            ind = np.random.randint(-len(self.ep_start_inds), self.ptr, size=batch_size)

        samp_ind = np.clip(ind, a_min=0, a_max=np.inf).astype(int)
        data = dict(
            obs=self.obs_buf[samp_ind],
            act=self.act_buf[samp_ind],
            prod=self.prod_buf[samp_ind],
            next_obs=self.next_obs_buf[samp_ind],
            tim=self.tim_buf[samp_ind],
            logbev=self.logbev_buf[samp_ind],
            logtarg=self.logtarg_buf[samp_ind],
            first_timestep=ind < 0,
        )

        if np.any(ind < 0):
            # Find any s_0 as s' and properly set them.
            start_ind = np.where(ind < 0)[0]
            data["next_obs"][start_ind] = self.obs_buf[
                np.asarray(self.ep_start_inds)[-ind[start_ind] - 1]
            ]
        return {k: torch.as_tensor(v, dtype=torch.float32) for k, v in data.items()}

    def delete_last_traj(self):
        self.ptr = self.path_start_idx


def load_policy(path, env):
    ac_kwargs = dict(hidden_sizes=[64, 32])

    ac = core.MLPActorCritic(env.observation_space, env.action_space, **ac_kwargs)
    checkpoint = torch.load(path)
    ac.load_state_dict(checkpoint["model_state_dict"])
    return ac


def maybe_collect_dataset(
    env,
    gamma,
    buffer_size,
    max_len,
    policy_path,
    random_weight,
    fold=1,
    load_dataset=None,
):
    save_buf = load_dataset is not None
    if load_dataset:
        if os.path.isfile(load_dataset):
            save_buf = False
            print("Loaded from existing buffer.")
            buf = pickle.load(open(load_dataset, "rb"))
            buf.set_size(buffer_size)
        os.makedirs(os.path.dirname(load_dataset), exist_ok=True)

    ac = load_policy(policy_path, env)
    obs_dim = env.observation_space.shape
    act_dim = env.action_space.shape

    buf = PPOBuffer(obs_dim, act_dim, buffer_size * max_len, fold, gamma)

    (o, _), ep_len = env.reset(), 0
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
        logtarg = (
            ac.pi._log_prob_from_distribution(pi, torch.as_tensor(a)).detach().numpy()
        )
        logbev = np.log(random_weight * unif + (1 - random_weight) * np.exp(logtarg))
        next_o, r, d, _, _ = env.step(a)
        ep_len += 1

        # save and log
        buf.store(o, a, r, next_o, ep_len - 1, logbev, logtarg)

        # Update obs (critical!)
        o = next_o

        terminal = d
        epoch_ended = ep_len == max_len - 1

        if terminal or epoch_ended:
            if terminal and not (epoch_ended):
                # print('Warning: trajectory ends early at %d steps.' % ep_len, flush=True)
                buf.delete_last_traj()
                (o, _), ep_ret, ep_len = env.reset(), 0, 0
                continue
            (o, _), ep_ret, ep_len = env.reset(), 0, 0
            num_traj += 1
            buf.finish_path()

    if save_buf:
        pickle.dump(buf, open(load_dataset, "wb"))

    return buf


def policy_evaluation(env_name, policy_path, gamma, max_len, total_trajs=100):
    env = gym.make(env_name)
    ac = load_policy(policy_path, env)

    (o, _), ep_len, ep_ret, ep_avg_ret = env.reset(), 0, 0, 0
    num_traj = 0
    rets = []
    avg_rets = []

    while num_traj < total_trajs:
        a, _, _ = ac.step(torch.as_tensor(o, dtype=torch.float32))
        next_o, r, d, _, _ = env.step(a)
        ep_ret += r * gamma**ep_len
        ep_avg_ret += r
        ep_len += 1
        # Update obs (critical!)
        o = next_o

        terminal = d
        epoch_ended = ep_len == max_len - 1

        if terminal or epoch_ended:
            num_traj += 1
            rets.append(ep_ret)
            avg_rets.append(ep_avg_ret)
            (o, _), ep_ret, ep_len, ep_avg_ret = env.reset(), 0, 0, 0
    return (1 - gamma) * np.mean(rets), np.var(rets), np.mean(avg_rets)
