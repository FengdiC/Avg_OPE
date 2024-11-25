import _pickle as pickle
import gymnasium as gym
import inspect
import numpy as np
import os
import sys
import torch

from torch.distributions.normal import Normal

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from gymnasium.spaces import Box, Discrete

import ppo.algo.core as core


def set_seed(seed):
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


class Buffer:
    """
    A buffer for storing trajectories experienced by a PPO agent interacting
    with the environment, and using Generalized Advantage Estimation (GAE-Lambda)
    for calculating the advantages of state-action pairs.
    """

    def __init__(self, obs_dim, act_dim, max_ep, max_len, fold):
        self.obs_buf = np.zeros((max_ep, max_len, obs_dim[0]), dtype=np.float32)
        self.next_obs_buf = np.zeros(
            (max_ep, max_len, obs_dim[0]), dtype=np.float32
        )
        try:
            self.act_buf = np.zeros((max_ep, max_len, act_dim[0]), dtype=np.float32)
        except:
            self.act_buf = np.zeros((max_ep, max_len), dtype=np.float32)
        self.rew_buf = np.zeros((max_ep, max_len), dtype=np.float32)
        self.tim_buf = np.zeros((max_ep, max_len), dtype=np.int32)
        self.logtarg_buf = np.zeros((max_ep, max_len), dtype=np.float32)
        self.prod_buf = np.zeros((max_ep, max_len), dtype=np.float32)
        self.logbev_buf = np.zeros((max_ep, max_len), dtype=np.float32)
        self.fold = fold
        self.ptr = 0
        self.max_ep = max_ep
        self.max_len = max_len
        self.max_size = max_ep * max_len
        self.ep_start_inds = [0]
        self.ep_i = 0

    def store(self, obs, act, rew, next_obs, tim, logbev, logtarg):
        """
        Append one timestep of agent-environment interaction to the buffer.
        """
        assert self.ptr < self.max_size  # buffer has to have room so you can store
        ep_i = self.ptr // self.max_len
        timestep_i = self.ptr % self.max_len

        self.obs_buf[ep_i, timestep_i] = obs
        self.next_obs_buf[ep_i, timestep_i] = next_obs
        self.act_buf[ep_i, timestep_i] = act
        self.rew_buf[ep_i, timestep_i] = rew
        self.tim_buf[ep_i, timestep_i] = tim
        self.logbev_buf[ep_i, timestep_i] = logbev
        self.logtarg_buf[ep_i, timestep_i] = logtarg
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

        # the next two lines implement GAE-Lambda advantage calculation
        deltas = self.logtarg_buf[self.ep_i] - self.logbev_buf[self.ep_i]
        self.prod_buf[self.ep_i] = np.append(
            0, core.discount_cumsum(deltas, 1)[:-1]
        )

        self.ep_i += 1

    def set_ep_len(self, max_ep, max_len):
        assert self.max_ep >= max_ep
        assert self.max_len >= max_len

        self.max_len = max_len
        self.max_ep = max_ep
        self.ep_i = max_ep
        self.ptr = max_ep * max_len

    def sample(self, batch_size, fold_num):
        interval = int(self.ptr / self.fold)
        if self.fold > 1:
            ind = np.random.randint(self.ptr - interval, size=batch_size)
            ind = ind + np.where(ind >= fold_num * interval, 1, 0) * interval
        else:
            ind = np.random.randint(-self.max_ep, self.ptr, size=batch_size)

        sample_ind = np.clip(ind, a_min=0, a_max=np.inf).astype(int)
        ep_i = sample_ind // self.max_len
        timestep_i = sample_ind % self.max_len

        data = dict(
            obs=self.obs_buf[ep_i, timestep_i],
            act=self.act_buf[ep_i, timestep_i],
            prod=self.prod_buf[ep_i, timestep_i],
            next_obs=self.next_obs_buf[ep_i, timestep_i],
            tim=self.tim_buf[ep_i, timestep_i],
            logbev=self.logbev_buf[ep_i, timestep_i],
            logtarg=self.logtarg_buf[ep_i, timestep_i],
            first_timestep=ind < 0,
        )

        if np.any(ind < 0):
            # Find any s_0 as s' and properly set them.
            start_ind = np.where(ind < 0)[0]
            data["next_obs"][start_ind] = data["obs"][start_ind]
        return {k: torch.as_tensor(v, dtype=torch.float32) for k, v in data.items()}

    def delete_last_traj(self):
        self.ptr = self.ep_i * self.max_len


def load_policy(path, env):
    ac_kwargs = dict(hidden_sizes=[64, 32])

    ac = core.MLPActorCritic(env.observation_space, env.action_space, **ac_kwargs)
    checkpoint = torch.load(path)
    ac.load_state_dict(checkpoint["model_state_dict"])
    return ac


def maybe_collect_dataset(
    env,
    max_ep,
    max_len,
    policy_path,
    random_weight,
    fold=1,
    load_dataset=None,
    mujoco=False,
):
    save_buf = load_dataset is not None
    if load_dataset:
        print("dataset path: {}".format(load_dataset))
        if os.path.isfile(load_dataset):
            save_buf = False
            print("Loaded from existing buffer.")
            buf = pickle.load(open(load_dataset, "rb"))
            buf.set_ep_len(max_ep, max_len)
        os.makedirs(os.path.dirname(load_dataset), exist_ok=True)

    ac = load_policy(policy_path, env)
    obs_dim = env.observation_space.shape
    act_dim = env.action_space.shape

    buf = Buffer(obs_dim, act_dim, max_ep, max_len, fold)

    (o, _), ep_len = env.reset(), 0
    num_traj = 0

    if isinstance(env.action_space, Box):
        action_range = env.action_space.high - env.action_space.low
        assert np.any(action_range > 0)
        unif = 1 / np.prod(action_range)
    elif isinstance(env.action_space, Discrete):
        unif = 1 / env.action_space.n

    while num_traj < max_ep:
        if mujoco:
            with torch.no_grad():
                std = torch.exp(ac.pi.log_std)
                mu = ac.pi.mu_net(torch.as_tensor(o, dtype=torch.float32))
                beh_pi = Normal(mu, std * random_weight)
                a = beh_pi.sample().numpy()
                pi = ac.pi._distribution(torch.as_tensor(o, dtype=torch.float32))
                logtarg = (
                    ac.pi._log_prob_from_distribution(pi, torch.as_tensor(a)).detach().numpy()
                )
                logbev = beh_pi.log_prob(torch.as_tensor(a)).sum(axis=-1).detach().numpy()
        else:
            if np.random.random() < random_weight:
                # random behaviour policy
                a = env.action_space.sample()
            else:
                a, _, _ = ac.step(torch.as_tensor(o, dtype=torch.float32))
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
            if epoch_ended:
                num_traj += 1
                rets.append(ep_ret)
                avg_rets.append(ep_avg_ret)
            (o, _), ep_ret, ep_len, ep_avg_ret = env.reset(), 0, 0, 0

    return (1 - gamma) * np.mean(rets), np.var(rets), np.mean(avg_rets)
