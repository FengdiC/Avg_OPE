import os, sys, inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
import copy
import numpy as np
from gym.spaces import Box, Discrete
from tqdm import tqdm
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

    def sample(self, batch_size, fold_num):
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


class WeightNet(nn.Module):
    def __init__(self, o_dim, hidden_sizes, activation, use_batch_norm=False):
        super(WeightNet, self).__init__()
        sizes = [o_dim] + list(hidden_sizes)
        layers = []
        for j in range(len(sizes) - 1):
            if use_batch_norm:
                layers += [nn.BatchNorm1d(sizes[j])]
            layers += [nn.Linear(sizes[j], sizes[j + 1]), nn.Tanh()]
        if use_batch_norm:
            layers += [nn.BatchNorm1d(sizes[-1])]
        self.body = nn.Sequential(*layers)
        self.weight = nn.Sequential(
            nn.Linear(sizes[-1], 1), activation()
        )  # nn.Identity()

    def forward(self, obs):
        obs = obs.float()
        body = self.body(obs)
        weight = self.weight(body)
        return torch.squeeze(weight)


# load target policy
def load(path, env):
    ac_kwargs = dict(hidden_sizes=[64, 32])

    ac = core.MLPActorCritic(env.observation_space, env.action_space, **ac_kwargs)
    checkpoint = torch.load(path)
    ac.load_state_dict(checkpoint["model_state_dict"])
    return ac


def eval_policy(env_name, path, gamma):
    env = gym.make(env_name)
    ac = load(path, env)
    if gamma is None:
        hyperparam = random_search(196)
        gamma = hyperparam["gamma"]

    (o, _), ep_len, ep_ret, ep_avg_ret = env.reset(), 0, 0, 0
    num_traj = 0
    rets = []
    avg_rets = []

    while num_traj < 100:
        a, _, logtarg = ac.step(torch.as_tensor(o, dtype=torch.float32))
        next_o, r, d, _, _ = env.step(a)
        ep_ret += r * gamma**ep_len
        ep_avg_ret += r
        ep_len += 1
        # Update obs (critical!)
        o = next_o

        terminal = d

        if terminal:
            num_traj += 1
            rets.append(ep_ret)
            avg_rets.append(ep_avg_ret)
            (o, _), ep_ret, ep_len, ep_avg_ret = env.reset(), 0, 0, 0
    return (1 - gamma) * np.mean(rets), np.var(rets), np.mean(avg_rets)


# sample behaviour dataset
# behaviour policy = (1- random_weight) * target_policy + random_weight * random_policy
# random_weight = 0.3  0.5  0.7
# classic control max_len=50 number_traj = 40 80 200
def collect_dataset(
    env,
    gamma,
    buffer_size=20,
    max_len=200,
    path="./exper/cartpole_998.pth",
    random_weight=0.2,
    fold=10,
):
    ac = load(path, env)
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
    return buf


# TODO: TD learning
def td_learning():
    """
    Two-timescale learning:
    - weight ratio
    - values

    NOTE: we need to apply importance sampling on the loss
    """

    pass


# train weight net
def train(
    lr,
    env,
    seed,
    path,
    hyper_choice,
    link,
    random_weight,
    l1_lambda,
    cop_discount,
    discount=0.95,
    checkpoint=5,
    epoch=1000,
    cv_fold=1,
    batch_size=256,
    buffer_size=20,
    max_len=50,
    use_batch_norm=False,
    use_target_network=False,
    tau=0.0005,
    **kwargs
):
    """
    With cop_discount -> 1, fixed-point solution -> d_pi/d_mu
    """
    # hyperparam = random_search(hyper_choice)
    # gamma = hyperparam['gamma']
    gamma = discount
    env = gym.make(env)

    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    env.reset(seed=seed)

    buf = collect_dataset(
        env,
        gamma,
        buffer_size=buffer_size,
        max_len=max_len,
        path=path,
        random_weight=random_weight,
        fold=1,
    )
    buf_test = collect_dataset(
        env,
        gamma,
        buffer_size=buffer_size,
        max_len=max_len,
        path=path,
        random_weight=random_weight,
        fold=1,
    )

    if link == "inverse" or link == "identity":
        weight = WeightNet(
            env.observation_space.shape[0],
            hidden_sizes=(256, 256),
            activation=nn.ReLU,
            use_batch_norm=use_batch_norm,
        )
    else:
        weight = WeightNet(
            env.observation_space.shape[0],
            hidden_sizes=(256, 256),
            activation=nn.Identity,
            use_batch_norm=use_batch_norm,
        )

    # Set up optimizers for policy and value function
    optimizer = Adam(weight.parameters(), lr)

    if use_target_network:
        target_weight = copy.deepcopy(weight)

        def update(fold_num):
            # sample minibatches
            data = buf.sample(batch_size, fold_num)

            obs, act, next_obs = data["obs"], data["act"], data["next_obs"]
            tim, prod = data["tim"], data["prod"]
            logbev, logtarg = data["logbev"], data["logtarg"]
            first_timestep = data["first_timestep"]

            c_next_obs = target_weight(next_obs)
            with torch.no_grad():
                c_obs = weight(obs)

            if link == "inverse":
                raise NotImplementedError
                # loss = ((1/weight(obs) - label) ** 2).mean()
            elif link == "identity":
                with torch.no_grad():
                    label = (
                        cop_discount * torch.exp((logtarg - logbev) + torch.log(c_obs))
                        + (1 - cop_discount)
                    ) ** (1 - first_timestep)
                    label = label.detach()
                loss = ((c_next_obs - label) ** 2).mean()
            elif link == "loglog":
                raise NotImplementedError
                # loss = ((torch.exp(torch.exp(weight(obs)))-1 - label) ** 2).mean()
            else:
                # By default log parameterization
                with torch.no_grad():
                    label = (
                        cop_discount * torch.exp((logtarg - logbev) + c_obs)
                        + (1 - cop_discount)
                    ) ** (1 - first_timestep)
                    label = label.detach()
                loss = ((torch.exp(c_next_obs) - label) ** 2).mean()

            l1_norm = sum(torch.linalg.norm(p, 1) for p in weight.parameters())
            loss = loss + l1_lambda * l1_norm

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Update target network
            for param, target_param in zip(
                weight.parameters(), target_weight.parameters()
            ):
                target_param.data.mul_(1.0 - tau)
                target_param.data.add_(param.data * tau)

    else:

        def update(fold_num):
            # sample minibatches
            data = buf.sample(batch_size, fold_num)

            obs, act, next_obs = data["obs"], data["act"], data["next_obs"]
            tim, prod = data["tim"], data["prod"]
            logbev, logtarg = data["logbev"], data["logtarg"]
            first_timestep = data["first_timestep"]

            # Since we might get s_0 as s', we should exclude predicting s for those situations
            non_first_timestep = np.where(1 - first_timestep)[0]
            all_obs = torch.cat((obs[non_first_timestep], next_obs), dim=0)
            c_all = weight(all_obs)
            c_next_obs = c_all[len(non_first_timestep) :]
            c_obs = torch.ones_like(c_next_obs)
            c_obs[non_first_timestep] = c_all[non_first_timestep]

            if link == "inverse":
                raise NotImplementedError
                # loss = ((1/weight(obs) - label) ** 2).mean()
            elif link == "identity":
                with torch.no_grad():
                    label = (
                        cop_discount * torch.exp((logtarg - logbev) + torch.log(c_obs))
                        + (1 - cop_discount)
                    ) ** (1 - first_timestep)
                    label = label.detach()
                loss = ((c_next_obs - label) ** 2).mean()
            elif link == "loglog":
                raise NotImplementedError
                # loss = ((torch.exp(torch.exp(weight(obs)))-1 - label) ** 2).mean()
            else:
                # By default log parameterization
                with torch.no_grad():
                    label = (
                        cop_discount * torch.exp((logtarg - logbev) + c_obs)
                        + (1 - cop_discount)
                    ) ** (1 - first_timestep)
                    label = label.detach()
                loss = ((torch.exp(c_next_obs) - label) ** 2).mean()

            l1_norm = sum(torch.linalg.norm(p, 1) for p in weight.parameters())
            loss = loss + l1_lambda * l1_norm

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # Let's record the estimated objective value = 1/n \sum_{i=1}^n est_ratio(s_i,a_i) r(s_i,a_i)
    # classic control: train 5k steps and checkpoint =5

    def eval(buffer):
        ratio = (
            weight(torch.as_tensor(buffer.obs_buf[: buffer.ptr], dtype=torch.float32))
            .detach()
            .numpy()
        )
        if link == "inverse":
            ratio = 1 / (ratio + 0.001)
        elif link == "identity":
            pass
        elif link == "loglog":
            ratio = np.exp(np.exp(ratio))
        else:
            ratio = np.exp(ratio)
        obj = np.mean(
            ratio
            * np.exp(buffer.logtarg_buf[: buffer.ptr] - buffer.logbev_buf[: buffer.ptr])
            * buffer.rew_buf[: buffer.ptr]
        )
        return obj * max_len * (1 - gamma)

    def eval_cv(buffer, fold_num):
        interval = int(buffer.ptr / buffer.fold)
        ind = range(fold_num * interval, (fold_num + 1) * interval, 1)
        ratio = (
            weight(torch.as_tensor(buffer.obs_buf[ind], dtype=torch.float32))
            .detach()
            .numpy()
        )
        if link == "inverse":
            ratio = 1 / (ratio + 0.001)
        elif link == "identity":
            pass
        elif link == "loglog":
            ratio = np.exp(np.exp(ratio)) - 1
        else:
            ratio = np.exp(ratio)
        obj = np.mean(
            ratio
            * np.exp(buffer.logtarg_buf[ind] - buffer.logbev_buf[ind])
            * buffer.rew_buf[ind]
        )
        return obj * max_len * (1 - gamma)

    objs_cv_mean = []
    for fold_num in range(cv_fold):
        objs, objs_test, objs_cv = [], [], []
        for steps in tqdm(range(epoch * checkpoint)):
            weight.train()
            update(fold_num)
            if steps % checkpoint == 0:
                # obj_cv = eval_cv(buf,fold_num)
                weight.eval()
                obj, obj_test = eval(buf), eval(buf_test)
                objs.append(obj)
                objs_test.append(obj_test)
                # objs_cv.append(np.around(obj_cv,decimals=4))
        # objs_cv_mean.append(objs_cv)
    return objs, objs_test
    # return np.around(np.mean(objs_cv_mean,axis=0),decimals=4)


def argsparser():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, default="./")
    parser.add_argument("--log_dir", type=str, default="./")
    parser.add_argument("--env", type=str, default="Hopper-v4")
    parser.add_argument("--seed", "-s", type=int, default=0)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--epoch", type=int, default=250)
    parser.add_argument("--array", type=int, default=1)

    parser.add_argument("--use_target_network", action="store_true")
    parser.add_argument("--tau", type=float, default=0.0005)
    parser.add_argument("--use_batch_norm", action="store_true")
    parser.add_argument("--cop_discount", type=float, default=0.99)
    parser.add_argument("--discount", type=float, default=0.99)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--buffer_size", type=int, default=20)
    parser.add_argument("--max_len", type=int, default=50)
    parser.add_argument("--link", type=str, default="log")
    parser.add_argument("--l1_lambda", type=float, default=1.0)
    parser.add_argument("--random_weight", type=float, default=0.3)
    parser.add_argument("--true_value", type=float, default=1.0)
    args = parser.parse_args()
    return args


def tune():
    """
    (1) The true value is estimated by sampling a large amount of trajectories under the target policy. The target policy is in the zip file

    (2) The behaviour policy will be (1- random_weight) * target_policy + random_weight*random_policy. The code is in the same file, method, collect(). We will use random_weight = 0.3, 0.5 and 0.7.

    (3) Let's record the estimated objective value = 1/n \sum_{i=1}^n est_ratio(s_i,a_i) r(s_i,a_i)  and (1-gamma) \sum_{i=1}^{100}q(s_{i,0},\pi_{targ}(s_{i,0}))on two datasets: both training and testing.
    And track the MSE between estimated value and the given true value; we will store the weight of the last model for each seed.

    (3)(b) Let's record every 5 steps and totally train for 10k steps for classic control, 250k for mujoco across 10 random seeds. [Check again with the original paper.]

    (4) The dataset size will be based on the number of trajectories, say 40 80 200 trajectories, each of 50 steps for classic control and 100 steps for mujoco.

    (5) Let's try for 3 discount factors: 0.8, 0.99 and 0.995.
    """
    random_weights = [0.3, 0.5, 0.7]
    discount_factors = [0.8, 0.99, 0.995]
    batch_sizes = [256, 512]
    links = ["default"]
    buffer_sizes = [40, 80, 200]

    args = argsparser()
    seeds = range(3)

    # One for each hyperparameter
    idx = np.unravel_index(args.array, (3, 3, 2, 1, 3))
    random_weight, discount_factor, batch_size, link, buffer_size = (
        random_weights[idx[0]],
        discount_factors[idx[1]],
        batch_sizes[idx[2]],
        links[idx[3]],
        buffer_sizes[idx[4]],
    )
    filename = (
        args.log_dir
        + "mse-tune-"
        + "random_weight_"
        + str(random_weight)
        + "-"
        + "discount_factor_"
        + str(discount_factor)
        + "-"
        + "buffer_size_"
        + str(buffer_size)
        + "-"
        + "link_"
        + str(link)
        + "-"
        + "batch_size_"
        + str(batch_size)
        + ".csv"
    )
    os.makedirs(args.log_dir, exist_ok=True)
    mylist = [str(i) for i in range(0, args.epoch * args.steps, args.steps)] + [
        "hyperparam"
    ]
    with open(filename, "w+", newline="") as file:
        # Step 4: Using csv.writer to write the list to the CSV file
        writer = csv.writer(file)
        writer.writerow(mylist)  # Use writerow for single list

    for alpha in [0.0005, 0.001, 0.002, 0.005, 0.01]:
        for lr in [0.0001, 0.0005, 0.001, 0.005]:
            result = []
            print("Finish one combination of hyperparameters!")
            for seed in seeds:
                cv = train(
                    lr=lr,
                    env=args.env,
                    seed=seed,
                    path=args.path,
                    hyper_choice=args.seed,
                    link=link,
                    random_weight=random_weight,
                    l1_lambda=alpha,
                    checkpoint=args.steps,
                    epoch=args.epoch,
                    cv_fold=10,
                    batch_size=batch_size,
                    buffer_size=buffer_size,
                    max_len=args.max_len,
                )
                print(
                    "Return result shape: ", cv.shape, ":::", args.steps, ":::", seeds
                )
                result.append(cv)
                # name = ['lr',lr,'alpha',alpha]
                # name = [str(s) for s in name]
                # name.append(str(seed))
                # print("hyperparam", '-'.join(name))
                # logger.logkv("hyperparam", '-'.join(name))
                # for n in range(cv.shape[0]):
                #     logger.logkv(str(n * args.steps), cv[n])
                # logger.dumpkvs()
            result = np.array(result)
            ret = np.around(np.mean(result, axis=0), decimals=4)
            var = np.around(np.var(result, axis=0), decimals=4)
            print("Mean shape: ", ret.shape, ":::", var.shape)
            name = ["lr", lr, "alpha", alpha]
            name = [str(s) for s in name]
            name_1 = name + ["mean"]
            name_2 = name + ["var"]
            mylist = [str(i) for i in list(ret)] + ["-".join(name_1)]
            with open(filename, "a", newline="") as file:
                # Step 4: Using csv.writer to write the list to the CSV file
                writer = csv.writer(file)
                writer.writerow(mylist)  # Use writerow for single list
            print("-".join(name_1))
            # logger.logkv("hyperparam", '-'.join(name_2))
            # for n in range(ret.shape[0]):
            #     logger.logkv(str(n * args.steps), var[n])
            # logger.dumpkvs()


if __name__ == "__main__":
    args = argsparser()
    res_train, res_test = train(**vars(args), hyper_choice=args.seed)
    print(res_train[-1])
    print(res_test[-1])

    res_targ = eval_policy(args.env, args.path, args.discount)
    print(res_targ[0])
