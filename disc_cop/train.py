import _pickle as pickle
import copy
import gymnasium as gym
import inspect
import numpy as np
import os
import sys
import torch
import torch.nn as nn

from tqdm import tqdm

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from torch.optim import Adam

from disc_cop.constants import MAX_BUFFER_SIZE
from disc_cop.envs import ENV_TO_FAMILY, ENV_ID_TO_NAME
from disc_cop.utils import maybe_collect_dataset, set_seed


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


# train weight net
def train_ratio(
    lr,
    env,
    seed,
    policy_path,
    link,
    random_weight,
    l1_lambda,
    cop_discount,
    discount=0.95,
    checkpoint=5,
    epoch=1000,
    cv_fold=1,
    batch_size=256,
    max_ep=20,
    max_len=50,
    use_batch_norm=False,
    use_target_network=False,
    tau=0.0005,
    load_dataset=None,
    baseline_path=None,
    save_path=None,
    device="cpu",
    **kwargs,
):
    """
    With cop_discount -> 1, fixed-point solution -> d_pi/d_mu
    """
    gamma = discount
    set_seed(seed)

    mujoco = ENV_TO_FAMILY[ENV_ID_TO_NAME[env]] == "mujoco"
    env = gym.make(env)
    env.reset(seed=seed)

    buf = maybe_collect_dataset(
        env,
        max_ep=max_ep,
        max_len=max_len,
        policy_path=policy_path,
        random_weight=random_weight,
        fold=1,
        load_dataset=os.path.join(
            load_dataset,
            "train-random_weight_{}-buffer_size_{}-max_len_{}.pkl".format(
                random_weight,
                MAX_BUFFER_SIZE,
                max_len,
            ),
        ),
        mujoco=mujoco,
    )
    buf_test = maybe_collect_dataset(
        env,
        max_ep=max_ep,
        max_len=max_len,
        policy_path=policy_path,
        random_weight=random_weight,
        fold=1,
        load_dataset=os.path.join(
            load_dataset,
            "test-random_weight_{}-buffer_size_{}-max_len_{}.pkl".format(
                random_weight,
                MAX_BUFFER_SIZE,
                max_len,
            ),
        ),
        mujoco=mujoco,
    )

    buf.to(device)
    buf_test.to(device)

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
    weight.to(device)

    # Set up optimizers for policy and value function
    optimizer = Adam(weight.parameters(), lr)

    """
    DEFINE LOSS
    """
    if use_target_network:
        target_weight = copy.deepcopy(weight)
        target_weight.to(device)

        def update(fold_num):
            # sample minibatches
            data = buf.sample(batch_size, fold_num)

            obs, act, next_obs = data["obs"], data["act"], data["next_obs"]
            tim, prod = data["tim"], data["prod"]
            logbev, logtarg = data["logbev"], data["logtarg"]
            first_timestep = data["first_timestep"]

            c_next_obs = weight(next_obs)
            with torch.no_grad():
                c_obs = target_weight(obs)

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
            non_first_timestep = torch.where(1 - first_timestep)[0]
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

    """
    EVALUATION
    """

    def eval(buffer):
        ratio = (
            weight(
                torch.as_tensor(
                    buffer.obs_buf.reshape((-1, buffer.obs_buf.shape[-1]))[: buffer.ptr],
                    dtype=torch.float32,
                    device=device,
                ),
            )
            .detach()
            .cpu()
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
            * np.exp(buffer.logtarg_buf.flatten()[: buffer.ptr] - buffer.logbev_buf.flatten()[: buffer.ptr])
            * buffer.rew_buf.flatten()[: buffer.ptr]
        )
        return obj * max_len * (1 - gamma)

    """
    TRAINING LOOP
    """
    run_dirname = (
        "random_weight_"
        + str(random_weight)
        + "-"
        + "discount_factor_"
        + str(gamma)
        + "-"
        + "max_ep_"
        + str(max_ep)
        + "-"
        + "max_len_"
        + str(max_len)
        + "-"
        + "link_"
        + str(link)
        + "-"
        + "batch_size_"
        + str(batch_size)
        + "-"
        + "bootstrap_target_"
        + ("target_network" if use_target_network else "cross_q")
        + "-"
        + "lr_"
        + str(lr)
        + "-"
        + "alpha_"
        + str(l1_lambda)
    )

    seed_prefix = "seed_{}".format(seed)
    baseline = None
    check_best = lambda obj_test, curr_best, steps: curr_best

    if save_path:
        os.makedirs(os.path.join(save_path, run_dirname), exist_ok=True)

    if baseline_path and os.path.isfile(baseline_path):
        baseline_rewards = pickle.load(open(baseline_path, "rb"))[seed]

        baseline = (1 - gamma) * np.mean(
            baseline_rewards[:, :max_len] * (gamma ** np.arange(max_len)[None])
        )

        def check_best(obj_test, curr_best, steps):
            loss = (obj_test - baseline) ** 2
            if loss < curr_best:
                curr_best = loss
                print("CURRENT BEST {} @ {}".format(curr_best, steps))
                if save_path:
                    torch.save(
                        {
                            "loss": loss,
                            "steps": steps,
                            "model_state_dict": {k: v.cpu() for k, v in weight.state_dict().items()},
                        },
                        open(
                            os.path.join(
                                save_path,
                                run_dirname,
                                "{}-curr_best.pt".format(
                                    seed_prefix
                                ),
                            ),
                            "wb",
                        ),
                    )
            return curr_best

    curr_best = np.inf
    for fold_num in range(cv_fold):
        objs, objs_test = [], []
        for steps in tqdm(range(epoch * checkpoint)):
            weight.train()
            update(fold_num)
            if steps % checkpoint == 0:
                weight.eval()
                obj, obj_test = eval(buf), eval(buf_test)
                objs.append(obj)
                objs_test.append(obj_test)
                curr_best = check_best(obj_test, curr_best, steps)

    if save_path:
        torch.save(
            {
                "loss": None,
                "steps": steps,
                "model_state_dict": {k: v.cpu() for k, v in weight.state_dict().items()},
            },
            open(
                os.path.join(
                    save_path,
                    run_dirname,
                    "{}-final.pt".format(seed_prefix)
                ),
                "wb",
            ),
        )

    return objs, objs_test
