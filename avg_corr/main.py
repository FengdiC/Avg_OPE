import numpy as np
from gym.spaces import Box, Discrete
import gym
import torch
import torch.nn as nn
from torch.optim import Adam
import ppo.algo.core as core
from ppo.algo.random_search import random_search

class weight_net(nn.Module):
    def __init__(self, o_dim, hidden_sizes,activation):
        super(weight_net, self).__init__()
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

# sample behaviour dataset randomly
def collect_dataset(env):


# train weight net
