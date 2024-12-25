import os, sys, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
from algo.ppo import argsparser,ppo
from algo import core

from utils import logger
import numpy as np
import gym
from algo.random_search import random_search,set_one_thread

args = argsparser()
seeds = range(3)

# Torch Shenanigans fix
set_one_thread()

logger.configure(args.log_dir, ['csv'], log_suffix='biased-ppo-tune-' + str(args.env))

returns = []
for seed in seeds:
    hyperparam = random_search(args.seed)
    # hyperparam['gamma'] = args.gamma
    checkpoint = 4000
    result = ppo(lambda: gym.make(args.env), actor_critic=core.MLPActorCritic,
                 ac_kwargs=dict(hidden_sizes=args.hid), gamma=hyperparam['gamma'], pi_lr = hyperparam["pi_lr"],
                 target_kl=hyperparam['target_kl'], vf_lr=hyperparam['vf_lr'], epochs=args.epochs,
                 seed=seed, naive=False, save_freq=args.save_freq,logger_kwargs=dict(log_dir=args.log_dir))

    ret = np.array(result)
    print(ret.shape)
    returns.append(ret)
    name = list(hyperparam.values())
    name = [str(s) for s in name]
    name.append(str(seed))
    print("hyperparam", '-'.join(name))
    logger.logkv("hyperparam", '-'.join(name))
    for n in range(ret.shape[0]):
        logger.logkv(str((n + 1) * checkpoint), ret[n])
    logger.dumpkvs()
