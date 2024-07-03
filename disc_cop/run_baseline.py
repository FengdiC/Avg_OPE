import _pickle as pickle
import os

from constants import HYPERPARAMETERS, LOG_DIR
from envs import ENVS, ENV_FAMILY_SPECIFICS
from utils import set_seed, policy_evaluation


def main(args):
    set_seed(args.seed)
    for env_family in ENVS:
        for env_name, env_config in ENVS[env_family].items():
            res = dict()
            for gamma in HYPERPARAMETERS["discount_factors"]:
                res[gamma] = policy_evaluation(
                    env_name=env_config[0],
                    policy_path=env_config[1],
                    gamma=gamma,
                    max_len=ENV_FAMILY_SPECIFICS[env_family]["max_len"],
                    total_trajs=500,
                )

            pickle.dump(
                res,
                open(os.path.join(LOG_DIR, "baseline-{}.pkl".format(env_name)), "wb"),
            )
