import _pickle as pickle
import argparse
import inspect
import os
import sys

from tqdm import tqdm

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from disc_cop.constants import HYPERPARAMETERS, LOG_DIR
from disc_cop.envs import ENVS, ENV_FAMILY_SPECIFICS, ENV_TO_FAMILY
from disc_cop.utils import set_seed, policy_evaluation


def main(args):
    env_name = args.env_name
    env_family = ENV_TO_FAMILY[env_name]
    env_config = ENVS[env_family][env_name]
    res = dict()
    for seed in tqdm(HYPERPARAMETERS["seeds"]):
        set_seed(seed)
        res[seed] = dict()
        for gamma in HYPERPARAMETERS["discount_factors"]:
            res[seed][gamma] = policy_evaluation(
                env_name=env_config[0],
                policy_path=env_config[1],
                gamma=gamma,
                max_len=ENV_FAMILY_SPECIFICS[env_family]["max_len"],
                total_trajs=args.total_trajs,
            )

    pickle.dump(
        res,
        open(os.path.join(LOG_DIR, "baseline-{}.pkl".format(env_name)), "wb"),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--env_name",
        type=str,
        required=True,
        choices=list(ENV_TO_FAMILY.keys()),
        help="The environment name",
    )
    parser.add_argument(
        "--total_trajs",
        type=int,
        default=500,
        help="The number of trajectories for policy evaluation",
    )
    args = parser.parse_args()
    main(args)
