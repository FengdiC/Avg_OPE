import _pickle as pickle
import argparse
import gym
import inspect
import os
import sys

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from itertools import product
from tqdm import tqdm

from disc_cop.constants import HYPERPARAMETERS
from disc_cop.envs import ENVS, ENV_FAMILY_SPECIFICS, ENV_TO_FAMILY
from disc_cop.utils import maybe_collect_dataset, set_seed


def main(args):
    env_name = args.env_name
    env_family = ENV_TO_FAMILY[env_name]
    env_config = ENVS[env_family][env_name]
    for seed in tqdm(HYPERPARAMETERS[env_family]["seeds"]):
        load_dataset = os.path.join(
            args.dataset_dir,
            "{}-seed_{}".format(
                env_name,
                seed,
            ),
        )

        max_num_samples = max(HYPERPARAMETERS[env_family]["buffer_sizes"])
        min_max_len = min(HYPERPARAMETERS[env_family]["max_lens"])
        for (
            policy_path,
            random_weight,
        ) in product(
            [env_config[1]],
            HYPERPARAMETERS[env_family]["random_weights"],
        ):

            set_seed(seed)

            max_ep = max_num_samples // min_max_len
            env = gym.make(env_config[0])
            env.reset(seed=seed)

            maybe_collect_dataset(
                env,
                max_ep=max_ep,
                max_len=min_max_len,
                policy_path=policy_path,
                random_weight=random_weight,
                fold=1,
                load_dataset=os.path.join(
                    load_dataset,
                    "train-random_weight_{}.pkl".format(random_weight),
                ),
            )
            maybe_collect_dataset(
                env,
                max_ep=max_ep,
                max_len=min_max_len,
                policy_path=policy_path,
                random_weight=random_weight,
                fold=1,
                load_dataset=os.path.join(
                    load_dataset,
                    "test-random_weight_{}.pkl".format(random_weight),
                ),
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
        "--dataset_dir",
        type=str,
        required=True,
        help="The directory storing all datasets",
    )
    args = parser.parse_args()
    main(args)
