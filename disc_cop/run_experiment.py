import _pickle as pickle
import argparse
import inspect
import os
import sys

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from disc_cop.train import train_ratio

from tqdm import tqdm


def main(args):
    assert os.path.isfile(args.config_path)
    experiment_info = pickle.load(open(args.config_path, "rb"))

    for seed in tqdm(experiment_info["seeds"]):
        experiment_info["results"][seed] = train_ratio(
            **experiment_info["hyperparameters"],
            load_dataset=os.path.join(
                args.dataset_dir,
                "{}-seed_{}".format(
                    experiment_info["env_name"],
                    seed,
                ),
            )
        )

    pickle.dump(
        experiment_info,
        open(args.config_path, "wb"),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_path",
        type=str,
        required=True,
        help="The pickle file containing the hyperparameters",
    )
    parser.add_argument(
        "--dataset_dir",
        type=str,
        required=True,
        help="The directory storing all datasets",
    )
    args = parser.parse_args()
    main(args)
