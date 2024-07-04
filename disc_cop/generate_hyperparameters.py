import _pickle as pickle
import inspect
import math
import os
import sys

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from itertools import product
from tqdm import tqdm

from disc_cop.constants import LOG_DIR, DATASET_DIR, HYPERPARAMETERS
from disc_cop.envs import ENVS, ENV_FAMILY_SPECIFICS


def generate_experiment_configs():
    """
    Generate experiment configurations.
    This outputs a directory of pickle files that specifies the hyperparameters
    """

    random_weights = HYPERPARAMETERS["random_weights"]
    discount_factors = HYPERPARAMETERS["discount_factors"]
    batch_sizes = HYPERPARAMETERS["batch_sizes"]
    links = HYPERPARAMETERS["links"]
    buffer_sizes = HYPERPARAMETERS["buffer_sizes"]
    bootstrap_targets = HYPERPARAMETERS["bootstrap_targets"]
    lrs = HYPERPARAMETERS["lrs"]
    alphas = HYPERPARAMETERS["alphas"]
    tau = HYPERPARAMETERS["tau"]
    seeds = HYPERPARAMETERS["seeds"]
    step_frequency = HYPERPARAMETERS["step_frequency"]

    for env_family in ENVS:
        env_specific = ENV_FAMILY_SPECIFICS[env_family]
        for env_name, env_config in ENVS[env_family].items():
            dat_content = ""
            for (
                random_weight,
                discount_factor,
                batch_size,
                link,
                buffer_size,
                bootstrap_target,
                lr,
                alpha,
            ) in tqdm(
                product(
                    random_weights,
                    discount_factors,
                    batch_sizes,
                    links,
                    buffer_sizes,
                    bootstrap_targets,
                    lrs,
                    alphas,
                ),
                postfix="Hyperparameter sweep for environment: {}".format(env_name),
            ):
                filename = (
                    "mse-tune-"
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
                    + "-"
                    + "bootstrap_target_"
                    + str(bootstrap_target)
                    + "-"
                    + "lr_"
                    + str(lr)
                    + "-"
                    + "alpha_"
                    + str(alpha)
                    + ".pkl"
                )
                os.makedirs(os.path.join(LOG_DIR, env_name), exist_ok=True)

                result = dict(
                    seeds=seeds,
                    env_name=env_name,
                    hyperparameters=dict(
                        lr=lr,
                        env=env_config[0],
                        policy_path=env_config[1],
                        link=link,
                        random_weight=random_weight,
                        l1_lambda=alpha,
                        checkpoint=step_frequency,
                        epoch=math.ceil(env_specific["train_steps"] / step_frequency),
                        cv_fold=1,
                        batch_size=batch_size,
                        buffer_size=buffer_size,
                        max_len=env_specific["max_len"],
                        use_batch_norm=bootstrap_target != "target_network",
                        use_target_network=bootstrap_target == "target_network",
                        discount=discount_factor,
                        cop_discount=discount_factor,
                        tau=tau,
                        baseline_path=os.path.join(
                            LOG_DIR, "baseline-{}.pkl".format(env_name)
                        ),
                        save_path=os.path.join(LOG_DIR, "saved_models", env_name),
                    ),
                    results=dict(),
                )

                pickle.dump(
                    result, open(os.path.join(LOG_DIR, env_name, filename), "wb")
                )

                dat_content += "export config_path={} dataset_dir={} \n".format(
                    os.path.join(LOG_DIR, env_name, filename),
                    DATASET_DIR,
                )

            with open(os.path.join(LOG_DIR, "{}.dat".format(env_name)), "w+") as f:
                f.writelines(dat_content)


if __name__ == "__main__":
    generate_experiment_configs()
