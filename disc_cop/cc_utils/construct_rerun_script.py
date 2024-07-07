"""
This script constructs a bash script that reruns all the incomplete/failed runs.
The list is provided by check_runs.py
"""

import inspect
import os
import sys

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from constants import LOG_DIR, DATASET_DIR, RUN_REPORT_DIR, CC_ACCOUNT, REPO_PATH
from envs import ENV_FAMILY_SPECIFICS, ENV_TO_FAMILY

rerun_variants = {
    "ant": [
        "random_weight_0.3-discount_factor_0.99-buffer_size_200-link_default-batch_size_512-bootstrap_target_cross_q-lr_0.005-alpha_0.0",
        "random_weight_0.7-discount_factor_0.8-buffer_size_80-link_default-batch_size_512-bootstrap_target_cross_q-lr_0.005-alpha_0.0",
        "random_weight_0.3-discount_factor_0.8-buffer_size_40-link_default-batch_size_512-bootstrap_target_cross_q-lr_0.005-alpha_0.0",
        "random_weight_0.3-discount_factor_0.99-buffer_size_80-link_default-batch_size_512-bootstrap_target_cross_q-lr_0.005-alpha_0.0",
        "random_weight_0.5-discount_factor_0.8-buffer_size_40-link_default-batch_size_512-bootstrap_target_cross_q-lr_0.005-alpha_0.0",
        "random_weight_0.3-discount_factor_0.995-buffer_size_200-link_default-batch_size_512-bootstrap_target_cross_q-lr_0.005-alpha_0.0",
        "random_weight_0.3-discount_factor_0.995-buffer_size_80-link_default-batch_size_512-bootstrap_target_cross_q-lr_0.005-alpha_0.0",
        "random_weight_0.7-discount_factor_0.8-buffer_size_200-link_default-batch_size_512-bootstrap_target_cross_q-lr_0.005-alpha_0.0",
        "random_weight_0.3-discount_factor_0.99-buffer_size_40-link_default-batch_size_512-bootstrap_target_cross_q-lr_0.005-alpha_0.0",
        "random_weight_0.5-discount_factor_0.99-buffer_size_40-link_default-batch_size_512-bootstrap_target_cross_q-lr_0.005-alpha_0.0",
        "random_weight_0.5-discount_factor_0.8-buffer_size_80-link_default-batch_size_512-bootstrap_target_cross_q-lr_0.005-alpha_0.0",
        "random_weight_0.3-discount_factor_0.995-buffer_size_40-link_default-batch_size_512-bootstrap_target_cross_q-lr_0.005-alpha_0.0",
        "random_weight_0.7-discount_factor_0.8-buffer_size_40-link_default-batch_size_512-bootstrap_target_cross_q-lr_0.005-alpha_0.0",
        "random_weight_0.3-discount_factor_0.8-buffer_size_80-link_default-batch_size_512-bootstrap_target_cross_q-lr_0.005-alpha_0.0",
        "random_weight_0.3-discount_factor_0.8-buffer_size_200-link_default-batch_size_512-bootstrap_target_cross_q-lr_0.005-alpha_0.0",
        "random_weight_0.5-discount_factor_0.8-buffer_size_200-link_default-batch_size_512-bootstrap_target_cross_q-lr_0.005-alpha_0.0",
    ],
    "halfcheetah": [
        "random_weight_0.5-discount_factor_0.8-buffer_size_80-link_default-batch_size_512-bootstrap_target_cross_q-lr_0.005-alpha_0.0",
        "random_weight_0.5-discount_factor_0.8-buffer_size_200-link_default-batch_size_256-bootstrap_target_cross_q-lr_0.005-alpha_0.0",
        "random_weight_0.5-discount_factor_0.8-buffer_size_200-link_default-batch_size_512-bootstrap_target_cross_q-lr_0.005-alpha_0.0",
        "random_weight_0.7-discount_factor_0.99-buffer_size_80-link_default-batch_size_512-bootstrap_target_cross_q-lr_0.005-alpha_0.0",
        "random_weight_0.3-discount_factor_0.8-buffer_size_200-link_default-batch_size_512-bootstrap_target_cross_q-lr_0.005-alpha_0.0",
        "random_weight_0.5-discount_factor_0.99-buffer_size_200-link_default-batch_size_512-bootstrap_target_cross_q-lr_0.005-alpha_0.0",
        "random_weight_0.5-discount_factor_0.995-buffer_size_200-link_default-batch_size_512-bootstrap_target_cross_q-lr_0.005-alpha_0.0",
        "random_weight_0.7-discount_factor_0.995-buffer_size_40-link_default-batch_size_512-bootstrap_target_cross_q-lr_0.005-alpha_0.0",
        "random_weight_0.5-discount_factor_0.995-buffer_size_80-link_default-batch_size_512-bootstrap_target_cross_q-lr_0.005-alpha_0.0",
        "random_weight_0.5-discount_factor_0.99-buffer_size_40-link_default-batch_size_512-bootstrap_target_cross_q-lr_0.005-alpha_0.0",
        "random_weight_0.3-discount_factor_0.99-buffer_size_40-link_default-batch_size_512-bootstrap_target_cross_q-lr_0.005-alpha_0.0",
        "random_weight_0.7-discount_factor_0.8-buffer_size_40-link_default-batch_size_512-bootstrap_target_cross_q-lr_0.005-alpha_0.0",
        "random_weight_0.3-discount_factor_0.8-buffer_size_80-link_default-batch_size_512-bootstrap_target_cross_q-lr_0.005-alpha_0.0",
        "random_weight_0.3-discount_factor_0.995-buffer_size_40-link_default-batch_size_512-bootstrap_target_cross_q-lr_0.005-alpha_0.0",
        "random_weight_0.7-discount_factor_0.995-buffer_size_80-link_default-batch_size_512-bootstrap_target_cross_q-lr_0.005-alpha_0.0",
        "random_weight_0.5-discount_factor_0.995-buffer_size_40-link_default-batch_size_512-bootstrap_target_cross_q-lr_0.005-alpha_0.0",
        "random_weight_0.5-discount_factor_0.99-buffer_size_80-link_default-batch_size_512-bootstrap_target_cross_q-lr_0.005-alpha_0.0",
        "random_weight_0.7-discount_factor_0.8-buffer_size_200-link_default-batch_size_512-bootstrap_target_cross_q-lr_0.005-alpha_0.0",
        "random_weight_0.3-discount_factor_0.99-buffer_size_200-link_default-batch_size_512-bootstrap_target_cross_q-lr_0.0001-alpha_0.0",
        "random_weight_0.7-discount_factor_0.8-buffer_size_80-link_default-batch_size_512-bootstrap_target_cross_q-lr_0.005-alpha_0.0",
        "random_weight_0.3-discount_factor_0.8-buffer_size_40-link_default-batch_size_512-bootstrap_target_cross_q-lr_0.005-alpha_0.0",
        "random_weight_0.3-discount_factor_0.995-buffer_size_80-link_default-batch_size_512-bootstrap_target_cross_q-lr_0.005-alpha_0.0",
        "random_weight_0.3-discount_factor_0.99-buffer_size_80-link_default-batch_size_512-bootstrap_target_cross_q-lr_0.005-alpha_0.0",
        "random_weight_0.7-discount_factor_0.99-buffer_size_40-link_default-batch_size_512-bootstrap_target_cross_q-lr_0.005-alpha_0.0",
        "random_weight_0.3-discount_factor_0.99-buffer_size_200-link_default-batch_size_512-bootstrap_target_cross_q-lr_0.005-alpha_0.0",
        "random_weight_0.5-discount_factor_0.8-buffer_size_40-link_default-batch_size_512-bootstrap_target_cross_q-lr_0.005-alpha_0.0",
        "random_weight_0.3-discount_factor_0.995-buffer_size_200-link_default-batch_size_512-bootstrap_target_cross_q-lr_0.005-alpha_0.0",
        "random_weight_0.7-discount_factor_0.99-buffer_size_200-link_default-batch_size_512-bootstrap_target_cross_q-lr_0.005-alpha_0.0",
        "random_weight_0.7-discount_factor_0.995-buffer_size_200-link_default-batch_size_512-bootstrap_target_cross_q-lr_0.005-alpha_0.0",
    ],
    "hopper": [
        "random_weight_0.7-discount_factor_0.995-buffer_size_80-link_default-batch_size_512-bootstrap_target_cross_q-lr_0.005-alpha_0.0",
        "random_weight_0.3-discount_factor_0.995-buffer_size_200-link_default-batch_size_512-bootstrap_target_cross_q-lr_0.005-alpha_0.0",
        "random_weight_0.5-discount_factor_0.995-buffer_size_40-link_default-batch_size_512-bootstrap_target_cross_q-lr_0.005-alpha_0.0",
        "random_weight_0.5-discount_factor_0.99-buffer_size_40-link_default-batch_size_512-bootstrap_target_cross_q-lr_0.005-alpha_0.0",
        "random_weight_0.3-discount_factor_0.995-buffer_size_80-link_default-batch_size_512-bootstrap_target_cross_q-lr_0.005-alpha_0.0",
        "random_weight_0.7-discount_factor_0.8-buffer_size_80-link_default-batch_size_512-bootstrap_target_cross_q-lr_0.005-alpha_0.0",
        "random_weight_0.3-discount_factor_0.8-buffer_size_40-link_default-batch_size_512-bootstrap_target_cross_q-lr_0.005-alpha_0.0",
        "random_weight_0.5-discount_factor_0.8-buffer_size_40-link_default-batch_size_512-bootstrap_target_cross_q-lr_0.005-alpha_0.0",
        "random_weight_0.3-discount_factor_0.99-buffer_size_200-link_default-batch_size_512-bootstrap_target_cross_q-lr_0.005-alpha_0.0",
        "random_weight_0.7-discount_factor_0.99-buffer_size_80-link_default-batch_size_512-bootstrap_target_cross_q-lr_0.005-alpha_0.0",
        "random_weight_0.3-discount_factor_0.8-buffer_size_200-link_default-batch_size_512-bootstrap_target_cross_q-lr_0.005-alpha_0.0",
        "random_weight_0.7-discount_factor_0.99-buffer_size_200-link_default-batch_size_512-bootstrap_target_cross_q-lr_0.005-alpha_0.0",
        "random_weight_0.7-discount_factor_0.99-buffer_size_40-link_default-batch_size_512-bootstrap_target_cross_q-lr_0.005-alpha_0.0",
        "random_weight_0.5-discount_factor_0.8-buffer_size_80-link_default-batch_size_512-bootstrap_target_cross_q-lr_0.005-alpha_0.0",
        "random_weight_0.5-discount_factor_0.99-buffer_size_200-link_default-batch_size_512-bootstrap_target_cross_q-lr_0.005-alpha_0.0",
        "random_weight_0.7-discount_factor_0.8-buffer_size_200-link_default-batch_size_512-bootstrap_target_cross_q-lr_0.005-alpha_0.0",
        "random_weight_0.5-discount_factor_0.99-buffer_size_80-link_default-batch_size_512-bootstrap_target_cross_q-lr_0.005-alpha_0.0",
        "random_weight_0.5-discount_factor_0.995-buffer_size_200-link_default-batch_size_512-bootstrap_target_cross_q-lr_0.005-alpha_0.0",
        "random_weight_0.5-discount_factor_0.995-buffer_size_80-link_default-batch_size_512-bootstrap_target_cross_q-lr_0.005-alpha_0.0",
        "random_weight_0.7-discount_factor_0.8-buffer_size_40-link_default-batch_size_512-bootstrap_target_cross_q-lr_0.005-alpha_0.0",
        "random_weight_0.3-discount_factor_0.8-buffer_size_80-link_default-batch_size_512-bootstrap_target_cross_q-lr_0.005-alpha_0.0",
        "random_weight_0.3-discount_factor_0.995-buffer_size_40-link_default-batch_size_512-bootstrap_target_cross_q-lr_0.005-alpha_0.0",
        "random_weight_0.3-discount_factor_0.99-buffer_size_80-link_default-batch_size_512-bootstrap_target_cross_q-lr_0.005-alpha_0.0",
    ],
    "swimmer": [
        "random_weight_0.3-discount_factor_0.995-buffer_size_200-link_default-batch_size_512-bootstrap_target_cross_q-lr_0.005-alpha_0.0",
        "random_weight_0.5-discount_factor_0.8-buffer_size_80-link_default-batch_size_512-bootstrap_target_cross_q-lr_0.005-alpha_0.0",
        "random_weight_0.3-discount_factor_0.99-buffer_size_200-link_default-batch_size_512-bootstrap_target_cross_q-lr_0.005-alpha_0.0",
        "random_weight_0.7-discount_factor_0.99-buffer_size_80-link_default-batch_size_512-bootstrap_target_cross_q-lr_0.005-alpha_0.0",
        "random_weight_0.5-discount_factor_0.8-buffer_size_200-link_default-batch_size_512-bootstrap_target_cross_q-lr_0.005-alpha_0.0",
        "random_weight_0.7-discount_factor_0.8-buffer_size_40-link_default-batch_size_512-bootstrap_target_cross_q-lr_0.005-alpha_0.0",
        "random_weight_0.5-discount_factor_0.995-buffer_size_40-link_default-batch_size_512-bootstrap_target_cross_q-lr_0.005-alpha_0.0",
        "random_weight_0.3-discount_factor_0.8-buffer_size_200-link_default-batch_size_512-bootstrap_target_cross_q-lr_0.005-alpha_0.0",
        "random_weight_0.7-discount_factor_0.995-buffer_size_80-link_default-batch_size_512-bootstrap_target_cross_q-lr_0.005-alpha_0.0",
        "random_weight_0.3-discount_factor_0.8-buffer_size_40-link_default-batch_size_512-bootstrap_target_cross_q-lr_0.005-alpha_0.0",
        "random_weight_0.3-discount_factor_0.995-buffer_size_40-link_default-batch_size_512-bootstrap_target_cross_q-lr_0.005-alpha_0.0",
        "random_weight_0.7-discount_factor_0.8-buffer_size_80-link_default-batch_size_512-bootstrap_target_cross_q-lr_0.005-alpha_0.0",
        "random_weight_0.7-discount_factor_0.995-buffer_size_40-link_default-batch_size_512-bootstrap_target_cross_q-lr_0.005-alpha_0.0",
        "random_weight_0.5-discount_factor_0.995-buffer_size_200-link_default-batch_size_512-bootstrap_target_cross_q-lr_0.005-alpha_0.0",
        "random_weight_0.3-discount_factor_0.99-buffer_size_80-link_default-batch_size_512-bootstrap_target_cross_q-lr_0.005-alpha_0.0",
        "random_weight_0.5-discount_factor_0.8-buffer_size_40-link_default-batch_size_512-bootstrap_target_cross_q-lr_0.005-alpha_0.0",
    ],
}

os.makedirs(
    os.path.join(RUN_REPORT_DIR, "reruns"),
    exist_ok=True,
)

num_runs = 0
dat_content = ""
for env_name, variants in rerun_variants.items():
    for variant in variants:
        curr_variant_path = os.path.join(
            LOG_DIR, env_name, "mse-tune-{}.pkl".format(variant)
        )

        dat_content += "export config_path={} dataset_dir={}\n".format(
            curr_variant_path,
            DATASET_DIR,
        )
        num_runs += 1

with open(os.path.join(LOG_DIR, "rerun.dat"), "w+") as f:
    f.writelines(dat_content)

sbatch_content = ""
sbatch_content += "#!/bin/bash\n"
sbatch_content += "#SBATCH --account={}\n".format(CC_ACCOUNT)
sbatch_content += "#SBATCH --time={}\n".format(
    ENV_FAMILY_SPECIFICS[ENV_TO_FAMILY[env_name]]["run_time"]
)
sbatch_content += "#SBATCH --cpus-per-task=1\n"
sbatch_content += "#SBATCH --mem=3G\n"
sbatch_content += "#SBATCH --array=1-{}\n".format(num_runs)
sbatch_content += "#SBATCH --output={}/%j.out\n".format(
    os.path.join(RUN_REPORT_DIR, "reruns")
)
sbatch_content += "module load python/3.10\n"
sbatch_content += "module load StdEnv/2020\n"
sbatch_content += "module load mujoco/2.2.2\n"
sbatch_content += "source ~/avg_ope/bin/activate\n"
sbatch_content += '`sed -n "${SLURM_ARRAY_TASK_ID}p"'
sbatch_content += " < {}`\n".format(os.path.join(LOG_DIR, "reruns.dat"))
sbatch_content += "echo ${SLURM_ARRAY_TASK_ID}\n"
sbatch_content += 'echo "Current working directory is `pwd`"\n'
sbatch_content += 'echo "Running on hostname `hostname`"\n'
sbatch_content += 'echo "Starting run at: `date`"\n'
sbatch_content += "python3 {}/disc_cop/run_experiment.py \\\n".format(REPO_PATH)
sbatch_content += "  --config_path=${config_path} \\\n"
sbatch_content += "  --dataset_dir=${dataset_dir}\n"
sbatch_content += 'echo "Program test finished with exit code $? at: `date`"\n'

with open(
    os.path.join(f"./run_all-reruns.sh"),
    "w+",
) as f:
    f.writelines(sbatch_content)
