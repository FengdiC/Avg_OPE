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

from constants import LOG_DIR, DATASET_DIR, RUN_REPORT_DIR, CC_ACCOUNT, REPO_PATH, USE_GPU
from envs import ENV_FAMILY_SPECIFICS, ENV_TO_FAMILY

rerun_variants = {
    "hopper": [
        "random_weight_2.0-discount_factor_0.95-max_ep_40-max_len_100-link_default-batch_size_512-bootstrap_target_target_network-lr_0.001-alpha_0.01",
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

with open(os.path.join(LOG_DIR, "reruns.dat"), "w+") as f:
    f.writelines(dat_content)

sbatch_content = ""
sbatch_content += "#!/bin/bash\n"
sbatch_content += "#SBATCH --account={}\n".format(CC_ACCOUNT)
sbatch_content += "#SBATCH --time={}\n".format(
    ENV_FAMILY_SPECIFICS[ENV_TO_FAMILY[env_name]]["run_time"]
)
sbatch_content += "#SBATCH --cpus-per-task=1\n"
sbatch_content += "#SBATCH --mem=3G\n"
if USE_GPU:
    sbatch_content += "#SBATCH --gres=gpu:1\n"
sbatch_content += "#SBATCH --array=1-{}\n".format(num_runs)
sbatch_content += "#SBATCH --output={}/%j.out\n".format(
    os.path.join(RUN_REPORT_DIR, "reruns")
)
if USE_GPU:
    sbatch_content += "module load StdEnv/2023\n"
    sbatch_content += "module load gcc gcccore/.12.3 opencv intel/2023.2.1 cuda/11.8 python/3.10 mpi4py\n"
    sbatch_content += "module load mujoco\n"
    sbatch_content += "source ~/avg_ope_cuda/bin/activate\n"
else:
    sbatch_content += "module load StdEnv/2023\n"
    sbatch_content += "module load gcc gcccore/.12.3 opencv intel/2023.2.1 python/3.10 mpi4py\n"
    sbatch_content += "module load mujoco\n"
    sbatch_content += "source ~/avg_ope/bin/activate\n"
sbatch_content += '`sed -n "${SLURM_ARRAY_TASK_ID}p"'
sbatch_content += " < {}`\n".format(os.path.join(LOG_DIR, "reruns.dat"))
sbatch_content += "echo ${SLURM_ARRAY_TASK_ID}\n"
sbatch_content += 'echo "Current working directory is `pwd`"\n'
sbatch_content += 'echo "Running on hostname `hostname`"\n'
sbatch_content += 'echo "Starting run at: `date`"\n'

sbatch_content += "unzip {}/datasets.zip -d $SLURM_TMPDIR\n".format(DATASET_DIR)

sbatch_content += "python3 {}/disc_cop/run_experiment.py \\\n".format(REPO_PATH)
if USE_GPU:
    sbatch_content += "  --device=cuda:0 \\\n"
sbatch_content += "  --config_path=${config_path} \\\n"
sbatch_content += "  --dataset_dir=${SLURM_TMPDIR}/datasets \n"
sbatch_content += 'echo "Program test finished with exit code $? at: `date`"\n'

with open(
    os.path.join(f"./run_all-reruns.sh"),
    "w+",
) as f:
    f.writelines(sbatch_content)
