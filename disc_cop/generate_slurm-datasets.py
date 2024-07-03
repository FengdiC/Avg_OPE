import os

from constants import LOG_DIR, RUN_REPORT_DIR, REPO_PATH, DATASET_DIR
from envs import ENV_TO_FAMILY

dat_content = ""
for env_name in ENV_TO_FAMILY:
    dat_content += "export env_name={} dataset_dir={} \n".format(
        env_name,
        DATASET_DIR,
    )

with open(os.path.join(LOG_DIR, "{}.dat".format(env_name)), "w+") as f:
    f.writelines(dat_content)

sbatch_content = ""
sbatch_content += "#!/bin/bash\n"
sbatch_content += "#SBATCH --account=def-schuurma\n"
sbatch_content += "#SBATCH --time=00:30:00\n"
sbatch_content += "#SBATCH --cpus-per-task=1\n"
sbatch_content += "#SBATCH --mem=3G\n"
sbatch_content += "#SBATCH --array=1-{}\n".format(len(ENV_TO_FAMILY))
sbatch_content += "#SBATCH --output={}/%j.out\n".format(
    os.path.join(RUN_REPORT_DIR, "generate_datasets")
)
sbatch_content += "module load python/3.9\n"
sbatch_content += "module load mujoco\n"
sbatch_content += "source ~/jaxl_env/bin/activate\n"
sbatch_content += '`sed -n "${SLURM_ARRAY_TASK_ID}p"'
sbatch_content += " < {}`\n".format(os.path.join(LOG_DIR, "generate_datasets.dat"))
sbatch_content += "echo ${SLURM_ARRAY_TASK_ID}\n"
sbatch_content += 'echo "Current working directory is `pwd`"\n'
sbatch_content += 'echo "Running on hostname `hostname`"\n'
sbatch_content += 'echo "Starting run at: `date`"\n'
sbatch_content += "python3 {}/generate_datasets.py \\\n".format(REPO_PATH)
sbatch_content += "  --env_name=${env_name} \\\n"
sbatch_content += "  --dataset_dir=${dataset_dir}\n"
sbatch_content += 'echo "Program test finished with exit code $? at: `date`"\n'

with open(
    os.path.join(f"./run_all-{env_name}.sh"),
    "w+",
) as f:
    f.writelines(sbatch_content)
