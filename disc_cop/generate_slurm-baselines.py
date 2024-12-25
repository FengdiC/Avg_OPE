import inspect
import os
import sys

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from disc_cop.constants import (
    LOG_DIR,
    RUN_REPORT_DIR,
    REPO_PATH,
    CC_ACCOUNT,
)
from disc_cop.envs import ENV_TO_FAMILY

dat_content = ""
for env_name in ENV_TO_FAMILY:
    dat_content += "export env_name={} \n".format(
        env_name,
    )

with open(os.path.join(LOG_DIR, "baselines.dat"), "w+") as f:
    f.writelines(dat_content)

sbatch_content = ""
sbatch_content += "#!/bin/bash\n"
sbatch_content += "#SBATCH --account={}\n".format(CC_ACCOUNT)
sbatch_content += "#SBATCH --time=03:00:00\n"
sbatch_content += "#SBATCH --cpus-per-task=1\n"
sbatch_content += "#SBATCH --mem=3G\n"
sbatch_content += "#SBATCH --array=1-{}\n".format(len(ENV_TO_FAMILY))
sbatch_content += "#SBATCH --output={}/%j.out\n".format(
    os.path.join(RUN_REPORT_DIR, "baselines")
)
sbatch_content += "module load StdEnv/2023\n"
sbatch_content += "module load gcc gcccore/.12.3 opencv intel/2023.2.1 python/3.10 mpi4py\n"
sbatch_content += "module load mujoco\n"
sbatch_content += "source ~/avg_ope/bin/activate\n"
sbatch_content += '`sed -n "${SLURM_ARRAY_TASK_ID}p"'
sbatch_content += " < {}`\n".format(os.path.join(LOG_DIR, "baselines.dat"))
sbatch_content += "echo ${SLURM_ARRAY_TASK_ID}\n"
sbatch_content += 'echo "Current working directory is `pwd`"\n'
sbatch_content += 'echo "Running on hostname `hostname`"\n'
sbatch_content += 'echo "Starting run at: `date`"\n'
sbatch_content += "python3 {}/disc_cop/run_baseline.py \\\n".format(REPO_PATH)
sbatch_content += "  --env_name=${env_name}\n"
sbatch_content += 'echo "Program test finished with exit code $? at: `date`"\n'

with open(
    os.path.join(f"./run_all-run_baselines.sh"),
    "w+",
) as f:
    f.writelines(sbatch_content)
