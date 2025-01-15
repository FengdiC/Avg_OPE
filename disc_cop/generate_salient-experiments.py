import inspect
import os
import sys

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from disc_cop.constants import LOG_DIR, REPO_PATH, DATASET_DIR, USE_GPU, NUM_GPUS, NUM_RUNS_PER_GPU
from disc_cop.envs import ENV_TO_FAMILY

sbatch_content = ""
sbatch_content += "#!/bin/bash\n"
run_i = 0

for env_name in ENV_TO_FAMILY:
    configs = [
        filename
        for filename in os.listdir(os.path.join(LOG_DIR, env_name))
        if filename.endswith(".pkl")
    ]

    with open(os.path.join(LOG_DIR, "{}.dat".format(env_name)), "r") as f:
        for line in f.readlines():
            print(line)
            if not line.startswith("export"):
                continue

            config_path = line.split(" ")[1].split("=")[1]
            
            sbatch_content += "python3 {}/disc_cop/run_experiment.py \\\n".format(REPO_PATH)
            if USE_GPU and NUM_GPUS > 0:
                device = run_i % NUM_GPUS
                sbatch_content += "  --device=cuda:{} \\\n".format(device)
            sbatch_content += "  --config_path={} \\\n".format(config_path)
            sbatch_content += "  --dataset_dir={}/datasets \\\n".format(DATASET_DIR)
            sbatch_content += "  --baseline_dir={} &\n".format(DATASET_DIR)

            run_i += 1

            if run_i % (NUM_RUNS_PER_GPU * NUM_GPUS) == 0:
                sbatch_content += "wait \n"

sbatch_content += "wait \n"
with open(
    os.path.join(f"./run_all.sh"),
    "w+",
) as f:
    f.writelines(sbatch_content)
