#!/bin/bash
#SBATCH --cpus-per-task=1  # Cores proportional to GPUs: 6 on Cedar, 16 on Graham.
#SBATCH --mem=3600M       # Memory proportional to GPUs: 32000 Cedar, 64000 Graham.
#SBATCH --time=0-72:00
#SBATCH --output=%N-%j.out
#SBATCH --account=def-ashique
#SBATCH --gpus-per-node=1
#SBATCH --array=0-100

# salloc --cpus-per-task=1 --mem=3600M --time=0-3:00 --account=def-ashique

source $HOME/ENV_1/bin/activate
module load StdEnv/2023
module load gcc gcccore/.12.3 opencv intel/2023.2.1 cuda/11.8 python/3.10 mpi4py
module load mujoco

SECONDS=0
echo

python run/tune.py --array $SLURM_ARRAY_TASK_ID --data_dir $SCRATCH/avg_corr/ \
--log_dir $SCRATCH/avg_mse/tune/ --steps 5 --epoch 10000 --max_len 100 --seed 280 &

echo "Baseline job $seed took $SECONDS"
sleep 72h