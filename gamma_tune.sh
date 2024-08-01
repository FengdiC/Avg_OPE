#!/bin/bash
#SBATCH --cpus-per-task=1  # Cores proportional to GPUs: 6 on Cedar, 16 on Graham.
#SBATCH --mem=3600M       # Memory proportional to GPUs: 32000 Cedar, 64000 Graham.
#SBATCH --time=0-3:00
#SBATCH --output=%N-%j.out
#SBATCH --account=def-ashique
#SBATCH --array=1-96

# salloc --cpus-per-task=1 --mem=3600M --time=0-3:00 --account=def-ashique

source $HOME/ENV_1/bin/activate
module load StdEnv/2023
module load gcc opencv intel/2023.2.1 cuda/11.8 python/3.10 mpi4py

SECONDS=0
echo
python avg_corr/gamma.py --path './exper/cartpole.pth' --env 'CartPole-v1' --array $SLURM_ARRAY_TASK_ID \
--log_dir $SCRATCH/avg_gamma/cartpole/ --steps 5 --epoch 2000 --max_len 50 --seed 32 &

echo "Baseline job $seed took $SECONDS"
sleep 72h