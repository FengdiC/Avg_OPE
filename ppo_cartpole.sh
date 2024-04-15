#!/bin/bash
#SBATCH --cpus-per-task=1  # Cores proportional to GPUs: 6 on Cedar, 16 on Graham.
#SBATCH --mem=1200M       # Memory proportional to GPUs: 32000 Cedar, 64000 Graham.
#SBATCH --time=0-3:00
#SBATCH --output=%N-%j.out
#SBATCH --account=def-ashique
#SBATCH --array=1-300

source $HOME/ENV_1/bin/activate
module load StdEnv/2023
module load gcc opencv intel/2023.2.1 cuda/11.8 python/3.10 mpi4py

SECONDS=0
#python ppo/ppo_tune.py --seed $SLURM_ARRAY_TASK_ID --log_dir $SCRATCH/avg_discount/acrobot --env 'Acrobot-v1' --epochs 100 --save_freq 15&
file="$SCRATCH/avg_discount/cartpole${SLURM_ARRAY_TASK_ID}"
echo file
python ppo/ppo_tune.py --seed $SLURM_ARRAY_TASK_ID --log_dir $file --env 'CartPole-v1' --epochs 100 --save_freq 100&

echo "Baseline job $seed took $SECONDS"
sleep 72h