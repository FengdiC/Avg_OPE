#!/bin/bash
#SBATCH --cpus-per-task=1  # Cores proportional to GPUs: 6 on Cedar, 16 on Graham.
#SBATCH --mem=7200M       # Memory proportional to GPUs: 32000 Cedar, 64000 Graham.
#SBATCH --time=0-24:00
#SBATCH --output=%N-%j.out
#SBATCH --account=rrg-ashique
#SBATCH --gpus-per-node=1
#SBATCH --array=0-125

# salloc --cpus-per-task=1 --mem=3600M --time=0-3:00 --account=def-ashique --gpus-per-node=1
# Did not tune for three discount factors

module load StdEnv/2023
module load gcc gcccore/.12.3 opencv intel/2023.2.1 cuda/11.8 python/3.10 mpi4py
module load mujoco
source $HOME/ENV_1/bin/activate

SECONDS=0
echo

#python run/eval_policies.py --log_dir $SCRATCH/avg_corr/

#python avg_corr/run_cartpole_td.py --log_dir $SCRATCH/avg_corr/td_err/ --steps 5 --epoch 2000 --max_len 50

for seed in  $(seq 0 9); do
  start_time=$SECONDS
  python dice_rl/scripts/run_neural_dice_classic.py --output_dir $SCRATCH/avg_corr/dice_gpu/classic/ \
  --array $SLURM_ARRAY_TASK_ID --steps 5 --epoch 5000 --max_trajectory_length 100 \
  --data_dir $SCRATCH/avg_corr/ --seed $seed

   wait

  elapsed_time=$((SECONDS - start_time))
  echo "Baseline job $seed took $elapsed_time seconds"
done

for seed in  $(seq 0 9); do
  start_time=$SECONDS
  python dice_rl/scripts/run_neural_dice.py --output_dir $SCRATCH/avg_corr/dice_gpu/mujoco/ \
  --array $SLURM_ARRAY_TASK_ID  --steps 5 --epoch 5000 --max_trajectory_length 100 \
  --data_dir $SCRATCH/avg_corr/ --seed $seed
   wait

  elapsed_time=$((SECONDS - start_time))
  echo "Baseline job $seed took $elapsed_time seconds"
done

echo "Baseline job took $SECONDS"
