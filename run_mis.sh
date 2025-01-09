#!/bin/bash
#SBATCH --cpus-per-task=1  # Cores proportional to GPUs: 6 on Cedar, 16 on Graham.
#SBATCH --mem=3600M       # Memory proportional to GPUs: 32000 Cedar, 64000 Graham.
#SBATCH --time=0-144:00
#SBATCH --output=%N-%j.out
#SBATCH --account=def-ashique
#SBATCH --array=0-125

# salloc --gpus-per-node=1  --cpus-per-task=1 --mem=3600M --time=0-3:00 --account=def-ashique
# Did not tune for three discount factors

module load StdEnv/2023
module load gcc gcccore/.12.3 opencv intel/2023.2.1 cuda/11.8 python/3.10 mpi4py
module load mujoco
source $HOME/ENV_1/bin/activate

SECONDS=0
echo

#python run/eval_policies.py --log_dir $SCRATCH/avg_corr/

#python avg_corr/run_cartpole_td.py --log_dir $SCRATCH/avg_corr/td_err/ --steps 5 --epoch 2000 --max_len 50

for seed in  $(seq 1 10); do
  echo "Baseline job $seed took $SECONDS"
  python SR-DICE/run_classic.py --log_dir $SCRATCH/avg_corr/MIS/ --policy "SR_DICE"\
  --array $SLURM_ARRAY_TASK_ID --steps 5 --epoch 5000  --data_dir $SCRATCH/avg_corr/ --seed $seed
done

for seed in  $(seq 1 10); do
  echo "Baseline job $seed took $SECONDS"
  python SR-DICE/run_mujoco.py --log_dir $SCRATCH/avg_corr/MIS/ --policy "SR_DICE"\
  --array $SLURM_ARRAY_TASK_ID  --steps 5 --epoch 40000 --data_dir $SCRATCH/avg_corr/ --seed $seed
done

echo "Baseline job $seed took $SECONDS"
sleep 144h


