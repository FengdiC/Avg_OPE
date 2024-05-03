#!/bin/bash
#SBATCH --cpus-per-task=1  # Cores proportional to GPUs: 6 on Cedar, 16 on Graham.
#SBATCH --mem=3600M       # Memory proportional to GPUs: 32000 Cedar, 64000 Graham.
#SBATCH --time=0-12:00
#SBATCH --output=%N-%j.out
#SBATCH --account=def-ashique
#SBATCH --array=1-36

# salloc --cpus-per-task=1 --mem=3600M --time=0-3:00 --account=def-ashique

source $HOME/ENV_1/bin/activate
module load StdEnv/2023
module load gcc opencv intel/2023.2.1 cuda/11.8 python/3.10 mpi4py

SECONDS=0
echo
for RANDOM_WEIGHT in 0.3 0.5 0.7
do
  for BATCH_SIZE in 256 512
  do
    for LINK in 'inverse' 'log'
    do
      for BUFFER in 40 80 200
      do
        python avg_corr/gamma.py --path './exper/cartpole.pth' --env 'CartPole-v1' \
        --log_dir $SCRATCH/avg_gamma/cartpole/ --batch_size $BATCH_SIZE \
        --link $LINK --random_weight $RANDOM_WEIGHT \
        --steps 5 --epoch 500 --buffer_size $BUFFER --max_len 50 --seed 32 &
      done
    done
  done
done

echo "Baseline job $seed took $SECONDS"
sleep 72h