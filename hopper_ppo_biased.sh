#!/bin/bash
#SBATCH --cpus-per-task=1  # Cores proportional to GPUs: 6 on Cedar, 16 on Graham.
#SBATCH --mem=3600M       # Memory proportional to GPUs: 32000 Cedar, 64000 Graham.
#SBATCH --time=0-72:00
#SBATCH --output=%N-%j.out
#SBATCH --account=def-ashique

source $HOME/ENV_1/bin/activate
module load StdEnv/2023
module load gcc opencv intel/2023.2.1 cuda/11.8 python/3.10 mpi4py

SECONDS=0
#python ppo/ppo_tune.py --seed 32 --log_dir $SCRATCH/avg_discount/acrobot --env 'Acrobot-v1' --epochs 100 --save_freq 15&
#python ppo/ppo_tune.py --seed 32 --log_dir $SCRATCH/avg_discount/cartpole --env 'CartPole-v1' --epochs 100 --save_freq 15&
#python ppo/ppo_tune.py --seed 165 --log_dir $SCRATCH/avg_discount/pendulum --env 'Pendulum-v1' --epochs 100 --save_freq 15&
#python ppo/ppo_tune.py --seed 196 --log_dir $SCRATCH/avg_discount/mountaincar --env 'MountainCarContinuous-v0' --epochs 100 --save_freq 15&

python ppo/ppo_tune.py --seed 280 --log_dir $SCRATCH/avg_discount/hopper --env 'Hopper-v4' --epochs 250 --save_freq 48&
python ppo/ppo_tune.py --seed 394 --log_dir $SCRATCH/avg_discount/swimmer --env 'Swimmer-v4' --epochs 250 --save_freq 48&
python ppo/ppo_tune.py --seed 16 --log_dir $SCRATCH/avg_discount/ant --env 'Ant-v4' --epochs 250 --save_freq 48&
python ppo/ppo_tune.py --seed 39 --log_dir $SCRATCH/avg_discount/walker --env 'Walker2d-v4' --epochs 250 --save_freq 48&
python ppo/ppo_tune.py --seed 280 --log_dir $SCRATCH/avg_discount/halfcheetah --env 'HalfCheetah-v4' --epochs 250 --save_freq 48&

echo "Baseline job $seed took $SECONDS"
sleep 72h