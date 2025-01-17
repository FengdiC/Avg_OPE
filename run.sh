#!/bin/bash
#SBATCH --cpus-per-task=1  # Cores proportional to GPUs: 6 on Cedar, 16 on Graham.
#SBATCH --mem=3600M       # Memory proportional to GPUs: 32000 Cedar, 64000 Graham.
#SBATCH --time=0-12:00
#SBATCH --output=%N-%j.out
#SBATCH --account=def-ashique
#SBATCH --array=0-125
#SBATCH --gpus-per-node=1

# salloc --cpus-per-task=1 --mem=3600M --time=0-3:00 --account=def-ashique --gpus-per-node=1
# Did not tune for three discount factors

source $HOME/ENV_1/bin/activate
module load StdEnv/2023
module load gcc gcccore/.12.3 opencv intel/2023.2.1 cuda/11.8 python/3.10 mpi4py
module load mujoco

# narval activate first; beluga, cedar activate last

SECONDS=0
echo

for seed in  $(seq 0 9); do
  start_time=$SECONDS
  python run/run_classic.py --log_dir $SCRATCH/avg_corr/avg_mse/ \
  --array $SLURM_ARRAY_TASK_ID  --steps 5 --epoch 5000 --data_dir $SCRATCH/avg_corr/ --seed $seed

  wait

  elapsed_time=$((SECONDS - start_time))
  echo "Baseline job $seed took $elapsed_time seconds"
done

for seed in  $(seq 0 9); do
  start_time=$SECONDS
  python run/run_mujoco.py --log_dir $SCRATCH/avg_corr/avg_mse/ \
  --array $SLURM_ARRAY_TASK_ID --steps 5 --epoch 40000 --data_dir $SCRATCH/avg_corr/ --seed $seed

  wait

  # Calculate elapsed time
  elapsed_time=$((SECONDS - start_time))
  echo "Baseline job $seed took $elapsed_time seconds"
done


echo "Baseline job $seed took $SECONDS"
sleep 144h

# python avg_corr/run_classic.py --log_dir $SCRATCH/avg_corr/classic/ --array 10  --steps 5 --epoch 2000 --max_len 50
# pip install torch torchvision numpy
# python -m pip install -U pip
# python -m pip install -U matplotlib
#  pip install pandas scipy gym==0.25.1
#  wget https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz
#  tar -xvzf mujoco210-linux-x86_64.tar.gz
#  mkdir .mujoco
#  mv mujoco210 ./.mujoco/mujoco210
#  pip install -U 'mujoco-py<2.2,>=2.1'
#  vi .bashrc
#  export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/fengdic/.mujoco/mujoco210/bin
#  source .bashrc
#  pip install "Cython<3"
#  python
#  import mujoco_py
#  module load mujoco
#  pip install imageio tqdm
#  pip install gymnasium>=0.29