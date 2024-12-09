#!/bin/bash
#SBATCH --cpus-per-task=1  # Cores proportional to GPUs: 6 on Cedar, 16 on Graham.
#SBATCH --mem=3600M       # Memory proportional to GPUs: 32000 Cedar, 64000 Graham.
#SBATCH --time=0-144:00
#SBATCH --output=%N-%j.out
#SBATCH --account=def-ashique
#SBATCH --array=1-600

# salloc --cpus-per-task=1 --mem=3600M --time=0-3:00 --account=def-ashique
# Did not tune for three discount factors

source $HOME/ENV_1/bin/activate
module load StdEnv/2023
module load gcc gcccore/.12.3 opencv intel/2023.2.1 cuda/11.8 python/3.10 mpi4py
module load mujoco

SECONDS=0
echo

#python run/eval_policies.py --log_dir $SCRATCH/avg_corr/

#python avg_corr/run_cartpole_td.py --log_dir $SCRATCH/avg_corr/td_err/ --steps 5 --epoch 2000 --max_len 50

python run/run_classic.py --log_dir $SCRATCH/avg_corr/classic/ \
--array $SLURM_ARRAY_TASK_ID  --steps 5 --epoch 5000 --max_len 50 &

python run/run_mujoco.py --log_dir $SCRATCH/avg_corr/mujoco/ \
--array $SLURM_ARRAY_TASK_ID --steps 5 --epoch 100000 --max_len 100 &

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
#  pip install imageio
#  pip install gymnasium>=0.29