# Instructions

## New installation
```
module load StdEnv/2023
module load gcc gcccore/.12.3 opencv intel/2023.2.1 cuda/11.8 python/3.10 mpi4py
module load mujoco

# ========================================================================
# First time setup to install mujoco binary
wget https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz
tar -xvzf mujoco210-linux-x86_64.tar.gz
mkdir .mujoco
mv mujoco210 ./.mujoco/mujoco210
pip install -U 'mujoco-py<2.2,>=2.1'
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/.mujoco/mujoco210/bin

python -m venv ~/avg_ope
source ~/avg_ope/bin/activate
pip install "Cython<3"
pip install imageio
pip install gymnasium>=0.29
```

## Installation
```
# ========================================================================
# First time setup to install mujoco binary
mkdir ~/mujocodwn
mkdir ~/.mujoco
wget https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz -P ~/mujocodwn
tar -xvzf ~/mujocodwn/mujoco210-linux-x86_64.tar.gz -C ~/.mujoco/
rm -rf ~/mujocodwn
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/.mujoco/mujoco210/bin:/usr/lib/nvidia
# ========================================================================

# ========================================================================
# If Compute Canada:
module load python/3.10
module load StdEnv/2020
module load mujoco/2.2.2
# ========================================================================

python -m venv ~/avg_ope
source ~/avg_ope/bin/activate

# ========================================================================
# If not Compute Canada:
pip install mujoco matplotlib numpy torch
pip install mujoco==2.3.3
#Otherwise
pip install torch --no-index
# ========================================================================

pip install tqdm
pip install gym
pip install mujoco-py

# ========================================================================
# Might need to do this when gym.make fails on MuJoCo
pip install "cython<3"
# ========================================================================
```

## Running Experiments
NOTE: Please modify `HOME_DIR` in `constants.py` to use the desired location for storing generated content.

### Example
In this example, we assume that `HOME_DIR = "/home/fakeuser"`.
```
# Generate offline dataset
python generate_datasets.py --env_name=ant --dataset_dir=./datasets

# Get on-policy policy evaluation
python run_baseline.py --env_name=ant --total_trajs=500

# Generate all hyperparameter configurations
python generate_hyperparameters.py

# Run disc COP
python run_experiment.py --config_path=/home/fakeuser/scratch/results/disc_cop/ant/mse-tune-random_weight_0.3-discount_factor_0.8-buffer_size_40-link_default-batch_size_256-bootstrap_target_cross_q-lr_0.001-alpha_0.0.pkl --dataset_dir=./datasets
```

### Generate Datasets
To generate offline datasets of an environment for ratio training, run
```
python generate_datasets.py --env_name=<ENV_NAME> --dataset_dir=<PATH/TO/DATASET/DIRECTORY>
```

`env_name` supports:
```
ant
halfcheetah
hopper
swimmer
walker
acrobot
cartpole
mountain_car
```

### Generate Configurations
Run `generate_hyperparameters.py` to generate all the configurations for sweeping.
By default, this should generate 432 hyperparameter configurations for each environment.

### Running an Experiment
To create baseline (i.e. on-policy policy evaluation), run
```
python run_baseline.py --env_name=<ENV_NAME_AS_ABOVE> --total_trajs=500
```

To run discounted COP, run
```
python run_experiment.py --config_path=<PATH/TO/CONFIG.pkl> --dataset_dir=<PATH/TO/DATASET/DIRECTORY>
```

- `config_path` is the path to a configuration generated similarly in `generate_hyperparameters.py`.
- `dataset_dir` is the path to a directory generated similarly in `generate_datasets.py`.

### Slurm
We have slurm scripts `generate_slurm-*.py` that generates bash scripts to execute on Compute Canada.
Please make sure to change `CC_ACCOUNT` in `constants.py` to use your Compute Canada account.
Note that `generate_slurm-experiments.py` requires `generate_hyperparameters.py` to be run beforehand.

Running all scripts should produce the following bash scripts:
```
run_all-acrobot.sh
run_all-ant.sh
run_all-cartpole.sh
run_all-generate_datasets.sh
run_all-halfcheetah.sh
run_all-hopper.sh
run_all-mountain_car.sh
run_all-run_baselines.sh
run_all-swimmer.sh
run_all-walker.sh
```

You may simply do `sbatch run_all-<FILE>.sh` to kick off the run.


## Experimental Status
Baseline tuning:
```
Cartpole
random_weight = 0.3
traj_len = 100
num_traj = 40
discount_factor = 0.95

Hopper
random_weight = 2.0
traj_len = 100
num_traj = 40
discount_factor = 0.95
```

Per-step metric for setting:
```
random_weight = 0.5
num_traj = 40
traj_len = 100
discount_factor = 0.95
```