# Install MuJoCo binary
mkdir ~/mujocodwn
mkdir ~/.mujoco
wget https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz -P ~/mujocodwn
tar -xvzf ~/mujocodwn/mujoco210-linux-x86_64.tar.gz -C ~/.mujoco/
rm -rf ~/mujocodwn

# Install Python packages
module load python/3.9
module load mujoco

python -m venv ~/avg_ope
source ~/avg_ope/bin/activate

python -m pip install -r requirements.txt