import inspect
import os
import sys

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from disc_cop.constants import REPO_PATH

ENVS = {
    "mujoco": {
        "ant": ("Ant-v4", "{}/exper/ant.pth".format(REPO_PATH)),
        "halfcheetah": (
            "HalfCheetah-v4",
            "{}/exper/halfcheetah_1.pth".format(REPO_PATH),
        ),
        "hopper": ("Hopper-v4", "{}/exper/hopper.pth".format(REPO_PATH)),
        "swimmer": ("Swimmer-v4", "{}/exper/swimmer.pth".format(REPO_PATH)),
        "walker": ("Walker2d-v4", "{}/exper/walker.pth".format(REPO_PATH)),
    },
    "classic_control": {
        "acrobot": ("Acrobot-v1", "{}/exper/acrobot.pth".format(REPO_PATH)),
        "cartpole": ("CartPole-v1", "{}/exper/cartpole.pth".format(REPO_PATH)),
        "mountain_car": (
            "MountainCar-v0",
            "{}/exper/mountaincar.pth".format(REPO_PATH),
        ),
    },
}

ENV_TO_FAMILY = {
    "ant": "mujoco",
    "halfcheetah": "mujoco",
    "hopper": "mujoco",
    "swimmer": "mujoco",
    "walker": "mujoco",
    "acrobot": "classic_control",
    "cartpole": "classic_control",
    "mountain_car": "classic_control",
}

ENV_FAMILY_SPECIFICS = {
    "mujoco": {
        "max_len": 100,
        "train_steps": 250_000,
        "run_time": "05:00:00",
    },
    "classic_control": {
        "max_len": 50,
        "train_steps": 10_000,
        "run_time": "02:00:00",
    },
}
