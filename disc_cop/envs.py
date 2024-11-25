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
            "MountainCarContinuous-v0",
            "{}/exper/mountaincar.pth".format(REPO_PATH),
        ),
    },
}

# ENVS = {
#     "mujoco": {
#         "hopper": ("Hopper-v4", "{}/exper/hopper.pth".format(REPO_PATH)),
#     },
#     "classic_control": {
#         "cartpole": ("CartPole-v1", "{}/exper/cartpole.pth".format(REPO_PATH)),
#     },
# }

ENV_ID_TO_NAME = {
    "Ant-v4": "ant",
    "HalfCheetah-v4": "halfcheetah",
    "Hopper-v4": "hopper",
    "Swimmer-v4": "swimmer",
    "Walker2d-v4": "walker",
    "Acrobot-v1": "acrobot",
    "CartPole-v1": "cartpole",
    "MountainCarContinuous-v0": "mountain_car",
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

# For tuning
# ENV_TO_FAMILY = {
#     "hopper": "mujoco",
#     "cartpole": "classic_control",
# }

ENV_FAMILY_SPECIFICS = {
    "mujoco": {
        "max_len": 100,
        "train_steps": 250_000,
        "run_time": "02:55:00",
    },
    "classic_control": {
        "max_len": 50,
        "train_steps": 10_000,
        "run_time": "02:00:00",
    },
}
