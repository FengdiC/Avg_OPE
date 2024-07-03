ENVS = {
    "mujoco": {
        "ant": ("Ant-v4", "./exper/ant.pth"),
        "halfcheetah": ("HalfCheetah-v4", "./exper/halfcheetah_1.pth"),
        "hopper": ("Hopper-v4", "./exper/hopper.pth"),
        "swimmer": ("Swimmer-v4", "./exper/swimmer.pth"),
        "walker": ("Walker2d-v4", "./exper/walker.pth"),
    },
    "classic_control": {
        "acrobot": ("Acrobot-v1", "./exper/acrobot.pth"),
        "cartpole": ("CartPole-v1", "./exper/cartpole.pth"),
        "mountain_car": ("MountainCarContinuous-v0", "./exper/mountaincar.pth"),
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
