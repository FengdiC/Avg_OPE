# Beluga
HOME_DIR = "/home/chanb"
LOG_DIR = f"{HOME_DIR}/scratch/disc_cop/results"
RUN_REPORT_DIR = f"{HOME_DIR}/scratch/disc_cop/run_reports"
DATASET_DIR = f"{HOME_DIR}/scratch/disc_cop/datasets"
REPO_PATH = f"{HOME_DIR}/src/Avg_OPE"
CC_ACCOUNT = "def-schuurma"
USE_SLURM = True
USE_GPU = True
RUN_ALL = False

# HOME_DIR = "/Users/chanb/research/ualberta/Avg_OPE"
# LOG_DIR = "/Users/chanb/research/ualberta/Avg_OPE/local/results"
# RUN_REPORT_DIR = "/Users/chanb/research/ualberta/Avg_OPE/local/run_reports"
# DATASET_DIR = "/Users/chanb/research/ualberta/Avg_OPE/local/datasets"
# REPO_PATH = "/Users/chanb/research/ualberta/Avg_OPE"
# CC_ACCOUNT = "def-schuurma"
# USE_SLURM = False

MAX_BUFFER_SIZE = 16000
if RUN_ALL:
    # RUN ALL
    HYPERPARAMETERS = dict(
        mujoco=dict(
            random_weights=[1.4, 1.8, 2.0, 2.4, 2.8],
            discount_factors=[0.8, 0.9, 0.95, 0.99 ,0.995],
            buffer_sizes=[2000, 4000, 8000, 16000],
            max_lens=[20, 50, 100, 200],
            batch_sizes=[512],
            links=["default"],
            bootstrap_targets=["target_network"],
            lrs=[0.005],
            alphas=[0.0],  # L1
            tau=0.0005,  # target network
            seeds=range(10),
            step_frequency=5,
        ),
        classic_control=dict(
            random_weights=[0.1, 0.2, 0.3, 0.4, 0.5],
            discount_factors=[0.8, 0.9, 0.95, 0.99 ,0.995],
            buffer_sizes=[2000, 4000, 8000, 16000],
            max_lens=[20, 40, 80, 100],
            batch_sizes=[512],
            links=["default"],
            bootstrap_targets=["target_network"],
            lrs=[0.005],
            alphas=[0.01],  # L1
            tau=0.0005,  # target network
            seeds=range(10),
            step_frequency=5,
        ),
        # gamma, buffer_size, random_weight, max_len
        manual_settings=[
            [0.8, 4000, 0.3, 40,],
            [0.9, 4000, 0.3, 40,],
            [0.95, 4000, 0.3, 40,],
            [0.99, 4000, 0.3, 40,],
            [0.995, 4000, 0.3, 40,],
            [0.95, 2000, 0.3, 40,],
            [0.95, 4000, 0.3, 40,],
            [0.95, 8000, 0.3, 40,],
            [0.95, 16000, 0.3, 40,],
            [0.95, 4000, 0.1, 40,],
            [0.95, 4000, 0.2, 40,],
            [0.95, 4000, 0.3, 40,],
            [0.95, 4000, 0.4, 40,],
            [0.95, 4000, 0.5, 40,],
            [0.95, 4000, 0.3, 20,],
            [0.95, 4000, 0.3, 40,],
            [0.95, 4000, 0.3, 80,],
            [0.95, 4000 ,0.3 ,100,],
        ]
    )
else:
    # CHOOSE BEST HYPERPARAM WITH CARTPOLE AND HOPPER
    HYPERPARAMETERS = dict(
        mujoco=dict(
            random_weights=[2.0],
            discount_factors=[0.95],
            buffer_sizes=[4000],
            max_lens=[100],
            # =============================================
            # TODO: Figure best config below
            batch_sizes=[512],
            links=["default"],
            bootstrap_targets=["target_network", "cross_q"],
            lrs=[0.0001, 0.0005, 0.001, 0.005],
            alphas=[0.0, 0.01, 0.1],  # L1
            tau=0.0005,  # target network
            # =============================================
            seeds=range(5),
            step_frequency=5,
        ),
        classic_control=dict(
            random_weights=[0.3],
            discount_factors=[0.95],
            buffer_sizes=[4000],
            max_lens=[100],
            # =============================================
            # TODO: Figure best config below
            batch_sizes=[512],
            links=["default"],
            bootstrap_targets=["target_network", "cross_q"],
            lrs=[0.0001, 0.0005, 0.001, 0.005],
            alphas=[0.0, 0.01, 0.1],  # L1
            tau=0.0005,  # target network
            # =============================================
            seeds=range(5),
            step_frequency=5,
        )
    )

"""
discount_factor_lists = [0.8, 0.9, 0.95, 0.99, 0.995]
size_lists = [2000, 4000, 8000, 16000]

weight_lists = [0.1, 0.2, 0.3, 0.4, 0.5]
length_lists = [20, 40, 80, 100]
env = ['CartPole-v1', 'Acrobot-v1']
path = ['./exper/cartpole.pth', './exper/acrobot.pth']
random_weight, length, discount_factor, size = (
    0.3,
    40,
    0.95,
    4000,
)
idx = np.unravel_index(args.array, (18, 2))
if idx[0] < 5:
    discount_factor = discount_factor_lists[idx[0]]
elif idx[0] < 9:
    size = size_lists[idx[0] - 5]
elif idx[0] < 14:
    random_weight = weight_lists[idx[0] - 9]
else:
    length = length_lists[idx[0] - 14]
env = env[idx[1]]
"""