HOME_DIR = "/home/chanb"
LOG_DIR = f"{HOME_DIR}/scratch/results/disc_cop"
RUN_REPORT_DIR = f"{HOME_DIR}/scratch/run_reports/disc_cop"
DATASET_DIR = f"{HOME_DIR}/scratch/datasets/disc_cop"
REPO_PATH = f"{HOME_DIR}/src/Avg_OPE"
CC_ACCOUNT = "def-schuurma"

# HOME_DIR = "/Users/chanb/research/ualberta/Avg_OPE"
# LOG_DIR = "/Users/chanb/research/ualberta/Avg_OPE/local/results"
# RUN_REPORT_DIR = "/Users/chanb/research/ualberta/Avg_OPE/local/run_reports"
# DATASET_DIR = "/Users/chanb/research/ualberta/Avg_OPE/local/datasets"
# REPO_PATH = "/Users/chanb/research/ualberta/Avg_OPE"
# CC_ACCOUNT = "def-schuurma"

# Run 1
# HYPERPARAMETERS = dict(
#     random_weights=[0.3, 0.5, 0.7],
#     discount_factors=[0.8, 0.99, 0.995],
#     batch_sizes=[256, 512],
#     links=["default"],
#     buffer_sizes=[40 * 50, 80 * 50, 200 * 50],
#     bootstrap_targets=["target_network", "cross_q"],
#     lrs=[0.0001, 0.0005, 0.001, 0.005],
#     alphas=[0.0],  # L1
#     tau=0.0005,  # target network
#     seeds=range(10),
#     step_frequency=5,
#     max_lens=[50],
# )

# Hyperparam sweep for Run 2
# HYPERPARAMETERS = dict(
#     random_weights=[0.7],
#     discount_factors=[0.95],
#     buffer_sizes=[2000],
#     max_lens=[50],
#     # =============================================
#     # TODO: Figure best config below
#     batch_sizes=[256, 512],
#     links=["default"],
#     bootstrap_targets=["target_network", "cross_q"],
#     lrs=[0.0001, 0.0005, 0.001, 0.005],
#     alphas=[0.0],  # L1
#     tau=0.0005,  # target network
#     # =============================================
#     seeds=range(10),
#     step_frequency=5,
# )

"""
Hopper: mse-tune-random_weight_0.7-discount_factor_0.95-buffer_size_40-link_default-batch_size_256-bootstrap_target_target_network-lr_0.001-alpha_0.0-max_len_50
Cartpole: mse-tune-random_weight_0.7-discount_factor_0.95-buffer_size_40-link_default-batch_size_256-bootstrap_target_target_network-lr_0.005-alpha_0.0-max_len_50
"""

# Run 2
HYPERPARAMETERS = dict(
    mujoco=dict(
        random_weights=[0.3, 0.4, 0.5, 0.6, 0.7],
        discount_factors=[0.8, 0.9, 0.95, 0.99 ,0.995],
        buffer_sizes=[2000, 4000, 8000, 16000, 32000],
        max_lens=[20, 50, 100, 200, 400],
        batch_sizes=[256],
        links=["default"],
        bootstrap_targets=["target_network"],
        lrs=[0.001],
        alphas=[0.0],  # L1
        tau=0.0005,  # target network
        seeds=range(10),
        step_frequency=5,
    ),
    classic_control=dict(
        random_weights=[0.3, 0.4, 0.5, 0.6, 0.7],
        discount_factors=[0.8, 0.9, 0.95, 0.99 ,0.995],
        buffer_sizes=[2000, 4000, 8000, 16000, 32000],
        max_lens=[20, 50, 100, 200, 400],
        batch_sizes=[256],
        links=["default"],
        bootstrap_targets=["target_network"],
        lrs=[0.005],
        alphas=[0.0],  # L1
        tau=0.0005,  # target network
        seeds=range(10),
        step_frequency=5,
    ),
)