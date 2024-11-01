HOME_DIR = "/home/chanb"
LOG_DIR = f"{HOME_DIR}/scratch/disc_cop/results"
RUN_REPORT_DIR = f"{HOME_DIR}/scratch/disc_cop/run_reports"
DATASET_DIR = f"{HOME_DIR}/scratch/disc_cop/datasets"
REPO_PATH = f"{HOME_DIR}/src/Avg_OPE"
CC_ACCOUNT = "def-schuurma"

# HOME_DIR = "/Users/chanb/research/ualberta/Avg_OPE"
# LOG_DIR = "/Users/chanb/research/ualberta/Avg_OPE/local/results"
# RUN_REPORT_DIR = "/Users/chanb/research/ualberta/Avg_OPE/local/run_reports"
# DATASET_DIR = "/Users/chanb/research/ualberta/Avg_OPE/local/datasets"
# REPO_PATH = "/Users/chanb/research/ualberta/Avg_OPE"
# CC_ACCOUNT = "def-schuurma"

# CHOOSE BEST HYPERPARAM WITH CARTPOLE AND HOPPER
HYPERPARAMETERS = dict(
    mujoco=dict(
        random_weights=[2.0],
        discount_factors=[0.95],
        buffer_sizes=[4000],
        max_lens=[100],
        # =============================================
        # TODO: Figure best config below
        batch_sizes=[256, 512],
        links=["default"],
        bootstrap_targets=["target_network", "cross_q"],
        lrs=[0.0001, 0.0005, 0.001, 0.005],
        alphas=[0.0, 0.01, 0.1],  # L1
        tau=0.0005,  # target network
        # =============================================
        seeds=range(10),
        step_frequency=5,
    ),
    classic_control=dict(
        random_weights=[0.5],
        discount_factors=[0.95],
        buffer_sizes=[4000],
        max_lens=[100],
        # =============================================
        # TODO: Figure best config below
        batch_sizes=[256, 512],
        links=["default"],
        bootstrap_targets=["target_network", "cross_q"],
        lrs=[0.0001, 0.0005, 0.001, 0.005],
        alphas=[0.0, 0.01, 0.1],  # L1
        tau=0.0005,  # target network
        # =============================================
        seeds=range(10),
        step_frequency=5,
    )
)

# RUN ALL
# HYPERPARAMETERS = dict(
#     mujoco=dict(
#         random_weights=[1.4, 1.8, 2.0, 2.4, 2.8],
#         discount_factors=[0.8, 0.9, 0.95, 0.99 ,0.995],
#         buffer_sizes=[2000, 4000, 8000, 16000],
#         max_lens=[20,50,100,200,400],
#         batch_sizes=[256],
#         links=["default"],
#         bootstrap_targets=["target_network"],
#         lrs=[0.001],
#         alphas=[0.0],  # L1
#         tau=0.0005,  # target network
#         seeds=range(10),
#         step_frequency=5,
#     ),
#     classic_control=dict(
#         random_weights=[0.1, 0.2, 0.3, 0.4, 0.5],
#         discount_factors=[0.8, 0.9, 0.95, 0.99 ,0.995],
#         buffer_sizes=[2000, 4000, 8000, 16000],
#         max_lens=[20,50,100,200,400],
#         batch_sizes=[256],
#         links=["default"],
#         bootstrap_targets=["target_network"],
#         lrs=[0.005],
#         alphas=[0.0],  # L1
#         tau=0.0005,  # target network
#         seeds=range(10),
#         step_frequency=5,
#     ),
# )