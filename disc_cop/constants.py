LOG_DIR = "/home/chanb/scratch/results/disc_cop"
RUN_REPORT_DIR = "/home/chanb/scratch/run_reports/disc_cop"
DATASET_DIR = "/home/chanb/scratch/datasets/disc_cop"
REPO_PATH = "/home/chanb/scratch/Avg_OPE/disc_cop"

HYPERPARAMETERS = dict(
    random_weights=[0.3, 0.5, 0.7],
    discount_factors=[0.8, 0.99, 0.995],
    batch_sizes=[256, 512],
    links=["default"],
    buffer_sizes=[40, 80, 200],
    bootstrap_targets=["target_network", "cross_q"],
    lrs=[0.0001, 0.0005, 0.001, 0.005],
    alphas=[0.0],  # L1
    tau=0.0005,  # target network
    seeds=range(10),
    step_frequency=5,
)
