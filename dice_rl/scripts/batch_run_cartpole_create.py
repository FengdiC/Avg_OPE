import os
import subprocess

# Define hyperparameters to sweep over
alpha_values = [0.7]
seed_values = [0, 1, 2, 3, 4]
gamma_values = [0.99]  # Example gamma values
num_trajectory_values = [200]  # Example num_trajectory values

save_dir_prefix = './results06241706/cartpole/'

for gamma in gamma_values:
    for alpha in alpha_values:
        for num_trajectory in num_trajectory_values:
            for seed in seed_values:
                # Command for the first loop
                for alpha_val in [alpha, 1.0]:
                    command1 = (
                        f"python scripts/create_dataset.py "
                        f"--save_dir={save_dir} "
                        f"--load_dir=./exper/cartpole.pth "
                        f"--env_name=CartPole-v1 "
                        f"--num_trajectory={num_trajectory} "
                        f"--max_trajectory_length=50 "
                        f"--alpha={alpha_val} "
                        f"--tabular_obs=0 "
                        f"--force "
                        f"--seed={seed}"
                    )
                    subprocess.run(command1, shell=True)

                # Command for the second loop
                for alpha_val in [alpha, 1.0]:
                    seed_plus_100 = seed + 100
                    command2 = (
                        f"python scripts/create_dataset.py "
                        f"--save_dir={save_dir} "
                        f"--load_dir=./exper/cartpole.pth "
                        f"--env_name=CartPole-v1 "
                        f"--num_trajectory={num_trajectory} "
                        f"--max_trajectory_length=50 "
                        f"--alpha={alpha_val} "
                        f"--tabular_obs=0 "
                        f"--force "
                        f"--seed={seed_plus_100}"
                    )
                    subprocess.run(command2, shell=True)
