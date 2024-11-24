import os

# Define hyperparameters to sweep over
alpha_values = [(1-i) for i in [0.1,0.2, 0.3,0.4,0.5]]
# seed_values = [100, 200]
seed_values = [5,6,7,8,9 ]
gamma_values = [0.8, 0.9, 0.95, 0.99 ,0.995]  # Example gamma values
num_trajectory_values = [2000,4000,8000,16000]  # Example num_trajectory values
max_len_values = [20,50,100,200,400]
max_num_bash = 5
num_bash = 0

save_dir_prefix = './results10021944/cartpole/'
bash_file_prefix = './bash10021944/cartpole/run_results10021944'
bash_file_index = 0

# Create the folder if it does not exist
os.makedirs(os.path.dirname(bash_file_prefix), exist_ok=True)

for gamma in gamma_values:
    for alpha in alpha_values:
        for num_trajectory in num_trajectory_values:
            for seed in seed_values:
                for max_trajectory_length in max_len_values:
                    # Command for the first loop
                    new_num_trajectory = int(num_trajectory / max_trajectory_length)
                    for alpha_val in [alpha,]:
                        command1 = (
                            f"python scripts/create_dataset.py "
                            f"--save_dir={save_dir_prefix} "
                            f"--load_dir=./exper/cartpole.pth "
                            f"--env_name=CartPole-v1 "
                            f"--num_trajectory={new_num_trajectory} "
                            f"--max_trajectory_length={max_trajectory_length} "
                            f"--alpha={alpha_val} "
                            f"--tabular_obs=0 "
                            f"--force "
                            f"--seed={seed} "
                            f"--gamma={gamma}"
                        )
                        bash_file_name = f"{bash_file_prefix}_{bash_file_index}.sh"
                        file_exists = os.path.isfile(bash_file_name)
                        with open(bash_file_name, 'a' if file_exists else 'w') as bash_file:
                            if not file_exists:
                                bash_file.write("#!/bin/bash\n")
                                os.chmod(bash_file_name, 0o755)
                            bash_file.write(command1 + "\n")
                        num_bash += 1
                        if num_bash == max_num_bash:
                            bash_file_index += 1
                            num_bash = 0

                    # Command for the second loop
                    for alpha_val in [alpha,]:
                        seed_plus_100 = seed + 100
                        command2 = (
                            f"python scripts/create_dataset.py "
                            f"--save_dir={save_dir_prefix} "
                            f"--load_dir=./exper/cartpole.pth "
                            f"--env_name=CartPole-v1 "
                            f"--num_trajectory={new_num_trajectory} "
                            f"--max_trajectory_length={max_trajectory_length} "
                            f"--alpha={alpha_val} "
                            f"--tabular_obs=0 "
                            f"--force "
                            f"--seed={seed_plus_100} "
                            f"--gamma={gamma}"
                        )
                        bash_file_name = f"{bash_file_prefix}_{bash_file_index}.sh"
                        file_exists = os.path.isfile(bash_file_name)
                        with open(bash_file_name, 'a' if file_exists else 'w') as bash_file:
                            if not file_exists:
                                bash_file.write("#!/bin/bash\n")
                                os.chmod(bash_file_name, 0o755)
                            bash_file.write(command2 + "\n")
                        num_bash += 1
                        if num_bash == max_num_bash:
                            bash_file_index += 1
                            num_bash = 0

print(f"Generated {bash_file_index} bash scripts.")
