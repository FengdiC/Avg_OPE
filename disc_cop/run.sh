#!/bin/bash

SECONDS=0
echo

for filename in ./log/cartpole/*.pkl; do
   python3 run_experiment.py --config_path $filename --dataset_dir '../dataset/'
   wait
done

for filename in ./log/acrobot/*.pkl; do
   python3 run_experiment.py --config_path $filename --dataset_dir '../dataset/'
   wait
done

echo "Baseline job $seed took $SECONDS"
wait