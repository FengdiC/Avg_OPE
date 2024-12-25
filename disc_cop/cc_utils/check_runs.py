"""
This script simply iterates through the saved models
"""

import inspect
import os
import sys

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from constants import LOG_DIR

from tqdm import tqdm
from pprint import pprint

dir_path = f"{LOG_DIR}/saved_models"
env_name = "hopper"

complete_dir_path = os.path.join(dir_path, env_name)
config_path = os.path.join(f"{LOG_DIR}", env_name)

incomplete_variants = []
for variant in tqdm(os.listdir(config_path)):
    variant = variant[9:-4]
    curr_variant_path = os.path.join(complete_dir_path, variant)
    if not os.path.isdir(curr_variant_path) or "seed_9-final.pt" not in os.listdir(curr_variant_path):
        incomplete_variants.append(variant)

pprint(incomplete_variants)
print(len(incomplete_variants))
