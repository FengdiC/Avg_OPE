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
env_name = "mountain_car"

complete_dir_path = os.path.join(dir_path, env_name)

incomplete_variants = []
for variant in tqdm(os.listdir(complete_dir_path)):
    if "seed_9-final.pt" not in os.listdir(os.path.join(complete_dir_path, variant)):
        incomplete_variants.append(variant)

pprint(incomplete_variants)
print(len(incomplete_variants))
