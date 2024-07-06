import os

from tqdm import tqdm
from pprint import pprint

dir_path = "/home/chanb/scratch/results/disc_cop/saved_models"
env_name = "halfcheetah"

complete_dir_path = os.path.join(dir_path, env_name)

incomplete_variants = []
for variant in tqdm(os.listdir(complete_dir_path)):
    if "seed_9-final.pt" not in os.listdir(os.path.join(complete_dir_path, variant)):
        incomplete_variants.append(variant)

pprint(incomplete_variants)
print(len(incomplete_variants))
