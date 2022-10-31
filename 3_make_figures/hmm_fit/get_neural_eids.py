import json
import os
import sys
import pickle
import numpy as np
from one_global import one
from pprint import pprint

animal_list = list(np.load("../../data/ibl/partially_processed/animal_list.npz")["arr_0"])
ind_fit_dir = '../../results/ibl_individual_fit'

animals_avail = []
animals_ = os.listdir(ind_fit_dir)
with open("../../data/ibl/animals_avail.txt", "w") as file:
    for a in animals_:
        if a in animal_list:
            animals_avail.append(a)
            file.write(f"{a}\n")

animal_eid_dict = dict([(animal, []) for animal in animals_avail])

neural_dir = "../../data/ibl/partially_processed/spike_counts"
eids = [file.split('.')[0] for file in os.listdir(neural_dir)]
for eid in eids:
    info = one.get_details(eid)
    try:
        animal_eid_dict[info['subject']].append(eid)
    except:
        continue

neural_dict = animal_eid_dict.copy()
for animal, eids in animal_eid_dict.items():
    if len(eids) == 0:
        neural_dict.pop(animal)
with open("../../data/ibl/partially_processed/neural_dict.pkl", "wb") as f:
    pickle.dump(neural_dict, f)

print("\n====== good animals and eids ======")
for animal, eids in neural_dict.items():
    print(f"------ {animal}")
    for eid in eids:
        print(f"\t{eid}")