# Download IBL dataset and begin processing it: identify unique animals in
# IBL dataset that enter biased blocks.  Save a dictionary with each animal
# and a list of their eids in the biased blocks

import numpy as np
import numpy.random as npr
import json
import pickle
from collections import defaultdict
import wget
from zipfile import ZipFile
import os
from pprint import pprint
npr.seed(65)

from one_global import one
print(one)

def path2eid(path):
    lab, _, subject, date, number = path.split('/')
    _eids, _info = one.search(laboratory=lab, date_range=[date,date], number=number, subject=subject, details=True)
    return _eids[0], _info[0]

if __name__ == '__main__':
    # parent directory where data are saved
    ibl_data_path = "../../data/ibl"

    # where to load eids from (generated previously)
    eids_dict_path = "../../data/ibl/partially_processed_old"

    # create directory for saving data:
    out_dir = f"{ibl_data_path}/partially_processed"
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # # TO CHANGE ACCORDINGLY LATER ON, FOR THE DATA WE NEED
    # # FOR NOW ASSUMING ACCESS TO DATA THROUGH ONE
    # if DOWNLOAD_DATA: # Warning: this step takes a while
    #     if not os.path.exists(ibl_data_path):
    #         os.makedirs(ibl_data_path)
    #     # download IBL data
    #     url = 'https://ndownloader.figshare.com/files/21623715'
    #     wget.download(url, ibl_data_path)
    #     # now unzip downloaded data:
    #     with ZipFile(ibl_data_path + "ibl-behavior-data-Dec2019.zip",
    #                  'r') as zipObj:
    #         # extract all the contents of zip file in ibl_data_path
    #         zipObj.extractall(ibl_data_path)

    # FOR DEBUG... SKIP THE SEARCH AND USE THE DICTS ALREADY SAVED
    try:
        with open(f"{out_dir}/eids_info.pkl", "rb") as f:
            eids, info = pickle.load(f)
        print("Loaded existing eids and info...")
    except:
        print("Failed to load eids and info. Loading Z. Ashwood list, converting and downloading...")
        with open(f"{eids_dict_path}/animal_eid_dict.json", "r") as f:
            animal_eid_dict = json.load(f)
        animal_list = np.load(f"{eids_dict_path}/animal_list.npz")

        labs = []
        animals = []
        dates = []
        eids = []
        info = []
        for _, paths in animal_eid_dict.items():
            for path in paths:
                eid, _info = path2eid(path)
                labs.append(_info['lab'])
                animals.append(_info['subject'])
                dates.append(_info['date'])
                eids.append(eid)
                info.append(_info)

        with open(f"{out_dir}/eids_info.pkl", "wb") as f:
            pickle.dump([eids,info], f)

    assert len(eids) > 0, "ONE search is in incorrect directory"
    animal_list = []
    animal_eid_dict = defaultdict(list)

    eid_info_dict = {}
    for i, eid in enumerate(eids):

        print("\n================================================")
        print(f"{i +1} of {len(eids)}")
        print("------------------------------------------------")
        print(eid)
        print("------------------------")
        pprint(info[i])
        try:
            trial_data = one.load_object(eid, 'trials', collection="alf")
            # spike_data = one.load_object(eid, 'spikes', collection="alf/probe00/pykilosort",
            #                                 attribute=["times","depths","amps"])
            bias_probs = trial_data.probabilityLeft
        except:
            # skip eid if there's some issue with `probabilityLeft`
            continue

        comparison = np.sort(np.unique(bias_probs)) == np.array([0.2, 0.5, 0.8])
        print(f'We{" DO NOT " if not np.all(comparison) else " "}have all bias probabilities')
        if not np.all(comparison):
            failed_eids.append(eid)
        # sessions with bias blocks
        if isinstance(comparison, np.ndarray):
            # update def of comparison to single True/False
            comparison = comparison.all()
        if comparison == True:
            animal = info[i]['subject']
            if animal not in animal_list:
                animal_list.append(animal)
            animal_eid_dict[animal].append(eid)

            eid_info_dict[eid] = info[i]

    # save
    _json = json.dumps(animal_eid_dict)
    with open(f"{out_dir}/animal_eid_dict.json", "w") as f:
        f.write(_json)
    
    with open(f"{out_dir}/eid_info_dict.pkl", "wb") as f:
        pickle.dump(eid_info_dict, f)
    
    np.savez(f'{out_dir}/animal_list.npz', animal_list)
