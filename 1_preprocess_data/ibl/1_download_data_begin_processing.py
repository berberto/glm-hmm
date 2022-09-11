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

if __name__ == '__main__':
    # parent directory where data are saved
    ibl_data_path = "../../data/ibl"

    # where to load eids from (generated previously)
    eids_dict_path = "../../data/ibl"

    # create directory for saving data:
    out_dir = f"{ibl_data_path}/partially_processed"
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
        
    # FOR DEBUG... SKIP THE SEARCH AND USE THE DICTS ALREADY SAVED
    try:
        with open(f"{eids_dict_path}/eids_info.pkl", "rb") as f:
            eids, info = pickle.load(f)
        print("Loaded existing eids and info...")
        print("These were converted from Ashwood et al. data, compatibly with one.api.ONE")
    except:
        raise Exception("Failed to load `eids` and `info`. You need to search yourself, mate.")
        # If this exception was thrown at you, you should write the search code below...
        # Something like...
        # >>> eids, info = one.search(dataset=["trials"],
        # >>>                         date_range=["2019-01-01","2019-12-31"], 
        # >>>                         datails=True)

    assert len(eids) > 0, "ONE search is in incorrect directory"
    animal_list = []
    animal_eid_dict = defaultdict(list)

    try:
        # load the eids and info for data in Ashwood et al that also have neural data
        with open(f"{out_dir}/eids_info_neural.pkl", "rb") as f:
            eids_neural, info_neural = pickle.load(f)
    except FileNotFoundError as e:
        # if can't find it, search them
        print(e)
        eids_neural = one.search(dataset=['trials', 'spikes'])
        idx = [eids.index(_eid) for _eid in eids_neural]
        info_neural = [info[i] for i in idx]
        # save info on file
        with open(f"{out_dir}/eids_info_neural.pkl", "wb") as f:
            pickle.dump([eids_neural, info_neural], f)

    paths_neural = []
    eid_info_dict = {}
    for i, eid in enumerate(eids_neural):

        print("\n====================================")
        print(f"{i +1} of {len(eids_neural)}")
        print("------------------------------------")
        print(eid)
        print("------------------------------------")
        pprint(info_neural[i])

        # load the behavioural data
        try:
            trial_data = one.load_object(eid, 'trials', collection="alf")
            # ?) only this is needed?
            bias_probs = trial_data.probabilityLeft
        except:
            # skip eid if there's some issue with `probabilityLeft`
            continue

        # if the behavioural data have no issue, then try and load the `spikes` data
        if eid in eids_neural:
            try:
                # first find the name of the probe to search in `alf` directory
                probe_insertions = one.load_dataset(eid, 'probes.description')
                probe_label = probe_insertions[0]['label']
                print(f"probe: {probe_label}")
                
                # then try to load the object -- care about `times`, `amps` and `depths` only
                # ?) wouldn't `times` and `depths` be enough?
                spikes = one.load_object(eid, 'spikes', collection=f"alf/{probe_label}",
                                    attribute=['times','amps','depths'])
                print("Spike data loaded")
                _path = one.eid2path(eid)
                paths_neural.append(str(_path))

                # # WHILE TESTING
                # # raster plot of spikes
                # from brainbox.plot import driftmap
                # import matplotlib.pyplot as plt
                # fig, ax = plt.subplots()
                # driftmap(spikes.times, spikes.depths, ax=ax, t_bin=0.1, d_bin=5);
                # fig.savefig(f"{_path}/raster.png", bbox_inches="tight")
                # plt.show()
                # plt.close(fig)

            except Exception as e:

                # if some error occurred in loading the data
                idx = eids_neural.index(eid)
                eids_neural.pop(idx)
                info_neural.pop(idx)

                print("Error loading spike data: ", e)
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

    with open(f"{out_dir}/paths_neural.txt", "w") as f:
        for _path in paths_neural:
            f.write(f"{_path}\n")

    with open(f"{out_dir}/eids_info_neural.pkl", "wb") as f:
        pickle.dump([eids_neural, info_neural], f)