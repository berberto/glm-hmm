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
        # print("These were converted from Ashwood et al. data, compatibly with one.api.ONE")
        print("These were found in a spreadsheet called ME")
    except:
        # raise Exception("Failed to load `eids` and `info`. You need to search yourself, mate.")
        # If this exception was thrown at you, you should write the search code below...
        # Something like...
        # >>> eids, info = one.search(dataset=["trials"],
        # >>>                         date_range=["2019-01-01","2019-12-31"], 
        # >>>                         datails=True)
        print("Failed to load `eids` and `info`.") 
        print("Search all eids for subjects with neural data...")
        # # 1) find all the eids with neural data
        # eids, info = one.search(dataset=['spikes'], details=True)
        # 1b) take eids from ME spreadsheet
        with open(f"{eids_dict_path}/eids_info_ME.pkl", "rb") as f:
            eids, info = pickle.load(f)
        print(f"Found {len(eids)} eids with 'spikes' data")
        # 2) for those, collect the name of the subjects
        subjects = []
        for _eid, _dict in zip(eids,info):
            subjects.append(_dict['subject'])
        subjects = list(set(subjects)) # unique subjects
        print(f"Found {len(subjects)} subjects with 'spikes' data")
        # 3) search all the eids for that subject, and discard subject if less than 30
        eids = []
        info = []
        for subject in subjects:
            _eids, _info = one.search(subject=[subject], details=True)
            print(f"{subject} -- {len(_eids)}")
            if len(_eids) < 30:
                continue
            eids += _eids
            info += _info
        # 4) save search
        with open(f"{eids_dict_path}/eids_info_AP.pkl", "wb") as f:
            pickle.dump([eids, info], f)

    print(f"Found a total of {len(eids)} eids")

    assert len(eids) > 0, "ONE search is in incorrect directory"
    animal_list = []
    animal_eid_dict = defaultdict(list)

    try:
        with open(f"{out_dir}/eids_info_neural.pkl", "rb") as f:
            eids_neural, info_neural = pickle.load(f)
        print("Loaded neural eids and info from that ME spreadsheet.")
    except FileNotFoundError as e:
        import pandas as pd
        df = pd.read_csv(f"{ibl_data_path}/ME.csv")
        print(df)

        eids_neural = list(df['eid'].to_numpy())     

        info_neural = []
        for i, eid in enumerate(eids_neural):
            print(i, eid)
            _d = one.get_details(eid)
            info_neural.append(_d)

        # pprint(info_neural)
        # # if can't find it, search them
        # print(e)
        # print("Among the eids found, search those with neural data")
        # eids_neural = []
        # info_neural = []
        # for _eid, _dict in zip(eids, info):
        #     try:
        #         probe_insertions = one.load_dataset(_eid, 'probes.description')
        #         eids_neural.append(_eid)
        #         info_neural.append(_dict)
        #     except:
        #         pass

        # save info on file
        with open(f"{out_dir}/eids_info_neural.pkl", "wb") as f:
            pickle.dump([eids_neural, info_neural], f)

    print(f"Found {len(eids_neural)} eids with neural data")

    paths_neural = []
    probes = []
    eid_info_dict = {}
    for i, eid in enumerate(eids):

        print("\n====================================")
        print(f"{i +1} of {len(eids)}")
        print("------------------------------------")
        print(eid)
        print("------------------------------------")
        pprint(info[i])

        # load the behavioural data
        try:
            trial_data = one.load_object(eid, 'trials', collection="alf")
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
                spikes = one.load_object(eid, 'spikes', collection=f"alf/{probe_label}",
                                    attribute=['times','amps','depths','clusters'])
                clusters = one.load_object(eid, 'clusters', collection=f"alf/{probe_label}")
                print("Spike data loaded")
                _path = one.eid2path(eid)
                paths_neural.append(str(_path))
                probes.append(probe_label)

                # # WHILE TESTING
                # # raster plot of spikes
                # from brainbox.plot import driftmap
                # import matplotlib.pyplot as plt
                # # fig, ax = plt.subplots()
                # # driftmap(spikes.times, spikes.depths, ax=ax, t_bin=0.1, d_bin=5);
                # # fig.savefig(f"{_path}/raster_depths.png", bbox_inches="tight")
                # # os.system(f"cp {_path}/raster_depths.png raster/{eid}_depths.png")
                # # # plt.show()
                # # plt.close(fig)
                # fig, ax = plt.subplots()
                # mask = np.where((spikes.times > 100) & (spikes.times < 101))
                # driftmap(spikes.times[mask], spikes.clusters[mask], ax=ax, t_bin=0.001, d_bin=1);
                # # fig.savefig(f"{_path}/raster_clusters.png", bbox_inches="tight")
                # # os.system(f"cp {_path}/raster_clusters.png raster/{eid}_clusters.png")
                # plt.show()
                # plt.close(fig)
                # exit()
                
                del spikes

            except Exception as e:

                # if some error occurred in loading the data
                idx = eids_neural.index(eid)
                eids_neural.pop(idx)
                info_neural.pop(idx)

                print("Error loading neural data: ", e)
                continue

        comparison = np.sort(np.unique(bias_probs)) == np.array([0.2, 0.5, 0.8])
        print(f'We{" DO NOT " if not np.all(comparison) else " "}have all bias probabilities')
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