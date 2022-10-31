import numpy as np
from brainbox.processing import bincount2D
from scipy.ndimage import uniform_filter

# AUTOMATIC RELEVANCE

# REFRACTORY PERIOD

def select_spikes_by (spike_vals, vals):
    '''
    returns array of integers with the IDs of the spikes that
    are assigned to specific values (eg cluster ID)
    '''
    spike_ids = np.array([],dtype=int)
    for val in vals:
        spike_ids = np.hstack((spike_ids, np.where(spike_vals == val)[0]))
    spike_ids = np.sort(np.unique(spike_ids)) # (?) perhaps unnecessary
    return spike_ids

def filter_clusters (spike_times, spike_clusters, T_BIN=0.02, min_fr=5, max_fr=100):
    # build 2D histogram of spiking times vs clusters (putative neurons)
    R, _times, _clusters = bincount2D(spike_times, spike_clusters, xbin=T_BIN)

    # rolling average of the firing rate (over window w) -- clusters are not averaged
    roll_afr = uniform_filter(R.astype(float), size=[1,int(1./T_BIN)], mode='wrap')/T_BIN # rate = counts/bin-width

    # c_ids = np.where((np.max(roll_afr, axis=1) < max_fr) & (np.max(roll_afr, axis=1) > min_fr))[0]
    c_ids = np.where(np.max(roll_afr, axis=1) < max_fr)[0]

    return np.take(_clusters, c_ids)


# before this, one should do quality control (eg measure drift, exclude high/low firing rates, ...)
def spike_counts_before_stim (stimOn_times, spike_times, spike_clusters, clusters=None, w=0.2):
    '''
    stimOn_times, ndarray (N_trials,): time [s] of stimulus onset for each trial
    spike_times, ndarray (N_spikes,): times [s] of spikes in the whole session
    spike_clusters, ndarray (N_spikes,): cluster ids of spike in the whole session
    clusters, ndarray (M,): indices of clusters to be selected
    w, float: time [s] window before stimulus onset where spikes are counted
    '''
    if clusters is None:
        clusters = np.arange( np.max(spike_clusters) + 1 )

    counts = np.zeros((len(stimOn_times), len(clusters)))
    for trial, on_time in enumerate(stimOn_times):
        # get the ids of the spikes within [on - w, on]
        spike_ids = np.where((spike_times < on_time) & (spike_times > on_time - w))
        # count the number of spikes for every cluster (number of times cluster occurring in `spike_clusters`)
        counts[trial] = np.take( np.bincount( spike_clusters[spike_ids], minlength=np.max(clusters) + 1 ), clusters )
    return counts


if __name__ == "__main__":

    import os
    import pickle
    from one_global import one
    from pprint import pprint

    print(one)

    out_dir = f"../../data/ibl/partially_processed"
    with open(f"{out_dir}/eids_info_neural.pkl", "rb") as f:
        eids_neural, info_neural = pickle.load(f)

    neural_dir = f"{out_dir}/spike_counts"
    os.makedirs(neural_dir, exist_ok=True)

    for i, eid in enumerate(eids_neural):
        print("\n====================================")
        print(f"{i +1} of {len(eids_neural)}")
        print("------------------------------------")
        print(eid)
        print("------------------------------------")
        pprint(info_neural[i])

        probe_insertions = one.load_dataset(eid, 'probes.description')

        print(f"# insertions = {len(probe_insertions)}")

        probe_label = probe_insertions[0]['label']
        spikes = one.load_object(eid, 'spikes', collection=f"alf/{probe_label}",
                            attribute=['times','clusters'])
        trials = one.load_object(eid, 'trials')
        stimOn_times = trials.stimOn_times
        probabilityLeft = trials.probabilityLeft
        spike_times = spikes.times
        spike_clusters = spikes.clusters

        # return a table (N_trials, N_clusters) with the spike count in a window
        # before stimulus onset for all trials
        spike_counts = spike_counts_before_stim(stimOn_times, spike_times, spike_clusters)

        # trial_ids = np.where(probabilityLeft == 0.5)[0]

        # # filter out cluster by average firing rates
        # cluster_ids = filter_clusters(spike_times, spike_clusters)

        # counts = np.take(spike_counts, trial_ids, axis=0)
        # counts = np.take(counts, cluster_ids, axis=1)

        np.save(f"{neural_dir}/{eid}.npy", spike_counts)
