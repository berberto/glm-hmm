# Create panels a-c of Figure 3 of Ashwood et al. (2020)
import json
import os
import sys
import pickle

sys.path.append('../')

from plotting_utils import load_glmhmm_data, load_cv_arr, load_data, \
    get_file_name_for_best_model_fold, partition_data_by_session, \
    create_violation_mask, get_marginal_posterior, get_was_correct, \
    get_prob_right

from regression import Regression
from sklearn.model_selection import train_test_split, KFold

def get_session_ids (eids):
    '''
    get a list of eids, and return the corresponding list
    of session_id in the format used by this code to 
    identify sessions (saved in '<animal>_processed.npz' files)
    '''
    session_ids = []
    for eid in eids:
        assert isinstance(eid, str), "'eid' must be in string format!"
        info = one.get_details(eid)
        session_id = f"{info['subject']}-{info['date']}-{info['number']:03d}"
        session_ids.append(session_id)
    return session_ids

from scipy.cluster.hierarchy import dendrogram
from sklearn.cluster import AgglomerativeClustering
def cluster_neurons (spike_counts):
    model = AgglomerativeClustering(distance_threshold=0, n_clusters=None, linkage="ward")
    model = model.fit(spike_counts)
    idx = np.argsort(model.labels_)
    return np.take(spike_counts, idx, axis=0)


import matplotlib.pyplot as plt
import numpy as np
cols = ['#ff7f00', '#4daf4a', '#377eb8', '#f781bf', '#a65628', '#984ea3',
        '#999999', '#e41a1c', '#dede00']

if __name__ == '__main__':

    from torch.nn import functional as F
    loss=F.cross_entropy
    # loss=F.mse_loss

    with open("../../data/ibl/partially_processed/neural_dict.pkl", "rb") as f:
        neural_dict = pickle.load(f)

    # manually select all the animals where "all" the
    # latent states are explored with "high" probability
    good_animals = [
        "DY_010",
        "DY_011",
        "ibl_witten_13",
        "ibl_witten_26",
        "ibl_witten_27",
        "ibl_witten_29",
        "KS014",
        "KS016",
        "KS023",
        "KS044",
        "NYU-39",
        "NYU-46",
        "SWC_043",
        "SWC_054",
        "UCLA012",
        "ZFM-01937",
        "ZFM-02369",
    ]

    from pprint import pprint
    from one_global import one
    # for animal in neural_dict.keys():
    for animal in good_animals:
        print(animal)
        data_dir = '../../data/ibl/data_for_cluster/data_by_animal'
        results_dir = f'../../results/ibl_individual_fit/{animal}'
        neural_dir = f'../../data/ibl/partially_processed/spike_counts'
        # figure_dir = '../../neural_hmm_fit/figures'
        figure_dir = f'../../neural_hmm_fit/{animal}'
        models_dir = f'{figure_dir}/{loss.__name__}'
        if not os.path.exists(figure_dir):
            os.makedirs(figure_dir)
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)

        # get the eids for this animal
        eids = neural_dict[animal]
        # convert them to the <subject>-<date>-<number> format
        session_ids = get_session_ids(eids)

        K = 3

        cv_file = f"{results_dir}/cvbt_folds_model.npz"
        cvbt_folds_model = load_cv_arr(cv_file)

        with open(f"{results_dir}/best_init_cvbt_dict.json", 'r') as f:
            best_init_cvbt_dict = json.load(f)

        # Get the file name corresponding to the best initialization for given K
        # value
        raw_file = get_file_name_for_best_model_fold(cvbt_folds_model, K,
                                                     results_dir,
                                                     best_init_cvbt_dict)
        hmm_params, lls = load_glmhmm_data(raw_file)

        # Save parameters for initializing individual fits
        weight_vectors = hmm_params[2]
        log_transition_matrix = hmm_params[1][0]
        init_state_dist = hmm_params[0][0]

        # get processed behaviour data for animal
        inpt, y, session = load_data(f"{data_dir}/{animal}_processed.npz")

        all_sessions = np.unique(session)

        # Create mask:
        # Identify violations for exclusion:
        violation_idx = np.where(y == -1)[0]
        nonviolation_idx, mask = create_violation_mask(violation_idx,
                                                       inpt.shape[0])
        y[np.where(y == -1), :] = 1
        inputs, datas, train_masks = partition_data_by_session(
            np.hstack((inpt, np.ones((len(inpt), 1)))), y, mask,
            session)

        posterior_probs = get_marginal_posterior(inputs, datas, train_masks,
                                                 hmm_params, K, range(K))

        fig = plt.figure(figsize=(6, 4.5))
        fig.suptitle(animal)
        plt.subplots_adjust(wspace=0.5, hspace=0.5)

        #
        #   GLM-HMM parameters for this animal
        #
        # transition probabilities
        plt.subplot(2, 3, 1)
        transition_matrix = np.exp(log_transition_matrix)
        plt.imshow(transition_matrix, vmin=-0.8, vmax=1, cmap='bone')
        for i in range(transition_matrix.shape[0]):
            for j in range(transition_matrix.shape[1]):
                text = plt.text(j,
                                i,
                                str(np.around(transition_matrix[i, j],
                                              decimals=2)),
                                ha="center",
                                va="center",
                                color="k",
                                fontsize=10)
        plt.xlim(-0.5, K - 0.5)
        plt.xticks(range(0, K),
                   ('1', '2', '3', '4', '4', '5', '6', '7', '8', '9', '10')[:K],
                   fontsize=10)
        plt.yticks(range(0, K),
                   ('1', '2', '3', '4', '4', '5', '6', '7', '8', '9', '10')[:K],
                   fontsize=10)
        plt.ylim(K - 0.5, -0.5)
        plt.ylabel("state t-1", fontsize=10)
        plt.xlabel("state t", fontsize=10)

        # GLM parameters
        weight_vectors = -hmm_params[2]

        plt.subplot2grid(shape=(3,2), loc=(0,1), colspan=2)
        M = weight_vectors.shape[2] - 1
        for k in range(K):
            plt.plot(range(M + 1),
                     weight_vectors[k][0][[0, 3, 1, 2]],
                     marker='o',
                     label="state " + str(k + 1),
                     color=cols[k],
                     lw=1,
                     alpha=0.7)
        plt.yticks([-2.5, 0, 2.5, 5], fontsize=10)
        plt.xticks(
            [0, 1, 2, 3],
            ['stimulus', 'bias', 'prev. \nchoice', 'win-stay-\nlose-switch'],
            fontsize=10,
            rotation=45)
        plt.ylabel("GLM weight", fontsize=10)
        plt.axhline(y=0, color="k", alpha=0.5, ls="--", lw=0.5)
        plt.gca().spines['right'].set_visible(False)
        plt.gca().spines['top'].set_visible(False)

        for k in range(K):
            plt.subplot(2, 3, k+4)
            # USE GLM WEIGHTS TO GET PROB RIGHT
            stim_vals, prob_right_max = get_prob_right(weight_vectors, inpt, k, 1,
                                                       1)
            _, prob_right_min = get_prob_right(weight_vectors, inpt, k, -1, -1)
            plt.plot(stim_vals,prob_right_max, '-',
                     color=cols[k], alpha=1, lw=1,
                     zorder=5)  # went R and was rewarded on previous trial
            plt.plot(stim_vals,get_prob_right(weight_vectors, inpt, k, -1, 1)[1], '--',
                     color=cols[k], alpha=0.5, lw=1)  # went L and was not rewarded on previous trial
            plt.plot(stim_vals,get_prob_right(weight_vectors, inpt, k, 1, -1)[1], '-',
                     color=cols[k], alpha=0.5, lw=1,
                     markersize=3)  # went R and was not rewarded on previous trial
            plt.plot(stim_vals, prob_right_min, '--', color=cols[k], alpha=1,
                     lw=1)  # went L and was rewarded on previous trial
            plt.xticks([min(stim_vals), 0, max(stim_vals)],
                       labels=['', '', ''],
                       fontsize=10)
            plt.yticks([0, 0.5, 1], ['', '', ''], fontsize=10)
            plt.ylabel('')
            plt.xlabel('')
            if k == 0:
                plt.title("state 1", # \n(\"engaged\")", 
                          fontsize=10, color=cols[k])
                plt.xticks([min(stim_vals), 0, max(stim_vals)],
                           labels=['-100', '0', '100'],
                           fontsize=10)
                plt.yticks([0, 0.5, 1], ['0', '0.5', '1'], fontsize=10)
                plt.ylabel('p("R")', fontsize=10)
                plt.xlabel('stimulus', fontsize=10)
            if k == 1:
                plt.title("state 2", # \n(\"biased left\")",
                          fontsize=10, color=cols[k])
                plt.xticks([min(stim_vals), 0, max(stim_vals)],
                           labels=['', '', ''],
                           fontsize=10)
                plt.yticks([0, 0.5, 1], ['', '', ''], fontsize=10)
            if k == 2:
                plt.title("state 3", # \n(\"biased right\")",
                          fontsize=10, color=cols[k])
                plt.xticks([min(stim_vals), 0, max(stim_vals)],
                           labels=['', '', ''],
                           fontsize=10)
                plt.yticks([0, 0.5, 1], ['', '', ''], fontsize=10)
            plt.axhline(y=0.5, color="k", alpha=0.45, ls=":", linewidth=0.5)
            plt.axvline(x=0, color="k", alpha=0.45, ls=":", linewidth=0.5)
            plt.gca().spines['right'].set_visible(False)
            plt.gca().spines['top'].set_visible(False)
            plt.ylim((-0.01, 1.01))
        fig.savefig(f"{figure_dir}/glm-hmm.svg")
        plt.close(fig)

        for i, (eid, sess) in enumerate(zip(eids, session_ids)):

            # a plot for each session
            fig = plt.figure(figsize=(8, 4))
            fig.suptitle(sess)
            plt.subplots_adjust(wspace=0.3, hspace=0.5)

            #
            #   Training Logistic regression and plot performance
            #
            n_folds = 5
            n_epochs = 100
            session_dir = f"{models_dir}/{eid}"
            if not os.path.exists(session_dir):
                os.makedirs(session_dir)

            idx_session = np.where(session == sess)[0]
            spike_counts = np.load(f"{neural_dir}/{eid}.npy")[:90].T # n_neurons x n_trials

            X_ = spike_counts.T
            y_ = posterior_probs[idx_session, :]
            posterior_probs_this_session_predicted = np.zeros_like(y_)
            
            # shuffle trials within session
            idx_shuffle = np.arange(len(X_))
            np.random.shuffle(idx_shuffle)
            X_ = X_[idx_shuffle]
            y_ = y_[idx_shuffle]

            y_mean = np.mean(y_,axis=0)

            train_losses = []
            test_losses = []

            plt.subplot(2, 3, 2)
            plt.xlabel("True")
            plt.ylabel("Predicted")
            av_score = 0
            # train and test with 5-fold cross validation
            for fold, (train_idx, test_idx) in enumerate(KFold(n_splits=n_folds).split(X_)):
                print("\nfold ", fold)
                model = Regression(X_[0], y_[0], models_dir=session_dir, loss=loss)
                X_train, X_test = X_[train_idx], X_[test_idx]
                y_train, y_test = y_[train_idx], y_[test_idx]
                train_loss, test_loss = model.train(X_train, y_train,
                            X_test=X_test, y_test=y_test, n_epochs=n_epochs, monitor=True)
                train_losses.append(train_loss)
                test_losses.append(test_loss)
                model.save(model_name=f"weights_{fold}")
                
                y_pred = model.predict(X_test)
                posterior_probs_this_session_predicted[test_idx] = y_pred
                for p_true, p_pred, color in zip(y_test.T, y_pred.T, cols):
                    plt.scatter(p_true, p_pred, s=2, color=color, alpha=.5)

                _min = min(y_test.min(),y_pred.min())
                _max = max(y_test.max(),y_pred.max())

                av_score += model.score(X_test, y_test, y_mean=y_mean)/n_folds

            for m, color in zip(y_mean, cols):
                plt.axhline(m, c=color, ls='--')
            plt.plot([_min,_max],[_min,_max], c='k', ls='--')

            plt.title(f"R2 score = {av_score:.4f}")
            plt.gca().spines['right'].set_visible(False)
            plt.gca().spines['top'].set_visible(False)

            train_losses = np.array(train_losses)
            test_losses = np.array(test_losses)
            np.save(f"{session_dir}/losses_train.npy", train_losses)
            np.save(f"{session_dir}/losses_test.npy", test_losses)

            # plot training curves
            ax = plt.subplot(2, 3, 3)
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Loss")
            train_mid = np.median(train_losses, axis=0)
            test_mid = np.median(test_losses, axis=0)
            train_low = np.min(train_losses, axis=0) # np.percentile(train_losses, 10, axis=0)
            train_high = np.max(train_losses, axis=0) # np.percentile(train_losses, 90, axis=0)
            test_low = np.min(test_losses, axis=0) # np.percentile(test_losses, 10, axis=0)
            test_high = np.max(test_losses, axis=0) # np.percentile(test_losses, 90, axis=0)

            ax.fill_between(np.arange(n_epochs), train_low, train_high, color='C0', alpha=0.2)
            ax.fill_between(np.arange(n_epochs), test_low, test_high, color='C1', alpha=0.2)

            ax.plot(train_mid, color='C0', lw=2, label="train loss")
            ax.plot(test_mid, color='C1', lw=2, label="test loss")
            # ax.set_title(f"{sess}\nR2 score = {av_score:.4f}")
            ax.legend()


            states_max_posterior = np.argmax(posterior_probs, axis=1)

            sess_to_plot = session_ids[:3]

            #
            #   plot spike count matrices
            #
            plt.subplot(2, 3, 1)
            eid = eids[session_ids.index(sess)]
            spike_counts = np.load(f"{neural_dir}/{eid}.npy")[:90].T # n_neurons x n_trials
            spike_counts = cluster_neurons(spike_counts)
            plt.imshow(spike_counts, origin='lower', aspect='auto',cmap='binary')
            plt.xticks([0, 45, 90], ["", "", ""], fontsize=10)
            ticks_pos = np.arange(0, len(spike_counts), 50)
            plt.yticks(ticks_pos, len(ticks_pos)*[""], fontsize=10)
            plt.gca().spines['right'].set_visible(False)
            plt.gca().spines['top'].set_visible(False)
            plt.ylabel("spike count\n200 ms before stim", fontsize=10)
            plt.xticks([0, 45, 90], ["0", "45", "90"], fontsize=10)

            #
            #   plot probabilities of hidden states over trials
            #
            plt.subplot(2, 3, 4)
            idx_session = np.where(session == sess)[0]
            this_inpt = inpt[idx_session, :]
            posterior_probs_this_session = posterior_probs[idx_session, :]
            # Plot trial structure for this session too:
            for k in range(K):
                plt.plot(posterior_probs_this_session[:, k],
                         label="State " + str(k + 1), lw=1,
                         color=cols[k])
            states_this_sess = states_max_posterior[idx_session]
            state_change_locs = np.where(np.abs(np.diff(states_this_sess)) > 0)[0]
            for change_loc in state_change_locs:
                plt.axvline(x=change_loc, color='k', lw=0.5, linestyle='--')
            plt.xticks([0, 45, 90], ["0", "45", "90"], fontsize=10)
            plt.yticks([0, 0.5, 1], ["0", "", "1"], fontsize=10)
            plt.ylim((-0.01, 1.01))
            plt.gca().spines['right'].set_visible(False)
            plt.gca().spines['top'].set_visible(False)
            plt.ylabel("p(state)", fontsize=10)
            plt.xlabel("trial #", fontsize=10)
            
            plt.subplot(2, 3, 5)
            idx_session = np.where(session == sess)[0]
            this_inpt = inpt[idx_session, :]
            posterior_probs_this_session = posterior_probs[idx_session, :]
            # Plot trial structure for this session too:
            for k in range(K):
                plt.plot(posterior_probs_this_session_predicted[:, k], 'o',
                         label="State " + str(k + 1), markersize=1.,
                         color=cols[k])
                plt.plot(posterior_probs_this_session_predicted[:, k],
                         label="State " + str(k + 1),  lw=.3,
                         color=cols[k], alpha=0.3)
            states_this_sess = states_max_posterior[idx_session]
            state_change_locs = np.where(np.abs(np.diff(states_this_sess)) > 0)[0]
            for change_loc in state_change_locs:
                plt.axvline(x=change_loc, color='k', lw=0.5, linestyle='--')
            plt.xticks([0, 45, 90], ["0", "45", "90"], fontsize=10)
            plt.yticks([0, 0.5, 1], ["", "", ""], fontsize=10)
            plt.ylim((-0.01, 1.01))
            plt.gca().spines['right'].set_visible(False)
            plt.gca().spines['top'].set_visible(False)
            plt.ylabel("p(state)", fontsize=10)

            #
            #   plot choices
            #
            plt.subplot(2, 3, 6)
            idx_session = np.where(session == sess)[0]
            this_inpt, this_y = inpt[idx_session, :], y[idx_session, :]
            was_correct, idx_easy = get_was_correct(this_inpt, this_y)
            this_y = this_y[:, 0] + np.random.normal(0, 0.03, len(this_y[:, 0]))
            # plot choice, color by correct/incorrect:
            locs_correct = np.where(was_correct == 1)[0]
            locs_incorrect = np.where(was_correct == 0)[0]
            plt.plot(locs_correct, this_y[locs_correct], 'o', color='black',
                     markersize=2, zorder=3, alpha=0.5)
            plt.plot(locs_incorrect, this_y[locs_incorrect], 'v', color='red',
                     markersize=2, zorder=4, alpha=0.5)

            states_this_sess = states_max_posterior[idx_session]
            state_change_locs = np.where(np.abs(np.diff(states_this_sess)) > 0)[0]
            for change_loc in state_change_locs:
                plt.axvline(x=change_loc, color='k', lw=0.5, linestyle='--')
            plt.ylim((-0.13, 1.13))
            plt.xticks([0, 45, 90], ["0", "45", "90"], fontsize=10)
            plt.yticks([0, 1], ["L", "R"], fontsize=10)
            plt.gca().spines['right'].set_visible(False)
            plt.gca().spines['top'].set_visible(False)
            plt.ylabel("choice", fontsize=10)


            fig.savefig(f"{session_dir}/session.svg")

            plt.close(fig)

        # exit()