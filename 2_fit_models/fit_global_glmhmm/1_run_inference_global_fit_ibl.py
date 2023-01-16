import sys
import os
import autograd.numpy as np
from glm_hmm_utils import load_cluster_arr, load_session_fold_lookup, \
    load_data, create_violation_mask, launch_glm_hmm_job

D = 1  # data (observations) dimension
C = 2  # number of output types/categories
N_em_iters = 300  # number of EM iterations
N_initializations = 20

if __name__ == '__main__':
    data_dir = '../../data/ibl/data_for_cluster'
    results_dir = '../../results/ibl_global_fit'

    num_folds = 5
    global_fit = True
    # perform mle => set transition_alpha to 1
    transition_alpha = 1
    prior_sigma = 100

    try:
        K, fold, iter = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
    except:
        raise Exception(f"Parameters from command line: `K`, `fold`, `iter`")


    def run (K, fold, iter):
        #  read in data and train/test split
        animal_file = f"{data_dir}/all_animals_concat.npz"
        session_fold_lookup_table = load_session_fold_lookup(
            f"{data_dir}/all_animals_concat_session_fold_lookup.npz")

        inpt, y, session = load_data(animal_file)
        #  append a column of ones to inpt to represent the bias covariate:
        inpt = np.hstack((inpt, np.ones((len(inpt),1))))
        y = y.astype('int')
        # Identify violations for exclusion:
        violation_idx = np.where(y == -1)[0]
        nonviolation_idx, mask = create_violation_mask(violation_idx,
                                                       inpt.shape[0])

        #  GLM weights to use to initialize GLM-HMM
        init_param_file = f"{results_dir}/GLM/fold_{str(fold)}/variables_of_interest_iter_0.npz"

        # create save directory for this initialization/fold combination:
        save_directory = f"{results_dir}/GLM_HMM_K_{str(K)}/fold_{str(fold)}/iter_{str(iter)}"
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)

        launch_glm_hmm_job(inpt,
                           y,
                           session,
                           mask,
                           session_fold_lookup_table,
                           K,
                           D,
                           C,
                           N_em_iters,
                           transition_alpha,
                           prior_sigma,
                           fold,
                           iter,
                           global_fit,
                           init_param_file,
                           save_directory)

    run(K, fold, iter)