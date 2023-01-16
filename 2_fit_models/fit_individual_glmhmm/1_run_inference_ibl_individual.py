#  Fit GLM-HMM to data from all IBL animals together.  These fits will be
#  used to initialize the models for individual animals
import os
import sys

import autograd.numpy as np

D = 1  # data (observations) dimension
C = 2  # number of output types/categories
N_em_iters = 300  # number of EM iterations

if __name__ == '__main__':
    global_data_dir = '../../data/ibl/data_for_cluster'
    data_dir = f"{global_data_dir}/data_by_animal"
    results_dir = '../../results/ibl_individual_fit/'
    sys.path.insert(0, '../fit_global_glmhmm/')

    from glm_hmm_utils import load_cluster_arr, load_session_fold_lookup, \
        load_animal_list, load_data, create_violation_mask, \
        launch_glm_hmm_job

    try:
        prior_sigma, transition_alpha, K, fold, iter = sys.argv[1:]
    except:
        raise Exception(f"Parameters from command line: `prior_sigma`, `transition_alpha`, `K`, `fold`, `iter`")

    prior_sigma =  float(prior_sigma)
    transition_alpha =  float(transition_alpha)
    K = int(K)
    fold = int(fold)
    iter = int(iter)

    def run (prior_sigma, transition_alpha, K, fold, iter):
        animal_list = load_animal_list(f"{data_dir}/animal_list.npz")

        for i, animal in enumerate(animal_list):
            print(animal)
            animal_file = f"{data_dir}/{animal}_processed.npz"
            session_fold_lookup_table = load_session_fold_lookup(
                f"{data_dir}/{animal}_session_fold_lookup.npz")

            global_fit = False

            inpt, y, session = load_data(animal_file)
            #  append a column of ones to inpt to represent the bias covariate:
            inpt = np.hstack((inpt, np.ones((len(inpt), 1))))
            y = y.astype('int')

            overall_dir = f"{results_dir}/{animal}"

            # Identify violations for exclusion:
            violation_idx = np.where(y == -1)[0]
            nonviolation_idx, mask = create_violation_mask(violation_idx,
                                                           inpt.shape[0])

            init_param_file = f"{global_data_dir}/best_global_params/best_params_K_{str(K)}.npz"

            # create save directory for this initialization/fold combination:
            save_directory = f"{overall_dir}/GLM_HMM_K_{str(K)}/fold_{str(fold)}/iter_{str(iter)}"
            if not os.path.exists(save_directory):
                os.makedirs(save_directory)

            launch_glm_hmm_job(inpt, y, session, mask, session_fold_lookup_table,
                               K, D, C, N_em_iters, transition_alpha, prior_sigma,
                               fold, iter, global_fit, init_param_file,
                               save_directory)

    run(prior_sigma, transition_alpha, K, fold, iter)