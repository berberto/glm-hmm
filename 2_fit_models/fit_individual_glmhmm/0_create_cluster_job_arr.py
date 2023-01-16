#  In order to facilitate parallelization of jobs, create a job array that
#  can be used on e.g. a cluster
import numpy as np
import os

prior_sigma = [2]
transition_alpha = [2]
K_vals = [2, 3, 4, 5]
num_folds = 5
N_initializations = 3

run = False

if __name__ == '__main__':
    with open("job_array_pars.txt", "w") as f:
        for K in K_vals:
            for i in range(num_folds):
                for j in range(N_initializations):
                    for sigma in prior_sigma:
                        for alpha in transition_alpha:
                            f.write(f"{sigma}\t{alpha}\t{K}\t{i}\t{j}\n")
                            if run:
                                os.system(f"1_run_inference_ibl_individual.py {sigma}\t{alpha}\t{K}\t{i}\t{j}")
