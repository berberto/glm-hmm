#  In order to facilitate parallelization of jobs, create a job array that
#  can be used on e.g. a cluster
import numpy as np
import os

K_vals = [2, 3, 4, 5]
num_folds = 5
num_iterations = 20

run = False

if __name__ == '__main__':
    cluster_job_arr = []
    with open("job_array_pars.txt", "w") as f:
        for K in K_vals[::-1]:
            for i in range(num_folds):
                for n in range(num_iterations):
                    f.write(f"{K}\t{i}\t{n}\n")
                    if run:
                        os.system(f"1_run_inference_global_fit_ibl.py  {K}  {i}  {n}")
