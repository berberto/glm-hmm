#!/bin/bash

#SBATCH --partition=cpu
#SBATCH --array=1-60
#SBATCH --job-name=hmm_ind
#SBATCH -n 1
#SBATCH --mem=16G
#SBATCH --time=6-00:00:00

parameters=`sed -n "${SLURM_ARRAY_TASK_ID} p" job_array_pars.txt`
parArray=($parameters)

sigma=${parArray[0]}
alpha=${parArray[1]}
K=${parArray[2]}
fold=${parArray[3]}
init=${parArray[4]}

python --version
python 1_run_inference_ibl_individual.py "$sigma" "$alpha" "$K" "$fold" "$init"
