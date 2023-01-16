import numpy as np
import os

def run (command):
    assert isinstance(command, str), "input must be a string"
    cols = os.get_terminal_size().columns
    print(cols*"="+"\n"+command+"\n"+cols*"=")
    error_code = os.system(command)
    if error_code:
        print(f"Command\n\t{command}\nterminated with error {error_code}")
        exit(error_code)

#
#   GET AND PREPROCESS DATA
#
os.chdir("1_preprocess_data/ibl")

# test configuration of one being used
run("python one_global.py")
# download the behavioural data and filter the 50-50 trials
run("python 1_download_data_begin_processing.py")
# get the covariates of the GLM for every trial in every session
run("python 2_create_design_mat.py")
# (this is of dubious use for the whole pipeline...)
run("python 3_get_response_time_data.py")
os.chdir("../..")


#
#   FIT GLM-HMM MODELS
#
os.chdir("2_fit_models")

# Fit the lapse model
os.chdir("fit_lapse_model")
for n in [1,2]:
    run(f"python 1_fit_lapse_model_all_ibl_animals_together.py {n}") # 
for n in [1,2]:
    run(f"python 2_fit_lapse_model_ibl_animals_separately.py {n}")   # 
os.chdir("..")

# Fit one single GLM to ...
os.chdir("fit_glm")
# ... all mice together (this gives an initialisation)
run("python 1_fit_glm_all_ibl_animals_together.py")         # 
# ... and then to mice separately
run("python 2_fit_glm_ibl_animals_separately.py")           # 
os.chdir("..")

# Fit the GLM-HMM model to all mice
os.chdir("fit_global_glmhmm")
run("python 0_create_cluster_job_arr.py")                   # 
# Here need to submit to cluster
#   sbatch submit.slurm
#   instead of run("python 1_run_inference_global_fit_ibl.py")             #
run("python 2_apply_post_processing.py")                    # 
run("python 3_get_best_params_for_individual_initialization.py")
os.chdir("..")

os.chdir("fit_individual_glmhmm")
run("python 0_create_cluster_job_arr.py")
# Here need to submit to cluster:
#   sbatch submit.slurm
#   instead of run("python 1_run_inference_ibl_individual.py")
run("python 2_apply_post_processing.py")
run("python 3_plot_best_params.py")
os.chdir("../..")
# exit()


os.chdir("3_make_figures")

os.chdir("figure_2")
run("python 1_calculate_predictive_accuracy.py")
run("python 2_make_top_plots_fig_2.py")
run("python 3_make_figure_2f.py")
run("python 4_make_figure_2g.py")
run("python 5_make_figure_2h.py"  )
os.chdir("..")

os.chdir("figure_3")
run("python 1_make_top_plots_fig_3.py")  
run("python 2_make_figure_3d.py")  
run("python 3_make_figure_3e.py")
os.chdir("..")

os.chdir("figure_4")
run("python 1_calculate_predictive_accuracy.py")
run("python 2_make_all_plots_fig_4.py")
os.chdir("..")

# os.chdir("figure_5")
# run("python 1_make_plots_5def.py")
# os.chdir("..")

os.chdir("figure_6")
run("python 1_plot_rt_q_q_ibl.py") 
run("python 2_create_bootstrap_distribution.py") 
run("python 3_plot_response_times_90th_percentile.py")
os.chdir("..")

# os.chdir("figure_7")
# run("python 1_make_plots_7def.py") 
# os.chdir("../..")
