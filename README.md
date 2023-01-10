# glm-hmm

Code to reproduce figures in ["Mice alternate between discrete strategies during perceptual decision-making"][manuscript] from Ashwood, Roy, Stone, IBL, Urai, Churchland, Pouget and Pillow (2020).  Note: while this code reproduces the figures/analyses for our paper, the easiest way to get started applying the GLM-HMM to your own data, is with [this notebook](https://github.com/zashwood/ssm/blob/master/notebooks/2b%20Input%20Driven%20Observations%20(GLM-HMM).ipynb). 

Code is ordered so that the IBL dataset discussed in the paper is preprocessed into the desired format by the scripts in `1_preprocess_data`. 

Within this directory, run the scripts in the order indicated by the number at the beginning of the file name (i.e run `1_download_data_begin_processing.py` first to obtain the IBL data locally and then run `2_create_design_mat.py` to obtain the design matrix used as input for all of the models discussed in our paper).

Next, you can fit the GLM, lapse and GLM-HMM models discussed in the paper using the code contained in `2_fit_models`.
As discussed in the paper, the GLM should be run first as the GLM fit is used to initialize the global GLM-HMM (the model that is fit with data from all animals).
The lapse model fits, while not used for any initialization purposes, should be run next so as to be able to perform model comparison with the global and individual GLM-HMMs.
The global GLM-HMM should be run next, as it is used to initialize the models for all individual animals.
Finally GLM-HMMs can be fit to the data from individual animals using the code in the associated directory.
          
Assuming that you have downloaded and preprocessed the datasets, and that you have fit all models on these datasets,  you can reproduce the figures of our paper corresponding to the IBL dataset by running the code contained in "3_make_figures".
In order to produce Figures 5 and 7, replace the IBL URL in the preprocessing pipeline scripts, with the URLs for the [Odoemene et al. (2018)](https://doi.org/10.14224/1.38944) and [Urai et al. (2017)](https://doi.org/10.6084/m9.figshare.4300043) datasets, and rerun the GLM, lapse and GLM-HMM models on these datasets before running the provided figure plotting code.

## Setup

### `iblenv` environment

First, create and setup the `iblenv` conda environment, following the [instructions here](https://github.com/int-brain-lab/iblenv).
```bash
cd /path/of/choice
conda create --name iblenv python=3.9 --yes
conda activate iblenv
git clone https://github.com/int-brain-lab/iblapps
pip install --editable iblapps
git clone https://github.com/int-brain-lab/iblenv
cd iblenv
pip install --requirement requirements.txt
```

### Install the `ssm` package

We use version 0.0.1 of the Bayesian State Space Modeling framework from Scott Linderman's lab to perform GLM-HMM inference. Within the `iblenv` environment, install the forked version of the `ssm` package available [here](https://github.com/zashwood/ssm).  This is a lightly modified version of the master branch of the ssm package available at [https://github.com/lindermanlab/ssm](https://github.com/lindermanlab/ssm). It is modified so as to handle violation trials as described in Section 4 of the manuscript. 
    
```bash
conda activate iblenv
cd /path/of/choice
git clone https://github.com/zashwood/ssm
cd ssm
pip install numpy cython
pip install -e .
```

## Usage

You can run the whole pipeline with `master.py` 

```bash
git clone https://github.com/berberto/glm-hmm.git
cd glm-hmm
python master.py
```

Not exactly the whole pipeline. Fit of the GLM-HMM globally or for individual mice is currently done via SLURM job submission. All the steps before and after these can be executed via the `master.py` script.


[manuscript]: https://www.biorxiv.org/content/10.1101/2020.10.19.346353v4.full.pdf