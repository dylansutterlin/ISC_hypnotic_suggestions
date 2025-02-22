'''
Goals
- extract data from preprocessed folder, and relevant metadata
- plot combined carpet plot to see QC 
- Import the data
- Import the design matrices
- Specify the contrasts to run
- Run GLM and save results 

'''

import os
import numpy as np
import pandas as pd
import sys
import nibabel as nib
import matplotlib.pyplot as plt

from nilearn.glm.first_level import FirstLevelModel
from nilearn.glm.second_level import SecondLevelModel
from nilearn.glm.contrasts import compute_contrast
from nilearn.image import mean_img, concat_imgs
from nilearn.plotting import plot_design_matrix, plot_stat_map

if not os.getcwd().endswith('ISC_hypnotic_suggestions'):

    script_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../scripts'))
    sys.path.append(script_dir)

import preproc_utils
import visu_utils 
import glm_utils as utils
from sklearn.utils import Bunch
# %% [markdown]
## load data
model_dir = r'/data/rainville/dSutterlin/projects/ISC_hypnotic_suggestions/results/imaging/preproc_data/model1_23subjects_zscore_sample_detrend_21-02-25'
setup = preproc_utils.load_json(os.path.join(model_dir, 'setup_parameters.json'))
data_info = preproc_utils.load_pickle(os.path.join(model_dir, 'data_info_regressors.pkl'))
masker_params = preproc_utils.load_json(os.path.join(model_dir, 'preproc_params.json'))
MAX_ITER = 2

subjects = setup['subjects'][:MAX_ITER]
regressors = data_info['regressors_names'][:MAX_ITER]
condition_names = setup['condition_names']
tr = setup['tr']

#%%
design_matrices = preproc_utils.load_design_matrices(setup.design_matrices)

contrasts = {
    "Analgesia_vs_Neutral": "ANA_sugg - N_ANA_sugg",
    "Hyperalgesia_vs_Neutral": "HYPER_sugg - N_HYPER_sugg",
    "Pain_vs_NoPain": "ANA_shock + HYPER_shock - N_ANA_shock - N_HYPER_shock",
}

# Storage for first-level models and contrast maps
first_level_models = {}
contrast_maps = {}

for sub in setup.subjects:
    print(f"Processing {sub}...")


    dm_combined = design_matrices_2runs[sub]
    nscans_ana = data.nscans["Ana"][sub]
    nscans_hyper = data.nscans["Hyper"][sub]
    dm_combined["Run_Intercept_Ana"] = np.concatenate([np.ones(nscans_ana), np.zeros(nscans_hyper)])
    dm_combined["Run_Intercept_Hyper"] = np.concatenate([np.zeros(nscans_ana), np.ones(nscans_hyper)])

    func1 = nib.load(data.func_imgs_paths[sub][0])
    func2 = nib.load(data.func_imgs_paths[sub][1])
    # func_imgs = [concat_imgs([])nib.load(img) for img in data.func_imgs_paths[sub]]
    
# Run GLM for each subject
for sub in setup.subjects:
    print(f"Processing {sub}...")

    # Load subject-specific fMRI runs
    func_imgs = [nib.load(img) for img in data.func_imgs_paths[sub]]
    
    # Load design matrix
    design_matrix = design_matrices_2runs[sub]

    # Load movement confounds
    confounds = data.nuis_params_paths[sub]

    # Fit first-level GLM
    first_level_model = FirstLevelModel(
        t_r=tr, mask_img=mask, smoothing_fwhm=None, standardize=False
    )
    first_level_model.fit(func_imgs, design_matrices=design_matrix, confounds=confounds)

    # Compute contrasts and store maps
    subject_contrasts = {}
    for contrast_name, contrast_formula in contrasts.items():
        contrast_map = first_level_model.compute_contrast(
            contrast_formula, output_type="z_score"
        )
        subject_contrasts[contrast_name] = contrast_map

        # Save contrast maps
        contrast_path = os.path.join(setup.save_dir, f"{sub}_{contrast_name}.nii.gz")
        contrast_map.to_filename(contrast_path)

    # Store subject models and contrast maps
    first_level_models[sub] = first_level_model
    contrast_maps[sub] = subject_contrasts

    print(f"Completed GLM for {sub}")


# from nilearn.glm.first_level import FirstLevelModel

# for i, sub in enumerate(setup.subjects):

#     # Load 4D functional images and concatenate
#     ana_img = nib.load(setup.ana_run[i])  # Analgesia run
#     hyper_img = nib.load(setup.hyper_run[i])  # Hyperalgesia run
#     full_4d_img = concat_imgs([ana_img, hyper_img])  # Concatenate both runs

# # Initialize the GLM model (you probably already did this in your pipeline)
# fmri_glm = FirstLevelModel()  # Adjust t_r as per your TR value
# fmri_glm = fmri_glm.fit(setup.ana_run[i], design_matrices=dm_combined[i])  # Replace `fmri_img` with your actual fMRI data
