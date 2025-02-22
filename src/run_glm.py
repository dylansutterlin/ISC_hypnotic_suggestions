'''
Goals
- extract data from preprocessed folder, and relevant metadata
- plot combined carpet plot to see QC 
- Import the data
- Import the design matrices
- Specify the contrasts to run
- Run GLM and save results 

'''
# %%
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
setup_dct = preproc_utils.load_json(os.path.join(model_dir, 'setup_parameters.json'))
data_info_dct = preproc_utils.load_pickle(os.path.join(model_dir, 'data_info_regressors.pkl'))
masker_params = preproc_utils.load_json(os.path.join(model_dir, 'preproc_params.json'))

MAX_ITER = 2

setup = Bunch(**setup_dct)
data = Bunch(**data_info_dct)
masker_params = Bunch(**masker_params)

subjects = setup.subjects[:MAX_ITER]
regressors_dct = data.regressors_per_conds
condition_names = setup.run_names
tr = setup.tr

reload(utils)
ref_img = nib.load(setup.ana_run[0])
mask = utils.load_data_mask(ref_img)

save_glm = os.path.join(setup.save_dir, 'GLM_results')
os.makedirs(save_glm, exist_ok=True)

# %% [markdown]
## First level localizer
from importlib import reload
reload(utils)

contrasts = {
    "Analgesia_vs_Neutral": "ANA_sugg - N_ANA_sugg",
    "Hyperalgesia_vs_Neutral": "HYPER_sugg - N_HYPER_sugg",
    "Pain_vs_NoPain": "ANA_shock + HYPER_shock - N_ANA_shock - N_HYPER_shock",
}

# Storage for first-level models and contrast maps
loc_contrasts_vectors = {}
first_level_models = {}
contrast_maps = {}
subjects_contrasts_files = {}

for sub, subject in enumerate(subjects):

    print(f"Processing {subject}...")
    subject = subjects[0]
    sub = 0
    # design matrices
    dm_combined = pd.read_csv(data.design_mat_2runs_files[sub], index_col=0)
    nscans_ana = data.nscans["Ana"][subject]
    nscans_hyper = data.nscans["Hyper"][subject]
    dm_combined["Run_Intercept_Ana"] = np.concatenate([np.ones(nscans_ana), np.zeros(nscans_hyper)])
    dm_combined["Run_Intercept_Hyper"] = np.concatenate([np.zeros(nscans_ana), np.ones(nscans_hyper)])
    #Func images
    func1 = nib.load(setup.ana_run[sub])
    func2 = nib.load(setup.hyper_run[sub])
    func_imgs = concat_imgs([func1, func2])
    # make localizer contrasts
    if sub == 0: plot=True
    else: plot=False
    loc_contrasts_vectors[subject] = utils.make_contrasts(dm_combined, regressors_dct, plot = plot)

    first_level_model = FirstLevelModel(t_r=tr, mask_img=mask, smoothing_fwhm=None, standardize=False)
    first_level_model.fit(func_imgs, design_matrices=dm_combined, confounds=None) # already in dm

    sub_contrasts = {}
    contrasts_files = {}
    MAX_CONT = len(loc_contrasts_vectors[subject].keys())

    i = 0
    for condition, contrast_vector in loc_contrasts_vectors[subject].items():
        
        if i < MAX_CONT:
            print(f"Localizer for {condition}...")
            contrast_map = first_level_model.compute_contrast(contrast_vector, output_type="z_score")
            sub_contrasts[condition] = contrast_map

            # Save contrast maps
            contrast_path = os.path.join(save_glm, f"{condition}_localizer_{subject}.nii.gz")
            contrasts_files[condition] = contrast_path
            contrast_map.to_filename(contrast_path)
            i += 1
        else :
            break

    first_level_models[subject] = first_level_model
    contrast_maps[subject] = sub_contrasts
    subjects_contrasts_files[subject] = contrasts_files

# %%
# from nilearn.glm.first_level import FirstLevelModel

# for i, sub in enumerate(setup.subjects):

#     # Load 4D functional images and concatenate
#     ana_img = nib.load(setup.ana_run[i])  # Analgesia run
#     hyper_img = nib.load(setup.hyper_run[i])  # Hyperalgesia run
#     full_4d_img = concat_imgs([ana_img, hyper_img])  # Concatenate both runs

# # Initialize the GLM model (you probably already did this in your pipeline)
# fmri_glm = FirstLevelModel()  # Adjust t_r as per your TR value
# fmri_glm = fmri_glm.fit(setup.ana_run[i], design_matrices=dm_combined[i])  # Replace `fmri_img` with your actual fMRI data
