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


if os.getcwd().endswith('ISC_hypnotic_suggestions'):
    print('Appending scripts/ to python path')
    script_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../scripts'))
    sys.path.append(script_dir)

import preproc_utils
import visu_utils 
import glm_utils as utils
import qc_utils
from sklearn.utils import Bunch
from importlib import reload
from nilearn import datasets

# %% [markdown]
## load data
model_name = r'model2_3subjects_zscore_sample_detrend_25-02-25'
model_dir = rf'/data/rainville/dSutterlin/projects/ISC_hypnotic_suggestions/results/imaging/preproc_data/{model_name}'
setup_dct = preproc_utils.load_json(os.path.join(model_dir, 'setup_parameters.json'))
data_info_dct = preproc_utils.load_pickle(os.path.join(model_dir, 'data_info_regressors.pkl'))
masker_params = preproc_utils.load_json(os.path.join(model_dir, 'preproc_params.json'))

MAX_ITER = 3

setup = Bunch(**setup_dct)
data = Bunch(**data_info_dct)
masker_params = Bunch(**masker_params)
glm_info = Bunch()
subjects = setup.subjects

if MAX_ITER == None :
    MAX_ITER= len(subjects) 
else: 
    subjects = subjects[:MAX_ITER]
    setup.subjects = subjects   

regressors_dct = data.regressors_per_conds
condition_names = setup.run_names
tr = setup.tr


# %%

# DEV of independent run processing using fixed effect
extra_regressors_idx = {
    'all_sugg': [0, 1],
    'all_shock': [2,3]
}

contrasts_spec = {
    "ANA_sugg_minus_N_ANA_sugg": (1.5, -1),
    "HYPER_sugg_minus_N_HYPER_sugg": (1.5, -1),
    "ANA_shock+HYPER_shock_minus_N_ANA_shock+N_HYPER_shock": (1, -1)
}


loc_contrasts_vectors1 = {}
loc_contrasts_vectors2 = {}
first_level_models = {}
contrast_maps = {}
first_lev_files = {}

for sub, subject in enumerate(subjects):

    print(f"Processing {subject}...({sub+1}/{len(subjects)})")
    # design matrices
    dm_ana = pd.read_csv(data.ana_design_mat_files[sub], index_col=0)
    dm_hyper = pd.read_csv(data.hyper_design_mat_files[sub], index_col=0)
    
    dm_combined = pd.read_csv(data.design_mat_2runs_files[sub], index_col=0)
    nscans_ana = data.nscans["Ana"][subject]
    nscans_hyper = data.nscans["Hyper"][subject]

    # remove columns from design matrix
    # rm_cols = ['reg6', 'reg7'] + [col for col in dm_combined.columns if col.startswith('drift')]
    # dm_combined = dm_combined.drop(rm_cols, axis=1)
    # rm_cols_ana = ['reg6', 'reg7'] + [col for col in dm_ana.columns if col.startswith('drift')]
    # dm_ana = dm_ana.drop(rm_cols_ana, axis=1)
    # rm_cols_hyper = ['reg6', 'reg7'] + [col for col in dm_hyper.columns if col.startswith('drift')]
    # dm_hyper = dm_hyper.drop(rm_cols_hyper, axis=1)

    # plot design matrix but only for subject 1
    if sub == 0: plot=True
    else: plot=False
    plot_design_matrix(dm_ana)
    plot_design_matrix(dm_hyper)

    #Func images
    func1 = nib.load(setup.ana_run[sub])
    func2 = nib.load(setup.hyper_run[sub])
    # func_imgs = concat_imgs([func1, func2])

    regressors_dct_ana = {cond : cols for cond, cols in regressors_dct.items() if setup.run_id[0] in cond}
    regressors_dct_hyper = {cond : cols for cond, cols in regressors_dct.items() if setup.run_id[1] in cond}

    # make localizer contrasts
    if sub == 0: plot=True
    else: plot=False

    loc_contrasts_vectors1[subject] = utils.make_localizer_vec_from_reg(dm_ana, regressors_dct_ana,extra_indices_dct= extra_regressors_idx, plot = plot)
    loc_contrasts_vectors2[subject] = utils.make_localizer_vec_from_reg(dm_hyper, regressors_dct_hyper,extra_indices_dct= extra_regressors_idx, plot = plot)
    #loc_contrasts_vectors[subject] = utils.make_localizer_vec_from_reg(dm_combined, regressors_dct,extra_indices_dct= extra_regressors_idx, plot = plot)
    
    loc_contrasts_vectors1[subject].update(utils.make_contrast_vec_from_reg(dm_combined, regressors_dct, contrasts_spec, plot=plot))
    
    flm_ana = FirstLevelModel(t_r=tr, mask_img=mask, smoothing_fwhm=None, standardize=False, signal_scaling=False)
    flm_ana.fit(func_img_ana, design_matrices=dm_ana, confounds=None)
    
    flm_hyper = FirstLevelModel(t_r=tr, mask_img=mask, smoothing_fwhm=None, standardize=False, signal_scaling=False)
    flm_hyper.fit(func_img_hyper, design_matrices=dm_hyper, confounds=None)
    # first_level_model = FirstLevelModel(t_r=tr, mask_img=mask, smoothing_fwhm=None, standardize=False, signal_scaling=False)
    # first_level_model.fit(func_imgs, design_matrices=dm_combined, confounds=None) # already in dm

    contrast_run1 = flm_ana.compute_contrast(contrast_vector, output_type='all')
    contrast_run2 = flm_hyper.compute_contrast(contrast_vector, output_type='all')
    
    # Combine the run-level results using fixed effects.
    contrast_imgs = [contrast_run1['effect_size'], contrast_run2['effect_size']]
    variance_imgs = [contrast_run1['effect_variance'], contrast_run2['effect_variance']]
    
    fixed_effect_contrast, fixed_effect_variance, fixed_effect_stat = compute_fixed_effects(
        contrast_imgs, variance_imgs, mask_img=mask
    )

    # .... last modif (not fully tested) 7th march 2025


    sub_contrasts = {}
    contrasts_files = {}
    MAX_CONT = len(loc_contrasts_vectors[subject].keys())

    i = 0
    for condition, contrast_vector in loc_contrasts_vectors[subject].items():
        
        os.makedirs(os.path.join(save_glm, condition), exist_ok=True)

        print(f"{condition}...")
        if i < MAX_CONT:
            print(f"contrast for {condition}...")
            contrast_map = first_level_model.compute_contrast(contrast_vector, output_type="z_score")
            sub_contrasts[condition] = contrast_map

            # Save contrast maps
            contrasts_files[condition] = os.path.join(save_glm, condition, f"firstlev_localizer_{subject}.nii.gz")
            contrast_map.to_filename(contrasts_files[condition])
            i += 1
        else :
            break

    first_level_models[subject] = first_level_model
    contrast_maps[subject] = sub_contrasts
    first_lev_files[subject] = contrasts_files

contrast_names = list(first_lev_files[subject].keys())
glm_info.first_level_models = first_level_models
glm_info.contrast_files_1level = first_lev_files

