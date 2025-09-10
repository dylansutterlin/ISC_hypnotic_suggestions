
# %% [markdown]
# # Script to extract timeseries and check quality of the data
# Includes Timeseries heatmaps, atlas fitting, report saving, GLM building
    # Support script : ../func.py

# %%
import sys
import os
import glob
import numpy as np
import pandas as pd
import pickle as pkl
import seaborn as sns
import json
from datetime import datetime

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from nilearn.plotting import plot_stat_map
from sklearn.utils import Bunch
from nilearn.maskers import (
    MultiNiftiMasker,
    NiftiMasker,
    MultiNiftiLabelsMasker,
    MultiNiftiMapsMasker,
)
import warnings
import nibabel as nib
import importlib
warnings.simplefilter("ignore")
from nilearn.image import load_img
from nilearn.plotting import plot_roi
from nilearn.maskers import MultiNiftiLabelsMasker, NiftiLabelsMasker
from nilearn.image import mean_img
from nilearn.plotting import plot_stat_map, view_img
from nilearn.glm.first_level import make_first_level_design_matrix
from nilearn.plotting import plot_design_matrix
import importlib
from nilearn.image import index_img, concat_imgs
from nilearn.glm.first_level import FirstLevelModel
# import preproc_utils as utils

from importlib import reload

# if os.getcwd().endswith('ISC_hypnotic_suggestions'):
#     from scripts import preproc_utils as utils
# else:
#     script_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../scripts'))
#     sys.path.append(script_dir)
#     import preproc_utils as utils
#     import visu_utils 

import src.isc_utils as isc_utils
import src.visu_utils as visu_utils
import src.preproc_utils as utils

import gc

# %%
# parent_dir = os.path.abspath(os.path.join(os.path.dirname("func.py"), ".."))
# sys.path.append(parent_dir)
# import func
# print(f"Python version: {sys.version}")
# %% [markdown]
# #### Load data

# %%
data_dir = '/data/rainville/Hypnosis_ISC/4D_data/full_run'
ana_run = glob.glob(os.path.join(data_dir, 'sub*', '*analgesia*.nii.gz'))
subjects = [os.path.basename(os.path.dirname(path)) for path in ana_run]
nsub = len(subjects) #all
do_carptet_plot = False

setup = Bunch(
    data_dir="/data/rainville/Hypnosis_ISC/4D_data/full_run",
    run_names = ["Analgesia", "Hyperalgesia"],
    ana_run=glob.glob(
        os.path.join(
            "/data/rainville/Hypnosis_ISC/4D_data/full_run",
            "sub*",
            "*analgesia*.nii.gz",
        )
    )[0:nsub],
    hyper_run=glob.glob(
        os.path.join(
            "/data/rainville/Hypnosis_ISC/4D_data/full_run",
            "sub*",
            "*hyperalgesia*.nii.gz",
        )
    )[0:nsub],
    behav_path="/data/rainville/dSutterlin/projects/resting_hypnosis/resting_state_hypnosis/atlases/Hypnosis_variables_20190114_pr_jc.xlsx",
    events_dir = "/data/rainville/HYPNOSIS_IMAGING_DATA/timestamps",
    confound_dir = "/data/rainville/HYPNOSIS_IMAGING_DATA/Nii",
    project_dir="/data/rainville/dSutterlin/projects/ISC_hypnotic_suggestions",
    results_dir=os.path.join(
        "/data/rainville/dSutterlin/projects/ISC_hypnotic_suggestions",
        "results/imaging/preproc_data",
    ),  
    subjects=subjects[0:nsub]

)
setup.APM_subjects = ['APM' + sub[4:] for sub in setup.subjects][0:nsub] # make APMXX format instead of subXX
setup.tr = 3

voxel_masker = Bunch(name="voxel_wise")

# Preprocessing parameters
# parcel_name = par"voxel_wise"
# atlas_name = "k50-2mm-parcel"
# parcel_type = 'labelsMasker'
mask = '/data/rainville/Hypnosis_ISC/masks/brainmask_91-109-91.nii'
voxel_masker.mask = mask
masker_params_dict = {
    "standardize": 'zscore_sample',
    "mask_img": mask,
    "detrend": False,
    "low_pass": None,
    "high_pass": 0.00234, # 1/428 sec. from desmarteaux 2021
    "t_r": setup.tr,
    "smoothing_fwhm": None,
    "standardize_confounds": True,
    "verbose": 3,
    "high_variance_confounds": False,
    "mask_strategy": "whole-brain-template", 
    "memory" : "nilearn_cache"
}
HRF_MODEL = 'spm'
DRIFT_MODEL = None

# initialize preproc model and save dirs
preproc_model_name = "model3_single-blocks_extraction-vols_{}subjects_{}_detrend-{}_{}".format(
    len(setup.subjects),masker_params_dict['standardize'],masker_params_dict['detrend'], datetime.today().strftime("%d-%m-%y")
)
save_dir = os.path.join(setup.results_dir, preproc_model_name)
if os.path.exists(save_dir):
    print(f"The folder '{save_dir}' already exists. Results will be overwritten!!.")
    error = input("Do you want to continue? (y/n)")
else:
    os.makedirs(save_dir, exist_ok=True)
setup.model_name = preproc_model_name
setup.save_dir = save_dir

# save meta data
masker_params_path = os.path.join(save_dir, "preproc_params.json")
utils.save_json(masker_params_path, masker_params_dict)

save_setup_p = os.path.join(setup.save_dir, "setup_parameters.json")
utils.save_json(save_setup_p, setup)

print(f"Directory {save_dir} created.")
print(f"Masker parameters saved to {masker_params_path}")

# %% [markdown]
# #### Import and load data!

# %% [markdown]
# #### Extract timestamps/events, movement files based on APM_subjects
# %%
# extract event files
events_dir = setup.events_dir
confound_dir = setup.confound_dir
run_names = setup.run_names
APM_subjects = setup.APM_subjects

events_ana = [
    utils.get_timestamps(
        data_dir, sub, events_dir, run_names[0], return_path=False
    ).sort_values(by="onset")
    for sub in APM_subjects
]
events_hyper = [
    utils.get_timestamps(
        data_dir, sub, events_dir, run_names[1], return_path=False
    ).sort_values(by="onset")
    for sub in APM_subjects
]

# ANA run first!
confound_files = []
for sub in APM_subjects:

    formatted_sub = f"{sub[:3]}_{sub[3:]}*"
    # print(formatted_sub)
    # Extend the confound_files list with the matching files
    confound_files.extend(
        glob.glob(os.path.join(confound_dir, "H*", "*", formatted_sub, "*_8nuisreg_*"))
    )
    # print(glob.glob(os.path.join(confound_dir, 'H*', '*', formatted_sub, '*_8nuisreg_*')))
print("len confound files : ", len(confound_files))

cf4 = []  # extra 4 sub with diff TRs in H2/APM*/*nuis.txt
for sub in APM_subjects:
    formatted_sub = f"{sub[:3]}_{sub[3:]}*"
    cf4.extend(
        glob.glob(os.path.join(confound_dir, "H*", formatted_sub, "*_8nuisreg_*"))
    )

print(f"found {len(cf4)} extra irreglar-sub condound files")
confound_files.extend(cf4)

# Reorder matching APM_subjects
ordered_confound_files = []
for sub in APM_subjects:
    formatted_sub = f"{sub[:3]}_{sub[3:]}"
    match = [f for f in confound_files if formatted_sub in f]
    if match:
        ordered_confound_files.append(match[0])  # Take the first match

confound_files = ordered_confound_files
confound_files_dct = {sub: file for sub, file in zip(subjects, confound_files)}
print("Total confound files : ", len(confound_files))

data = Bunch()
data.events_dfs = {}
data.events_dfs["Ana"] = events_ana
data.events_dfs["Hyper"] = events_hyper
data.confound_files = confound_files

# %% [markdown]
# Ensure same length and order of all variables

# %% [markdown]
# #### Split confound file based on run scans (up = Ana run, lower = Hyper run)

# %%
# Get the number of scans for each subject
data.nscans = {}
data.nscans["Ana"] = {
    sub: nib.load(img).shape[-1] for sub, img in zip(subjects, setup.ana_run)
}
data.nscans["Hyper"] = {
    sub: nib.load(img).shape[-1] for sub, img in zip(subjects, setup.hyper_run)
}

# Split movement regressor file in two according to scan number of each
# output loaded np arrays fir each run
ana_confounds = []
hyper_confounds = []
for i, sub in enumerate(setup.subjects):

    conf_path = confound_files[i]
    confounds = np.array(pd.read_csv(conf_path, sep="\s+", header=None))

    print(confounds.shape, type(confounds), "len % 2 :", len(confounds) / 2)

    nscan_ana = data.nscans["Ana"][sub]
    up_ana = confounds[0:nscan_ana, :]

    nscan_hyper = data.nscans["Hyper"][sub]
    low_hyper = confounds[nscan_ana : nscan_ana + nscan_hyper, :]

    print("low_ana shape:", up_ana.shape)
    print("low_hyper shape:", low_hyper.shape)

    ana_confounds.append(up_ana)
    hyper_confounds.append(low_hyper)

    # up, low = split_conf_matrix(confound_files[i], nscan, run_names[i]
    #    )

data.confounds_Ana = ana_confounds
data.confounds_Hyper = hyper_confounds

# %% [markdown]
# ### Check TRs per condition

# %% [markdown]
# #### utilstion to count TRs for suggestion conditions, for neutral and modulation suggestion. Added to utils.py file

# %% [markdown]
# #### Number of TRs (total condition duration / TR) TR=3 in this dataset

# %%
reload(utils)

# hyper conditionregerss_mvmnt_carpet
TRs_df, fig_hyper = utils.count_plot_TRs_2conditions(
    setup.subjects,
    events_hyper,
    TR=3,
    neutral_pattern="N_HYPER.*instrbk",
    modulation_pattern="HYPER.*instrbk",
    title="Neutral and Hyperalgesia Suggestion TRs per Subjects",
    neutral_color="skyblue",
    modulation_color="salmon",
    save_path=os.path.join(
        setup.save_dir, "TRs-per-conds_Hyper_{}-subjects.png".format(len(events_hyper))
    ),
)

plt.close(fig_hyper)

# Ana condition
TRs_df, fig_ana = utils.count_plot_TRs_2conditions(
    setup.subjects,
    events_ana,
    TR=3,
    neutral_pattern="N_ANA.*instrbk",
    modulation_pattern="ANA.*instrbk",
    title="Neutral and Analgesia Suggestion TRs per Subjects",
    neutral_color="skyblue",
    modulation_color="forestgreen",
    save_path=os.path.join(
        setup.save_dir, "TRs-per-conds_Ana_{}-subjects.png".format(len(events_ana))
    ),
)
plt.close(fig_ana)
# %% [markdown]
# ## First level model
# %% [markdown]
# #### Compute and save design matrice (joint runs)
# Returns dict [sub] : combined_dm

# %%
setup.keys(), voxel_masker.keys()

# %%
run_names_short = ['Ana','Hyper'] # [setup.conditions[0][0:3], setup.conditions[1][0:5]] # Ana and Hyper
design_matrices_2runs = {}
design_matrices_2runs_files = []
dm_ana_df = []
dm_hyper_df = []
dm_ana_files = []
dm_hyper_files = []

save_path = os.path.join(setup.save_dir, "design_matrices_TxT")
os.makedirs(save_path, exist_ok=True)


for i, sub in enumerate(setup.subjects):

    tr = masker_params_dict["t_r"]
    # names = setup.run_names
    # ts_ana = voxel_masker.preproc_2d_Ana[i]
    # ts_hyper = voxel_masker.preproc_2d_Hyper[i]
    nscans_ana = nib.load(setup.ana_run[i]).shape[-1] #ts_ana.shape[0]
    frame_time_ana = np.arange(nscans_ana) * tr
    nscans_hyper = nib.load(setup.hyper_run[i]).shape[-1] #ts_hyper.shape[0]
    frame_time_hyper = np.arange(nscans_hyper) * tr

    # Assert that nscans match for both runs
    # warnings == ts_hyper.shape[0], f"Mismatch in nscans for subject {sub}"

    if masker_params_dict["high_pass"] is None:
        masker_params_dict["high_pass"] = 0.01 #default nilearn

    dm_ana = make_first_level_design_matrix(
        frame_time_ana,
        data.events_dfs["Ana"][i],
        hrf_model= HRF_MODEL,
        drift_model=DRIFT_MODEL,
        high_pass=masker_params_dict["high_pass"],
        add_regs=data.confounds_Ana[i],
        min_onset=0,
        oversampling=50 #default
    )

    dm_hyper = make_first_level_design_matrix(
        frame_time_hyper,
        data.events_dfs["Hyper"][i],
        hrf_model=HRF_MODEL,
        drift_model=DRIFT_MODEL,
        high_pass=masker_params_dict["high_pass"],
        add_regs=data.confounds_Hyper[i],
        min_onset=0,
        oversampling=50 #default
     )
    
    #rename movement reg 
    dm_ana.columns = [f"Ana_{col}" if col.startswith("nuis") else col for col in dm_ana.columns]
    dm_hyper.columns = [f"Hyper_{col}" if col.startswith("nuis") else col for col in dm_hyper.columns]


    # Combine design matrices
    zero_padding_ana = pd.DataFrame(0, index=dm_ana.index, columns=dm_hyper.columns)
    dm_ana_padded = pd.concat([dm_ana, zero_padding_ana], axis=1)

    zero_padding_hyper = pd.DataFrame(0, index=dm_hyper.index, columns=dm_ana.columns)
    dm_hyper_padded = pd.concat([zero_padding_hyper, dm_hyper], axis=1)

    dm_combined = pd.concat([dm_ana_padded, dm_hyper_padded], axis=0)
    design_matrices_2runs[sub] = dm_combined

    # save individual design matrices
    dm_ana_df.append(dm_ana)
    dm_hyper_df.append(dm_hyper)
    dm_ana_files.append(os.path.join(save_path, f"{sub}-Ana-design_matrix.csv"))
    dm_hyper_files.append(os.path.join(save_path, f"{sub}-Hyper-design_matrix.csv"))
    dm_ana_df[i].to_csv(dm_ana_files[-1])
    dm_hyper_df[i].to_csv(dm_hyper_files[-1])

    # Save combined design matrix as CSV
    csv_name = f'{sub}-{"-".join(run_names_short)}-design_matrix.csv'
    dm_combined_csv_path = os.path.join(save_path, csv_name)
    dm_combined.to_csv(dm_combined_csv_path)
    design_matrices_2runs_files.append(dm_combined_csv_path)

    # Save combined design matrix as PNG
    png_name = f'{sub}-{"-".join(run_names_short)}-design_matrix.png'
    dm_combined_png_path = os.path.join(save_path, png_name)

    plot_design_matrix(dm_combined, rescale=True)
    plt.title(f"Combined Design Matrix for {sub}")
    plt.tight_layout()
    plt.savefig(dm_combined_png_path)
    plt.close()

    print(
        f"Saved design matrix for subject {sub}: CSV ({csv_name}) and PNG ({png_name})."
    )

# %%
    # Plot for the last subject
    if i == len(setup.subjects) - 1:

        plot_design_matrix(dm_ana, rescale=True)
        plt.title(f"Design Matrix for {sub} - Ana", fontsize=16)
        plt.tight_layout()
        plt.show()

        plot_design_matrix(dm_hyper, rescale=True)
        plt.title(f"Design Matrix for {sub} - Hyper", fontsize=16)
        plt.tight_layout()
        plt.show()

        plot_design_matrix(dm_combined, rescale=True)
        plt.title(f"Combined Design Matrix for {sub}", fontsize=16)
        plt.tight_layout()
        plt.show()

data.design_mat_2runs_files = design_matrices_2runs_files
data.ana_design_mat_files = dm_ana_files
data.hyper_design_mat_files = dm_hyper_files

#%%
# return design_matrices_2runs

# %% [markdown]
# ### From DMs, Extract TRs for each condition, concatenate and save
# Combine the timeseries for both runs

# %%
# conditions = ['ANA', 'N_ANA', 'HYPER', 'N_HYPER']
condition_names = ['ANA_sugg', 'N_ANA_sugg', 'HYPER_sugg', 'N_HYPER_sugg',
                    'ANA_shock', 'N_ANA_shock', 'HYPER_shock', 'N_HYPER_shock']

# Trial wise model 8 AVRIL
condition_names = [trial_cond for trial_cond in list(dm_combined.columns) if 'instrbk' in trial_cond]          

data.condition_names = condition_names
indices_all_cond = []
kept_columns = []
full_confounds = [np.vstack([ana, hyper]) for ana, hyper in zip(data.confounds_Ana, data.confounds_Hyper)]
data.full_confounds = full_confounds

# Extract TRs indices for regressors of all subjects (dict for each sub)
for sub in setup.subjects:

    print(f"==Extracting TRs indices for regr essors of {sub}==")
    dm_combined = design_matrices_2runs[sub]
    cond_indices = {}
    kept_col = {}
    
    for cond in condition_names:
        if 'sugg' in cond:
            cond_indices[cond], kept_col[cond] = utils.create_tr_masks_for_suggestion(
                dm_combined, regressor=cond
            )
        elif 'shock' in cond:
            cond_indices[cond], kept_col[cond] = utils.create_tr_masks_for_shock(
                dm_combined, regressor=cond
            )
        elif 'instrbk' in cond:
            cond_indices[cond], kept_col[cond] = utils.create_tr_mask_for_single_trial(
                dm_combined, trial_regressor=cond
            )

        indices_all_cond.append(cond_indices) # list of dict per sub
        kept_columns.append(kept_col)

data.regressors_per_conds = kept_columns[0] # regressors are the same for all subj

# %%
# Visualize segmented events
sub_i = 0
sub = setup.subjects[sub_i]
save_tr_visu = os.path.join(setup.save_dir, f"TRs_visualization_{sub}")
os.makedirs(save_tr_visu, exist_ok=True)
fig_paths = []
for cond in condition_names:

    test_shock = kept_columns[sub_i][cond]
    ind_shock = indices_all_cond[sub_i][cond]
    extracted_timeseries = design_matrices_2runs[sub][list(test_shock)]

    x_values = np.arange(len(extracted_timeseries))

    plt.figure(figsize=(10, 5))
    for regressor in test_shock:
        plt.plot(x_values, extracted_timeseries[regressor], label=regressor, alpha=0.7)
    n_tr = len(ind_shock)
    plt.scatter(ind_shock, [0] * len(ind_shock), color="red", marker="o", label=f"Kept TRs : {n_tr}")

    plt.xlabel("Time (TRs)")
    plt.ylabel("Regressor Value")
    plt.title(f"design matrix regressors for {cond}")
    plt.legend()
    plt.grid(True)
    fig_path = os.path.join(save_tr_visu, f"{cond}_TRs.png")
    fig_paths.append(fig_path)
    plt.savefig(fig_path)
    
    plt.show()

# %% 
#save combined figures
reload(visu_utils)
visu_utils.plot_images_grid(fig_paths, 'All conditions extracted TRs', save_to=os.path.join(save_tr_visu, "grid_view_all_TRs.png"))
# %%

# %% [markdown]
## Get timeseries for each cond indices

#%%
extracted_volumes_per_cond = []
nuis_reg_per_cond = []

# Get TR indices for each conditions and concatenate volumes
for i, sub in enumerate(setup.subjects):
    print(f"Processing subject {sub}" ) #: Extracting NIfTI volumes for {condition_names}")

    ana_img = nib.load(setup.ana_run[i])  # Analgesia run
    hyper_img = nib.load(setup.hyper_run[i])  # Hyperalgesia run
    full_4d_img = concat_imgs([ana_img, hyper_img])  # Concatenate both runs
    sliced_imgs_per_cond = {}
    sliced_confounds_per_cond = {}

    for cond in condition_names:
        # Extract TR indices from design matrix
        sliced_imgs_per_cond[cond] = index_img(full_4d_img, indices_all_cond[i][cond])
        # Extract movement regressors 
        sliced_confounds_per_cond[cond] = full_confounds[i][indices_all_cond[i][cond]]

        if cond == 'ANA_sugg': 
            print(f'Remove 3 last TRs for {cond} !!!')
            sliced_imgs_per_cond[cond] = index_img(sliced_imgs_per_cond[cond], list(range(sliced_imgs_per_cond[cond].shape[-1] - 3)))
            sliced_confounds_per_cond[cond] = sliced_confounds_per_cond[cond][:-3,:]  # Trim last 3 TRs
        if cond == 'N_ANA_shock':
            print(f'Remove last (1) TR for {cond} !!!')
            sliced_imgs_per_cond[cond] = index_img(sliced_imgs_per_cond[cond], list(range(sliced_imgs_per_cond[cond].shape[-1] - 1)))
            sliced_confounds_per_cond[cond] = sliced_confounds_per_cond[cond][:-1,:]  # Trim last TR
    print('shape of extracted images : ', {k: v.shape for k, v in sliced_imgs_per_cond.items()})
    # Store extracted images and confounds
    extracted_volumes_per_cond.append(sliced_imgs_per_cond)
    nuis_reg_per_cond.append(sliced_confounds_per_cond) # out  [dict where keys are cond : np.array, ...]

print(f"Extracted NIfTI volumes and nuisance regressors for all subjects: {condition_names}")
# %% [markdown]
# #### Save concatenated timeseries for each condition
#
# output func_imgs_paths : list of dict containing paths to 4D nifti files for each condition

# %%

save_dir_4D = os.path.join(setup.save_dir, f"extracted_4D_per_cond_{nsub}sub")
os.makedirs(save_dir_4D, exist_ok=True)
func_imgs_paths = {}
nuis_params_paths = {}

# reconstruct the 4D nifti files for each subject
for i, sub in enumerate(setup.subjects):

    print(f"[{sub}] : Saving 4D timeseries for cond {list(condition_names)}")
    cond_names = list(extracted_volumes_per_cond[i].keys())
    sub_data = extracted_volumes_per_cond[i]
    sub_nuis = nuis_reg_per_cond[i]
    subject_folder = os.path.join(save_dir_4D, sub)
    os.makedirs(subject_folder, exist_ok=True) 
    func_cond_paths = []
    sub_imgs_shape = []

    for ncond, cond in enumerate(condition_names):

        nifti_imgs = sub_data[cond]
        nscans = nifti_imgs.shape[-1]
        print('imgs having shape :', nifti_imgs.shape)
        nifti_save_path = os.path.join(subject_folder, f"{cond}_{nscans}-vol.nii.gz")
        nifti_imgs.to_filename(nifti_save_path)
        func_cond_paths.append(nifti_save_path)

    # Save nuisance regressors
    sub_nuis_reg = sub_nuis[cond]
    # n_reg = sub_nuis_reg.shape[0]
    nuis_reg_save_path = os.path.join(subject_folder, f"mvmnt_reg_dct_{len(condition_names)}conds.pkl")
    utils.save_pickle(nuis_reg_save_path, nuis_reg_per_cond[i])
    func_imgs_paths[sub] = func_cond_paths 
    nuis_params_paths[sub] = nuis_reg_save_path

data.func_imgs_paths = func_imgs_paths
data.nuis_params_paths = nuis_params_paths

# %%
# save data
save_data_p = os.path.join(setup.save_dir, 'data_info_regressors.pkl')
utils.save_pickle(save_data_p, data)

# %% [markdown]
# Quality check
### Carpet plot etc. 
# %%
mean_img = False

if mean_img:
    # mean images for each condition and plot
    mean_img_dir = os.path.join(setup.save_dir, "mean_activation")
    os.makedirs(mean_img_dir, exist_ok=True)

    # Compute mean image per condition
    for condition in condition_names:

        img_list = [subject_imgs[condition] for subject_imgs in extracted_volumes_per_cond if condition in subject_imgs]
        mean_condition_img = mean_img(img_list)
        display = plot_stat_map(
            mean_condition_img,
            title=f"Mean Image - {condition}",
            threshold=1e-3,
            dim=0.8,
            display_mode='mosaic'

        )
        mean_image_plot_path = os.path.join(mean_img_dir, f"mean_img_{condition}.png")
        display.savefig(mean_image_plot_path)
        plt.close()

        print(f"Saved mean image plot for {condition} at {mean_image_plot_path}")
    print(f"\nAll mean images saved in {mean_img_dir}")
# %%
# ========================
# %% [markdown]
# QC check
from src import qc_utils

do_carptet_plot = True
if do_carptet_plot:

    # Atlas and mask resample  reports
    from nilearn.image import index_img
    
    # reload(qc_utils)

    # assert afine of data
    for cond in condition_names:
        func_cond = [subject_imgs[cond] for subject_imgs in extracted_volumes_per_cond]
        same_aff = qc_utils.assert_same_affine(func_cond, setup.subjects)
    if same_aff:
        ref_img = index_img(extracted_volumes_per_cond[0][condition_names[0]], 0)
        data_shape = ref_img.shape[0:3] #3D
        data_affine = ref_img.affine
        print(f"Data shape : {data_shape}, Data affine : {data_affine}")

    # %%
    from nilearn.image import resample_to_img, binarize_img
    from nilearn.plotting import plot_roi
    from nilearn.datasets import fetch_atlas_schaefer_2018
    from nilearn import datasets
   

    data_mask = nib.load('/data/rainville/Hypnosis_ISC/masks/brainmask_91-109-91.nii')
    data_mask = datasets.load_mni152_brain_mask()

    _ = qc_utils.assert_same_affine(func_cond, setup.subjects, data_mask)
    plot_roi(data_mask, title='data_mask', display_mode='ortho', draw_cross=False)

    # %%
    resamp_mask = qc_utils.resamp_to_img_mask(data_mask, ref_img)
    _ = qc_utils.assert_same_affine(func_cond, setup.subjects, resamp_mask)
    resamp_mask.to_filename(os.path.join(setup.save_dir, 'resampled_to_data_MNI_mask.nii.gz'))
    plot_roi(resamp_mask, bg_img = ref_img, title='resampled_mask', display_mode='ortho', draw_cross=False)

    lan800_prob = nib.load(os.path.join(setup.project_dir, 'masks/lipkin2022_lanA800', 'LanA_n806.nii'))
    lan800_mask = binarize_img(lan800_prob, threshold=.30)
    resamp_lan800 = qc_utils.resamp_to_img_mask(lan800_mask, ref_img)
    plot_roi(resamp_lan800,bg_img=ref_img, title='lan800', display_mode='ortho', draw_cross=False)

    atlas_data = fetch_atlas_schaefer_2018(n_rois = 200, resolution_mm=2)
    atlas_native = nib.load(atlas_data['maps'])
    atlas = qc_utils.resamp_to_img_mask(atlas_native, ref_img)
    plot_roi(atlas,bg_img=ref_img, title='atlas', display_mode='ortho', draw_cross=False)
    _ = qc_utils.assert_same_affine(func_cond, setup.subjects, atlas)

    #%%
    #==================
    # plot ROI timserie 

    from nilearn.maskers import NiftiLabelsMasker

    labels = atlas_data['labels']
    labels = [label.decode('utf-8') if isinstance(label, bytes) else label for label in labels]
    
    roi_name = '7Networks_LH_SomMot_2' # L STG, high ISC
    roi_idx = labels.index(roi_name) 
    roi_ts_filenames = []

    masker = NiftiLabelsMasker(
        labels_img=atlas,
        standardize=True,
        detrend=False,
        t_r=3,
        standardize_confounds = True,
        memory_level=0,
        verbose=0,
    )
    # masker.fit(atlas)

    ts_path = os.path.join(setup.save_dir, 'roi_timecourse')
    os.makedirs(ts_path, exist_ok=True)

    for cond in condition_names:

        plt.figure(figsize=(12, 6))
        plt.title(f"Time series for {roi_name} - {cond}")
        
        for sub_idx, sub in enumerate(setup.subjects):

            # Get time series for this subject and condition
            ts = masker.fit_transform(
                extracted_volumes_per_cond[sub_idx][cond],
                confounds=nuis_reg_per_cond[sub_idx][cond]
            )
            roi_ts = ts[:, roi_idx]
            plt.plot(roi_ts, label=f"sub-{sub}")

        plt.xlabel("Time (TRs)")
        plt.ylabel("Signal (a.u.)")
        plt.legend()
        plt.tight_layout()
        file_path = os.path.join(ts_path, f"{roi_name}_{cond}_time_series.png")
        plt.savefig(file_path)
        roi_ts_filenames.append(file_path)
        plt.show()
    
   #save grid images for all conditions
    n_subs = len(setup.subjects)
    visu_utils.plot_images_grid(roi_ts_filenames, f'STG {n_subs} timecourse', save_to=os.path.join(ts_path, f"grid_view_roi_timecourse_{n_subs}subs.png"))

        
    # %%
    #==================================
    # Carpet plot unprocessed timseries
    from nilearn.plotting import plot_carpet

    MAX_ITER = len(condition_names)
    MASK = resamp_lan800

    qc_path = os.path.join(setup.save_dir, 'QC_carpet')
    carpet_files_per_cond = {}
    os.makedirs(qc_path, exist_ok=True)

    for cond in condition_names[:MAX_ITER]:

        qc_cond_path = os.path.join(qc_path, cond)
        os.makedirs(qc_cond_path, exist_ok=True)
        carpet_files_per_cond[cond] = []
        
        for sub, subject in enumerate(setup.subjects):

        # for cond in condition_names:
            file_path = os.path.join(qc_cond_path, f'{subject}_{cond}_carpet_detrend.png')
            imgs = extracted_volumes_per_cond[sub][cond]
            display = plot_carpet(
                imgs,
                mask_img=MASK,
                detrend=True,
                t_r=3,
                standardize=True,
                title=f"global patterns {subject} in cond {cond}",
            )
            display.savefig(file_path) 
            carpet_files_per_cond[cond].append(file_path)
            display.show()

    for cond in condition_names[:MAX_ITER]:
        n_subs = len(setup.subjects)
        visu_utils.plot_images_grid(carpet_files_per_cond[cond], f'Carpet plot {cond}, {n_subs} subj.', save_to=os.path.join(qc_path, f"grid_view_carpet_{cond}_{n_subs}subs.png"))

    # %% [markdown]
    # ======================
    # Fit transform images to get residuals (rm confounds)
    regress_mvmnt_carpet = True
    MAX_ITER = len(condition_names) 
    MASK = resamp_lan800

    if regress_mvmnt_carpet:
        transformed_dir = os.path.join(setup.save_dir, "transformed_2d_imgs")
        os.makedirs(transformed_dir, exist_ok=True)

        # exclude_subject = 'sub-02'
        masker_params_dict.update({'mask_img': MASK})
        masker = NiftiMasker( **masker_params_dict, dtype='float64', memory_level = 0)

        sub_timeseries_all_cond = {}
        fitted_maskers = {}

        for i, cond in enumerate(condition_names[:MAX_ITER]):
            sub_timeseries = []
            sub_maskers = []
            gc.collect()

            for i, sub in enumerate(setup.subjects):
                print(sub)
                # if sub == exclude_subject:  # Skip sub-02
                #     print(f"Skipping {sub}")
                #     continue 

                sub_imgs = extracted_volumes_per_cond[i][cond]
                sub_imgs_data = sub_imgs.get_fdata().astype(np.float32)
                sub_imgs_float32 = nib.Nifti1Image(sub_imgs_data, sub_imgs.affine, sub_imgs.header)

                sub_reg = nuis_reg_per_cond[i][cond]
                ts = masker.fit_transform(sub_imgs_float32, confounds=sub_reg)
                print(ts.shape) # cmt
                sub_timeseries.append(ts)
                sub_maskers.append(masker)

            sub_timeseries_all_cond[cond] = sub_timeseries # np.stack(sub_timeseries, axis=-1) #TR x ROI x subjects 
            fitted_maskers[cond] = sub_maskers
            # print(sub_timeseries_all_cond[cond].shape)

            cond_path = os.path.join(transformed_dir, f'{cond}_{i+1}sub.npz')

            # np.savez_compressed(cond_path, sub_timeseries_all_cond[cond])
        ncond = len(condition_names)
        masker_path = os.path.join(transformed_dir, f'maskers_dct_all_{ncond}cond_{i}sub.pkl')
        # utils.save_pickle(masker_path, fitted_maskers)

        # %%
        # Plot carpet on inverse transformed data with regressed movement

        qc_path_reg = os.path.join(setup.save_dir, 'QC_carpet_regressed_conf')
        os.makedirs(qc_path_reg, exist_ok=True)
        reg_carpet_files_per_cond = {}

        for i, cond in enumerate(condition_names[:MAX_ITER]):

            qc_cond_path = os.path.join(qc_path_reg, cond)
            os.makedirs(qc_cond_path, exist_ok=True)
            reg_carpet_files_per_cond[cond] = []
            
            for sub, subject in enumerate(setup.subjects):

                file_path = os.path.join(qc_cond_path, f'{subject}_{cond}_reg-carpet_detrend.png')
                masker_i = fitted_maskers[cond][sub]
                imgs = masker_i.inverse_transform(sub_timeseries_all_cond[cond][sub]) #[:,:,i])
                display = plot_carpet(
                    imgs,
                    mask_img=MASK,
                    detrend=False,
                    t_r=3,
                    standardize=True,
                    title=f"global patterns {subject} in cond {cond}",
                )
                display.savefig(file_path) 
                reg_carpet_files_per_cond[cond].append(file_path)
                display.show()

        # %%
        for cond in condition_names[:MAX_ITER]:
            n_subs = len(setup.subjects)
            visu_utils.plot_images_grid(reg_carpet_files_per_cond[cond],
                                        f'Carpet plot {cond}, {n_subs} subj.',
                                        save_to=os.path.join(qc_path_reg, f"grid_view_carpet_{cond}_{n_subs}subs.png"))
    # %%
    # %% 
# save data


# %% [markdown]
# # Parcellation and ROI extraction

# %% [markdown]
# ## Quality check on ROI timeseries

# %% [markdown]
# #### **Probabilistic atlas**
#----------------------------
# %%


# #  atlas-related variables
# atlas = {
#     "path": os.path.join(setup.project_dir, "masks", "k50_2mm.nii/", "*.nii"),
#     "name": "k50_2mm-parcel",
#     "masker_params": masker_params_dict,
# }

# atlas["img"] = load_img(atlas["path"])

# # Plot the atlas ROI
# plot_roi(atlas["img"], title=atlas["name"])
# print(np.unique(atlas["img"].get_fdata(), return_counts=True))

# # Prepare masker using the parameters in the dictionary
# atlas["labels"] = np.unique(atlas["img"].get_fdata() + 1)
# atlas["labels"] = atlas["labels"][atlas["labels"] != 0]
# atlas['labels'] = np.arange(1, 46)
# atlas["MultiMasker"] = NiftiLabelsMasker(
#     atlas["img"], labels=atlas["labels"]
#     )
# atlas['save'] = os.path.join(setup.save_dir,atlas['name'])

# # Print masker parameters for verification
# print(atlas["MultiMasker"].get_params())

print('Done with all!!')
# %% [markdown]
# ##### DiFuMo64

# %%
# atlas_path = "/data/rainville/dSutterlin/projects/ISC_hypnotic_suggestions/masks/DiFuMo256/3mm/maps.nii.gz"
# atlas_dict_path = "/data/rainville/dSutterlin/projects/ISC_hypnotic_suggestions/masks/DiFuMo256/labels_256_dictionary.csv"
# difumo = nib.load(atlas_path)
# atlas_df = pd.read_csv(atlas_dict_path)
# print("atlas loaded with N ROI : ", difumo.shape[-1])


# %% [markdown]
# ## Apply parcellation and plot reports
# 1) group
# 2) individual report + ROI x TRs heatmaps

# %% [markdown]
# #### 1) Group masker fitting on atlas

# %% [markdown]
# #### Apply group label masker and generate report
