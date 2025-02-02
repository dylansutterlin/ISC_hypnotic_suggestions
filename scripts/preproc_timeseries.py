
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
import preproc_utils as utils
from nilearn.image import index_img, concat_imgs
from nilearn.glm.first_level import FirstLevelModel
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
nsub = 23 #len(subjects)

setup = Bunch(
    data_dir="/data/rainville/Hypnosis_ISC/4D_data/full_run",
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
    APM_subjects=["APM" + sub[4:] for sub in subjects][0:nsub],  # Format to APMXX
    subjects=subjects[0:nsub],
    project_dir="/data/rainville/dSutterlin/projects/ISC_hypnotic_suggestions",
    results_dir=os.path.join(
        "/data/rainville/dSutterlin/projects/ISC_hypnotic_suggestions",
        "results/imaging/preproc_data",
    ),  # Path to results
)

voxel_masker = Bunch(name="voxel_wise")

# Preprocessing parameters
parcel_name = "voxel_wise"
atlas_name = "k50-2mm-parcel"
# parcel_type = 'labelsMasker'
mask = '/data/rainville/Hypnosis_ISC/masks/brainmask_91-109-91.nii'
masker_params_dict = {
    "standardize": 'zscore_sample',
    "mask_img": mask,
    "detrend": True,
    "low_pass": None,
    "high_pass": None,  # 1/428 sec.
    "t_r": 3,
    "smoothing_fwhm": None,
    "standardize_confounds": True,
    "verbose": 3,
    "high_variance_confounds": True,
    "mask_strategy": "whole-brain-template",  # ignore for atlas maskers
}

# initialize preproc model and save dirs
preproc_model_name = "model2_{}subjects_{}_detrend_{}".format(
    len(setup.subjects),masker_params_dict['standardize'], datetime.today().strftime("%d-%m-%y")
)
save_dir = os.path.join(setup.results_dir, preproc_model_name)
if os.path.exists(save_dir):
    print(f"The folder '{save_dir}' already exists. Results will be overwritten!!.")
    error = input("Do you want to continue? (y/n)")
else:
    os.makedirs(save_dir, exist_ok=True)

setup.save_dir = save_dir
# os.makedirs(os.path.join(save_dir, "voxel_wise"), exist_ok=True)
# save masker parameters
masker_params_path = os.path.join(save_dir, "preproc_params.json")
with open(masker_params_path, "w") as f:
    json.dump(masker_params_dict, f, indent=4)

print(f"Directory {save_dir} created.")
print(f"Masker parameters saved to {masker_params_path}")

# %% [markdown]
# # GLM and data structure

# %% [markdown]
# #### Import and load data!

# %% [markdown]
# #### Extract timestamps/events, movement files based on APM_subjects
# %%
APM_subjects = ['APM' + sub[4:] for sub in setup.subjects][0:nsub] # make APMXX format instead of subXX
conditions = ["Analgesia", "Hyperalgesia"]
events_dir = "/data/rainville/HYPNOSIS_IMAGING_DATA/timestamps"
confound_dir = "/data/rainville/HYPNOSIS_IMAGING_DATA/Nii"

# Add to bunch
setup.conditions = conditions
setup.events_dir = events_dir
setup.confound_dir = confound_dir

events_ana = [
    utils.get_timestamps(
        data_dir, sub, events_dir, conditions[0], return_path=False
    ).sort_values(by="onset")
    for sub in APM_subjects
]
# events_ana_dct = {
#     sub: utils.get_timestamps(
#         data_dir, sub, events_dir, conditions[0], return_path=False
#     ).sort_values(by="onset")
#     for sub in APM_subjects
# }
# prefered dict but less easy to index
events_hyper = [
    utils.get_timestamps(
        data_dir, sub, events_dir, conditions[1], return_path=False
    ).sort_values(by="onset")
    for sub in APM_subjects
]
# events_hyper_dct = {
#     sub: utils.get_timestamps(
#         data_dir, sub, events_dir, conditions[1], return_path=False
#     ).sort_values(by="onset")
#     for sub in APM_subjects
# }

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

setup.events_dfs = {}
setup.events_dfs["Ana"] = events_ana
setup.events_dfs["Hyper"] = events_hyper
setup.confound_files = confound_files

# %% [markdown]
# Ensure same length and order of all variables

# %% [markdown]
# #### Split confound file based on run scans (up = Ana run, lower = Hyper run)

# %%
# Get the number of scans for each subject
setup.nscans = {}
setup.nscans["Ana"] = {
    sub: nib.load(img).shape[-1] for sub, img in zip(subjects, setup.ana_run)
}
setup.nscans["Hyper"] = {
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

    nscan_ana = setup.nscans["Ana"][sub]
    up_ana = confounds[0:nscan_ana, :]

    nscan_hyper = setup.nscans["Hyper"][sub]
    low_hyper = confounds[nscan_ana : nscan_ana + nscan_hyper, :]

    print("low_ana shape:", up_ana.shape)
    print("low_hyper shape:", low_hyper.shape)

    ana_confounds.append(up_ana)
    hyper_confounds.append(low_hyper)

    # up, low = split_conf_matrix(confound_files[i], nscan, run_names[i]
    #    )
setup.confound_files = confound_files
setup.confounds_Ana = ana_confounds
setup.confounds_Hyper = hyper_confounds

# %% [markdown]
# #### Process voxel-wise utils images/timeseries

# %%
# create folders for voxel wise model : e.g. /model/voxel_wise
# voxel_masker.obj = NiftiMasker(**masker_params_dict)
# voxel_masker.save = os.path.join(setup.save_dir, voxel_masker.name)

# ana_masked_timeseries, fitted_masks_ana = (
#     utils.extract_timeseries_and_generate_individual_reports(
#         setup.subjects,
#         setup.ana_run,
#         voxel_masker.obj,
#         voxel_masker.name,
#         voxel_masker.save,
#         confounds=setup.confounds_Ana,
#         confounf_files=confound_files,
#         condition_name="Analgesia",
#         do_heatmap=False,
#     )
# )

# hyper_masked_timeseries, fitted_masks_hyper = (
#     utils.extract_timeseries_and_generate_individual_reports(
#         setup.subjects,
#         setup.hyper_run,
#         voxel_masker.obj,
#         voxel_masker.name,
#         voxel_masker.save,
#         confounds=setup.confounds_Hyper,
#         confounf_files=confound_files,
#         condition_name="Hyperalgesia",
#         do_heatmap=False,
#     )
# )

# # store 2D timeseries in subject's dict.
# voxel_masker.preproc_2d_Ana = ana_masked_timeseries  # {sub : ts for sub, ts in zip(subjects, ana_masked_timeseries)}
# voxel_masker.preproc_2d_Hyper = hyper_masked_timeseries  # {sub : ts for sub, ts in zip(subjects, hyper_masked_timeseries)}

# voxel_masker.fitted_mask_Ana = (
#     fitted_masks_ana  # {sub : mask for sub, mask in zip(subjects, fitted_masks_ana)}
# )
# voxel_masker.fitted_mask_Hyper = fitted_masks_hyper  # {sub : mask for sub, mask in zip(subjects, fitted_masks_hyper)}

# voxel_masker.preproc_2d_cond_Descrip = "preproc 2d timeseries for ANA and HYPER conditions and fitted masks for all volumes : 372-377 TRs/subject"

# # Combine the timeseries for both runs
# all_run_2d_timeseries = [
#     np.vstack([ts1, ts2])
#     for ts1, ts2 in zip(voxel_masker.preproc_2d_Ana, voxel_masker.preproc_2d_Hyper)
# ]
# print(len(all_run_2d_timeseries), all_run_2d_timeseries[0].shape)

# ana_filename = os.path.join(voxel_masker.save, '{}_2d_timeseries_{}sub_Ana.npz'.format(voxel_masker.name, len(subjects)))
# hyper_filename = os.path.join(voxel_masker.save, '{}_2d_timeseries_{}sub_Hyper.npz'.format(voxel_masker.name, len(subjects)))

# #unpack dict in npz file
# np.savez_compressed(ana_filename, **voxel_masker.preproc_2d_Ana )
# np.savez_compressed(hyper_filename, **voxel_masker.preproc_2d_Hyper)

# print(f"Saved ANA timeseries to: {ana_filename}")
# print(f"Saved HYPER timeseries to: {hyper_filename}")

# ana_fitted_mask_path = os.path.join(results_dir, 'firstLevel/Ana_fittedMaskers_{}_{}-subjects.pkl'.format(parcel_name, len(subjects)))
# hyper_fitted_mask_path = os.path.join(results_dir, 'firstLevel/Hyper_fittedMaskers_{}_{}-subjects.pkl'.format(parcel_name,len(subjects)))

# with open(ana_fitted_mask_path, 'wb') as f:
#     pkl.dump(fitted_mask_ana, f)

# with open(hyper_fitted_mask_path, 'wb') as f:
#     pkl.dump(fitted_mask_hyper, f)

# print(f"Saved ANA fitted maskers to: {ana_fitted_mask_path}")
# print(f"Saved HYPER fitted maskers to: {hyper_fitted_mask_path}")


# %% [markdown]
# ### Check TRs per condition

# %% [markdown]
# #### utilstion to count TRs for suggestion conditions, for neutral and modulation suggestion. Added to utils.py file

# %% [markdown]
# #### Number of TRs (total condition duration / TR) TR=3 in this dataset

# %%
len(events_hyper), setup.save_dir


# hyper condition
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
run_names = ["Ana", "Hyper"]
# Create a dictionary to store combined design matrices for all subjects
design_matrices_2runs = {}
design_matrices_2runs_files = []
# Directory to save design matrices
save_path = os.path.join(setup.save_dir, "design_matrices_TxT")
os.makedirs(save_path, exist_ok=True)

for i, sub in enumerate(setup.subjects):
    tr = masker_params_dict["t_r"]
    names = setup.conditions

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
    # Create design matrices for each run
    dm_ana = make_first_level_design_matrix(
        frame_time_ana,
        setup.events_dfs["Ana"][i],
        hrf_model="spm",
        drift_model="cosine",
        high_pass=masker_params_dict["high_pass"],
        add_regs=setup.confounds_Ana[i],
    )

    dm_hyper = make_first_level_design_matrix(
        frame_time_hyper,
        setup.events_dfs["Hyper"][i],
        hrf_model="spm",
        drift_model="cosine",
        high_pass=masker_params_dict["high_pass"],
        add_regs=setup.confounds_Hyper[i],
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

    # Save combined design matrix as CSV
    csv_name = f'{sub}-{"-".join(run_names)}-design_matrix.csv'
    dm_combined_csv_path = os.path.join(save_path, csv_name)
    dm_combined.to_csv(dm_combined_csv_path)
    design_matrices_2runs_files.append(dm_combined_csv_path)

    # Save combined design matrix as PNG
    png_name = f'{sub}-{"-".join(run_names)}-design_matrix.png'
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

setup.design_mat_2runs_files = design_matrices_2runs_files

#%%
# return design_matrices_2runs
# from nilearn.glm.first_level import FirstLevelModel

# for i, sub in enumerate(setup.subjects):

#     # Load 4D functional images and concatenate
#     ana_img = nib.load(setup.ana_run[i])  # Analgesia run
#     hyper_img = nib.load(setup.hyper_run[i])  # Hyperalgesia run
#     full_4d_img = concat_imgs([ana_img, hyper_img])  # Concatenate both runs

# # Initialize the GLM model (you probably already did this in your pipeline)
# fmri_glm = FirstLevelModel()  # Adjust t_r as per your TR value
# fmri_glm = fmri_glm.fit(setup.ana_run[i], design_matrices=dm_combined[i])  # Replace `fmri_img` with your actual fMRI data

# %% [markdown]
# ### From DMs, Extract TRs for each condition, concatenate and save
# Combine the timeseries for both runs

# %%
# conditions = ['ANA', 'N_ANA', 'HYPER', 'N_HYPER']
regressors_names = ['ANA_sugg', 'N_ANA_sugg', 'HYPER_sugg', 'N_HYPER_sugg',
                    'ANA_shock', 'N_ANA_shock', 'HYPER_shock', 'N_HYPER_shock']
# Extract TRs indices for regressors of all subjects (dict for each sub)
# reload(utils)
indices_all_cond = []
kept_columns = []
full_confounds = [np.vstack([ana, hyper]) for ana, hyper in zip(setup.confounds_Ana, setup.confounds_Hyper)]

for sub in setup.subjects:
    print(f"==Extracting TRs indices for regr essors of {sub}==")
    dm_combined = design_matrices_2runs[sub]

    cond_indices = {}
    kept_col = {}
    for cond in regressors_names:
        if 'sugg' in cond:
            cond_indices[cond], kept_col[cond] = utils.create_tr_masks_for_suggestion(
                dm_combined, regressor=cond
            )
        elif 'shock' in cond:
            cond_indices[cond], kept_col[cond] = utils.create_tr_masks_for_shock(
                dm_combined, regressor=cond
            )

        indices_all_cond.append(cond_indices) # list of dict per sub
        kept_columns.append(kept_col)

# %%
# Visualize segmented events
for cond in regressors_names:

    test_shock = kept_col[cond]
    ind_shock = cond_indices[cond]
    extracted_timeseries = dm_combined[list(test_shock)]

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
    plt.show()


# %% [markdown]
## Get timeseries for each cond indices

#%%
extracted_volumes_per_cond = []
nuis_reg_per_cond = []

for i, sub in enumerate(setup.subjects):
    print(f"Processing subject {sub}: Extracting NIfTI volumes for {regressors_names}")

    # Load 4D functional images and concatenate
    ana_img = nib.load(setup.ana_run[i])  # Analgesia run
    hyper_img = nib.load(setup.hyper_run[i])  # Hyperalgesia run
    full_4d_img = concat_imgs([ana_img, hyper_img])  # Concatenate both runs

    # Dictionary to store sliced imgs per condition
    sliced_imgs_per_cond = {}
    sliced_confounds_per_cond = {}

    for cond in regressors_names:
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

print(f"Extracted NIfTI volumes and nuisance regressors for all subjects: {regressors_names}")
# %% [markdown]
# #### Save concatenated timeseries for each condition
#
# output func_imgs_paths : list of dict contan

# %%

# Save the 4D NIfTI files for each condition
save_dir_4D = os.path.join(setup.save_dir, f"extracted_4D_per_cond_{nsub}sub")
os.makedirs(save_dir_4D, exist_ok=True)
func_imgs_paths = {}
nuis_params_paths = {}
# reconstruct the 4D nifti files for each subject
for i, sub in enumerate(setup.subjects):
    print(f"[{sub}] : Saving 4D timeseries for cond {list(regressors_names)}")
    cond_names = list(extracted_volumes_per_cond[i].keys())
    sub_data = extracted_volumes_per_cond[i]
    sub_nuis = nuis_reg_per_cond[i]
    subject_folder = os.path.join(save_dir_4D, sub)
    os.makedirs(subject_folder, exist_ok=True) 

    func_cond_paths = []
    sub_imgs_shape = []

    for ncond, cond in enumerate(regressors_names):
        nifti_imgs = sub_data[cond]
        nscans = nifti_imgs.shape[-1]
        nifti_save_path = os.path.join(subject_folder, f"{cond}_{nscans}-vol.nii.gz")
        nifti_imgs.to_filename(nifti_save_path)
        
        func_cond_paths.append(nifti_save_path)

    # Save nuisance regressors
    sub_nuis_reg = sub_nuis[cond]
    # n_reg = sub_nuis_reg.shape[0]
    nuis_reg_save_path = os.path.join(subject_folder, f"mvmnt_reg_dct_{len(regressors_names)}conds.pkl")
    utils.save_data(nuis_reg_save_path, nuis_reg_per_cond[i])

    func_imgs_paths[sub] = func_cond_paths 
    nuis_params_paths[sub] = nuis_reg_save_path
setup.func_imgs_paths = func_imgs_paths
setup.nuis_params_paths = nuis_params_paths

# %%
# mean images for each condition and plot
mean_img_dir = os.path.join(setup.save_dir, "mean_activation")
os.makedirs(mean_img_dir, exist_ok=True)

# Compute mean image per condition
for condition in regressors_names:
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
#%%
# Carpet plot unprocessed timseries
from nilearn.plotting import plot_carpet

qc_path = os.path.join(setup.save_dir, 'QC_carpet')
os.makedirs(qc_path, exist_ok=True)
for sub, subject in enumerate(setup.subjects):
    for cond in regressors_names:
        file_path = os.path.join(qc_path, f'{subject}_{cond}_carpet_detrend.png')
        imgs = extracted_volumes_per_cond[sub][cond]
        display = plot_carpet(
            imgs,
            detrend=True,
            t_r=3,
            standardize=True,
            title=f"global patterns {subject} in cond {cond}",
        )
        display.savefig(file_path) 
        display.show()

# %% [markdown]
# ======================
# Fit transform images
transformed_dir = os.path.join(setup.save_dir, "transformed_2d_imgs")
os.makedirs(transformed_dir, exist_ok=True)
exclude_subject = 'sub-02'
# for eachsubjects, fit transform the 4D images and stack in the -1 dim the arrays, save to npz
masker = NiftiMasker(**masker_params_dict)

sub_timeseries_all_cond = {}
fitted_maskers = {}
for cond in regressors_names:
    sub_timeseries = []
    sub_maskers = []
    for i, sub in enumerate(setup.subjects):
        if sub == exclude_subject:  # Skip sub-02
            print(f"Skipping {sub}")
            continue 
        sub_imgs = extracted_volumes_per_cond[i][cond]
        sub_reg = nuis_reg_per_cond[i][cond]
        ts = masker.fit_transform(sub_imgs, confounds=sub_reg)
        sub_timeseries.append(ts)
        sub_maskers.append(masker)
    sub_timeseries_all_cond[cond] = np.stack(sub_timeseries, axis=-1) #TR x ROI x subjects 
    fitted_maskers[cond] = sub_maskers
    print(sub_timeseries_all_cond[cond].shape)

    cond_path = os.path.join(transformed_dir, f'{cond}_{i+1}sub.npz')
    np.savez_compressed(cond_path, sub_timeseries_all_cond[cond])
ncond = len(regressors_names)
masker_path = os.path.join(transformed_dir, f'maskers_dct_all_{ncond}cond_{i}sub.pkl')
utils.save_data(masker_path, fitted_maskers)
# %%
# Plot carpet on inverse transformed data with regressed movement

qc_path_reg = os.path.join(setup.save_dir, 'QC_carpet_regressed_conf')
os.makedirs(qc_path_reg, exist_ok=True)
for sub, subject in enumerate(setup.subjects):
    for cond in regressors_names:
        file_path = os.path.join(qc_path, f'{subject}_{cond}_carpet_detrend.png')
        imgs = extracted_volumes_per_cond[sub][cond]
        display = plot_carpet(
            imgs,
            detrend=True,
            t_r=3,
            standardize=True,
            title=f"global patterns {subject} in cond {cond}",
        )
        display.savefig(file_path)  # Save manually
        display.show()

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

# %%
'''
#%%
importlib.reload(func)

ROI_timseries_dct = {}
fitted_ROImasks = []
for condition, img_paths in condition_concat_imgs.items():
    print(f"\nComputing mean image for condition: {condition}")

    ROI_timeseries, fitted_ROImasks = utils.extract_timeseries_and_generate_individual_reports(
        setup.subjects,
        img_paths,
        atlas['MultiMasker'],
        atlas['name'],
        atlas['save'],
        confounds=None, #already preprocessed
        condition_name="Analgesia",
        do_heatmap=False,
    )

    ROI_timseries_dct[condition] = ROI_timeseries
    fitted_ROImasks.append(fitted_ROImasks)
    
    mean_condition_img = mean_img(fitter_ROImasks.inverse_transform(ROI_timeseries))

    # Plot and save the mean image
    display = plot_stat_map(
        mean_condition_img,
        title=f"{atlas['name']} {condition} mean image",
        display_mode="mosaic",
        threshold=1e-3,
        dim=0.8,
    )

    # html_view = view_img(mean_condition_img, threshold='98%', cut_coords=[-42, -16, 52])

    mean_image_path = os.path.join(
        atlas['save'],
        f"{atlas['name']}-mean_img_{condition}_{len(img_paths)}-subjects.png",
    )
    display.savefig(mean_image_path)
    plt.show()
    plt.close()






# %%
importlib.reload(func)

# Create directory for reports


utils.generate_multinifti_report(
    setup.ana_run, atlas["img"], atlas["name"], report_dir, condition_name="Analgesia"
)
utils.generate_multinifti_report(
    setup.hyper_run, atlas["img"], atlas["name"], report_dir, condition_name="Hyperalgesia"
)


# %% [markdown]
# #### 2) Individual atlas report (label masker)
# -->Save ROI x TRs heatmap + individual LabelMasker report in /results/cond/

# %% [markdown]
# #### Generate and save individual report and heatmaps for two runs

# %%

label_masker = NiftiLabelsMasker(mask_nifti, standardize=True, detrend=True)
masker_params = label_masker.get_params()

parcel_name = "k50-2mm-parcel"

ana_masked_timeseries, fitted_mask_ana = (
    utils.extract_timeseries_and_generate_individual_reports(
        setup.subjects,
        ana_run,
        label_masker,
        parcel_name,
        project_dir,
        condition_name="Analgesia",
    )
)

hyper_masked_timeseries, fitted_mask_hyper = (
    utils.extract_timeseries_and_generate_individual_reports(
        setup.subjects,
        hyper_run,
        label_masker,
        parcel_name,
        project_dir,
        condition_name="Hyperalgesia",
    )
)


# %% [markdown]
# ### Save masked timeseries and fitted masker

# %%
from importlib import reload

reload(func)

results_dir = os.path.join(project_dir, "results/imaging")
ana_ts_dict = {sub: ts for sub, ts in zip(setup.subjects, ana_masked_timeseries)}
hyper_ts_dict = {sub: ts for sub, ts in zip(setup.subjects, hyper_masked_timeseries)}

ana_filename = os.path.join(
    results_dir, "firstLevel/ana_masked_{}sub.npz".format(len(setup.subjects))
)
hyper_filename = os.path.join(
    results_dir, "firstLevel/hyper_masked_{}sub.npz".format(len(setup.subjects))
)

np.savez_compressed(ana_filename, **ana_ts_dict)
np.savez_compressed(hyper_filename, **hyper_ts_dict)

print(f"Saved ANA timeseries to: {ana_filename}")
print(f"Saved HYPER timeseries to: {hyper_filename}")

ana_fitted_mask_path = os.path.join(
    results_dir,
    "firstLevel/Ana_fittedMaskers_{}_{}-subjects.pkl".format(
        parcel_name, len(setup.subjects)
    ),
)
hyper_fitted_mask_path = os.path.join(
    results_dir,
    "firstLevel/Hyper_fittedMaskers_{}_{}-subjects.pkl".format(
        parcel_name, len(setup.subjects)
    ),
)

with open(ana_fitted_mask_path, "wb") as f:
    pkl.dump(fitted_mask_ana, f)

with open(hyper_fitted_mask_path, "wb") as f:
    pkl.dump(fitted_mask_hyper, f)

print(f"Saved ANA fitted maskers to: {ana_fitted_mask_path}")
print(f"Saved HYPER fitted maskers to: {hyper_fitted_mask_path}")


# %% [markdown]
# create a list containing all run timeseries 2d arrays for all subjects [sub_1(total TRs X ROi)...]

# %%
all_sugg_timeseries_per_sub = []
for i in range(len(setup.subjects)):
    ts1, ts2, ts3, ts4 = [extracted_timeseries_per_cond[i][cond] for cond in cond_names]
    all_sugg_timeseries_per_sub.append(np.vstack([ts1, ts2, ts3, ts4]))


# %%
importlib.reload(func)
import json

n_rois = all_run_2d_timeseries[0].shape[-1]
roi_names = [f"ROI-{idx}" for idx in range(n_rois)]

param_id = np.random.randint(0, 10000)
print(f"Parameter ID: {param_id}")
save_to = os.path.join(
    quality_check_save,
    "mean_timecourses-ROIs",
    f"{parcel_name}_mean_timecourses-paramId-{param_id}",
)
os.makedirs(save_to, exist_ok=True)

save_name = "mean_timecourses-ROIs_allSubj.png"
utils.plot_timecourses_from_ls(
    all_sugg_timeseries_per_sub, roi_names, save_to=False, n_rows=16, n_cols=3
)

# save masker params
masker_params_path = os.path.join(
    quality_check_save, f"{parcel_name}_masker_paramId-{param_id}.txt"
)
utils.write_masker_params(masker_params, masker_params_path)


# %% [markdown]
# Mean images

# %%
voxel_masker.save, voxel_masker.name  # keys()

# %%
# extract TRs for each condition
condition_keys = ["ANA", "N_ANA", "HYPER", "N_HYPER"]
condition = "N_ANA"


# %%
import pandas as pd
import matplotlib.pyplot as plt
from nilearn.glm.first_level import make_first_level_design_matrix
from nilearn.plotting import plot_design_matrix
import numpy as np


def create_design_matrix(events, confounds_file, TR, n_scans, title):
    """
    Create and plot a design matrix for one run.

    Parameters:
    -----------
    events : pd.DataFrame
        DataFrame containing onset, duration, and trial_type columns for the run.

    confounds_file : str
        Path to the confounds file for the run.

    TR : int
        Repetition time (TR) in seconds.

    n_scans : int
        Number of scans in the fMRI run.

    title : str
        Title for the design matrix plot.

    Returns:
    --------
    design_matrix : pd.DataFrame
        The design matrix for the run.

    fig : matplotlib.figure.Figure
        The figure object of the design matrix plot.
    """
    # Frame times for each TR
    frame_times = np.arange(n_scans) * TR

    # Create the design matrix
    design_matrix = make_first_level_design_matrix(
        frame_times=frame_times,
        events=events,
        add_regs=confounds.values,
        add_reg_names=confounds.columns,
    )

    # Plot the design matrix
    fig, ax = plt.subplots(figsize=(12, 6))
    plot_design_matrix(design_matrix, ax=ax)
    ax.set_title(title)
    plt.tight_layout()

    return design_matrix, fig


# Parameters for one subject
TR = 3  # Adjust based on your fMRI TR
n_scans_hyper = 200  # Adjust based on the number of scans in the hyper run
n_scans_ana = 180  # Adjust based on the number of scans in the analgesia run

subject_index = 0  # Replace with the desired subject index

# Paths to confounds
confounds_hyper = hyper_run[subject_index]
confounds_ana = ana_run[subject_index]

# Events for hyper and analgesia runs
events_hyper_subject = events_hyper[subject_index]
events_ana_subject = events_ana[subject_index]

# Create and plot the design matrix for the hyper run
dm_hyper, fig_hyper = create_design_matrix(
    events_hyper_subject,
    confounds_hyper,
    TR,
    n_scans_hyper,
    "Design Matrix for Hyper Run",
)

# Create and plot the design matrix for the analgesia run
dm_ana, fig_ana = create_design_matrix(
    events_ana_subject,
    confounds_ana,
    TR,
    n_scans_ana,
    "Design Matrix for Analgesia Run",
)

# Save figures
fig_hyper.savefig(f"design_matrix_hyper_subject_{subject_index + 1}.png", dpi=300)
fig_ana.savefig(f"design_matrix_ana_subject_{subject_index + 1}.png", dpi=300)


# %%
from nilearn.plotting import plot_prob_atlas

mask_nifti = mask_x.to_nifti()
view = plot_prob_atlas(
    mask_nifti, threshold=None
)  # Use threshold=None to display the whole mask
view

# %% [markdown]
# **Extract data in ROI**

# %%
# func_ana = Brain_Data(ana_files)
n_subj = 6
# limit computation time

roi = 4
roi_mask = mask_x[roi]

# file_list = glob.glob(os.path.join(data_dir, 'fmriprep', '*', 'func', f'*crop*{scan}*nii.gz'))
ana_data = []
for count, (sub, f) in enumerate(zip(setup.subjects, ana_files)):
    if count > n_subj:
        break
    else:
        print(sub)
        data = Brain_Data(f)
        ana_data.append(data.apply_mask(roi_mask))

# %% [markdown]
# **Hyper align**

# %%
ana_data

# %%
hyperalign = align(ana_data[:6], method="procrustes")

# %%
hyperalign

# %%
voxel_index = 50

voxel_unaligned = pd.DataFrame(
    [x.data[:, voxel_index] for x in ana_data]
).T  # x is subject i
voxel_aligned = pd.DataFrame(
    [x.data[:, voxel_index] for x in hyperalign["transformed"]]
).T

f, a = plt.subplots(nrows=2, figsize=(15, 5), sharex=True)
a[0].plot(voxel_unaligned, linestyle="-", alpha=0.2)
a[0].plot(np.mean(voxel_unaligned, axis=1), linestyle="-", color="navy")
a[0].set_ylabel("Unaligned Voxel", fontsize=16)
a[0].yaxis.set_ticks([])

a[1].plot(voxel_aligned, linestyle="-", alpha=0.2)
a[1].plot(np.mean(voxel_aligned, axis=1), linestyle="-", color="navy")
a[1].set_ylabel("Aligned Voxel", fontsize=16)
a[1].yaxis.set_ticks([])

plt.xlabel("Voxel Time Course (TRs)", fontsize=16)
a[0].set_title(
    f"Unaligned Voxel ISC: r={Adjacency(voxel_unaligned.corr(), matrix_type='similarity').mean():.02}",
    fontsize=18,
)
a[1].set_title(
    f"Aligned Voxel ISC: r={Adjacency(voxel_aligned.corr(), matrix_type='similarity').mean():.02}",
    fontsize=18,
)


# %% [markdown]
# **ISC distribution : N voxel pair-wise correlation, meaned**

# %% [markdown]
# Unaligned voxel ISC

# %%
ana_data[0].data.flatten().shape

# %%
ana_data[0].data.shape[1]

# %%
import numpy as np
from nltools.data import Adjacency

unaligned_isc = {}

# compute mean ISC for each unaligned voxel to plot distribution
for voxel_index in range(
    ana_data[0].data.shape[1]
):  # Assuming all_data is a list of Brain_Data objects
    # Extract the time series for this voxel across all subjects
    voxel_time_series = np.array(
        [x.data[:372, voxel_index] for x in ana_data]
    )  # shape (n_subjects, n_timepoints)
    voxel_corr = np.corrcoef(voxel_time_series)  # matrix shape (n_subjects, n_subjects)
    triu_indices = np.triu_indices_from(voxel_corr, k=1)  # mean of triangle
    unaligned_isc[voxel_index] = voxel_corr[triu_indices].mean()

plt.hist(unaligned_isc.values(), label="Unaligned ISC")
plt.axvline(
    x=np.mean(list(unaligned_isc.values())),
    linestyle="--",
    color="red",
    linewidth=2,
    label=f"Mean Unaligned ISC: {np.mean(list(unaligned_isc.values())):.2f}",
)
plt.ylabel("Frequency", fontsize=16)
plt.xlabel("Voxel ISC Values", fontsize=16)
plt.title("Unaligned ISC Distribution", fontsize=18)
plt.legend()
plt.show()

# Print the mean ISC value for unaligned data
print(f"Mean Unaligned ISC (voxel-wise): {np.mean(list(unaligned_isc.values())):.2f}")


# %%
plt.hist(hyperalign["isc"].values(), label="Aligned ISC")
plt.axvline(
    x=np.mean(list(hyperalign["isc"].values())),
    linestyle="--",
    color="red",
    linewidth=2,
)
plt.ylabel("Frequency", fontsize=16)
plt.xlabel("Voxel ISC Values", fontsize=16)
plt.legend()
plt.title("Hyperalignment ISC", fontsize=18)

print(f"Mean ISC: {np.mean(list(hyperalign['isc'].values())):.2}")

# %% [markdown]
# **Compare a slice of ROI**

# %%
tr_index = 100

f, a = plt.subplots(ncols=5, nrows=2, figsize=(15, 6), sharex=True, sharey=True)
for i in range(5):
    sns.heatmap(
        np.rot90(ana_data[i][tr_index].to_nifti().dataobj[30:60, 10:28, 37]),
        cbar=False,
        cmap="RdBu_r",
        ax=a[0, i],
    )
    a[0, i].set_title(f"Subject: {i+1}", fontsize=18)
    a[0, i].axes.get_xaxis().set_visible(False)
    a[0, i].yaxis.set_ticks([])
    sns.heatmap(
        np.rot90(
            hyperalign["transformed"][i][tr_index].to_nifti().dataobj[30:60, 10:28, 37]
        ),
        cbar=False,
        cmap="RdBu_r",
        ax=a[1, i],
    )
    a[1, i].axes.get_xaxis().set_visible(False)
    a[1, i].yaxis.set_ticks([])

a[0, 0].set_ylabel("Unaligned Voxels", fontsize=16)
a[1, 0].set_ylabel("Aligned Voxels", fontsize=16)

plt.tight_layout()

# %%
data_dir = "/data/rainville/Hypnosis_ISC/4D_data/segmented"

'''
