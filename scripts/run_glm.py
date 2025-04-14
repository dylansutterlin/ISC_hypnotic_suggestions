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
from nilearn import plotting
from nilearn.glm.thresholding import threshold_stats_img

# if os.getcwd().endswith('ISC_hypnotic_suggestions'):
#     print('Appending scripts/ to python path')
#     script_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../scripts'))
#     sys.path.append(script_dir)

# if os.getcwd().endswith('src'):
#     os.chdir(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
#     print('changed dir to ', os.getcwd())


print('current working dir : ', os.getcwd())
# %%
from src import preproc_utils, visu_utils, qc_utils
import src.glm_utils as utils

from sklearn.utils import Bunch
from importlib import reload
from nilearn import datasets
from datetime import datetime
# %% [markdown]
## load data
preproc_model_name = r'model2_23subjects_zscore_sample_detrend_25-02-25' #r'model2_3subjects_zscore_sample_detrend_25-02-25'
model_dir = rf'/data/rainville/dSutterlin/projects/ISC_hypnotic_suggestions/results/imaging/preproc_data/{preproc_model_name}'
model_name = "model3_23subjects_nuis_nodrift_{}".format(datetime.today().strftime("%d-%m-%y"))

model_name = 'model3_23subjects_nuis_nodrift_31-03-25' #final model, reproduced Desmarteaux et al., 2019 !! 31 mars

#load manually
# model_name = r'model3_23subjects_allnuis_nodrift_10-03-25'

#%%
setup_dct = preproc_utils.load_json(os.path.join(model_dir, 'setup_parameters.json'))
data_info_dct = preproc_utils.load_pickle(os.path.join(model_dir, 'data_info_regressors.pkl'))
masker_params = preproc_utils.load_json(os.path.join(model_dir, 'preproc_params.json'))

MAX_ITER = None

setup = Bunch(**setup_dct)
setup.run_id = ['ANA', 'HYPER']
data = Bunch(**data_info_dct)
masker_params = Bunch(**masker_params)
glm_info = Bunch()
subjects = setup.subjects.sort()

if MAX_ITER == None :
    MAX_ITER= len(subjects) 
else: 
    subjects = subjects[:MAX_ITER]
    setup.subjects = subjects   

regressors_dct = data.regressors_per_conds
condition_names = setup.run_names
tr = setup.tr

reload(utils)
ref_img = nib.load(setup.ana_run[0])
mask = utils.load_data_mask(ref_img)
mni_temp = datasets.load_mni152_template(resolution=None)
mni_bg = qc_utils.resamp_to_img_mask(mni_temp, mask)

save_glm = os.path.join(setup.project_dir, 'results', 'imaging', 'GLM', model_name)
os.makedirs(save_glm, exist_ok=True)
setup.save_dir = save_glm

# save_glm = os.path.join(setup.save_dir, 'GLM_results')
# os.makedirs(save_glm, exist_ok=True)

# %% [markdown]
## First level localizer
from importlib import reload
reload(utils)

# combined runs GLM. Pre fixed effect try march 7th 25
extra_regressors_idx = {
    'all_sugg': [0, 1, 2, 3],
    'all_shock': [4, 5, 6, 7]
}

contrasts_spec = {
    "ANA_sugg_minus_N_ANA_sugg": (1.5, -1),
    "HYPER_sugg_minus_N_HYPER_sugg": (1.5, -1),
    "HYPER_sugg_minus_ANA_sugg": (1, -1),
    "N_HYPER_sugg_minus_N_ANA_sugg": (1, -1),
    "ANA_sugg+HYPER_sugg_minus_N_ANA_sugg+N_HYPER_sugg": (1.5, -1),
    "N_ANA_shock_minus_N_HYPER_shock": (1, -1), # eq. N cond?
    "ANA_shock_minus_N_ANA_shock": (1, -1),
    "HYPER_shock_minus_N_HYPER_shock": (1, -1),
    "ANA_shock+HYPER_shock_minus_N_ANA_shock+N_HYPER_shock": (1, -1)
}

loc_contrasts_vectors = {}
first_level_models = {}
contrast_maps = {}
first_lev_files = {}

for sub, subject in enumerate(subjects):

    print(f"Processing {subject}...({sub+1}/{len(subjects)})")
    # design matrices
    dm_combined = pd.read_csv(data.design_mat_2runs_files[sub], index_col=0)
    nscans_ana = data.nscans["Ana"][subject]
    nscans_hyper = data.nscans["Hyper"][subject]
    
    # remove columns from design matrix
    rm_cols = []
    rm_cols = [col for col in dm_combined.columns if col.startswith('drift')]
    dm_combined = dm_combined.drop(rm_cols, axis=1)
   
    # plot design matrix but only for subject 1
    if sub == 0:
        plot_design_matrix(dm_combined, output_file=os.path.join(save_glm, 'sub1_design_matrix.png'))
        print(f"Removing columns: {rm_cols}")
    
    #Func images
    func1 = nib.load(setup.ana_run[sub])
    func2 = nib.load(setup.hyper_run[sub])
    func_imgs = concat_imgs([func1, func2])
    # make localizer contrasts
    if sub == 0: plot=True
    else: plot=False
    loc_contrasts_vectors[subject] = utils.make_localizer_vec_from_reg(dm_combined, regressors_dct,extra_indices_dct= extra_regressors_idx, plot = plot, save_to=save_glm)
    loc_contrasts_vectors[subject].update(utils.make_contrast_vec_from_reg(dm_combined, regressors_dct, contrasts_spec, plot=plot, save_to=save_glm))
  
    first_level_model = FirstLevelModel(t_r=tr, mask_img=mask, smoothing_fwhm=None, standardize=False, signal_scaling=False)
    first_level_model.fit(func_imgs, design_matrices=dm_combined, confounds=None) # already in dm

    sub_contrasts = {}
    contrasts_files = {}
    MAX_CONT = len(loc_contrasts_vectors[subject].keys())

    i = 0
    for condition, contrast_vector in loc_contrasts_vectors[subject].items():
        
        os.makedirs(os.path.join(save_glm, condition), exist_ok=True)

        # print(f"{condition}...")
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


# extra_regressors_idx = {
#     'all_sugg': [0, 1, 2, 3],
#     'all_shock': [4, 5, 6, 7]
# }

# %%

# cond = 'all_sugg'
# first_lev_files = glm_info.contrast_files_1level
# for subject in subjects:
#     img = first_lev_files[subject][cond]
#     display = plotting.plot_stat_map(
#         img,
#         title=f"first level stats unc. {cond}",
#         threshold=3.0,            
#         display_mode='ortho')     

#     plt.show()
#     plt.close()
# %% [markdown]
## Second level localizer
# %%

from nilearn.glm.second_level import SecondLevelModel
from nilearn.plotting import plot_stat_map, plot_glass_brain
from nilearn.glm.second_level import make_second_level_design_matrix
from nilearn.plotting import plot_design_matrix

# --- SECOND-LEVEL ANALYSIS FOR LOCALIZER EFFECTS ---

# group_dm = make_second_level_design_matrix(
#     subjects, confounds=None
# )
group_dm = pd.DataFrame({"Intercept": [1] * len(subjects)})

fig, ax1 = plt.subplots(1, 1, figsize=(3, 4), constrained_layout=True)

ax = plot_design_matrix(group_dm, axes=ax1)
ax.set_ylabel("maps")
ax.set_title("Second level design matrix", fontsize=12)
plt.show()

glm_second_save = os.path.join(setup.save_dir, 'second_level')
os.makedirs(glm_second_save, exist_ok=True)

group_contrast_maps = {}
group_contrast_files = {}
second_level_models = {}

for cond in contrast_names:
    print(f"Processing group-level analysis for {cond}...")
    
    contrast_imgs = [contrast_maps[sub][cond] for sub in subjects]

    second_level_model = SecondLevelModel(smoothing_fwhm=None)
    second_level_model.fit(contrast_imgs, design_matrix=group_dm)
    second_level_models[cond] = second_level_model

    all_contrasts = second_level_model.compute_contrast(output_type="all")
    utils.save_pickle(os.path.join(glm_second_save, f"{cond}_all_effects.pkl"), all_contrasts)

    group_z_map = all_contrasts['z_score']
    group_contrast_path = os.path.join(glm_second_save, f"group_{cond}.nii.gz")
    group_z_map.to_filename(group_contrast_path)
    print(f"Saved group-level contrast map for {cond} to {group_contrast_path}")
    
    # reports
    #thresh_img, thresh = threshold_stats_img(group_z_map, alpha=0.05, height_control='fdr', cluster_threshold= 5)
    second_level_report = second_level_model.generate_report('Intercept', 
                                                             threshold=3, # not used 
                                                             alpha=0.05,
                                                             height_control='fdr',
                                                             title=f"Second-level GLM Report for {cond}",
                                                             cluster_threshold=5
                                                            )
    os.makedirs(os.path.join(glm_second_save, "html_reports"), exist_ok=True)
    second_level_report.save_as_html(os.path.join(glm_second_save,'html_reports', f"{cond}.html"))

    plot_stat_map(group_z_map, title=f"Unc. group-Level Activation for {cond}",
                  threshold=3.0)
    plt.show()
    
    plot_glass_brain(group_z_map, colorbar=True, threshold=3.0,
                     title=f"Unc. group-Level Glass Brain for {cond}")
    plt.show()

    group_contrast_maps[cond] = group_z_map
    group_contrast_files[cond] = group_contrast_path

glm_info.second_level_models = second_level_models
glm_info.group_contrast_files = group_contrast_files

utils.save_pickle(os.path.join(glm_second_save, 'results_paths.pkl'), glm_info)


# %%

# %%

reload(qc_utils)
print('Dot product...')
conditions_nps = {}
pain_reg = ['ANA_shock', 'N_ANA_shock', 'HYPER_shock', 'N_HYPER_shock']
signature_folder = os.path.join(setup.project_dir,'masks/mvpa_signatures')

# for cond in pain_reg:
        
#     cond_files = {subj: contrasts[cond] for subj, contrasts in first_lev_files.items() if subj != 'sub-47'}

#     cond_dot = qc_utils.compute_similarity(cond_files, signature_folder, pattern = 'NPS', metric='dot_product', resample_to_mask=True)
#     conditions_nps[cond] = cond_dot

# # Print the results

# print("Dot Product Similarity for pain contrasts")
# print(conditions_nps)

# utils.save_pickle(os.path.join(setup.save_dir, 'GLM_results', 'NPS_dot_pain.pkl'), conditions_nps)
# %%
# utils.save_json(os.path.join(setup.save_dir, 'GLM_results', 'glm_info.json'), glm_info)

print("Done with all GLM processing!")
# %%

def prep_visu_glm(results_p):

    # dot_p = os.path.join(results_p, 'GLM_results', 'NPS_dot_pain.pkl')
    # dot = utils.load_pickle(dot_p)

    res_p = os.path.join(results_p, 'GLM_results', 'second_level', 'results_paths.pkl')
    res = utils.load_pickle(res_p)

    return res

model_res = r'/data/rainville/dSutterlin/projects/ISC_hypnotic_suggestions/results/imaging/GLM/model3_23subjects_nuis_nodrift_31-03-25'
res = utils.load_pickle(os.path.join(model_res, '/second_level/results_paths.pkl')
# res_model = 'model2_23subjects_zscore_sample_detrend_25-02-25'
# res_model = 'model2_23subjects_nodrift_10-03-25'

# res_dir = r'/data/rainville/dSutterlin/projects/ISC_hypnotic_suggestions/results/imaging/GLM/model2_23subjects_allnuis78_drift_25-02-25'
# dot, res = prep_visu_glm(setup.save_dir)

model_first = res['first_level_models']
file_firstlev = res['contrast_files_1level']
file_group = res['group_contrast_files']
model_second = res['second_level_models']

all_regs = list(file_firstlev[subjects[0]].keys())
sugg_reg = [reg for reg in all_regs if 'sugg' in reg] 
pain_reg = [reg for reg in all_regs if 'shock' in reg] 
 

# # Generate reports
# reports = {}
# for cond in sugg_reg:

#     model = model_second[cond]
#     # contrast = nib.load(file_group[cond])
#     group_report = model.generate_report('Intercept',
#          title="Group-level GLM Report"
#     )
#     reports[cond] = group_report
#     # group_report.save_as_html("group_level_report.html")

# %%
visu_on = True

if visu_on:
    # VISU 1st
    # from nilearn import plotting
    # cond = 'all_sugg'

    # for subject in subjects:
    #     img = file_firstlev[subject][cond]
    #     display = plotting.plot_stat_map(
    #         img,
    #         title=f"first level stats unc. {cond}",
    #         threshold=3.0,            
    #         display_mode='ortho')     

    #     plt.show()

    # VISU 2nd
    apply_thresh = True
    stats_imgs = {}
    views_second = {}
    cuts_second = {}
    for condition in sugg_reg:
        
        if apply_thresh:
            img, thresh = threshold_stats_img(
                file_group[condition], alpha=0.05, height_control='fdr', cluster_threshold=0
                )   
            title = f"Second level stats corr. {condition}"
            print(f'thresh for {condition} is {thresh}')
        else: 
            thresh = 3.0  
            img = file_group[condition]
            title = f"Second level stats unc. {condition}"

        stats_imgs[condition] = img
        display = plotting.plot_stat_map(
            img,
            title=f"second level stats cor {condition}",
            threshold=thresh,            
            display_mode='ortho')   
        
        view = plotting.view_img(img, threshold=3.0, title=f"Second level stats unc. {condition}")
        cuts_second[condition] = display
        views_second[condition] = view

# %%
# all regressors 
apply_thresh = True
stats_imgs = {}
views_second = {}
cuts_second = {}
for condition in all_regs:
    
    if apply_thresh:
        img, thresh = threshold_stats_img(
            file_group[condition], alpha=0.05, height_control='fdr', cluster_threshold=0
            )   
        title = f"Second level stats corr. {condition}"
        print(f'thresh for {condition} is {thresh}')
    else: 
        thresh = 3.0  
        img = file_group[condition]
        title = f"Second level stats unc. {condition}"

    stats_imgs[condition] = img
    display = plotting.plot_stat_map(
        img,
        title=f"second level stats cor {condition}",
        threshold=thresh,            
        display_mode='ortho')   
    
    view = plotting.view_img(img, threshold=3.0, title=f"Second level stats unc. {condition}")
    cuts_second[condition] = display
    views_second[condition] = view


        #plt.show()
    # corr_maps = {}
    # for condition in all_regs:
    #     corr_maps[condition] = (thresh_img, threshold)

    # display reports and thresholded maps 
    # for condition in all_regs:
    #     if 'sugg' in condition:
    #         img, thresh = corr_maps[condition]
    #         display = plotting.plot_stat_map(
    #             img,
    #             title=f"second level stats corr. {condition}",
    #             threshold=thresh,            
    #             display_mode='ortho')   
    #         plt.show()

# %%
# Load 1st level maps for further analyses
from glob import glob as glob
from nilearn.maskers import NiftiLabelsMasker
from nilearn.image import binarize_img
from nilearn.datasets import fetch_atlas_schaefer_2018
from src import qc_utils

model_res = r'/data/rainville/dSutterlin/projects/ISC_hypnotic_suggestions/results/imaging/GLM/model3_23subjects_nuis_nodrift_31-03-25'
project_dir = setup.project_dir
results_dir = setup['save_dir']

mvpa_save_to = os.path.join(results_dir, 'mvpa_similarity')
os.makedirs(mvpa_save_to, exist_ok=True)

# load maps 
all_shock_maps = glob(os.path.join(model_res, 'all_shock', 'firstlev_localizer_*.nii.gz'))
all_sugg = glob(os.path.join(model_res, 'all_sugg', 'firstlev_localizer_*.nii.gz'))

def build_subject_dict(file_list):
    """Build a dict: {subject_id: filepath} from a list of NIfTI file paths."""
    subject_dict = {}
    for path in file_list:
        fname = os.path.basename(path)
        subj_id = fname.split('_')[-1].replace('.nii.gz', '')  # expects '..._sub-01.nii.gz'
        subject_dict[subj_id] = path
    return subject_dict

sugg_dict = build_subject_dict(all_sugg) # not sorted!!
shock_dict = build_subject_dict(all_shock_maps)

# Intersect 
shared_subjects = sorted(set(sugg_dict) & set(shock_dict))

# load atlas for ROI
full_mask = nib.load(os.path.join(project_dir, 'masks/lipkin2022_lanA800', 'LanA_n806.nii'))
mask_native = binarize_img(full_mask, threshold=0.20)
mask_path = os.path.join(mvpa_save_to, f'bin_lanA800_{0.2}thresh.nii.gz')
resamp_mask = qc_utils.resamp_to_img_mask(mask_native, ref_img)
resamp_mask.to_filename(mask_path)

# atlas_data = fetch_atlas_schaefer_2018(n_rois = 100, resolution_mm=2)
# atlas_native = nib.load(atlas_data['maps'])
atlas_native = nib.load('/data/rainville/dSutterlin/projects/ISC_hypnotic_suggestions/masks/sensaas/SENSAAS_MNI_ICBM_152_2mm.nii')
atlas_data = pd.read_csv('/data/rainville/dSutterlin/projects/ISC_hypnotic_suggestions/masks/sensaas/SENSAAS_description.csv')
atlas = qc_utils.resamp_to_img_mask(atlas_native, nib.load(all_sugg[0]))
view_img(atlas, threshold=0.5, title='SENSAAS atlas')

#sensaas ids and labels
roi_index = atlas_data['Index'].values
atlas_data['annot_abbreviation'] = atlas_data['Abbreviation'] + '_' + atlas_data['Hemisphere']
labels = atlas_data['annot_abbreviation'].values


print(atlas.shape, mask.shape, mask_native.shape)
# %%

from nilearn.image import load_img
from sklearn.metrics.pairwise import cosine_similarity

def extract_multivoxel_patterns_by_subject(sugg_dict, shock_dict, atlas_img, mask = None):
    """
    Extract voxelwise ROI patterns (by index) within language ROIs for each subject.

    Parameters
    ----------
    sugg_dict : dict
        Maps subject ID to suggestion GLM image path.
    shock_dict : dict
        Maps subject ID to shock/pain GLM image path.
    atlas_img : Nifti1Image
        Schaefer parcellation image.
    lang_mask_img : Nifti1Image
        Binary mask indicating language ROIs.

    Returns
    -------
    pattern_dict : dict
        pattern_dict[subject][roi_idx]['suggestion'] / ['pain'] = voxel vector
    """

    atlas_data = atlas_img.get_fdata()
    # lang_mask = lang_mask_img.get_fdata()

    pattern_dict = {}

    shared_subjects = sorted(set(sugg_dict) & set(shock_dict))

    # all_roi_indices = [int(r) for r in np.unique(atlas_data) if r != 0 and np.any((atlas_data == r) & (lang_mask > 0))]
    all_roi_indices = [int(r) for r in np.unique(atlas_data) if r != 0]

    similarity_matrix = np.full((len(shared_subjects), len(all_roi_indices)), np.nan)
    subj_list = []

    for subj_idx, subj in enumerate(shared_subjects):
    
        sugg_img = load_img(sugg_dict[subj])
        shock_img = load_img(shock_dict[subj])

        sugg_data = sugg_img.get_fdata()
        shock_data = shock_img.get_fdata()

        subj_list.append(subj)

        for roi_pos, roi_idx in enumerate(all_roi_indices):
            roi_mask = atlas_data == roi_idx
            if not roi_mask.any():
                continue

            sugg_vec = sugg_data[roi_mask].flatten()
            shock_vec = shock_data[roi_mask].flatten()

            # Only compute if both vectors have nonzero norm
            if np.linalg.norm(sugg_vec) > 0 and np.linalg.norm(shock_vec) > 0:
                sim = cosine_similarity(sugg_vec.reshape(1, -1), shock_vec.reshape(1, -1))[0, 0]
                similarity_matrix[subj_idx, roi_pos] = sim

    similarity_df = pd.DataFrame(similarity_matrix, index=subj_list, columns=all_roi_indices)

    return similarity_df


similarity_df = extract_multivoxel_patterns_by_subject(
    sugg_dict, shock_dict, atlas
)

#%%

mean_sim = similarity_df.mean(axis=0)
masker = NiftiLabelsMasker(labels_img=atlas, standardize=False)
masker.fit()  

similarity_img = masker.inverse_transform(mean_sim)

view_img(similarity_img, threshold=0, title='Cosine Similarity Projection',
              cmap='coolwarm',  colorbar=True)


#%%
similarity_img, _, _ = project_isc_to_brain(
    atlas_path=atlas,
    isc_median=similarity_full_vector,
    atlas_labels=labels,  # should match index ordering
    p_values=None,  # No thresholding
    p_threshold=0.01,  # Irrelevant here
    title='Mean Cosine Similarity (Suggestion vs Pain)',
    color='coolwarm',
    save_path=None,
    show=True
)

# Step 2: Create full vector (1 value per atlas ROI label)
all_atlas_vals = np.unique(atlas_img.get_fdata()).astype(int)
all_atlas_vals = all_atlas_vals[all_atlas_vals != 0]

# Build full-length vector: assign NaN or 0 to ROIs not in similarity_df
similarity_full_vector = np.zeros(int(np.max(all_atlas_vals)))
similarity_full_vector[:] = np.nan

for roi_val, sim_val in zip(roi_indices, similarity_vec):
    similarity_full_vector[roi_val - 1] = sim_val  # assuming 1-based indexing

# Step 3: Fit masker to the atlas

# Step 4: Inverse transform the full similarity vector to brain space
# Get only the part of the vector that matches the unique ROI labels in atlas
labels_in_atlas = np.unique(atlas_img.get_fdata()).astype(int)
labels_in_atlas = labels_in_atlas[labels_in_atlas != 0]
similarity_subset = [similarity_full_vector[i - 1] for i in labels_in_atlas]

similarity_img = masker.inverse_transform(np.array(similarity_subset).reshape(1, -1))

# Step 5: Plot
plot_stat_map(similarity_img, threshold=None, title='Cosine Similarity Projection',
              cmap='coolwarm', display_mode='z', cut_coords=7, colorbar=True)

#%%


#%%

# Load Atlas regions
# ------------------
from nilearn import plotting
from nilearn.plotting import view_img
import nibabel as nib
from nilearn.image import math_img
 

coords = plotting.find_parcellation_cut_coords(labels_img=atlas)
 
# Build DataFrame of region + coordinates
df_labels_coords = pd.DataFrame({
    'index': list(range(len(labels))),
    'region': labels,
    'x': [round(c[0], 2) for c in coords],
    'y': [round(c[1], 2) for c in coords],
    'z': [round(c[2], 2) for c in coords],
})
df_labels_coords
# df_labels_coords[df_labels_coords['region'].str.contains('somMot', case=False, na=False)] # for only somatosensory areas
 
# ------------------
# plot only selected roi to validate the location aMCC _ SMA
def plot_roi_by_label(label_name, atlas, df_labels_coords):
    try:
        row = df_labels_coords[df_labels_coords['region'] == label_name].iloc[0]
        region_index = int(row['index'])
        region_value = region_index + 1  # Atlas values start at 1
 
        mask_img = math_img("img == %d" % region_value, img=atlas)
 
        print(f"Found region: {label_name} (Index {region_index})")
        print(f"Coordinates: x={row['x']}, y={row['y']}, z={row['z']}")
 
        return view_img(mask_img, threshold=0.5, title=f"{label_name}")
 
    
    except IndexError:
        print(f"Region label '{label_name}' not found in atlas.")
   
    
amcc_name = labels[99]
# sma_name =
view_amcc = plot_roi_by_label(amcc_name, atlas, df_labels_coords)
# view_sma = plot_roi_by_label(amcc_name, atlas, df_labels_coords)
view_amcc

#%%

import numpy as np
import nibabel as nib
import pandas as pd

def count_voxels_per_roi(atlas_img, lang_mask_img):
    """
    Count the number of voxels per ROI after applying a language mask.

    Parameters
    ----------
    atlas_img : Nifti1Image
        Schaefer parcellation image.
    lang_mask_img : Nifti1Image
        Binary language mask image.

    Returns
    -------
    roi_voxel_counts : pd.DataFrame
        DataFrame with ROI index and voxel count.
    """
    atlas_data = atlas_img.get_fdata()
    lang_data = lang_mask_img.get_fdata()

    unique_rois = np.unique(atlas_data)
    voxel_counts = []

    for roi in unique_rois:
        if roi == 0:
            continue  # skip background
        roi_mask = (atlas_data == roi) & (lang_data > 0)
        count = np.sum(roi_mask)
        voxel_counts.append((int(roi), count))

    df_counts = pd.DataFrame(voxel_counts, columns=['ROI_index', 'Voxel_count'])
    return df_counts.sort_values('Voxel_count', ascending=False)

roi_voxel_df = count_voxels_per_roi(atlas, resamp_mask)
print(roi_voxel_df)


# %% NPS
'''
# dot_p = r'/data/rainville/dSutterlin/projects/ISC_hypnotic_suggestions/results/imaging/preproc_data/model1_23subjects_zscore_sample_detrend_21-02-25/GLM_results/NPS_dot_pain.pkl'
# dot = utils.load_pickle(dot_p)
# print(dot)

from scipy.stats import ttest_1samp

for cond in dot.keys():

    shock_similarity = np.array(dot[cond]).ravel()

    t_stat, p_val = ttest_1samp(shock_similarity, 0)
    print(f"One-sample t-test for {cond}: t = {t_stat:.3f}, p = {p_val:.3f}")

# %%

# res_p = r'/data/rainville/dSutterlin/projects/ISC_hypnotic_suggestions/results/imaging/preproc_data/model1_23subjects_zscore_sample_detrend_21-02-25/GLM_results/second_level/results_paths.pkl'
# res = utils.load_pickle(res_p)


# %% NPS
# signature_folder = os.path.join(setup.project_dir,'masks/mvpa_signatures')
# conditions_nps = {}
# for cond in pain_reg:
        
#     cond_files = {subj: contrasts[cond] for subj, contrasts in file_firstlev.items() if subj != 'sub-47'}

#     cond_dot = qc_utils.compute_similarity(cond_files, signature_folder, pattern = 'NPS', metric='dot_product', resample_to_mask=True)
#     conditions_nps[cond] = cond_dot

#     shock_similarity = np.array(cond_dot).ravel()
#     t_stat, p_val = ttest_1samp(shock_similarity, 0)
#     print(f"One-sample t-test for {cond}: t = {t_stat:.3f}, p = {p_val:.3f}")


# %%
from nilearn import plotting
views = []
for condition in all_regs:
    img = file_group[condition]
    # Create an interactive view of the statistical map.
    # Alternatively, you can open the view in your browser:
    # view.open_in_browser()

# %%
# p_folder = r'/data/rainville/dSutterlin/projects/ISC_hypnotic_suggestions/results/imaging/preproc_data/model1_23subjects_zscore_sample_detrend_21-02-25/GLM_results'

# for condition in list(regressors_dct.keys()):
#     img = f'{condition}_localizer_sub-47.nii.gz'
#     display = plotting.plot_stat_map(
#         os.path.join(p_folder, img),
#         title=f"Stat Map: {condition}",
#         threshold=3.0,            
#         display_mode='ortho')     

#     plt.show()
'''