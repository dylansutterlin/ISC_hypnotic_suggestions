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
subjects = setup.subjects
subjects.sort()

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
#============================
# VISUALIZATION
#============================

def prep_visu_glm(results_p):

    # dot_p = os.path.join(results_p, 'GLM_results', 'NPS_dot_pain.pkl')
    # dot = utils.load_pickle(dot_p)

    res_p = os.path.join(results_p, 'GLM_results', 'second_level', 'results_paths.pkl')
    res = utils.load_pickle(res_p)

    return res

model_res = r'/data/rainville/dSutterlin/projects/ISC_hypnotic_suggestions/results/imaging/GLM/model3_23subjects_nuis_nodrift_31-03-25'
res = utils.load_pickle(os.path.join(model_res, 'second_level/results_paths.pkl'))
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

#%%
#%% NPS
from scipy.stats import ttest_1samp
import seaborn as sns

signature_folder = os.path.join(setup.project_dir,'masks/mvpa_signatures')
conditions_nps = {}
for cond in ['all_shock']:
        
    cond_files = {subj: contrasts[cond] for subj, contrasts in file_firstlev.items() if subj != 'sub-47'}

    cond_dot = qc_utils.compute_similarity(cond_files, signature_folder, pattern = 'NPS', metric='dot_product')
    conditions_nps[cond] = cond_dot

    shock_similarity = np.array(cond_dot).ravel()
    t_stat, p_val = ttest_1samp(shock_similarity, 0)
    print(f"One-sample t-test for {cond}: t = {t_stat:.3f}, p = {p_val:.3f}")

import matplotlib.pyplot as plt

plt.figure(figsize=(8, 6))
sns.violinplot(data=cond_dot, x='similarity', inner='point', color='skyblue')
plt.title('Violin Plot of NPS Similarity (All Shock)')
plt.xlabel('NPS Similarity')
plt.tight_layout()
plt.show()

# %%

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
    reg_list = ['all_sugg', 'all_shock']
    # VISU 2nd
    apply_thresh = True
    stats_imgs = {}
    views_second = {}
    cuts_second = {}
    for condition in reg_list:
        
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
        
        view = plotting.view_img(img, threshold=thresh, title=f"2d-lev FDR {condition}")
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
#========================================
# Load 1st level maps for further analyses
#==========================================
from glob import glob as glob
from nilearn.maskers import NiftiLabelsMasker
from nilearn.image import binarize_img
from nilearn.plotting import view_img
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

def extract_multivoxel_patterns_by_subject(sugg_dict, shock_dict, atlas_img, all_roi_indices,mask = None):
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
  
    similarity_dict = {}

    for subj_idx, subj in enumerate(shared_subjects):
    
        sugg_img = load_img(sugg_dict[subj])
        shock_img = load_img(shock_dict[subj])

        sugg_data = sugg_img.get_fdata()
        shock_data = shock_img.get_fdata()

        # subj_list.append(subj)

        subj_dict = {}
        for roi_idx in all_roi_indices: # ROI idx != position !! hence go not in idx order, but idx as ids
            roi_mask = atlas_data == roi_idx
            if not roi_mask.any():
                continue

            sugg_vec = sugg_data[roi_mask].flatten()
            shock_vec = shock_data[roi_mask].flatten()

            if np.linalg.norm(sugg_vec) > 0 and np.linalg.norm(shock_vec) > 0:
                sim = cosine_similarity(sugg_vec.reshape(1, -1), shock_vec.reshape(1, -1))[0, 0]
                subj_dict[roi_idx] = sim

        similarity_dict[subj] = subj_dict

    similarity_df = pd.DataFrame.from_dict(similarity_dict, orient='index')

    return similarity_df

def project_vector_to_atlas(vector, roi_index, atlas):
    #will replace the label integer in vol with vector value
    atlas_fdata = atlas.get_fdata()  # Always use this for consistency
    sim_vol = np.zeros_like(atlas_fdata)

    for roi in roi_index:
        if roi == 0:
            continue
        value = vector.get(roi, 0.0)
        sim_vol[atlas_fdata == roi] = value  # <== Must use atlas_fdata here

    sim_img = nib.Nifti1Image(sim_vol, affine=atlas.affine, header=atlas.header)

    return sim_img

similarity_df  = extract_multivoxel_patterns_by_subject(
    sugg_dict, shock_dict, atlas, roi_index
)

#%%

mean_sim = similarity_df.mean(axis=0)
mean_sim_df = pd.DataFrame(mean_sim, columns=['mean_similarity'])
mean_sim_df.index = labels

threshold = 0.30
above_thresh = mean_sim_df[mean_sim_df['mean_similarity'] > threshold].copy()
# get thresholded df!
region_details = atlas_data.set_index('annot_abbreviation').loc[above_thresh.index]
final_df = above_thresh.join(region_details[['Network', 'Region', 'Abbreviation', 'Xmm', 'Ymm', 'Zmm']])
final_df['Full_Name'] = final_df['Region'] + " (" + final_df['Abbreviation'] + ")"
final_df = final_df.sort_values('mean_similarity', ascending=False)


sim_img = project_vector_to_atlas(mean_sim, roi_index, atlas)
view = view_img(sim_img, threshold=0.3,vmax=mean_sim.max(), cmap='coolwarm', title='Projected similarity')
view

plot_stat_map(sim_img, threshold=0, title='Cosine Similarity Projection',
                  colorbar=True, display_mode='x', cut_coords=5,
                  cmap='coolwarm', bg_img=mni_bg, black_bg=False, vmax = sim_img.get_fdata().max())
# %%
# =============
# Interaction of cosine similarity with behavioral
import seaborn as sns
import matplotlib.pyplot as plt

behav_df = pd.read_csv(
    os.path.join(setup['project_dir'], f'results/behavioral_data_cleaned.csv'),
    index_col=0
)
behav_df.index.name = 'subjects'
behav_df = behav_df.sort_index()
APM_subjects = ['APM' + sub[4:] for sub in subjects] # make APMXX format instead of subXX

behav_vars = [
    'Chge_hypnotic_depth', 'SHSS_score', 'raw_change_HYPER',
    'raw_change_ANA', 'total_chge_pain_hypAna',
    'Mental_relax_absChange', 'Abs_diff_automaticity'
]

behav_corr_df = behav_df[behav_vars].copy()
corr_matrix = behav_corr_df.corr()

plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', square=True,
            linewidths=0.5, cbar_kws={'label': 'Pearson r'})

plt.title('Correlation Matrix of Behavioral Variables', fontsize=14)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

behav_subset = behav_df[behav_vars].copy()

sns.set(style='whitegrid', context='notebook')
g = sns.pairplot(behav_subset, kind='scatter', diag_kind='kde', height=2.5,
                 plot_kws={'alpha': 0.6, 's': 40, 'edgecolor': 'k'},
                 diag_kws={'shade': True})

g.fig.suptitle('Pairwise Distributions and Relationships of Behavioral Variables', y=1.02)
plt.tight_layout()
plt.show()

#%%

from scipy.stats import pearsonr

y_name = 'SHSS_score' #total_chge_pain_hypAna
y = behav_df[y_name].values
rois = similarity_df.columns

r_vals = []
p_vals = []

for roi in rois:
    x = similarity_df[roi].values
    mask = ~np.isnan(x) & ~np.isnan(y)
    if mask.sum() > 2:
        r, p = pearsonr(x[mask], y[mask])
    else:
        r, p = np.nan, np.nan
    r_vals.append(r)
    p_vals.append(p)

corr_df = pd.DataFrame({
    'ROI_index': rois.astype(int),
    'correl': r_vals,
    'p_value': p_vals
})

plt.figure(figsize=(7, 4))
sns.histplot(corr_df['p_value'].dropna(), bins=20, color='skyblue')
plt.axvline(0.05, color='red', linestyle='--', label='p = 0.05')
plt.xlabel('p-value')
plt.ylabel('Count')
plt.title(f'Distribution of pearson p-values (Similarity × {y_name})')
plt.legend()
plt.tight_layout()
plt.show()

brain_r_pain = project_vector_to_atlas(corr_df['correl'], roi_index, atlas)
view_interaction = view_img(brain_r_pain, threshold=0, title=f'Cosine x {y_name}',
              cmap='coolwarm', colorbar=True)

plot_stat_map(brain_r_pain, threshold=0, title=f'Cosine x {y_name} interaction',
                  colorbar=True, display_mode='x', cut_coords=5,
                  cmap='coolwarm', bg_img=mni_bg, black_bg=False)

#%% 
#====================================
# Pairwise approach to pattern similarity
#====================================


#%%
#======================================
# REGRESSION
#========================================
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import ShuffleSplit
from sklearn.metrics import r2_score
from sklearn.linear_model import Ridge, Lasso
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA

def roi_regression(sugg_dict, behav_df, atlas_img, target_col, roi_idx):
    atlas_data = atlas_img.get_fdata()
    shared_subjects = sorted(sugg_dict.keys())

    X = []
    y = []

    for subj in shared_subjects:
        apm_subj = 'APM' + subj[4:]
        img = load_img(sugg_dict[subj])
        img_data = img.get_fdata()
        roi_mask = atlas_data == roi_idx
        if not roi_mask.any():
            continue
        pattern = img_data[roi_mask].flatten()
        X.append(pattern)
        y_val = behav_df.loc[apm_subj, target_col]
        y.append(y_val)

    X = np.array(X)
    y = np.array(y)
    print(f"X shape: {X.shape}, y shape: {y.shape}")
    # Remove subjects with NaNs
    valid_mask = ~np.isnan(y)
    X = X[valid_mask]
    y = y[valid_mask]

    model = make_pipeline(StandardScaler(), PCA(n_components=0.95), Lasso())
    cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=42)
    scores = cross_val_score(model, X, y, cv=cv, scoring='explained_variance')
    
    print(f"ROI {roi_idx} — Mean R²: {scores.mean():.3f}, STD: {scores.std():.3f}")
    return scores

top5_rois = ['HIPP2_Right', 'F3O1_Right', 'prec3_Left', 'HIPP2_Left', 'f2_2_Left']
roi_indices = atlas_data.set_index('annot_abbreviation').loc[list(final_df.index), 'Index']

roi_scores = {}
for abbr, roi_idx in roi_indices.items():
    print(f"\nRunning regression for ROI: {abbr} (Index: {roi_idx})")
    scores = roi_regression(
        sugg_dict=shock_dict,
        behav_df=behav_df,
        atlas_img=atlas,
        target_col='total_chge_pain_hypAna',
        roi_idx=roi_idx
    )
    roi_scores[abbr] = scores

roi_scores_df = pd.DataFrame.from_dict(roi_scores, orient='index')
roi_scores_df.columns = [f'Split_{i}' for i in range(roi_scores_df.shape[1])]

# Add summary stats
roi_scores_df['Mean_R2'] = roi_scores_df.mean(axis=1)
roi_scores_df['STD_R2'] = roi_scores_df.std(axis=1)

# Sort by mean R²
roi_scores_df = roi_scores_df.sort_values('Mean_R2', ascending=False)

# Plot
plt.figure(figsize=(10, 6))
sns.barplot(x=roi_scores_df.index.astype(str), y=roi_scores_df['Mean_R2'], palette='viridis')
plt.axhline(0, color='black', linestyle='--')
plt.title("ROI-wise Cross-Validated R² Scores")
plt.ylabel("Mean R²")
plt.xlabel("ROI Index")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

    
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


%% NPS
signature_folder = os.path.join(setup.project_dir,'masks/mvpa_signatures')
conditions_nps = {}
for cond in pain_reg:
        
    cond_files = {subj: contrasts[cond] for subj, contrasts in file_firstlev.items() if subj != 'sub-47'}

    cond_dot = qc_utils.compute_similarity(cond_files, signature_folder, pattern = 'NPS', metric='dot_product', resample_to_mask=True)
    conditions_nps[cond] = cond_dot

    shock_similarity = np.array(cond_dot).ravel()
    t_stat, p_val = ttest_1samp(shock_similarity, 0)
    print(f"One-sample t-test for {cond}: t = {t_stat:.3f}, p = {p_val:.3f}")


# %%
from nilearn import plotting
import seaborn as sns
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