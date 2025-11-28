# %%
from nilearn.image import math_img
from nilearn.plotting import plot_stat_map
import os
import seaborn as sns
from nilearn import plotting
from nilearn.plotting import view_img

import time
import matplotlib.pyplot as plt
from importlib import reload
import nibabel as nib
import numpy as np
import pandas as pd
from nilearn.image import concat_imgs
from brainiak.isc import (
    isc,
    bootstrap_isc,
    permutation_isc,
    compute_summary_statistic,
    phaseshift_isc,
)
from nilearn.maskers import MultiNiftiMapsMasker, MultiNiftiMasker
from nilearn.datasets import fetch_atlas_schaefer_2018
from sklearn.utils import Bunch
from nilearn.plotting import view_img_on_surf
from nilearn.maskers import NiftiLabelsMasker
from nilearn.plotting import plot_glass_brain

from nilearn.plotting import find_parcellation_cut_coords

import src.isc_utils as isc_utils
import src.visu_utils as visu_utils

reload(visu_utils)
reload(isc_utils)


# %% Load the data
model_names = {
    "single_trial_language_mask": "model_single-trial_sugg_23-sub_schafer-200-2mm_mask-lanA800_pairWise-True_preproc_reg-mvmnt-True-8",
    "single_trial_whole-brain": "model_single-trial-wb_sugg_23-sub_schafer-200-2mm_mask-whole-brain_pairWise-True_preproc_reg-mvmnt-True-8",
    "model5_sugg_tian": "model5-with-subcort_sugg_23-sub_schafer-200-2mm_mask-whole-brain_pairWise-True_preproc_reg-mvmnt-True-8",
    "model1_single-trial": "model1_single-trial-wb_sugg_23-sub_schafer_tian-200-2mm_mask-whole-brain_pairWise-True_preproc_reg-mvmnt-True-8",
    "model2_sugg": "model2_sugg_23-sub_schafer_tian-200-2mm_mask-whole-brain_pairWise-True_preproc_reg-mvmnt-True-8",
    "model3_shock": "model3_shock_23-sub_schafer_tian-200-2mm_mask-whole-brain_pairWise-True_preproc_reg-mvmnt-True-8",
}

model_is = "model2_sugg"
project_dir = "/data/rainville/dSutterlin/projects/ISC_hypnotic_suggestions"

color_isc = "Reds"
# base_path = "/data/rainville/dSutterlin/projects/ISC_hypnotic_suggestions/results/data/test_data_sugg_3sub"
preproc_model_data = "23subjects_zscore_sample_detrend_FWHM6_low-pass428_10-12-24/suggestion_blocks_concat_4D_23sub"
base_path = os.path.join(
    project_dir, "results/imaging/preproc_data", preproc_model_data
)
model_name = model_names[model_is]
# preproc_model_name =  model_names['model1_sugg'] #'model3_shock_23-sub_schafer-200-2mm_mask-whole-brain_pairWise-True_preproc_reg-mvmnt-True-8'

results_dir = os.path.join(project_dir, f"results/imaging/ISC/{model_name}")
setup = isc_utils.load_json(os.path.join(results_dir, "setup_parameters.json"))

# create visualization dir.
save_rsa_isc_mvpa_dir = os.path.join(project_dir, "results/imaging/IS-RSA_ISC-MVPA")
os.makedirs(save_rsa_isc_mvpa_dir, exist_ok=True)

parcel_name = setup["atlas_name"]
do_pairWise = setup["do_pairwise"]
n_boot = setup["n_boot"]

# post_hoc_dir = os.path.join(results_dir, "post_hoc_results")
# os.makedirs(post_hoc_dir, exist_ok=True)

# %%
# all_results_paths = utils.load_json(os.path.join(results_dir, "result_paths.json"))
# atlas_name = 'Difumo256' # change to setup['atlas_name']
# n_sub = setup['n_sub']
shaeffer_only = False
if shaeffer_only:

    atlas_data = fetch_atlas_schaefer_2018(n_rois=200, resolution_mm=2)
    atlas = nib.load(atlas_data["maps"])
    atlas_path = atlas_data[
        "maps"
    ]  # os.path.join(project_dir,os.path.join(project_dir, 'masks', 'k50_2mm', '*.nii*'))
    # labels_bytes = list(atlas_data['labels'])
    full_labels = [
        str(label, "utf-8") if isinstance(label, bytes) else str(label)
        for label in atlas_data["labels"]
    ]
    roi_index = [full_labels.index(lbl) + 1 for lbl in full_labels]
    id_labels_dct = dict(zip(roi_index, full_labels))

# ---------------
# up to date scheaffer + subcortical regions
else:
    atlas = nib.load(
        "/data/rainville/dSutterlin/projects/ISC_hypnotic_suggestions/masks/Tian2020_schaeffer200_subcortical16/combined_schaefer200_tian16_DSG.nii.gz"
    )
    id_labels_dct = isc_utils.load_json(
        "/data/rainville/dSutterlin/projects/ISC_hypnotic_suggestions/masks/Tian2020_schaeffer200_subcortical16/roi_labels_dict_DSG.json"
    )
    # remove background
    id_labels_dct.pop("0")
    full_labels = list(id_labels_dct.values())
    roi_index = list(id_labels_dct.keys())

atlas_masker = NiftiLabelsMasker(
    labels_img=atlas, labels=full_labels, standardize=False
)
atlas_masker.fit()

if "SENSAAS" in setup["atlas_name"]:
    roi_coords = list(
        zip(
            atlas_data["Xmm"].astype(float),
            atlas_data["Ymm"].astype(float),
            atlas_data["Zmm"].astype(float),
        )
    )
else:
    roi_coords = find_parcellation_cut_coords(labels_img=atlas)

from nilearn import datasets, plotting

bg_mni = datasets.load_mni152_template(resolution=1)

# %%
# ================================
# BEHAHVIORAL DATA
# ================================
import seaborn as sns
from src import preproc_utils

isc_results_roi = {}

behav_df = pd.read_csv(
    os.path.join(setup["project_dir"], f"results/behavioral_data_cleaned.csv"),
    index_col=0,
)
behav_df.index.name = "subjects"
behav_df = behav_df.sort_index()

xlsx_path = os.path.join(project_dir, "masks/Hypnosis_variables_20190114_pr_jc.xlsx")
subjects = list(setup["subjects"])
apm_subjects = ["APM" + subj[4:] for subj in subjects]
Y, rawY = preproc_utils.load_process_y(xlsx_path, subjects)

#%%

# %%
# LOAD PATH 1 MVPA sim suggestions 
# =================

from src.mvpa_rsa_utils import compute_inter_subject_mvpa_similarity, build_subject_dict

save_path = os.path.join( project_dir, f"results/imaging/RSA/mvpa_IS-RSA_SHSS_sugg-contrast-based_tian216")
os.makedirs(save_path, exist_ok=True)

n_perm_rsa = 10000
NEURAL_SIM = 'dot'
# load results
load_path = os.path.join(
    save_path, f"IS-RSA_mvpa-sugg_SHSS{n_perm_rsa}perm.pkl"
        )
sugg_results_dct = isc_utils.load_pickle(load_path)
sugg_sim_matrices_dct = isc_utils.load_pickle(os.path.join(
    save_path, f"similarity_mat_IS-RSA-mvpa_sugg-{NEURAL_SIM}_SHSS_behav_{n_perm_rsa}perm.pkl"
    )
)

combined_conditions = ['ANA_sugg_minus_N_ANA_sugg', 'HYPER_sugg_minus_N_HYPER_sugg']
# %%
# Initialize PPath 1 (MVPA suggestion)
sugg_rsa_stats_imgs_mvpa = {}
sugg_rsa_tables_mvpa_fdr = {}
sugg_rsa_fdr_imgs_mvpa = {}
sugg_rsa_liberal_imgs_mvpa = {}
sugg_rsa_liberal_tables_mvpa = {}
# RSA_MODEL_NAME = "sugg_contrast_SHSS"
# save_tables_rsa = os.path.join(save_path, RSA_MODEL_NAME, "tables")
# os.makedirs(save_tables_rsa, exist_ok=True)
# save_plots_rsa = os.path.join(save_path, RSA_MODEL_NAME, "plots")
# os.makedirs(save_plots_rsa, exist_ok=True)

all_p_values = []
for sim in sugg_results_dct.keys():  # similarity metric (e.g., 'euclidean', 'annak')
    for cond in sugg_results_dct[sim].keys():  # e.g., 'Analgesia', 'Hyperalgesia'

        model_id = sim + "_" + cond
        rsa_df = sugg_results_dct[sim][cond]
        p_values = rsa_df["p_values"].values
        all_p_values.extend(p_values)

p_thresh_global = isc_utils.fdr(np.array(all_p_values), q=0.05)
print("Global FDR threshold across all models:", p_thresh_global)

# Main loop to visualize results and threshold
for sim in sugg_results_dct.keys():  # similarity metric (e.g., 'euclidean', 'annak')
    for cond in sugg_results_dct[sim].keys():  # e.g., 'Analgesia', 'Hyperalgesia'

        model_id = sim + "_" + cond
        rsa_df = sugg_results_dct[sim][cond]
        # Ensure the dataframe rows are in the same order as the atlas labels

        correlations = rsa_df["spearman_r"].values
        p_values = rsa_df["p_values"].values
        region_id = rsa_df["ROI"].values

        for correction in ["FDR", 'unc01'] : #, "FDR", "bonf"]:
            show = False

            if correction == "unc01":
                p_thresh = 0.01
                p_thresh_str = "0.01"

            elif correction == "within_FDR":
                p_thresh = isc_utils.fdr(p_values, q=0.05)
                p_thresh_str = "within_FDR" + str(round(p_thresh, 4))
                print("Within-model FDR threshold:", p_thresh)
                show = False
                
            elif correction == "FDR":
                p_thresh = p_thresh_global  # or load global value
                p_thresh_str = "FDR" + str(round(p_thresh, 4))
                show = True

            elif correction == "bonf":
                p_thresh = isc_utils.bonferroni(p_values, alpha=0.05)
                p_thresh_str = "Bonf" + str(round(p_thresh, 4))

            rsa_img, rsa_thresh, sig_df, stats_img = (
                visu_utils.project_isc_to_brain_perm(
                    atlas_img=atlas,
                    isc_median=correlations,
                    atlas_labels=id_labels_dct,
                    roi_coords=roi_coords,
                    p_values=p_values,
                    p_threshold=p_thresh,
                    title=f"RSA MVPA {sim} – {cond} ({p_thresh_str})",
                    save_path=None,
                    show=show,
                    display_mode="x",       
                    cut_coords_plot=(5),
                    color="coolwarm",
                )
            )
            if correction == 'unc01':
                sugg_rsa_liberal_imgs_mvpa[model_id] = (rsa_img, rsa_thresh)
                sugg_rsa_liberal_tables_mvpa[model_id] = sig_df
                
            if correction == "FDR" and sig_df.shape[0] > 0:
                sugg_rsa_fdr_imgs_mvpa[model_id] = (rsa_img, rsa_thresh)
                sugg_rsa_tables_mvpa_fdr[model_id] = sig_df

                sig_df.columns = [
                    "ROI",
                    "Label",
                    "Spearman r",
                    "p-values",
                    "Coordinates (X,Y,Z)",
                ]
                fname = f"RSA_MVPA_{sim}_{cond}_table_{p_thresh_str}.csv"
                # sig_df.to_csv(os.path.join(save_tables_rsa, fname), index=False)

                print(f"Sig ROIs for {sim} – {cond}:", sig_df["Label"].to_list(), "\n")

#%%
# Create seed regions for RSA-ISC-MVPA analysis
seed_regions_ana_context = {}
seed_regions_hyper_context = {}

seed_regions_ana_matrices = {}
seed_regions_hyper_matrices = {}

# only positive seeds 
# for sim_model in rsa_tables_fdr.keys(): # ONLY POSITIVE REGIONS
#     if 'ana' in sim_model:
#         region_ids = list(rsa_tables_fdr[sim_model]['ROI'][rsa_tables_fdr[sim_model]['Spearman r']>0].values)
#         region_name = list(rsa_tables_fdr[sim_model]['Label'][rsa_tables_fdr[sim_model]['Spearman r']>0].values)

#         cond_dct = {id:name for id, name in zip(region_ids, region_name)}
#         seed_regions_ana_context = {**seed_regions_ana_context, **cond_dct}
#     elif 'hyper' in sim_model:
#         region_ids = list(rsa_tables_fdr[sim_model]['ROI'][rsa_tables_fdr[sim_model]['Spearman r']>0].values)
#         region_name = list(rsa_tables_fdr[sim_model]['Label'][rsa_tables_fdr[sim_model]['Spearman r']>0].values)
#         cond_dct = {id:name for id, name in zip(region_ids, region_name)}
#         seed_regions_hyper_context = {**seed_regions_hyper_context, **cond_dct}

# # get matrices
# for region_id, region in seed_regions_ana_context.items():
#     seed_regions_ana_matrices[region] = isc_matrices['ana_run'][region]
# for region_id, region in seed_regions_hyper_context.items():
#     seed_regions_hyper_matrices[region] = isc_matrices['hyper_run'][region]

# all seeds positive and negative FDR corr. is-rsa SHSS ~ mv pain results 
for sim_model in sugg_rsa_tables_mvpa_fdr.keys():
    if 'ANA' in sim_model:
        region_ids = list(sugg_rsa_tables_mvpa_fdr[sim_model]['ROI'].values)
        region_name = list(sugg_rsa_tables_mvpa_fdr[sim_model]['Label'].values)

        cond_dct = {id:name for id, name in zip(region_ids, region_name)}
        seed_regions_ana_context = {**seed_regions_ana_context, **cond_dct}
    elif 'HYPER' in sim_model:
        region_ids = list(sugg_rsa_tables_mvpa_fdr[sim_model]['ROI'].values)
        region_name = list(sugg_rsa_tables_mvpa_fdr[sim_model]['Label'].values)
        cond_dct = {id:name for id, name in zip(region_ids, region_name)}
        seed_regions_hyper_context = {**seed_regions_hyper_context, **cond_dct}

# get matrices
for region_id, region in seed_regions_ana_context.items():
    seed_regions_ana_matrices[region] = sugg_sim_matrices_dct['euclidean']['ANA_sugg_minus_N_ANA_sugg'][int(region_id)]
for region_id, region in seed_regions_hyper_context.items():
    seed_regions_hyper_matrices[region] = sugg_sim_matrices_dct['euclidean']['HYPER_sugg_minus_N_HYPER_sugg'][int(region_id)]


#%%
# LOAD path 2 results: IS-RSA SHSS MVPA to produce target rois (restrict search)
# =================
# VISUALISE PAIN-contrast ~ SHSS
# IS-RSA SHSS ~ ISC-PAIN
from src.mvpa_rsa_utils import compute_inter_subject_mvpa_similarity, build_subject_dict

pain_results_dct_all = {}
n_perm_rsa = 10000
similarity_matrices_dct_all = {}
for NEURAL_SIM in ['dot', 'cosine']: #, 'dot']: #, 'cosine']: #, 'dot']: #['euclidean', 'dot']: #'cosine', 'pearson']:
    # NEURAL_SIM = 'euclidean'  # 'cosine' or 'dot' or 'euclidean'
    # BEHAV_ID = "SHSS"  # "SHSS" or "chge-pain"
    effect_type = 'stat' #'effect_size' #'z_score'
    save_path = f"/data/rainville/dSutterlin/projects/ISC_hypnotic_suggestions/results/imaging/RSA/IS-RSA-mvpa-{NEURAL_SIM}-{effect_type}_contrast-based_tian216" #_{ANNAK_VERSION}-annak"
    os.makedirs(save_path, exist_ok=True)

    BEHAV_ID = "SHSS_score"  # "SHSS" or "chge-pain" or "pain_diff_Ana" or "pain_diff_Hyper" or "total_change_pain"
    pain_conditions = ["ANA_shock_minus_N_ANA_shock", "HYPER_shock_minus_N_HYPER_shock"]

    # load results
    load_path = os.path.join(
        save_path, f"IS-RSA-mvpa_pain-{NEURAL_SIM}_behav-{BEHAV_ID}_{n_perm_rsa}perm.pkl"
    )
    pain_results_dct_all[NEURAL_SIM] = isc_utils.load_pickle(load_path)
    similarity_matrices_dct_all[NEURAL_SIM] = isc_utils.load_pickle(os.path.join(
        save_path, f"similarity_mat_IS-RSA-mvpa_pain-{NEURAL_SIM}_behav-{BEHAV_ID}_{n_perm_rsa}perm.pkl"
    ))

# main results with dot-based similarity
NEURAL_SIM = 'dot'
pain_results_dct = pain_results_dct_all[NEURAL_SIM]
similarity_matrices_dct = similarity_matrices_dct_all[NEURAL_SIM]
save_path = f"/data/rainville/dSutterlin/projects/ISC_hypnotic_suggestions/results/imaging/RSA/IS-RSA-mvpa-{NEURAL_SIM}-{effect_type}_contrast-based_tian216" #_{ANNAK_VERSION}-annak"

rsa_stats_imgs_fdr = {}
rsa_tables_mvpa_fdr = {}
rsa_stats_imgs_liberal = {}
rsa_stats_imgs_mvpa_unc = {}
rsa_tables_mvpa_liberal = {}

RSA_MODEL_NAME = f"shock_contrast_{BEHAV_ID}"
save_tables_shock_shss = os.path.join(save_path, RSA_MODEL_NAME, "tables")
os.makedirs(save_tables_shock_shss, exist_ok=True)
save_plots_shock_shss = os.path.join(save_path, RSA_MODEL_NAME, "plots")
os.makedirs(save_plots_shock_shss, exist_ok=True)

all_p_values = []
condition_wise_p_values = {} # 216 * model / condition
for sim in pain_results_dct.keys():  # similarity metric (e.g., 'euclidean', 'annak')
    for cond in pain_results_dct[sim].keys():  # e.g., 'Analgesia', 'Hyperalgesia'
        
        model_id = sim + "_" + cond
        rsa_df = pain_results_dct[sim][cond]
        # Ensure the dataframe rows are in the same order as the atlas labels

        correlations = rsa_df["spearman_r"].values
        p_values = rsa_df["p_values"].values
        region_id = rsa_df["ROI"].values
        all_p_values.append(p_values)
        condition_wise_p_values[model_id] = p_values

# condition-wise p values
ana_p_values = np.array([])
hyper_p_values = np.array([])
for model_id in condition_wise_p_values.keys():
    if 'ANA' in model_id:
        ana_p_values = np.concatenate((ana_p_values, condition_wise_p_values[model_id]))
    elif 'HYP' in model_id:
        hyper_p_values = np.concatenate((hyper_p_values, condition_wise_p_values[model_id]))

# global FDR
all_p_values = np.concatenate(all_p_values)
global_fdr_thresh = isc_utils.fdr(all_p_values, q=0.05)
print(f"Global FDR correction pain~SHSS is : {global_fdr_thresh}")

for sim in pain_results_dct.keys():  # similarity metric (e.g., 'euclidean', 'annak')
    for cond in pain_results_dct[sim].keys():  # e.g., 'Analgesia', 'Hyperalgesia'
        
        model_id = sim + "_" + cond
        rsa_df = pain_results_dct[sim][cond]
        # Ensure the dataframe rows are in the same order as the atlas labels

        correlations = rsa_df["spearman_r"].values
        p_values = rsa_df["p_values"].values
        region_id = rsa_df["ROI"].values

        for correction in ["unc01", "within_FDR","within_cond_FDR", "FDR", "bonf"]:
            show = False

            if correction == "unc01":
                p_thresh = 0.01
                p_thresh_str = "0.01"
                show = False
            elif correction == "within_FDR":
                p_thresh = isc_utils.fdr(p_values, q=0.05)
                p_thresh_str = "within_FDR" + str(round(p_thresh, 4))
                show=True
            elif correction == "within_cond_FDR": # whole brain * 2 models
                condition_vals = ana_p_values if 'ANA' in model_id else hyper_p_values
                p_thresh = isc_utils.fdr(condition_vals, q=0.05)
                p_thresh_str = "within_cond_FDR" + str(round(p_thresh, 4))
                show = False
            elif correction == "FDR":
                p_thresh = global_fdr_thresh  # or load global value
                p_thresh_str = "FDR" + str(round(p_thresh, 4))
                show = True

            elif correction == "bonf":
                p_thresh = isc_utils.bonferroni(p_values, alpha=0.05)
                p_thresh_str = "Bonf" + str(round(p_thresh, 4))
                show = False
            print(f"\ncorrection: {correction} with p-thresh: {p_thresh}")

            rsa_img, rsa_thresh, sig_df, unc_img = (
                visu_utils.project_isc_to_brain_perm(
                    atlas_img=atlas,
                    isc_median=correlations,
                    atlas_labels=id_labels_dct,
                    roi_coords=roi_coords,
                    p_values=p_values,
                    p_threshold=p_thresh,
                    title=f"RSA MVPA {sim} – {cond} ({p_thresh_str})",
                    save_path=None,
                    show=show,
                    display_mode="x",
                    cut_coords_plot=(5),
                    color="coolwarm",
                )
            )
            
            rsa_stats_imgs_mvpa_unc[model_id] = unc_img
            
            if correction == 'unc01':
                rsa_stats_imgs_liberal[model_id] = (rsa_img, rsa_thresh)
                rsa_tables_mvpa_liberal[model_id] = sig_df
                
            if correction == "within_FDR" and sig_df.shape[0] > 0:
                rsa_stats_imgs_fdr[model_id] = (rsa_img, rsa_thresh)
                rsa_tables_mvpa_fdr[model_id] = sig_df

                sig_df.columns = [
                    "ROI",
                    "Label",
                    "Spearman r",
                    "p-values",
                    "Coordinates (X,Y,Z)",
                ]
                fname = f"RSA_MVPA_{sim}_{cond}_table_{p_thresh_str}.csv"
                # sig_df.to_csv(os.path.join(save_tables_shock_shss, fname), index=False)
                # #save unc. stats img
                # nib.save(unc_img, os.path.join(save_tables_shock_shss, f"RSA_MVPA_{sim}_{cond}_unc_stats_img.nii.gz"))
                # print(f"Sig ROIs for {sim} – {cond}:", sig_df["Label"].to_list(), "\n")
                view = view_img(rsa_img, threshold=rsa_thresh, title=f"RSA MVPA {sim} - {cond} ({p_thresh_str})")
                view.save_as_html(os.path.join(save_plots_shock_shss, f"viewer_RSA_{sim}_{cond}_{p_thresh_str}.html"))

#%%
# SAVE TARGET REGION FROM SIGNIFICANT PATH 2 (RSA SHSS~suggestion_ISC)
# Use liberal thresholding as a good compromise between choosing 
# target regions only FDR sig, vs. whole brain

# FDR region only
target_roi_ana = {}
target_roi_hyper = {}
for model_id in rsa_tables_mvpa_fdr.keys():
    if 'ANA' in model_id:
        region_ids = list(rsa_tables_mvpa_fdr[model_id]['ROI'].values)
        region_name = list(rsa_tables_mvpa_fdr[model_id]['Label'].values)
        cond_dct = {id:name for id, name in zip(region_ids, region_name)}
        target_roi_ana = {**target_roi_ana, **cond_dct}

    elif 'HYPER' in model_id:
        region_ids = list(rsa_tables_mvpa_fdr[model_id]['ROI'].values)
        region_name = list(rsa_tables_mvpa_fdr[model_id]['Label'].values)
        cond_dct = {id:name for id, name in zip(region_ids, region_name)}
        target_roi_hyper = {**target_roi_hyper, **cond_dct}

target_roi_ana = {}
target_roi_hyper = {}
for model_id in rsa_tables_mvpa_fdr.keys():
    if 'ANA' in model_id:
        region_ids = list(rsa_tables_mvpa_liberal[model_id]['ROI'].values)
        region_name = list(rsa_tables_mvpa_liberal[model_id]['Label'].values)

        cond_dct = {id:name for id, name in zip(region_ids, region_name)}
        target_roi_ana = {**target_roi_ana, **cond_dct}
    elif 'HYPER' in model_id:
        region_ids = list(rsa_tables_mvpa_liberal[model_id]['ROI'].values)
        region_name = list(rsa_tables_mvpa_liberal[model_id]['Label'].values)
        cond_dct = {id:name for id, name in zip(region_ids, region_name)}
        target_roi_hyper = {**target_roi_hyper, **cond_dct}

# manually add left NAc because of Desmarteau et al., 2021
target_roi_ana['202'] = 'AMY-rh'
target_roi_ana['205'] = 'NAc-rh'
target_roi_ana['210'] = 'AMY-lh'
target_roi_ana['213'] = 'NAc-lh'

target_roi_hyper['202'] = 'AMY-rh'
target_roi_hyper['205'] = 'NAc-rh'
target_roi_hyper['210'] = 'AMY-lh'
target_roi_hyper['213'] = 'NAc-lh'

#%%
# initialize results for ISC-MVPA RSA
# ================================

# final mvpa sim model defined before (dot model = final)
pain_results_dct = pain_results_dct_all[NEURAL_SIM]
similarity_matrices_mvpa = similarity_matrices_dct_all[NEURAL_SIM]['euclidean']
# save_mvpa_rsa = f"/data/rainville/dSutterlin/projects/ISC_hypnotic_suggestions/results/imaging/RSA/IS-RSA-mvpa-{final_metric}-{effect_type}_contrast-based_tian216" #_{ANNAK_VERSION}-annak"
save_rsa_isc_mvpa = os.path.join(save_rsa_isc_mvpa_dir, f"sugg-pain_mvpa-{effect_type}-maps_{NEURAL_SIM}-neural-sim_liberal01_targets")
os.makedirs(save_rsa_isc_mvpa, exist_ok=True)

# Have been overwrite by trial model code !
save_plots_rsa = os.path.join(save_rsa_isc_mvpa, "plots_rsa")
os.makedirs(save_plots_rsa, exist_ok=True)
save_tables_rsa = os.path.join(save_rsa_isc_mvpa, "tables_rsa")
os.makedirs(save_tables_rsa, exist_ok=True)


#%%
# PERFORM RSA between seed ISC matrices and MVPA similarity matrices
from tqdm import tqdm

n_perm_rsa = 5
LOAD_RESULTS_ONLY = False
TAIL = 2
neural_sim = 'dot'  # for visu only, NEURAL_SIM is defined earlier !

if LOAD_RESULTS_ONLY:
    rsa_isc_mvpa_results = isc_utils.load_pickle(os.path.join(
        save_rsa_isc_mvpa, f"IS-RSA_sugg_pain_mvpa-{neural_sim}_{n_perm_rsa}perm_{TAIL}tails.pkl"
    ))
    seed_matrices_conds = {}
    target_matrices_conds = {}

    for cond in combined_conditions:
        seed_matrices_conds[cond] = isc_utils.load_pickle(os.path.join(
            save_rsa_isc_mvpa, f"seed_similarity_ISC_mat_{cond}.pkl"))
        
        target_matrices_conds[cond] = isc_utils.load_pickle(os.path.join(save_rsa_isc_mvpa, f"target_similarity_MVPA_mat_{cond}.pkl"))

    print("Loaded existing results")
else:
    rsa_isc_mvpa_results = {}
    distributions_all = {}
    for cond in tqdm(combined_conditions, desc="Conditions"):
        if 'ANA' in cond:
            seed_matrices = seed_regions_ana_matrices
            labels_roi_dct = seed_regions_ana_context
            target_mvpa_matrices_full_brain = similarity_matrices_mvpa['ANA_shock_minus_N_ANA_shock']
            target_mvpa_matrices = {roi_id: mat for roi_id, mat in target_mvpa_matrices_full_brain.items() if str(roi_id) in target_roi_ana.keys()}
        
        elif 'HYPER' in cond:
            seed_matrices = seed_regions_hyper_matrices
            labels_roi_dct = seed_regions_hyper_context
            target_mvpa_matrices_full_brain = similarity_matrices_mvpa['HYPER_shock_minus_N_HYPER_shock']
            target_mvpa_matrices = {roi_id: mat for roi_id, mat in target_mvpa_matrices_full_brain.items() if str(roi_id) in target_roi_hyper.keys()}
       
        print("Performing RSA on : ", cond)
        rsa_isc_mvpa_results[cond] = {}
        distributions_all[cond] = {}
        
        for seed_id, seed_name in tqdm(labels_roi_dct.items(), desc="Seed regions"):
            seed_matrix = np.array(seed_matrices[seed_name])
            np.fill_diagonal(seed_matrix, 0)  # fill diag with 0 for similarity
            seed_vector = isc_utils.isc_matrix_to_vector(seed_matrix) # upper element/ squareform
            
            # RSA computation + perm
            rsa_rows = []  # build df

            for i, (target_id, target_matrix) in tqdm(
                enumerate(target_mvpa_matrices.items()),
                desc="RSA ROIs",
            ):  
                target_matrix = np.array(target_matrix)
                np.fill_diagonal(target_matrix, 0)  # fill diag with 0 for similarity
                target_vector = isc_utils.isc_matrix_to_vector(target_matrix) # upper element/ squareform
    
                r, p, dist = isc_utils.matrix_permutation(
                    seed_vector,
                    target_vector,
                    n_permute=n_perm_rsa,
                    metric="spearman",
                    how="upper",
                    tail=TAIL,
                    return_perms=True,
                )
                # 2 tail when testing behavioral model, as AnnaK can show reverse pattern
                rsa_rows.append(
                    {
                        "ROI": target_id,
                        "label": id_labels_dct[str(target_id)],
                        "spearman_r": r,
                        "p_values": round(p, 5),
                        "Coordinates": tuple(np.round(roi_coords[i], 0).astype(int)),
                    }
                )

            rsa_isc_mvpa_results[cond][seed_name] = pd.DataFrame(rsa_rows)
            distributions_all[cond][seed_name] = dist
            print(
                "Max r, mean and fdr",
                rsa_isc_mvpa_results[cond][seed_name]["spearman_r"].max(),
                rsa_isc_mvpa_results[cond][seed_name]["spearman_r"].mean(),
                isc_utils.fdr(rsa_isc_mvpa_results[cond][seed_name]["p_values"].to_numpy()),
            )
            print("Significant ROIs", rsa_isc_mvpa_results[cond][seed_name][
                rsa_isc_mvpa_results[cond][seed_name]["p_values"] < 0.01
            ])
            print(f"Done with {cond} - {seed_name} (id{seed_id})\n")

        save_to = os.path.join(
        save_rsa_isc_mvpa, f"IS-RSA_sugg_pain_mvpa-{NEURAL_SIM}_{n_perm_rsa}perm_{TAIL}tails.pkl"
        )
        isc_utils.save_data(save_to, rsa_isc_mvpa_results)
        isc_utils.save_data(os.path.join(
            save_rsa_isc_mvpa, f"seed_similarity_ISC_mat_{cond}.pkl"
        ), seed_matrices)
        isc_utils.save_data(os.path.join(save_rsa_isc_mvpa, f"target_similarity_MVPA_mat_{cond}.pkl"), target_mvpa_matrices)
        isc_utils.save_data(os.path.join(save_rsa_isc_mvpa, f"IS-RSA_sugg_pain_mvpa-pain-{NEURAL_SIM}_distributions_{cond}.pkl"), distributions_all)
        print(f"Saved RSA results to {save_to}")

#%%
# VISUALIZATION 
reload(visu_utils)

cond_p_values = {}
global_p_values = []
combined_seed_dfs = {}
for cond in combined_conditions:
    cond_results = rsa_isc_mvpa_results[cond]
    cond_p_values[cond] = []
    if 'ana' in cond: #manual addition 24 oct.
        restrict_target_rois = list(target_roi_ana.values())
    else:
        restrict_target_rois = list(target_roi_hyper.values())
    
    for seed_name in cond_results.keys():
        seed_df = cond_results[seed_name]
        #restrict
        seed_df = seed_df[seed_df['label'].isin(restrict_target_rois)]
        cond_p_values[cond].extend(seed_df['p_values'].values)
        combined_seed_dfs[f"{cond}_{seed_name}"] = seed_df
    cond_p_values[cond] = np.array(cond_p_values[cond])

    global_p_values.extend(cond_p_values[cond])
    fdr_cond = isc_utils.fdr(cond_p_values[cond], q=0.05)
    print(f"within condition FDR threshold for {cond} : {fdr_cond:.4f}")

global_p_thresh = isc_utils.fdr(np.array(global_p_values), q=0.05)
print(f"Global FDR threshold across conditions : {global_p_thresh:.4f}")

# correction = "FDR"  # "unc01", "within_FDR", "cond-wise_FDR", "FDR", "bonf"
#%%
liberal_rsa_imgs = {}
fdr_rsa_imgs = {}
sig_seed_dfs = {}
target_dfs = {}

def find_dct_key_by_value(dct, value):
    for k, v in dct.items():
        if v == value:
            return k
    return None

for correction in ['within_FDR']: #, 'unc01']:
    print(f'\n Correction : {correction}')
    for cond in combined_conditions:
        
        print(f"Projecting results for {cond}\n--------------------")
        cond_results = rsa_isc_mvpa_results[cond]
        sig_seed_dfs[cond] = {}

        restrict_target_rois = target_roi_ana if 'ANA' in cond else target_roi_hyper
        restrict_rois_ids = list(restrict_target_rois.keys())
        restrict_rois_ids_int = [int(i) for i in restrict_rois_ids]
    
        restrict_rois_names = list(restrict_target_rois.values())

        for seed_name in cond_results.keys():
            seed_id = find_dct_key_by_value(id_labels_dct, seed_name)
            print(f"Seed region (id): {seed_name} ({seed_id}) \n")
             # get significant results
            # get seed to whole brain (216) RSA dfs
            seed_df = cond_results[seed_name]
            target_dfs[seed_name] = seed_df
            # full_brain_correlations = seed_df["spearman_r"].values
            # full_brain_p_values = seed_df["p_values"].values
            
            # # constraining to certain predifined ROIs (from sig/unc MVPA RSA during pain)
            # print('restrincting tests to target ROIS : ', restrict_rois_names)
            # seed_df_restricted_target = seed_df[seed_df['ROI'].isin(restrict_rois_ids_int)]
            # target_indices = list(seed_df_restricted_target['ROI'].isin(restrict_rois_ids_int).index)
            # target_rois_p_values = full_brain_p_values[target_indices]
            brain_correlations = seed_df["spearman_r"].values
            brain_p_values = seed_df["p_values"].values

            if correction == "FDR":
                p_thresh = global_p_thresh
                p_thresh_str = "FDR" + str(round(p_thresh, 4))
                show = True
                print(f"FDR threshold for {cond} - {seed_name}: {p_thresh:.4f}")
            elif correction == "within_FDR":
                p_thresh = isc_utils.fdr(brain_p_values, q=0.05) # only on tested target rois
                p_thresh_str = "within_FDR" + str(round(p_thresh, 4))
                print(f"Within FDR threshold for {cond} - {seed_name}: {p_thresh:.4f}")
                show = True
            elif correction == "unc01":
                p_thresh = 0.01
                p_thresh_str = "0.01"
                show = True

            if brain_p_values.min() > p_thresh:
                print(f"No significant ROIs for {cond} - {seed_name} at p<{p_thresh}")
                continue
            # keep only target elements
            # from nilearn.image import new_img_like
            # def filter_atlas(atlas_img, atlas_labels_dict, keep_ids):
            #     atlas_data = atlas_img.get_fdata()
            #     mask_data = np.isin(atlas_data, keep_ids) * atlas_data
            #     filtered_img = new_img_like(atlas_img, mask_data)

            #     # Build new labels dict for the kept IDs
            #     new_labels = {
            #         roi: label for roi, label in atlas_labels_dict.items() if roi in keep_ids
            #     }
            #     return filtered_img, new_labels
            # filt_atlas, filt_id_labels_dct = filter_atlas(atlas, id_labels_dct, restrict_rois_ids)

            # project to brain
            rsa_img, rsa_thresh, sig_df, unc_img = visu_utils.project_isc_to_brain_perm(
                atlas_img=atlas,
                isc_median=brain_correlations,
                atlas_labels={k: v for k, v in id_labels_dct.items() if k in restrict_rois_ids},
                roi_coords=roi_coords,
                p_values=brain_p_values,
                p_threshold=p_thresh,  #!!
                title=f"IS-RSA SHSS~{cond} ({p_thresh_str})",  # "RSA-ISC: Suggestion-Pain Similarity (FDR<.05)",
                save_path=None,
                show=show,
                display_mode="x",
                cut_coords_plot=(5),  # (-52, -40, 34),
                color="coolwarm",
            )
            liberal_rsa_imgs[f"{cond}_{seed_name}"] = (rsa_img, rsa_thresh)
            if correction == "unc01":
                fdr_rsa_imgs[f"{cond}_{seed_name}"] = (rsa_img, rsa_thresh)

            if sig_df.shape[0]>0:
                sig_df = sig_df.sort_values('r Difference', ascending=False)
                sig_df.columns = [
                    "ROI",
                    "Label",
                    "Spearman r",
                    "p-values",
                    "Coordinates (X,Y,Z)",
                ]
                sig_seed_dfs[cond][seed_name] = sig_df
                print(f"\nSignificant results for {cond} - {seed_name} : \n", sig_df)
                sig_df.to_csv(
                    os.path.join(
                        save_tables_rsa,
                        f"IS-RSA-ISC-sugg_mvpa-pain-{neural_sim}_{cond}_seed-{seed_name}_table_{p_thresh_str}.csv",
                    ),
                    index=False,
                )
                view = plotting.view_img(
                    rsa_img,
                    threshold=rsa_thresh,
                    title=f"IS-RSA SHSS~{cond} seed-{seed_name} ({p_thresh_str})",
                    bg_img=bg_mni,
                    cmap="coolwarm",
                )
                view.save_as_html(
                    os.path.join(
                        save_plots_rsa,
                        f"view_{neural_sim}_{cond}_isc-mvpa_seed-{seed_name}_{p_thresh_str}.html"
                    )
                )

#%%
# #--==============================================================================
# # %%

# y_conditions = [
#     "SHSS_score",
#     "raw_change_ANA",
#     "raw_change_HYPER",
#     "total_chge_pain_hypAna",
# ]


# def plot_simmat(simmat, sim_method="Eucledian", y_name="SHSS_score"):
#     plt.figure(figsize=(10, 8))
#     plt.imshow(simmat, vmin=0, vmax=1, cmap="coolwarm")
#     plt.colorbar(label="Similarity")
#     plt.title(f"{sim_method} {y_name} similarity matrix")
#     plt.xlabel("Subject rank")
#     plt.ylabel("Subject rank")
#     plt.show()


# for metric in ["euclidean", "annak"]:
#     for y_name in ["SHSS_score"]:

#         y = Y[y_name].values
#         y = (y - np.mean(y)) / np.std(y)
#         sim_behav = isc_utils.compute_behav_similarity(
#             y, metric=metric, vectorize=False
#         )

#         ranks = np.argsort(y)  # gives indices that sort behavior low → high
#         sim_ranked = sim_behav[np.ix_(ranks, ranks)]

#         plot_simmat(sim_ranked, sim_method="Eucledian", y_name=y_name)

# # %%
# # ================================
# # Multivariate behavioral similarity
# # ================================
# from sklearn.preprocessing import StandardScaler

# sugg_cols = [
#     "SHSS_score",
#     "Chge_hypnotic_depth",
#     "Mental_relax_absChange",
#     "Abs_diff_automaticity",
# ]

# # pain_cols = [
# #     "VAS_Nana_Int",
# #     "VAS_Ana_Int",
# #     "VAS_Nhyper_Int",
# #     "VAS_Hyper_Int",
# #     "VAS_Nana_UnP",
# #     "VAS_Ana_UnP",
# #     "VAS_Nhyper_UnP",
# #     "VAS_Hyper_UnP"
# # ]

# pain_cols = ["raw_change_HYPER", "raw_change_ANA"]

# scaler = StandardScaler()
# Y_sugg = scaler.fit_transform(np.array(Y[sugg_cols].values, dtype=float))
# Y_pain = scaler.fit_transform(np.array(Y[pain_cols].values, dtype=float))

# plt.figure()
# plt.imshow(Y_sugg)
# plt.colorbar(label="scores")
# plt.xticks(np.arange(len(sugg_cols)), sugg_cols, rotation=45, ha="right")
# plt.yticks(np.arange(Y.shape[0]), Y.index)
# plt.title("Behavioral Features Heatmap")
# plt.tight_layout()

# plt.figure()
# plt.imshow(Y_pain)
# plt.colorbar(label="scores")
# plt.xticks(np.arange(len(pain_cols)), pain_cols, rotation=45, ha="right")
# plt.yticks(np.arange(Y.shape[0]), Y.index)
# plt.title("Behavioral Features Heatmap")
# plt.tight_layout()

# # Compute pairwise cosine similarity using the previously defined function
# cosine_sim_sugg = isc_utils.compute_behav_similarity(
#     Y_sugg, metric="cosine", vectorize=False
# )
# cosine_sim_pain = isc_utils.compute_behav_similarity(
#     Y_pain, metric="cosine", vectorize=False
# )

# plt.figure()
# plt.imshow(cosine_sim_sugg, cmap="coolwarm")
# plt.colorbar(label="Cosine Similarity")
# plt.title("Cosine Similarity: multivariate hypnonic scores")
# plt.tight_layout()

# plt.figure()
# plt.imshow(cosine_sim_pain, cmap="coolwarm")
# plt.colorbar(label="Cosine Similarity")
# plt.title("Cosine Similarity: multivariate pain scores")
# plt.tight_layout()

# # test correlation between univariate and multivariate similarity

# # suggestion
# y = Y["SHSS_score"].values
# y = (y - np.mean(y)) / np.std(y)
# sim_behav_vec = isc_utils.compute_behav_similarity(
#     y, metric="euclidean", vectorize=True
# )
# cosine_vec_sugg = isc_utils.compute_behav_similarity(
#     Y_sugg, metric="cosine", vectorize=True
# )
# r, p = spearmanr(cosine_vec_sugg, sim_behav_vec)
# print(
#     f"Spearman correlation between uni - multivariate SUGG var.: {r:.4f}, p-value: {p:.4f}"
# )

# y = Y["total_chge_pain_hypAna"].values
# y = (y - np.mean(y)) / np.std(y)
# sim_behav_vec = isc_utils.compute_behav_similarity(
#     y, metric="euclidean", vectorize=True
# )
# cosine_vec_pain = isc_utils.compute_behav_similarity(
#     Y_pain, metric="cosine", vectorize=True
# )

# r, p = spearmanr(cosine_vec_pain, sim_behav_vec)
# print(
#     f"Spearman correlation between uni - multivariate PAIN var.: {r:.4f}, p-value: {p:.4f}"
# )

# # %%

# # sns.clustermap(sim_sugg,
# #                metric='euclidean',
# #                method='average',
# #                cmap='coolwarm',
# #                xticklabels=True,
# #                yticklabels=True)

# # %%
# # ================================
# # Atlas
# # ================================
# SCHAEFER_ONLY = True

# atlas = nib.load(
#     "/data/rainville/dSutterlin/projects/ISC_hypnotic_suggestions/masks/Tian2020_schaeffer200_subcortical16/combined_schaefer200_tian16_DSG.nii.gz"
# )
# id_labels_dct = isc_utils.load_json(
#     "/data/rainville/dSutterlin/projects/ISC_hypnotic_suggestions/masks/Tian2020_schaeffer200_subcortical16/roi_labels_dict_DSG.json"
# )
# # remove background
# id_labels_dct.pop("0")
# labels = list(id_labels_dct.values())
# roi_index = list(id_labels_dct.keys())

# atlas_masker = NiftiLabelsMasker(labels_img=atlas, labels=labels, standardize=False)
# atlas_masker.fit()

# if SCHAEFER_ONLY:
#     atlas_data = fetch_atlas_schaefer_2018(n_rois=200, resolution_mm=2)
#     atlas = nib.load(atlas_data["maps"])
#     # atlas_path = atlas_data['maps'] #os.path.join(project_dir,os.path.join(project_dir, 'masks', 'k50_2mm', '*.nii*'))
#     # labels_bytes = list(atlas_data['labels'])
#     labels = [
#         str(label, "utf-8") if isinstance(label, bytes) else str(label)
#         for label in atlas_data["labels"]
#     ]
#     roi_index = [i + 1 for i in range(len(labels))]  # no background, 0
#     atlas_labels = dict(zip(roi_index, labels))

# else:
#     atlas_masker = NiftiLabelsMasker(
#         labels_img=atlas, atlas_labels=labels, standardize=False
#     )
#     atlas_masker.fit()
#     atlas_labels = dict(zip(roi_index, labels))
#     labels = list(atlas_labels.values())

# coords = find_parcellation_cut_coords(labels_img=atlas)
# # #%% VISU
# # from nilearn import plotting
# # reload(visu_utils)

# # rsa_path = '/data/rainville/dSutterlin/projects/ISC_hypnotic_suggestions/results/imaging/RSA/2025-05-21_23'
# # # rsa_isc = isc_utils.load_pickle(os.path.join(save_path, f'rsa_isc_pain-behav_sugg-pain_{n_perm_rsa}perm.pkl'))
# # rsa_isc = isc_utils.load_pickle('/data/rainville/dSutterlin/projects/ISC_hypnotic_suggestions/results/imaging/RSA/rsa_pain-behav_sugg-pain_10000perm.pkl')
# # rsa_isc_sugg = isc_utils.load_pickle(os.path.join(rsa_path, 'rsa_cosine-sugg-behav_isc-sugg5000perm.pkl' ))

# # views={}
# # for cond in ['HYPER', 'ANA', 'all_sugg', 'neutral']:

# #     print('cond', cond)
# #     rsa_df = rsa_isc[cond].sort_index(ascending=True) # to match the atlas labels

# #     # === Prepare variables for projection ===
# #     correlations = rsa_df['spearman_r'].values
# #     p_values = rsa_df['p_values'].values
# #     roi_labels = rsa_df['ROI'].values  # assumes label matches atlas
# #     fdr_p = isc_utils.fdr(p_values, q=0.05)
# #     print(f'FDR threshold: {fdr_p:.4f}')

# #     # === Map ROI label names to atlas index ===
# #     label_to_index = {label: idx for idx, label in enumerate(labels)}
# #     roi_indices = [label_to_index[roi] for roi in roi_labels]

# #     # === Create full-length arrays aligned with atlas ===
# #     # full_r_values = np.zeros(len(labels))
# #     # full_p_values = np.ones(len(labels))

# #     # for idx, roi_idx in enumerate(roi_indices):
# #     #     full_r_values[roi_idx] = correlations[idx]
# #     #     full_p_values[roi_idx] = p_values[idx]

# #     unc_p = 0.01
# #     # === Visualize with your existing function ===
# #     rsa_img, rsa_thresh, sig_labels = visu_utils.project_isc_to_brain_perm(
# #         atlas_img=atlas,
# #         isc_median=correlations,
# #         atlas_labels=atlas_labels,
# #         roi_coords = coords,
# #         p_values=p_values,
# #         p_threshold=fdr_p, #!!
# #         title=None, #"RSA-ISC: Suggestion-Pain Similarity (FDR<.05)",
# #         save_path=None,
# #         show=True,
# #         display_mode='x',
# #         cut_coords_plot=None, #(-52, -40, 34),
# #         color='Reds'
# #     )

# #     views[cond] = plotting.view_img(rsa_img, threshold=rsa_thresh, title=f"RSA suggestion - pain similarity {cond}", colorbar=True,symmetric_cmap=False, cmap = 'Reds')


# # %%
# # =================================
# # IS-RSA for ISC-SUGGESTION  & ISC-PAIN

# do_brain_sugg_with_pain = False
# if do_brain_sugg_with_pain:

#     model_sugg = model_names["model1_sugg_200_run"]
#     model_pain = model_names["model3_shock_200_run"]

#     rsa_per_cond = {}
#     for cond in tqdm(conditions):
#         print(f"Running RSA for condition: {cond}")
#         # sugg_path = f'/data/rainville/dSutterlin/projects/ISC_hypnotic_suggestions/results/imaging/ISC/{model_sugg}/concat_suggs_1samp_boot/isc_results_{cond}_{n_perm}boot_pairWiseTrue.pkl'
#         # pain_path = f'/data/rainville/dSutterlin/projects/ISC_hypnotic_suggestions/results/imaging/ISC/{model_pain}/concat_suggs_1samp_boot/isc_results_{cond}_{n_perm}boot_pairWiseTrue.pkl'

#         if cond in ["ANA", "HYPER", "NANA"]:
#             sugg_path = f"/data/rainville/dSutterlin/projects/ISC_hypnotic_suggestions/results/imaging/ISC/{model_sugg}/{cond}/isc_results_{cond}_{n_perm}boot_pairWiseTrue.pkl"
#             pain_path = f"/data/rainville/dSutterlin/projects/ISC_hypnotic_suggestions/results/imaging/ISC/{model_pain}/{cond}/isc_results_{cond}_{n_perm}boot_pairWiseTrue.pkl"
#         else:
#             sugg_path = f"/data/rainville/dSutterlin/projects/ISC_hypnotic_suggestions/results/imaging/ISC/{model_sugg}/concat_suggs_1samp_boot/isc_results_{cond}_{n_perm}boot_pairWiseTrue.pkl"
#             pain_path = f"/data/rainville/dSutterlin/projects/ISC_hypnotic_suggestions/results/imaging/ISC/{model_pain}/concat_suggs_1samp_boot/isc_results_{cond}_{n_perm}boot_pairWiseTrue.pkl"

#         isc_sugg = pd.DataFrame(isc_utils.load_pickle(sugg_path)["isc"], columns=labels)
#         isc_pain = pd.DataFrame(isc_utils.load_pickle(pain_path)["isc"], columns=labels)

#         assert isc_sugg.shape == isc_pain.shape

#         rsa_rows = []  # build df
#         for roi_idx, roi in enumerate(isc_sugg.columns):
#             sugg_vec = isc_sugg[roi].values
#             pain_vec = isc_pain[roi].values
#             r, p, dist = isc_utils.matrix_permutation(
#                 sugg_vec,
#                 pain_vec,
#                 n_permute=n_perm_rsa,
#                 metric="spearman",
#                 how="upper",
#                 tail=n_tail_rsa,
#                 return_perms=True,
#             )

#             rsa_rows.append(
#                 {
#                     "ROI": roi,
#                     "spearman_r": r,
#                     "p_values": round(p, 5),
#                     "x": coords[roi_idx][0],
#                     "y": coords[roi_idx][1],
#                     "z": coords[roi_idx][2],
#                 }
#             )

#         rsa_per_cond[cond] = pd.DataFrame(rsa_rows).sort_values(
#             by="spearman_r", ascending=False
#         )
#         print(
#             "Max r, mean and fdr",
#             rsa_per_cond[cond]["spearman_r"].max(),
#             rsa_per_cond[cond]["spearman_r"].mean(),
#             isc_utils.fdr(rsa_per_cond[cond]["p_values"].to_numpy()),
#         )

#     save_to = os.path.join(save_path, f"rsa_isc-sugg_isc-pain_{n_perm_rsa}perm.pkl")
#     isc_utils.save_data(save_to, rsa_per_cond)
#     print(f"Saved RSA results to {save_to}")
#     print("-----Done with rsa!-----")

# # #%%
# # # RSA ISFC
# # reload(isc_utils)
# # isfc_rsa_per_cond = {}

# # isfc_model_sugg = model_names['model5_isfc_sugg']
# # isfc_model_pain = model_names['model5_isfc_shock']
# # n_perm = 5000 # for loading

# # for cond in conditions:
# #     print(f'Running ISFC RSA for condition: {cond}')
# #     sugg_path = f'/data/rainville/dSutterlin/projects/ISC_hypnotic_suggestions/results/imaging/ISC/{model_sugg}/concat_suggs_1samp_boot/isc_results_{cond}_{n_perm}boot_pairWiseTrue.pkl'
# #     pain_path = f'/data/rainville/dSutterlin/projects/ISC_hypnotic_suggestions/results/imaging/ISC/{model_pain}/concat_suggs_1samp_boot/isc_results_{cond}_{n_perm}boot_pairWiseTrue.pkl'

# #     isfc_sugg = pd.DataFrame(isc_utils.load_pickle(sugg_path)['isc'], columns=labels)
# #     isfc_pain = pd.DataFrame(isc_utils.load_pickle(pain_path)['isc'], columns=labels)

# #     assert isc_sugg.shape == isfc_pain.shape

# #     rsa_rows = [] # build df
# #     for roi_idx, roi in enumerate(isc_sugg.columns):
# #         sugg_vec = isfc_sugg[roi].values
# #         pain_vec = isfc_pain[roi].values
# #         r, p, dist = isc_utils.matrix_permutation(sugg_vec, pain_vec, n_permute=n_perm_rsa, metric="spearman", how="upper", tail=n_tail_rsa, return_perms=True)

# #         rsa_rows.append({
# #             'ROI': roi,
# #             'spearman_r': r,
# #             'p_values': round(p, 5),
# #             'x': coords[roi_idx][0],
# #             'y': coords[roi_idx][1],
# #             'z': coords[roi_idx][2]
# #         })

# #     isfc_rsa_per_cond[cond] = pd.DataFrame(rsa_rows).sort_values(by='spearman_r', ascending=False)
# #     print('Max r, mean and fdr', isfc_rsa_per_cond[cond]['spearman_r'].max(), isfc_rsa_per_cond[cond]['spearman_r'].mean(), isc_utils.fdr(isfc_rsa_per_cond[cond]['p_values'].to_numpy()))

# # save_to = os.path.join(save_path, f'rsa_isfc_sugg-pain_{n_perm_rsa}perm.pkl')
# # isc_utils.save_data(save_to, isfc_rsa_per_cond)
# # print(f'Saved RSA results to {save_to}')
# # print('-----Done with ISFC rsa!-----')

# # %%
# reload(isc_utils)
# from tqdm import tqdm

# # =================================
# # IS-RSA for SUGGESTION (behav suggestion + behav pain)
# # =================================

# # check if all cond are in isc_results
# save_to = os.path.join(save_path, f"rsa_cosine-behav_isc-sugg{n_perm_rsa}perm.pkl")
# pairwise_behav = {"sugg": cosine_vec_sugg, "pain": cosine_vec_pain}
# rsa_behav_per_cond = {}

# for domain in pairwise_behav.keys():
#     rsa_behav_per_cond[domain] = {}
#     # conditions = ['ANA', 'HYPER', 'all_sugg'] # 'neutral']
#     for cond in conditions:
#         if cond in ["ANA", "HYPER", "NANA"]:
#             pain_path = f"/data/rainville/dSutterlin/projects/ISC_hypnotic_suggestions/results/imaging/ISC/{model_sugg}/{cond}/isc_results_{cond}_{n_perm}boot_pairWiseTrue.pkl"
#         else:
#             pain_path = f"/data/rainville/dSutterlin/projects/ISC_hypnotic_suggestions/results/imaging/ISC/{model_sugg}/concat_suggs_1samp_boot/isc_results_{cond}_{n_perm}boot_pairWiseTrue.pkl"

#         vec_behav_sim = pairwise_behav[domain]  # specify if sugg or pain
#         isc_pairwise = pd.DataFrame(
#             isc_utils.load_pickle(pain_path)["isc"], columns=labels
#         )

#         rsa_rows = []  # build df
#         for roi_idx, roi in tqdm(
#             enumerate(isc_pairwise.columns), total=len(isc_pairwise.columns)
#         ):
#             isc_roi = isc_pairwise[roi].values
#             r, p, dist = isc_utils.matrix_permutation(
#                 vec_behav_sim,
#                 isc_roi,
#                 n_permute=n_perm_rsa,
#                 metric="spearman",
#                 how="upper",
#                 tail=n_tail_rsa,
#                 return_perms=True,
#             )

#             rsa_rows.append(
#                 {
#                     "ROI": roi,
#                     "spearman_r": r,
#                     "p_values": round(p, 5),
#                     "x": coords[roi_idx][0],
#                     "y": coords[roi_idx][1],
#                     "z": coords[roi_idx][2],
#                 }
#             )

#         rsa_behav_per_cond[domain][cond] = pd.DataFrame(rsa_rows).sort_values(
#             by="spearman_r", ascending=False
#         )
#         print(
#             "Max r, mean and fdr",
#             rsa_behav_per_cond[domain][cond]["spearman_r"].max(),
#             rsa_behav_per_cond[domain][cond]["spearman_r"].mean(),
#             isc_utils.fdr(rsa_behav_per_cond[domain][cond]["p_values"].to_numpy()),
#         )

# isc_utils.save_data(save_to, rsa_behav_per_cond)
# print(f"Saved RSA results to {save_to}")


# # %%
# # =================================
# # IS-RSA for PAIN
# # =================================

# # check if all cond are in isc_results
# save_to = os.path.join(save_path, f"rsa_cosine-behav_isc-pain{n_perm_rsa}perm.pkl")
# conditions = ["ANA", "NANA", "HYPER", "modulation", "all_sugg", "neutral"]
# pairwise_behav = {"sugg-sim": cosine_vec_sugg, "pain-sim": cosine_vec_pain}
# rsa_behav_per_cond = {}

# for domain in pairwise_behav.keys():
#     rsa_behav_per_cond[domain] = {}

#     for cond in conditions:
#         if cond in ["ANA", "HYPER", "NANA"]:
#             pain_path = f"/data/rainville/dSutterlin/projects/ISC_hypnotic_suggestions/results/imaging/ISC/{model_pain}/{cond}/isc_results_{cond}_{n_perm}boot_pairWiseTrue.pkl"
#         else:
#             pain_path = f"/data/rainville/dSutterlin/projects/ISC_hypnotic_suggestions/results/imaging/ISC/{model_pain}/concat_suggs_1samp_boot/isc_results_{cond}_{n_perm}boot_pairWiseTrue.pkl"

#         vec_behav_sim = pairwise_behav[domain]  # specify if sugg or pain
#         isc_pairwise = pd.DataFrame(
#             isc_utils.load_pickle(pain_path)["isc"], columns=labels
#         )

#         rsa_rows = []  # build df
#         for roi_idx, roi in enumerate(isc_pairwise.columns):
#             isc_roi = isc_pairwise[roi].values
#             r, p, dist = isc_utils.matrix_permutation(
#                 vec_behav_sim,
#                 isc_roi,
#                 n_permute=n_perm_rsa,
#                 metric="spearman",
#                 how="upper",
#                 tail=n_tail_rsa,
#                 return_perms=True,
#             )

#             rsa_rows.append(
#                 {
#                     "ROI": roi,
#                     "spearman_r": r,
#                     "p_values": round(p, 5),
#                     "x": coords[roi_idx][0],
#                     "y": coords[roi_idx][1],
#                     "z": coords[roi_idx][2],
#                 }
#             )

#         rsa_behav_per_cond[domain][cond] = pd.DataFrame(rsa_rows).sort_values(
#             by="spearman_r", ascending=False
#         )
#         print(
#             "Max r, mean and fdr",
#             rsa_behav_per_cond[domain][cond]["spearman_r"].max(),
#             rsa_behav_per_cond[domain][cond]["spearman_r"].mean(),
#             isc_utils.fdr(rsa_behav_per_cond[domain][cond]["p_values"].to_numpy()),
#         )

# isc_utils.save_data(save_to, rsa_behav_per_cond)
# print(f"Saved RSA results to {save_to}")


# # %%
# print(f"Saved all RSA results to {save_path}")
# print("Done with all RSA!")
# # %%

# %%
