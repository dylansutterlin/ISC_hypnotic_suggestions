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

# # Sensaas
# # condition_target = ['all_sugg', 'A'
# atlas = nib.load('/data/rainville/dSutterlin/projects/ISC_hypnotic_suggestions/masks/sensaas/SENSAAS_MNI_ICBM_152_2mm.nii')
# atlas_data = pd.read_csv('/data/rainville/dSutterlin/projects/ISC_hypnotic_suggestions/masks/sensaas/SENSAAS_description.csv')
# # atlas = qc_utils.resamp_to_img_mask(atlas_native, ref_img)
# # view_img(atlas, threshold=0.5, title='SENSAAS atlas')
# #sensaas ids and labels
# roi_index = atlas_data['Index'].values
# atlas_data['annot_abbreviation'] = atlas_data['Abbreviation'] + '_' + atlas_data['Hemisphere']
# full_labels = atlas_data['annot_abbreviation'].values
# id_labels_dct = dict(zip(atlas_data['Index'], atlas_data['annot_abbreviation']))

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
# Load 1 sample ISC for runs/contexts : Hyper & Analgesia-context
# Will be used for plotting region ISC matrices
# stack contrast imgs in `contrast_imgs[cont]`

# Reload BLOCK model
model_name = model_names["model2_sugg"]
results_dir = os.path.join(project_dir, f"results/imaging/ISC/{model_name}")
setup = isc_utils.load_json(os.path.join(results_dir, "setup_parameters.json"))

result_key = "isc_results"
interactive_views = {}
combined_conditions = [
    "ana_run",
    "hyper_run",
]  # 'neutral']#setup['combined_conditions'] #['all_sugg', 'modulation', 'neutral']

isc_matrices = {}
iscs_cond = {}
isc_p_cond = {}
fdr_isc_imgs = {}
unc_isc_maps = {}
liberal_isc_imgs = {}
sig_isc_matrices = {}
sig_isc_tables = {}

# Load ISC results
for i, cond in enumerate(combined_conditions):
    print(
        f"Fetching 1 sample ISC values from BLOCK model, for contextual effect.\n {cond}"
    )
    isc_bootstrap = isc_utils.load_pickle(
        os.path.join(
            results_dir,
            f"concat_suggs_1samp_boot/isc_results_{cond}_{n_boot}boot_pairWise{do_pairWise}.pkl",
        )
    )
    isc_rois = pd.DataFrame(isc_bootstrap["isc"], columns=full_labels)
    isc_matrices[cond] = {
        region: isc_utils.vector_to_isc_matrix(isc_rois.iloc[:, col_j], diag=0)
        for col_j, region in enumerate(isc_rois.columns)
    }

    isc_median = isc_bootstrap["observed"]
    ci = isc_bootstrap["confidence_intervals"]
    p_values = isc_bootstrap["p_values"]
    dist = isc_bootstrap["distribution"]
    n_boot = setup["n_boot"]

    iscs_cond[cond] = isc_median  # one sample variables (vs. delta_iscs for contrasts)
    isc_p_cond[cond] = p_values

# FDR thresh across all conditions since
all_ps = []
all_iscs = []
for cond in combined_conditions:
    p = isc_p_cond[cond]
    all_ps.extend(p)
    all_iscs.extend(iscs_cond[cond])  # used for global scale/colorbar in plots
fdr_all_1samp = isc_utils.fdr(np.array(all_ps), q=0.05)
print("\nFDR across all conditions : ", fdr_all_1samp)


#%%
# LOAD RSA from ISC to produce seed regions
# # %%
# IS-RSA SHSS~ISC
# ===============
# Load results and save tables

load_dir = os.path.join(
    project_dir, "results/imaging/RSA/isc-RSA_ext_conds_sugg-pain_tian216_2tails"
)

RESULT_FILE = "rsa_SHSS-behav_isc-sugg10000perm.pkl"
rsa_isc_sugg = isc_utils.load_pickle(os.path.join(load_dir, RESULT_FILE))
rsa_isc_sugg['euclidean'] = rsa_isc_sugg.pop('euclidian') # fix typo in original script 

load_dir = os.path.join(
    project_dir, "results/imaging/RSA/isc-RSA_tian216_sugg-pain_combined_conds"
)

save_rsa = os.path.join(load_dir, "post-hoc_VISU-tables")
os.makedirs(save_rsa, exist_ok=True)

save_tables_rsa = os.path.join(save_rsa, "tables_rsa")
os.makedirs(save_tables_rsa, exist_ok=True)
save_plots_rsa = os.path.join(save_rsa, "plots_rsa")
os.makedirs(save_plots_rsa, exist_ok=True)

n_perm = 10000

rsa_tables_fdr = {}  # changed for `euclidean`` in original script
rsa_table_unc = {}
rsa_dfs = {}
models = ["annak", "euclidean"] #potentially euclidian !!
model = "annak"

rsa_fdr_imgs = {}
rsa_unc_stats_imgs = {}
rsa_liberal_imgs = {}
rsa_p_values = []

cond_p_thresh = {}
# Load rsa results and find FDR thresh
model_p_values = []
condition_wise_p_values = {}

for cond in [
        "ana_run",
        "hyper_run",
    ]:  # , 'HYPER', 'ANA']: #, 'all_sugg', 'neutral']:
    for model in models:

        model_cond_key = f"{model}_{cond}"

        print("\ncond", model_cond_key,)
        rsa_df = rsa_isc_sugg[model][cond].sort_index(
            ascending=True
        )  # to match the atlas labels
        rsa_dfs[model_cond_key] = rsa_df

        # === Prepare variables for projection ===
        correlations = rsa_df["spearman_r"].values
        p_values = rsa_df["p_values"].values
        rsa_p_values.extend(p_values)

        model_p_values.extend(p_values)

        fdr_p = isc_utils.fdr(p_values, q=0.05)
        condition_wise_p_values[model_cond_key] = p_values
        print(f"Within FDR threshold: {fdr_p:.4f}")
        print(f"Min p values : {p_values.min():.4f}")
        print(f"max spearman: {correlations.max():.4f}")

    cond_p_thresh[cond] = isc_utils.fdr(np.array(model_p_values), q=0.05)
    print(f"FDR threshold for condition {cond}: {cond_p_thresh[cond]:.4f}\n")

# compute across condition FDR thresh
fdr_all_rsa = isc_utils.fdr(np.array(rsa_p_values), q=0.05)
print(f"Across conditions FDR threshold: {fdr_all_rsa:.4f}")

# condition-wise p values
ana_p_values = np.array([])
hyper_p_values = np.array([])
for model_id in condition_wise_p_values.keys():
    if 'ANA' in model_id:
        ana_p_values = np.concatenate((ana_p_values, condition_wise_p_values[model_id]))
    elif 'HYP' in model_id:
        hyper_p_values = np.concatenate((hyper_p_values, condition_wise_p_values[model_id]))

# Project RSA values to brain and save tables
for correction in ["unc01", "within_FDR", "cond-wise_FDR", "FDR", "bonf"]:
    for (
        rsa_cond
    ) in (
        rsa_dfs.keys()
    ):  # ['ana_run', 'hyper_run']: #, 'HYPER', 'ANA']: #, 'all_sugg', 'neutral']:

        rsa_df = rsa_dfs[rsa_cond]
        correlations = rsa_df["spearman_r"].values
        p_values = rsa_df["p_values"].values
        region_id = rsa_df["ROI"].values

        show = False
        if correction == "unc01":
            p_thresh = 0.01
            p_thresh_str = "0.01"

        if correction == "within_FDR":
            p_thresh = isc_utils.fdr(p_values, q=0.05)
            p_thresh_str = "within_FDR" + str(round(p_thresh, 4))
            print(f"Within FDR threshold for {rsa_cond}: {p_thresh:.4f}")

        if correction == "FDR":
            p_thresh = fdr_all_rsa
            p_thresh_str = "FDR" + str(round(p_thresh, 4))
            print(f"FDR threshold for {rsa_cond}: {p_thresh:.4f}")
            show = True

        if correction == "cond-wise_FDR":
            model = rsa_cond.split("_")[0]
            cond_p_vals = ana_p_values if 'ana' in rsa_cond else hyper_p_values
            p_thresh = isc_utils.fdr(cond_p_vals, q=0.05)
            p_thresh_str = f"cond-wise_FDR_{cond}" + str(round(p_thresh, 4))
            show = False
            print(f"Condition-wise FDR threshold for {rsa_cond}: {p_thresh:.4f}")

        if correction == "bonf":
            p_thresh = isc_utils.bonferroni(p_values, alpha=0.05)
            p_thresh_str = "Bonf" + str(round(p_thresh, 4))

        rsa_img, rsa_thresh, sig_df, unc_img = visu_utils.project_isc_to_brain_perm(
            atlas_img=atlas,
            isc_median=correlations,
            atlas_labels=id_labels_dct,
            roi_coords=roi_coords,
            p_values=p_values,
            p_threshold=p_thresh,  #!!
            title=f"IS-RSA SHSS~{rsa_cond} ({p_thresh_str})",  # "RSA-ISC: Suggestion-Pain Similarity (FDR<.05)",
            save_path=None,
            show=show,
            display_mode="x",
            cut_coords_plot=(5),  # (-52, -40, 34),
            color="coolwarm",
        )

        rsa_unc_stats_imgs[rsa_cond] = (unc_img, rsa_thresh)

        if correction == "unc01":
            rsa_table_unc[rsa_cond] = sig_df
            rsa_liberal_imgs[rsa_cond] = (rsa_img, rsa_thresh)

        if correction == "FDR" and sig_df.shape[0] > 0:
            
            rsa_fdr_imgs[rsa_cond] = (rsa_img, rsa_thresh)
            rsa_tables_fdr[rsa_cond] = sig_df

            sig_df.columns = [
            "ROI",
            "Label",
            "Spearman r",
            "p-values",
            "Coordinates (X,Y,Z)",
            ]
            
            sig_df.to_csv(
                os.path.join(
                    save_tables_rsa, f"RSA_{rsa_cond}_table_{p_thresh_str}.csv"
                ),
                index=False,
            )
            # save uncorrected stats img
            rsa_img.to_filename(
                os.path.join(
                    save_tables_rsa, f"RSA_stats_img_{rsa_cond}_uncorrected.nii.gz"
                )
            )

            print("Sig ROIs : ", sig_df["Label"].to_list(), "\n")
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

# all seeds positive and negative
for sim_model in rsa_tables_fdr.keys(): # ONLY POSITIVE REGIONS
    if 'ana' in sim_model:
        region_ids = list(rsa_tables_fdr[sim_model]['ROI'].values)
        region_name = list(rsa_tables_fdr[sim_model]['Label'].values)

        cond_dct = {id:name for id, name in zip(region_ids, region_name)}
        seed_regions_ana_context = {**seed_regions_ana_context, **cond_dct}
    elif 'hyper' in sim_model:
        region_ids = list(rsa_tables_fdr[sim_model]['ROI'].values)
        region_name = list(rsa_tables_fdr[sim_model]['Label'].values)
        cond_dct = {id:name for id, name in zip(region_ids, region_name)}
        seed_regions_hyper_context = {**seed_regions_hyper_context, **cond_dct}

# get matrices
for region_id, region in seed_regions_ana_context.items():
    seed_regions_ana_matrices[region] = isc_matrices['ana_run'][region]
for region_id, region in seed_regions_hyper_context.items():
    seed_regions_hyper_matrices[region] = isc_matrices['hyper_run'][region]


#%%
# LOAD MVPA SIM MATRICES for RSA
# ================================

from src.mvpa_rsa_utils import compute_inter_subject_mvpa_similarity, build_subject_dict

results_dct_all = {}
similarity_matrices_dct_all = {}
for NEURAL_SIM in ['dot', 'cosine']: #, 'dot']: #, 'cosine']: #, 'dot']: #['euclidean', 'dot']: #'cosine', 'pearson']:
    # NEURAL_SIM = 'euclidean'  # 'cosine' or 'dot' or 'euclidean'
    # BEHAV_ID = "SHSS"  # "SHSS" or "chge-pain"
    n_perm_rsa = 10000
    effect_type = 'stat' #'effect_size' #'z_score'
    save_mvpa_rsa = f"/data/rainville/dSutterlin/projects/ISC_hypnotic_suggestions/results/imaging/RSA/IS-RSA-mvpa-{NEURAL_SIM}-{effect_type}_contrast-based_tian216" #_{ANNAK_VERSION}-annak"
    os.makedirs(save_mvpa_rsa, exist_ok=True)

    BEHAV_ID = "SHSS_score"  # "SHSS" or "chge-pain" or "pain_diff_Ana" or "pain_diff_Hyper" or "total_change_pain"
    pain_conditions = ["ANA_shock_minus_N_ANA_shock", "HYPER_shock_minus_N_HYPER_shock"]
   
    # load results
    load_path = os.path.join(
        save_mvpa_rsa, f"IS-RSA-mvpa_pain-{NEURAL_SIM}_behav-{BEHAV_ID}_{n_perm_rsa}perm.pkl"
    )
    results_dct_all[NEURAL_SIM] = isc_utils.load_pickle(load_path)
    similarity_matrices_dct_all[NEURAL_SIM] = isc_utils.load_pickle(os.path.join(
        save_mvpa_rsa, f"similarity_mat_IS-RSA-mvpa_pain-{NEURAL_SIM}_behav-{BEHAV_ID}_{n_perm_rsa}perm.pkl"
    ))

# main results with dot-based similarity
final_metric = 'cosine'
results_dct = results_dct_all[final_metric]
similarity_matrices_mvpa = similarity_matrices_dct_all[final_metric]['euclidean']
# save_mvpa_rsa = f"/data/rainville/dSutterlin/projects/ISC_hypnotic_suggestions/results/imaging/RSA/IS-RSA-mvpa-{final_metric}-{effect_type}_contrast-based_tian216" #_{ANNAK_VERSION}-annak"
save_rsa_isc_mvpa = os.path.join(save_rsa_isc_mvpa_dir, f"isc_mvpa-{effect_type}-maps_{final_metric}-neural-sim") 
os.makedirs(save_rsa_isc_mvpa, exist_ok=True)

# Have been overwrite by trial model code !
save_plots_rsa = os.path.join(save_rsa_isc_mvpa, "plots_rsa")
os.makedirs(save_plots_rsa, exist_ok=True)
save_tables_rsa = os.path.join(save_rsa_isc_mvpa, "tables_rsa")
os.makedirs(save_tables_rsa, exist_ok=True)



#%%
# PERFORM RSA between seed ISC matrices and MVPA similarity matrices
from tqdm import tqdm

n_perm_rsa = 10000
LOAD_RESULTS_ONLY = False

if LOAD_RESULTS_ONLY:
    rsa_isc_mvpa_results = isc_utils.load_pickle(os.path.join(
        save_rsa_isc_mvpa, f"IS-RSA-ISC-sugg_mvpa-pain-{NEURAL_SIM}_{n_perm_rsa}perm.pkl"
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
    for cond in tqdm(combined_conditions, desc="Conditions"):
        if 'ana' in cond:
            seed_matrices = seed_regions_ana_matrices
            labels_roi_dct = seed_regions_ana_context
            target_mvpa_matrices = similarity_matrices_mvpa['ANA_shock_minus_N_ANA_shock']
        elif 'hyper' in cond:
            seed_matrices = seed_regions_hyper_matrices
            labels_roi_dct = seed_regions_hyper_context
            target_mvpa_matrices = similarity_matrices_mvpa['HYPER_shock_minus_N_HYPER_shock']
        print("Performing RSA on : ", cond)
        rsa_isc_mvpa_results[cond] = {}

        for seed_id, seed_name in tqdm(labels_roi_dct.items(), desc="Seed regions"):
            seed_matrix = seed_matrices[seed_name]
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
                    tail=2,
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
            print(
                "Max r, mean and fdr",
                rsa_isc_mvpa_results[cond][seed_name]["spearman_r"].max(),
                rsa_isc_mvpa_results[cond][seed_name]["spearman_r"].mean(),
                isc_utils.fdr(rsa_isc_mvpa_results[cond][seed_name]["p_values"].to_numpy()),
            )
            print("Significant ROIs", rsa_isc_mvpa_results[cond][seed_name][
                rsa_isc_mvpa_results[cond][seed_name]["p_values"] < 0.01
            ])
            print(f"Done with {cond} - {seed_name}\n")

        save_to = os.path.join(
        save_rsa_isc_mvpa, f"IS-RSA-ISC-sugg_mvpa-pain-{NEURAL_SIM}_{n_perm_rsa}perm.pkl"
        )
        isc_utils.save_data(save_to, rsa_isc_mvpa_results)
        isc_utils.save_data(os.path.join(
            save_rsa_isc_mvpa, f"seed_similarity_ISC_mat_{cond}.pkl"
        ), seed_matrices)
        isc_utils.save_data(os.path.join(save_rsa_isc_mvpa, f"target_similarity_MVPA_mat_{cond}.pkl"), target_mvpa_matrices)
        print(f"Saved RSA results to {save_to}")

# #%%
# VISUALIZATION 
cond_p_values = {}
global_p_values = []
for cond in combined_conditions:
    cond_results = rsa_isc_mvpa_results[cond]
    cond_p_values[cond] = []

    for seed_name in cond_results.keys():
        seed_df = cond_results[seed_name]
        cond_p_values[cond].extend(seed_df['p_values'].values)
    cond_p_values[cond] = np.array(cond_p_values[cond])

    global_p_values.extend(cond_p_values[cond])
    fdr_cond = isc_utils.fdr(cond_p_values[cond], q=0.05)
    print(f"within condition FDR threshold for {cond} : {fdr_cond:.4f}")

global_p_thresh = isc_utils.fdr(np.array(global_p_values), q=0.05)
print(f"Global FDR threshold across conditions : {global_p_thresh:.4f}")

# correction = "FDR"  # "unc01", "within_FDR", "cond-wise_FDR", "FDR", "bonf"

liberal_rsa_imgs = {}
fdr_rsa_imgs = {}
for correction in ['FDR', 'unc01']:
    print(f'\n Correction : {correction}')
    for cond in combined_conditions:
        print(f"Projecting results for {cond}\n--------------------")
        cond_results = rsa_isc_mvpa_results[cond]
        for seed_name in cond_results.keys():
            seed_df = cond_results[seed_name]
            sig_df = seed_df[seed_df['p_values']<global_p_thresh]

            correlations = seed_df["spearman_r"].values
            p_values = seed_df["p_values"].values
            region_id = seed_df["ROI"].values

            if correction == "FDR":
                p_thresh = global_p_thresh
                p_thresh_str = "FDR" + str(round(p_thresh, 4))
                show = False
                print(f"FDR threshold for {cond} - {seed_name}: {p_thresh:.4f}")
            elif correction == "unc01":
                p_thresh = 0.01
                p_thresh_str = "0.01"
                show = True

            # project to brain
            rsa_img, rsa_thresh, sig_df, unc_img = visu_utils.project_isc_to_brain_perm(
                atlas_img=atlas,
                isc_median=correlations,
                atlas_labels=id_labels_dct,
                roi_coords=roi_coords,
                p_values=p_values,
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
                    print(f"\nSignificant results for {cond} - {seed_name} : \n", sig_df)

                    sig_df.to_csv(
                        os.path.join(
                            save_tables_rsa,
                            f"IS-RSA-ISC-sugg_mvpa-pain-{final_metric}_{cond}_seed-{seed_name}_table_{p_thresh_str}.csv",
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
                            f"view_{final_metric}_{cond}_isc-mvpa_seed-{seed_name}_{p_thresh_str}.html"
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
