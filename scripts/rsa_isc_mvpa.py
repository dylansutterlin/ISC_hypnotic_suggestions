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

# create output dir.
save_rsa_isc_mvpa_dir = os.path.join(project_dir, "results/imaging/IS-RSA_ISC-MVPA")
os.makedirs(save_rsa_isc_mvpa_dir, exist_ok=True)

parcel_name = setup["atlas_name"]
do_pairWise = setup["do_pairwise"]
n_boot = setup["n_boot"]

# post_hoc_dir = os.path.join(results_dir, "post_hoc_results")
# os.makedirs(post_hoc_dir, exist_ok=True)

    # %% Load atlas: Schaefer + Tian 216 ROIs
from src.nilearn_helper import Atlas
from nilearn import datasets

# Combined Schaefer + Tian subcortical (216 ROIs)
atlas_obj = Atlas("schaefer_tian216")
coords = atlas_obj.get_coords()
id_labels_dct = atlas_obj.id_labels_dct
labels_roi_dct = id_labels_dct # ctl H to change ...
atlas_labels = list(atlas_obj.id_labels_dct.values())
atlas = atlas_obj.maps
atlas_masker = atlas_obj.get_masker(
    atlas_obj.maps,
    atlas_obj.probabilistic,
    kwargs={
        "smoothing_fwhm": None,
        "standardize": True,
        "memory_level": 2,

    }
)

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

# Reload ISC BLOCK model --> get ISC matrices
model_name = model_names["model2_sugg"]
results_dir = os.path.join(project_dir, f"results/imaging/ISC/{model_name}")
setup = isc_utils.load_json(os.path.join(results_dir, "setup_parameters.json"))

result_key = "isc_results"
interactive_views = {}
combined_conditions = [
    #"ANA",
     "ana_run",
     "hyper_run" ]  # 'neutral']#setup['combined_conditions'] #['all_sugg', 'modulation', 'neutral']

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
    if cond =='ANA':
        isc_bootstrap = isc_utils.load_pickle(os.path.join(results_dir, f"{cond}/isc_results_{cond}_{n_boot}boot_pairWise{do_pairWise}.pkl"))
    else:
            
        isc_bootstrap = isc_utils.load_pickle(
            os.path.join(
                results_dir,
                f"concat_suggs_1samp_boot/isc_results_{cond}_{n_boot}boot_pairWise{do_pairWise}.pkl",
            )
        )
    
    isc_rois = pd.DataFrame(isc_bootstrap["isc"], columns=atlas_labels)
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

load_dir = os.path.join(
    project_dir, "results/imaging/RSA/isc-RSA_ext_conds_sugg-pain_tian216_2tails"
)

RESULT_FILE = "rsa_SHSS-behav_isc-sugg10000perm.pkl"
rsa_isc_sugg = isc_utils.load_pickle(os.path.join(load_dir, RESULT_FILE))
rsa_isc_sugg['euclidean'] = rsa_isc_sugg.pop('euclidian') # fix typo in original script 

# load_dir = os.path.join(
#     project_dir, "results/imaging/RSA/isc-RSA_tian216_sugg-pain_combined_conds"
# ) # Check final model modif nov. 25 !! See results/X/folder for cp scripts

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
models = ["annak", "euclidean"] #potentially `euclidian` !!
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
        #"ANA",
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
    if 'ANA' in model_id or 'ana' in model_id:
        ana_p_values = np.concatenate((ana_p_values, condition_wise_p_values[model_id]))
    elif 'HYP' in model_id or 'hyp' in model_id:
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
            show=True

        if correction == "FDR":
            p_thresh = fdr_all_rsa
            p_thresh_str = "FDR" + str(round(p_thresh, 4))
            print(f"FDR threshold for {rsa_cond}: {p_thresh:.4f}")
            show = False

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
            roi_coords=coords,
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
# LOAD path 2 results: IS-RSA SHSS MVPA to produce target rois (restrict search)
#%%
# =================
# VISUALISE PAIN-contrast ~ SHSS
# IS-RSA SHSS ~ ISC-PAIN
from src.mvpa_rsa_utils import compute_inter_subject_mvpa_similarity, build_subject_dict

results_dct_all = {}
n_perm_rsa = 10000
similarity_matrices_dct_all = {}
for NEURAL_SIM in ['dot', 'cosine']: #, 'dot']: #, 'cosine']: #, 'dot']: #['euclidean', 'dot']: #'cosine', 'pearson']:
    # NEURAL_SIM = 'euclidean'  # 'cosine' or 'dot' or 'euclidean'
    # BEHAV_ID = "SHSS"  # "SHSS" or "chge-pain"
    effect_type = 'stat' #'effect_size' #'z_score'
    save_path = f"/data/rainville/dSutterlin/projects/ISC_hypnotic_suggestions/results/imaging/RSA/IS-RSA-mvpa-{NEURAL_SIM}-{effect_type}_contrast-based_tian216_reproduced" #_{ANNAK_VERSION}-annak"
    os.makedirs(save_path, exist_ok=True)

    BEHAV_ID = "SHSS_score"  # "SHSS_score" or "chge-pain" or "pain_diff_Ana" or "pain_diff_Hyper" or "total_change_pain"
    pain_conditions = ["ANA_shock_minus_N_ANA_shock", "HYPER_shock_minus_N_HYPER_shock"]
   
    # load results
    load_path = os.path.join(
        save_path, f"IS-RSA-mvpa_pain-{NEURAL_SIM}_behav-{BEHAV_ID}_{n_perm_rsa}perm.pkl"
    )
    results_dct_all[NEURAL_SIM] = isc_utils.load_pickle(load_path)
    similarity_matrices_dct_all[NEURAL_SIM] = isc_utils.load_pickle(os.path.join(
        save_path, f"similarity_mat_IS-RSA-mvpa_pain-{NEURAL_SIM}_behav-{BEHAV_ID}_{n_perm_rsa}perm.pkl"
    ))

# main results with dot-based similarity
NEURAL_SIM = 'dot'
results_dct = results_dct_all[NEURAL_SIM]
similarity_matrices_mvpa = similarity_matrices_dct_all[NEURAL_SIM]['euclidean'] # Euclidean = Annak here ! 
save_path = f"/data/rainville/dSutterlin/projects/ISC_hypnotic_suggestions/results/imaging/RSA/IS-RSA-mvpa-{NEURAL_SIM}-{effect_type}_contrast-based_tian216_reproduced" #_{ANNAK_VERSION}-annak"

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
for sim in results_dct.keys():  # similarity metric (e.g., 'euclidean', 'annak')
    for cond in results_dct[sim].keys():  # e.g., 'Analgesia', 'Hyperalgesia'
        
        model_id = sim + "_" + cond
        rsa_df = results_dct[sim][cond]
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

for sim in results_dct.keys():  # similarity metric (e.g., 'euclidean', 'annak')
    for cond in results_dct[sim].keys():  # e.g., 'Analgesia', 'Hyperalgesia'
        
        model_id = sim + "_" + cond
        rsa_df = results_dct[sim][cond]
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
                show=False
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
                    roi_coords=coords,
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
                
            if correction == "FDR" and sig_df.shape[0] > 0:
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
# initialize results for ISC-MVPA RSA
# ================================

# save_mvpa_rsa = f"/data/rainville/dSutterlin/projects/ISC_hypnotic_suggestions/results/imaging/RSA/IS-RSA-mvpa-{final_metric}-{effect_type}_contrast-based_tian216" #_{ANNAK_VERSION}-annak"
save_rsa_isc_mvpa = os.path.join(save_rsa_isc_mvpa_dir, f"isc_mvpa-{effect_type}-maps_{NEURAL_SIM}-neural-sim_reproduced") #_liberal01_targets")
os.makedirs(save_rsa_isc_mvpa, exist_ok=True)

# Have been overwrite by trial model code !
save_plots_rsa = os.path.join(save_rsa_isc_mvpa, "plots_rsa")
os.makedirs(save_plots_rsa, exist_ok=True)
save_tables_rsa = os.path.join(save_rsa_isc_mvpa, "tables_rsa")
os.makedirs(save_tables_rsa, exist_ok=True)

if BEHAV_ID == "pain_diff_Ana" or BEHAV_ID == "SHSS_score":
    combined_conditions = ["ana_run", "hyper_run"]

#%%
# SAVE TARGET REGION FROM SIGNIFICANT PATH 2 (RSA SHSS~suggestion_ISC)
# target regions only FDR sig, vs. whole brain

# Diff. condition test families. No suppresion effects, only pos-pos or neg- neg across seed-target tests
isc_mvpa_cond_family = ['ANA_annak_positive', 'ANA_annak_negative', 'ANA_euclidean_positive',
                       'HYPER_annak_positive', 'HYPER_annak_negative']
target_roi_all_conditions = {}
target_roi_mvpa_matrices = {}

for cond in combined_conditions:

    if 'ana' in cond:

        target_mvpa_matrices_all_regions = similarity_matrices_mvpa['ANA_shock_minus_N_ANA_shock']

        # Annak pos + negative effects
        if 'annak_ANA_shock_minus_N_ANA_shock' in rsa_tables_mvpa_fdr.keys():

            mvpa_sig_df = rsa_tables_mvpa_fdr['annak_ANA_shock_minus_N_ANA_shock']

            pos_region_ids = list(mvpa_sig_df['ROI'][mvpa_sig_df['Spearman r']>0].values)
            pos_region_name = list(mvpa_sig_df['Label'][mvpa_sig_df['Spearman r']>0].values)

            neg_region_ids = list(mvpa_sig_df['ROI'][mvpa_sig_df['Spearman r']<0].values)
            neg_region_name = list(mvpa_sig_df['Label'][mvpa_sig_df['Spearman r']<0].values)

            # Stack in respective condition family dct
            target_roi_all_conditions['ANA_annak_positive'] = {id:name for id, name in zip(pos_region_ids, pos_region_name)}
            target_roi_all_conditions['ANA_annak_negative'] = {id:name for id, name in zip(neg_region_ids, neg_region_name)}

            target_roi_mvpa_matrices['ANA_annak_positive'] = {id:target_mvpa_matrices_all_regions[id] for id, name in zip(pos_region_ids, pos_region_name)}
            target_roi_mvpa_matrices['ANA_annak_negative'] = {id:target_mvpa_matrices_all_regions[id] for id, name in zip(neg_region_ids, neg_region_name)}

        if 'euclidean_ANA_shock_minus_N_ANA_shock' in rsa_tables_mvpa_fdr.keys():

            mvpa_sig_df = rsa_tables_mvpa_fdr['euclidean_ANA_shock_minus_N_ANA_shock']

            region_ids = list(mvpa_sig_df['ROI'].values)
            region_name = list(mvpa_sig_df['Label'].values)
            target_roi_all_conditions['ANA_euclidean_positive'] = {id:name for id, name in zip(region_ids, region_name)}
            target_roi_mvpa_matrices['ANA_euclidean_positive'] = {id:target_mvpa_matrices_all_regions[id] for id, name in zip(region_ids, region_name)}

    if 'hyper' in cond:

        target_mvpa_matrices_all_regions = similarity_matrices_mvpa['HYPER_shock_minus_N_HYPER_shock']

        # Annak pos + negative effects
        if 'annak_HYPER_shock_minus_N_HYPER_shock' in rsa_tables_mvpa_fdr.keys():

            mvpa_sig_df = rsa_tables_mvpa_fdr['annak_HYPER_shock_minus_N_HYPER_shock']

            pos_regions_ids = list(mvpa_sig_df['ROI'][mvpa_sig_df['Spearman r']>0].values)
            pos_region_name = list(mvpa_sig_df['Label'][mvpa_sig_df['Spearman r']>0].values)

            neg_region_ids = list(mvpa_sig_df['ROI'][mvpa_sig_df['Spearman r']<0].values)
            neg_region_name = list(mvpa_sig_df['Label'][mvpa_sig_df['Spearman r']<0].values)

            target_roi_all_conditions['HYPER_annak_positive'] = {id:name for id, name in zip(pos_regions_ids, pos_region_name)}
            target_roi_all_conditions['HYPER_annak_negative'] = {id:name for id, name in zip(neg_region_ids, neg_region_name)}

            target_roi_mvpa_matrices['HYPER_annak_positive'] = {id:target_mvpa_matrices_all_regions[id] for id, name in zip(pos_regions_ids, pos_region_name)}
            target_roi_mvpa_matrices['HYPER_annak_negative'] = {id:target_mvpa_matrices_all_regions[id] for id, name in zip(neg_region_ids, neg_region_name)}
        
        if 'euclidean_HYPER_shock_minus_N_HYPER_shock' in rsa_tables_mvpa_fdr.keys():

            mvpa_sig_df = rsa_tables_mvpa_fdr['euclidean_HYPER_shock_minus_N_HYPER_shock']

            region_ids = list(mvpa_sig_df['ROI'].values)
            region_name = list(mvpa_sig_df['Label'].values)
            target_roi_all_conditions['HYPER_euclidean_positive'] = {id:name for id, name in zip(region_ids, region_name)}
            target_roi_mvpa_matrices['HYPER_euclidean_positive'] = {id:target_mvpa_matrices_all_regions[id] for id, name in zip(region_ids, region_name)}
#--------------------
# SEED regions
# Seed regions from path 1 : RSA from ISC matrices and suggestibility
# Split seeds into pos/neg effects to perform RSA with target with same direction of FX

seed_roi_all_conditions = {}
seed_regions_isc_matrices = {}

for cond in combined_conditions:

    if 'ana' in cond:
        
        if 'annak_ana_run' in rsa_tables_fdr.keys():

            rsa_sig_df = rsa_tables_fdr['annak_ana_run']

            pos_region_ids = list(rsa_sig_df['ROI'][rsa_sig_df['Spearman r']>0].values)
            pos_region_name = list(rsa_sig_df['Label'][rsa_sig_df['Spearman r']>0].values)

            neg_region_ids = list(rsa_sig_df['ROI'][rsa_sig_df['Spearman r']<0].values)
            neg_region_name = list(rsa_sig_df['Label'][rsa_sig_df['Spearman r']<0].values)

            seed_roi_all_conditions['ANA_annak_positive'] = {id:name for id, name in zip(pos_region_ids, pos_region_name)}
            seed_roi_all_conditions['ANA_annak_negative'] = {id:name for id, name in zip(neg_region_ids, neg_region_name)}

            # isc matrices
            seed_regions_isc_matrices['ANA_annak_positive'] = {id:isc_matrices['ana_run'][name] for id, name in zip(pos_region_ids, pos_region_name)}
            seed_regions_isc_matrices['ANA_annak_negative'] = {id:isc_matrices['ana_run'][name] for id, name in zip(neg_region_ids, neg_region_name)}

        if 'euclidean_ana_run' in rsa_tables_fdr.keys():
            
            rsa_sig_df = rsa_tables_fdr['euclidean_ana_run']

            region_ids = list(rsa_sig_df['ROI'].values)
            region_name = list(rsa_sig_df['Label'].values)
            seed_roi_all_conditions['ANA_euclidean_positive'] = {id:name for id, name in zip(region_ids, region_name)}

            seed_regions_isc_matrices['ANA_euclidean_positive'] = {id:isc_matrices['ana_run'][name] for id, name in zip(region_ids, region_name)}

    if 'hyper' in cond:

        for model in rsa_tables_fdr.keys():

            if 'annak_hyper_run' in model:

                rsa_sig_df = rsa_tables_fdr['annak_hyper_run']

                pos_region_ids = list(rsa_sig_df['ROI'][rsa_sig_df['Spearman r']>0].values)
                pos_region_name = list(rsa_sig_df['Label'][rsa_sig_df['Spearman r']>0].values)

                neg_region_ids = list(rsa_sig_df['ROI'][rsa_sig_df['Spearman r']<0].values)
                neg_region_name = list(rsa_sig_df['Label'][rsa_sig_df['Spearman r']<0].values)

                seed_roi_all_conditions['HYPER_annak_positive'] = {id:name for id, name in zip(pos_region_ids, pos_region_name)}
                seed_roi_all_conditions['HYPER_annak_negative'] = {id:name for id, name in zip(neg_region_ids, neg_region_name)}

                seed_regions_isc_matrices['HYPER_annak_positive'] = {id:isc_matrices['hyper_run'][name] for id, name in zip(pos_region_ids, pos_region_name)}
                seed_regions_isc_matrices['HYPER_annak_negative'] = {id:isc_matrices['hyper_run'][name] for id, name in zip(neg_region_ids,neg_region_name)}
            
            # if 'euclidean_hyper_run' in model:
            #     rsa_sig_df = rsa_tables_fdr['euclidean_hyper_run']

            #     region_ids = list(rsa_sig_df['ROI'].values)
            #     region_name = list(rsa_sig_df['Label'].values)
            #     seed_roi_all_conditions['HYPER_euclidean_positive'] = {id:name for id, name in zip(region_ids, region_name)}
            #     seed_regions_isc_matrices['HYPER_euclidean_positive'] = {id:isc_matrices['hyper_run'][name] for id, name in zip(region_ids, region_name)}


#%%
# PERFORM RSA between seed ISC matrices and MVPA similarity matrices
from tqdm import tqdm


n_perm_rsa = 10000
LOAD_RESULTS_ONLY = False
TAIL=1
neural_sim = 'dot' # for visu only, NEURAL_SIM is defined earlier !

if LOAD_RESULTS_ONLY:

    # rsa_isc_mvpa_results = isc_utils.load_pickle(os.path.join(
    #     save_rsa_isc_mvpa, f"IS-RSA-ISC-sugg_mvpa-pain-{neural_sim}_{n_perm_rsa}perm_{TAIL}tails.pkl"
    # ))
    seed_isc_matrices_conds = {}
    target_matrices_conds = {}
    rsa_isc_mvpa_results = {}

    for test_cond in isc_mvpa_cond_family:

        # load results dict
        load_from = os.path.join(
        save_rsa_isc_mvpa, f"results_{test_cond}-{NEURAL_SIM}_{n_perm_rsa}perm_{TAIL}tails.pkl"
        )
        rsa_isc_mvpa_results[test_cond] = isc_utils.load_pickle(load_from)

        seed_isc_matrices_conds[test_cond] = isc_utils.load_pickle(os.path.join(
            save_rsa_isc_mvpa, f"seed_ISC_matrices_{test_cond}.pkl"))
        
        target_matrices_conds[test_cond] = isc_utils.load_pickle(os.path.join(save_rsa_isc_mvpa, f"target_MVPA_mat_{test_cond}.pkl"))

    print("Loaded existing results")

else:

# isc_mvpa_cond_family = ['ANA_annak_positive', 'ANA_annak_negative', 'ANA_euclidean_positive',
#                        'HYPER_annak_positive', 'HYPER_annak_negative', 'HYPER_euclidean_positive']

    rsa_isc_mvpa_results = {}
    distributions_all = {}

    for test_cond in tqdm(isc_mvpa_cond_family, desc="Conditions"):

        seed_regions_id_name = seed_roi_all_conditions[test_cond]
        seed_isc_matrices = seed_regions_isc_matrices[test_cond]

        target_regions_id_name = target_roi_all_conditions[test_cond]
        target_mvpa_matrices = target_roi_mvpa_matrices[test_cond]
        
        print("Performing RSA on : ", test_cond)

        rsa_isc_mvpa_results[test_cond] = {}
        distributions_all[test_cond] = {}

        for seed_id, seed_name in tqdm(seed_regions_id_name.items(), desc="Seed regions"):

            seed_region_matrix = seed_isc_matrices[seed_id]
            np.fill_diagonal(seed_region_matrix, 0)  # fill diag with 0 for similarity
            seed_vector = isc_utils.isc_matrix_to_vector(seed_region_matrix) # upper element/ squareform
            
            # RSA computation + perm
            rsa_rows = []  # build df

            for i, (target_roi_id, target_mvpa_matrix) in tqdm(
                enumerate(target_mvpa_matrices.items()),
                desc="RSA ROIs",
            ):  
                target_mvpa_matrix = np.array(target_mvpa_matrix)
                np.fill_diagonal(target_mvpa_matrix, 0)  # fill diag with 0 for similarity
                target_vector = isc_utils.isc_matrix_to_vector(target_mvpa_matrix) # upper element/ squareform
    
                r, p, dist = isc_utils.matrix_permutation(
                    seed_vector,
                    target_vector,
                    n_permute=n_perm_rsa,
                    metric="spearman",
                    how="upper",
                    tail=TAIL,
                    return_perms=True,
                )
                
                region_coord_idx = list(id_labels_dct.keys()).index(int(target_roi_id))
                rsa_rows.append(
                    {
                        "ROI": target_roi_id,
                        "label": id_labels_dct[target_roi_id],
                        "spearman_r": r,
                        "p_values": round(p, 5),
                        "Coordinates": tuple(np.round(coords[region_coord_idx], 0).astype(int)),
                    }
                )

            rsa_isc_mvpa_results[test_cond][seed_name] = pd.DataFrame(rsa_rows)
            distributions_all[test_cond][seed_name] = dist

            print(
                "Max r, mean and fdr",
                rsa_isc_mvpa_results[test_cond][seed_name]["spearman_r"].max(),
                rsa_isc_mvpa_results[test_cond][seed_name]["spearman_r"].mean(),
                isc_utils.fdr(rsa_isc_mvpa_results[test_cond][seed_name]["p_values"].to_numpy()),
            )
            print("Significant ROIs", rsa_isc_mvpa_results[test_cond][seed_name][
                rsa_isc_mvpa_results[test_cond][seed_name]["p_values"] < 0.01
            ])
            print(f"Done with {test_cond} - {seed_name}\n")

        save_to = os.path.join(
        save_rsa_isc_mvpa, f"results_{test_cond}-{NEURAL_SIM}_{n_perm_rsa}perm_{TAIL}tails.pkl"
        )

        isc_utils.save_data(save_to, rsa_isc_mvpa_results[test_cond])
        isc_utils.save_data(os.path.join(
            save_rsa_isc_mvpa, f"seed_ISC_matrices_{test_cond}.pkl"
        ), seed_isc_matrices)

        isc_utils.save_data(os.path.join(save_rsa_isc_mvpa, f"target_MVPA_mat_{test_cond}.pkl"), target_mvpa_matrices)
        isc_utils.save_data(os.path.join(save_rsa_isc_mvpa, f"IS-RSA-ISC-sugg_mvpa-pain-{NEURAL_SIM}_distributions_{test_cond}.pkl"), distributions_all[test_cond])
        print(f"Saved RSA results to {save_to}")

#%%
# VISUALIZATION 
reload(visu_utils)

cond_p_values = {}
global_p_values = []
combined_seed_dfs = {}

for test_cond in isc_mvpa_cond_family:

    cond_results = rsa_isc_mvpa_results[test_cond]
    cond_p_values[test_cond] = []

    for seed_name in cond_results.keys():
        seed_df = cond_results[seed_name]
        cond_p_values[test_cond].extend(seed_df['p_values'].values)
        #restrict
        # seed_df = seed_df[seed_df['label'].isin(restrict_target_rois)]
        # cond_p_values[cond].extend(seed_df['p_values'].values)
        # combined_seed_dfs[f"{cond}_{seed_name}"] = seed_df

    cond_p_values[test_cond] = np.array(cond_p_values[test_cond])

    global_p_values.extend(cond_p_values[test_cond])
    fdr_cond = isc_utils.fdr(cond_p_values[test_cond], q=0.05)
    print(f"within condition FDR threshold for {test_cond} : {fdr_cond:.4f}")

global_p_thresh = isc_utils.fdr(np.array(global_p_values), q=0.05)
print(f"Global FDR threshold across conditions : {global_p_thresh:.4f}")

correction = "FDR"  # "unc01", "within_FDR", "cond-wise_FDR", "FDR", "bonf"
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

for correction in ['family-wise_FDR']: #'within_FDR', 'FDR', 
    print(f'\n Correction : {correction}')

    for test_cond in rsa_isc_mvpa_results.keys():
        
        print(f"Projecting results for {test_cond}\n--------------------")
        cond_results = rsa_isc_mvpa_results[test_cond]

        # extract p values the test_condition
        family_wise_p_values = []
        for seed_name in cond_results.keys():
            seed_df = cond_results[seed_name]
            family_wise_p_values.extend(seed_df['p_values'].values)
        
        family_wise_p_values_p_values = np.array(family_wise_p_values)
        family_wise_thresh = isc_utils.fdr(family_wise_p_values_p_values, q=0.05)
        print(f"Family-wise FDR threshold for {test_cond} : {family_wise_thresh:.4f}")

        sig_seed_dfs[test_cond] = {}
        target_dfs = {}

        for seed_name in cond_results.keys():
            seed_id = find_dct_key_by_value(id_labels_dct, seed_name)
            print(f"Seed region (id): {seed_name} ({seed_id}) \n")
             # get significant results
            # get seed to whole brain (216) RSA dfs
            seed_df = cond_results[seed_name]
            target_dfs[seed_name] = seed_df
            possible_regions_ids = {int(roi_id): roi_name for roi_id, roi_name in target_roi_all_conditions[test_cond].items()}

            brain_correlations = seed_df["spearman_r"].values
            brain_p_values = seed_df["p_values"].values

            if correction == "FDR":
                p_thresh = global_p_thresh
                p_thresh_str = "FDR" + str(round(p_thresh, 4))
                show = True
                print(f"FDR threshold for {test_cond} - {seed_name}: {p_thresh:.4f}")

            elif correction == "family-wise_FDR":
                print('Family wise correlation !!')
                p_thresh = family_wise_thresh
                p_thresh_str = "family-wise_FDR" + str(round(p_thresh, 4))
                show = True
                print(f"Family-wise FDR threshold for {test_cond} - {seed_name}: {p_thresh:.4f}")

            elif correction == "within_FDR":
                p_thresh = isc_utils.fdr(brain_p_values, q=0.05) # only on tested target rois
                p_thresh_str = "within_FDR" + str(round(p_thresh, 4))
                print(f"Within FDR threshold for {test_cond} - {seed_name}: {p_thresh:.4f}")
                show = True

            elif correction == "unc01":
                p_thresh = 0.01
                p_thresh_str = "0.01"
                show = True

            if brain_p_values.min() > p_thresh:
                print(f"No significant ROIs for {test_cond} - {seed_name} at p<{p_thresh}")
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
                atlas_labels=possible_regions_ids,
                roi_coords=coords,
                p_values=brain_p_values,
                p_threshold=p_thresh,  #!!
                title=f"IS-RSA SHSS~{test_cond} ({p_thresh_str})",  # "RSA-ISC: Suggestion-Pain Similarity (FDR<.05)",
                save_path=None,
                show=show,
                display_mode="x",
                cut_coords_plot=(5),  # (-52, -40, 34),
                color="coolwarm",
            )
            liberal_rsa_imgs[f"{test_cond}_{seed_name}"] = (rsa_img, rsa_thresh)

            if correction == "unc01":
                fdr_rsa_imgs[f"{test_cond}_{seed_name}"] = (rsa_img, rsa_thresh)

            if sig_df.shape[0]>0 and correction == "family-wise_FDR":
                sig_df = sig_df.sort_values('r Difference', ascending=False)
                sig_df.columns = [
                    "ROI",
                    "Label",
                    "Spearman r",
                    "p-values",
                    "Coordinates (X,Y,Z)",
                ]
                sig_seed_dfs[test_cond][seed_name] = sig_df
                print(f"\nSignificant results for {test_cond} - {seed_name} : \n", sig_df)
                sig_df.to_csv(
                    os.path.join(
                        save_tables_rsa,
                        f"IS-RSA-ISC-sugg_mvpa-pain-{neural_sim}_{test_cond}_seed-{seed_name}_table_{p_thresh_str}.csv",
                    ),
                    index=False,
                )
                view = plotting.view_img(
                    rsa_img,
                    threshold=rsa_thresh,
                    title=f"IS-RSA SHSS~{test_cond} seed-{seed_name} ({p_thresh_str})",
                    bg_img=bg_mni,
                    cmap="coolwarm",
                )
                view.save_as_html(
                    os.path.join(
                        save_plots_rsa,
                        f"view_{neural_sim}_{test_cond}_isc-mvpa_seed-{seed_name}_{p_thresh_str}.html"
                    )
                )

