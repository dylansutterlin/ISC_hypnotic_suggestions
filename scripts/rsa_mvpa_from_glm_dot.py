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
from src import preproc_utils, visu_utils, qc_utils
import src.glm_utils as utils

from sklearn.utils import Bunch
from importlib import reload
from nilearn import datasets, image
from datetime import datetime

print("current working dir : ", os.getcwd())
# %% [markdown]
## load data
preproc_model_name = r'model2_23subjects_zscore_sample_detrend_25-02-25' #r'model2_3subjects_zscore_sample_detrend_25-02-25'
model_dir = rf"/data/rainville/dSutterlin/projects/ISC_hypnotic_suggestions/results/imaging/preproc_data/{preproc_model_name}"
preproc_model_name = r"model2_23subjects_zscore_sample_detrend_25-02-25"  # r'model2_3subjects_zscore_sample_detrend_25-02-25'

MODEL_NAME = "model3_final-isc_23subjects_nuis_nodrift_31-03-25"  # final model, reproduced Desmarteaux et al., 2019 !! 31 mars

setup_dct = preproc_utils.load_json(os.path.join(model_dir, "setup_parameters.json"))
data_info_dct = preproc_utils.load_pickle(
    os.path.join(model_dir, "data_info_regressors.pkl")
)
masker_params = preproc_utils.load_json(os.path.join(model_dir, "preproc_params.json"))

MAX_ITER = None

setup = Bunch(**setup_dct)
setup.run_id = ["ANA", "HYPER"]
data = Bunch(**data_info_dct)
masker_params = Bunch(**masker_params)
glm_info = Bunch()
subjects = setup.subjects
subjects.sort()

if MAX_ITER == None:
    MAX_ITER = len(subjects)
else:
    subjects = subjects[:MAX_ITER]
    setup.subjects = subjects

regressors_dct = data.regressors_per_conds
condition_names = setup.run_names
tr = setup.tr

reload(utils)
ref_img = nib.load(setup.ana_run[0])
single_img = image.index_img(ref_img, 0)
mask = utils.load_data_mask(ref_img)

mni_temp = datasets.load_mni152_template(resolution=1)
mni_bg = qc_utils.resamp_to_img_mask(mni_temp, mask)

# Output directories for adhoc analyses directly in GLM folder
save_glm = os.path.join(setup.project_dir, "results", "imaging", "GLM", MODEL_NAME)
os.makedirs(save_glm, exist_ok=True)
setup.save_dir = save_glm

save_glm_adhoc = os.path.join(save_glm, "adHoc_analyses_results")
os.makedirs(save_glm_adhoc, exist_ok=True)
save_glm_adhoc_plots = os.path.join(save_glm_adhoc, "plots")
os.makedirs(save_glm_adhoc_plots, exist_ok=True)
save_glm_adhoc_tables = os.path.join(save_glm_adhoc, "tables")
os.makedirs(save_glm_adhoc_tables, exist_ok=True)
save_glm_adhoc_rsa = os.path.join(save_glm_adhoc, "is-rsa_pattern_similarity")
os.makedirs(save_glm_adhoc_rsa, exist_ok=True)

# %%
# ========================================
# Load 1st level maps for further analyses
# ==========================================
from glob import glob as glob
from nilearn.maskers import NiftiLabelsMasker
from nilearn.image import binarize_img
from nilearn.plotting import view_img
from nilearn.datasets import fetch_atlas_schaefer_2018
from nilearn.plotting import find_parcellation_cut_coords
from src import qc_utils, isc_utils

model_res = r"/data/rainville/dSutterlin/projects/ISC_hypnotic_suggestions/results/imaging/GLM/model3_final-isc_23subjects_nuis_nodrift_31-03-25"
model_res = r"/data/rainville/dSutterlin/projects/ISC_hypnotic_suggestions/results/imaging/GLM/model3_final-isc_23subjects_nuis_nodrift_31-03-25"
project_dir = setup.project_dir
results_dir = setup["save_dir"]

# # combine Schaeffer + Tian subcortical atlas or just schaefer
SCHAEFER_ONLY = False

atlas = nib.load(
    "/data/rainville/dSutterlin/projects/ISC_hypnotic_suggestions/masks/Tian2020_schaeffer200_subcortical16/combined_schaefer200_tian16_DSG.nii.gz"
)
id_labels_dct = isc_utils.load_json(
    "/data/rainville/dSutterlin/projects/ISC_hypnotic_suggestions/masks/Tian2020_schaeffer200_subcortical16/roi_labels_dict_DSG.json"
)
# remove background
id_labels_dct.pop("0")
labels = list(id_labels_dct.values())
roi_index = [i + 1 for i in range(len(labels))]  # no background, 0

atlas_masker = NiftiLabelsMasker(labels_img=atlas, labels=labels, standardize=False)
atlas_masker.fit()

if SCHAEFER_ONLY:
    atlas_data = fetch_atlas_schaefer_2018(n_rois=200, resolution_mm=2)
    atlas = nib.load(atlas_data["maps"])
    # atlas_path = atlas_data['maps'] #os.path.join(project_dir,os.path.join(project_dir, 'masks', 'k50_2mm', '*.nii*'))
    # labels_bytes = list(atlas_data['labels'])
    labels = [
        str(label, "utf-8") if isinstance(label, bytes) else str(label)
        for label in atlas_data["labels"]
    ]
    # roi_index = [i+1 for i in range(len(labels))] # no background, 0
    atlas_labels = dict(zip(roi_index, labels))

else:
    atlas_masker = NiftiLabelsMasker(
        labels_img=atlas, atlas_labels=labels, standardize=False
    )
    atlas_masker.fit()
    atlas_labels = dict(zip(roi_index, labels))
    labels = list(atlas_labels.values())

coords = find_parcellation_cut_coords(labels_img=atlas)
labels_roi_dct = dict(zip(roi_index, labels))

ATLAS = atlas
# %%
from sklearn.preprocessing import StandardScaler
from src import preproc_utils
import seaborn as sns
import statsmodels.api as sm

reload(preproc_utils)
reload(isc_utils)
xlsx_path = r"/data/rainville/dSutterlin/projects/ISC_hypnotic_suggestions/masks/Hypnosis_variables_20190114_pr_jc.xlsx"
subjects = list(setup["subjects"])
apm_subjects = ["APM" + subj[4:] for subj in subjects]
print(apm_subjects)

Y, rawY = preproc_utils.load_process_y(xlsx_path, subjects)
#%% load updated BEHAV variables
from sklearn.preprocessing import StandardScaler
xlsx_preproc = os.path.join(project_dir, 'masks/Preprocessed_sugg_pain_behavioral.xlsx' )
behavioral_df = pd.read_excel(xlsx_preproc)  
scaler = StandardScaler(with_std=True) # keep variance, just mean center

behav_interest = {}
behav_interest['pain_diff_Ana'] = scaler.fit_transform(np.array(behavioral_df['pain_diff_Ana'], dtype='float').reshape(-1,1))
behav_interest['pain_diff_Hyper'] = scaler.fit_transform(np.array(behavioral_df['pain_diff_Hyper'], dtype='float').reshape(-1,1))
behav_interest['total_change_pain'] = scaler.fit_transform(np.array(behavioral_df['total_chge_pain_hypAna'], dtype='float').reshape(-1,1))
behav_interest['SHSS_score'] = scaler.fit_transform(np.array(behavioral_df['SHSS_score'], dtype='float').reshape(-1,1))

# %%
# ===========================
# MAIN CODE
from tqdm import tqdm
from nilearn.glm.thresholding import threshold_stats_img

mvpa_save_to = os.path.join(results_dir, "mvpa_similarity")
os.makedirs(mvpa_save_to, exist_ok=True)

n_perm_rsa = 10000

# UNIVARIATE pairwise similarities : NN & AnnaK
y = Y["SHSS_score"].values
y = (y - np.mean(y)) / np.std(y)
sim_behav_vec = isc_utils.compute_behav_similarity(
    y, metric="euclidean", vectorize=True
)
sim_behav_vec_annak = isc_utils.compute_behav_similarity(
    y, metric="annak", vectorize=True
)

# compute pairwise beahv for pain modualtion
y_pain = Y["total_chge_pain_hypAna"].values
y_pain = (y_pain - np.mean(y_pain)) / np.std(y_pain)

sim_behav_vec_pain = isc_utils.compute_behav_similarity(
    y_pain, metric="euclidean", vectorize=True
)
sim_behav_vec_annak_pain = isc_utils.compute_behav_similarity(
    y_pain, metric="annak", vectorize=True
)
# secondary pain variables

annak_sim_matrices = {}
annak_sim_matrices['SHSS_score'] = isc_utils.compute_behav_similarity(y, metric="annak", vectorize=False)
annak_sim_matrices['total_change_pain'] = isc_utils.compute_behav_similarity(behav_interest['total_change_pain'], metric="annak", vectorize=False)
annak_sim_matrices['pain_diff_Ana'] = isc_utils.compute_behav_similarity(behav_interest['pain_diff_Ana'], metric="annak", vectorize=False)
annak_sim_matrices['pain_diff_Hyper'] = isc_utils.compute_behav_similarity(behav_interest['pain_diff_Hyper'], metric="annak", vectorize=False)

NN_sim_matrices = {}
NN_sim_matrices['SHSS_score'] = isc_utils.compute_behav_similarity(y, metric="euclidean", vectorize=False)
NN_sim_matrices['total_change_pain'] = isc_utils.compute_behav_similarity(behav_interest['total_change_pain'], metric="euclidean", vectorize=False)
NN_sim_matrices['pain_diff_Ana'] = isc_utils.compute_behav_similarity(behav_interest['pain_diff_Ana'], metric="euclidean", vectorize=False)
NN_sim_matrices['pain_diff_Hyper'] = isc_utils.compute_behav_similarity(behav_interest['pain_diff_Hyper'], metric="euclidean", vectorize=False)

#%%visualization
for var, mat in NN_sim_matrices.items():

    ranked_indices = np.argsort(pd.Series(behav_interest[var].ravel()).values)

    ord_mat = mat[np.ix_(ranked_indices, ranked_indices)]
    ord_mat = (ord_mat * 2) - 1 # rescale between -1 and 1
    visu_utils.plot_simmat_behav(ord_mat, title=f"NN similarity matrix {var}",)

# %%
# PUBLICATION
# FINAL MODEL SUGGESTION SHSS ~ PAIN CONTRAST
# IS-RSA SHSS ~ ISC-PAIN
from src.mvpa_rsa_utils import compute_inter_subject_mvpa_similarity, build_subject_dict

for NEURAL_SIM in ['dot']: #['euclidean', 'dot', 'cosine', 'pearson']:
    specific_re_run = False
    # NEURAL_SIM = 'euclidean'  # 'cosine' or 'dot' or 'euclidean'
    # BEHAV_ID = "SHSS"  # "SHSS" or "chge-pain"
    effect_type = 'stat' #'effect_size' #'z_score'
    save_path = f"/data/rainville/dSutterlin/projects/ISC_hypnotic_suggestions/results/imaging/RSA/IS-RSA-mvpa-{NEURAL_SIM}-{effect_type}_sugg-pain_contrast-based_tian216"
    os.makedirs(save_path, exist_ok=True)
    BEHAV_ID = "chge-pain"  # "SHSS" or "chge-pain"
    if specific_re_run is True:
        
        print(f"Running IS-RSA for neural sim: {NEURAL_SIM}")
        for BEHAV_ID in ["SHSS", "chge-pain"]:
            print(f"Running IS-RSA for BEHAV: {BEHAV_ID}")
            if BEHAV_ID == "SHSS":
                sim_model = {
                    "euclidean": sim_behav_vec,
                    "annak": sim_behav_vec_annak,
                }
            if BEHAV_ID == "chge-pain":
                sim_model = {
                    "euclidean": sim_behav_vec_pain,
                    "annak": sim_behav_vec_annak_pain,
                }

            pain_conditions = {"ANA_shock_minus_N_ANA_shock", "HYPER_shock_minus_N_HYPER_shock"}
            results_dct = {}
            similarity_matrices_dct = {}
            
            for sim, behav_sim_vec_i in sim_model.items():
                print(f"Performing IS-RSA with {sim} similarity metric")
                results_dct[sim] = {}
                similarity_matrices_dct[sim] = {}
                for cond in tqdm(pain_conditions):
                    print("Performing RSA on : ", cond)
                    # load maps
                    # all_shock_maps = glob(os.path.join(model_res, 'all_shock', 'firstlev_localizer_*.nii.gz'))
                    pain_maps = glob(
                        os.path.join(
                            model_res, "first_level", cond, f"firstlev_{effect_type}_*.nii.gz"
                        )
                    )
                    pain_dict = build_subject_dict(pain_maps)  # not sorted!!
                    pain_dict = {sub: pain_dict[sub] for sub in subjects}

                    # mvpa similarity
                    similarity_matrices, vec_similarity_df = (
                        compute_inter_subject_mvpa_similarity(
                            pain_dict, atlas_img=ATLAS, labels_roi_dct=labels_roi_dct,
                            sim_method=NEURAL_SIM
                        )
                    )

                    # RSA computation + perm
                    rsa_rows = []  # build df

                    for i, roi in tqdm(
                        enumerate(vec_similarity_df.columns),
                        total=len(vec_similarity_df.columns),
                        desc="RSA ROIs",
                    ):

                        mvpa_sim_vec = vec_similarity_df[roi].values
                        r, p, dist = isc_utils.matrix_permutation(
                            behav_sim_vec_i,
                            mvpa_sim_vec,
                            n_permute=n_perm_rsa,
                            metric="spearman",
                            how="upper",
                            tail=2,
                            return_perms=True,
                        )
                        # 2 tail when testing behavioral model, as AnnaK can show reverse pattern
                        rsa_rows.append(
                            {
                                "ROI": roi,
                                "spearman_r": r,
                                "p_values": round(p, 5),
                                "Coordinates": tuple(np.round(coords[i], 0).astype(int)),
                            }
                        )

                    results_dct[sim][cond] = pd.DataFrame(rsa_rows)
                    similarity_matrices_dct[sim][cond] = similarity_matrices
                    print(
                        "Max r, mean and fdr",
                        results_dct[sim][cond]["spearman_r"].max(),
                        results_dct[sim][cond]["spearman_r"].mean(),
                        isc_utils.fdr(results_dct[sim][cond]["p_values"].to_numpy()),
                    )

            save_to = os.path.join(
                save_path, f"IS-RSA-mvpa_pain-{NEURAL_SIM}_behav-{BEHAV_ID}_{n_perm_rsa}perm.pkl"
            )
            isc_utils.save_data(save_to, results_dct)
            isc_utils.save_data(os.path.join(
                save_path, f"similarity_mat_IS-RSA-mvpa_pain-{NEURAL_SIM}_behav-{BEHAV_ID}_{n_perm_rsa}perm.pkl"
            ), similarity_matrices_dct)
            print(f"Saved RSA results to {save_to}")

    else:
        # load results

        load_path = os.path.join(
            save_path, f"IS-RSA-mvpa_pain-{NEURAL_SIM}_behav-{BEHAV_ID}_{n_perm_rsa}perm.pkl"
        )
        results_dct = isc_utils.load_pickle(load_path)
        similarity_matrices_dct = isc_utils.load_pickle(os.path.join(
            save_path, f"similarity_mat_IS-RSA-mvpa_pain-{NEURAL_SIM}_behav-{BEHAV_ID}_{n_perm_rsa}perm.pkl"
        ))
# %%
# =================
# VISUALISE PAIN-contrast ~ SHSS
rsa_stats_imgs_mvpa = {}
rsa_tables_mvpa_fdr = {}
rsa_stats_imgs_liberal = {}
rsa_tables_mvpa_unc= {}
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
            rsa_img, rsa_thresh, sig_df, unc_img = (
                visu_utils.project_isc_to_brain_perm(
                    atlas_img=ATLAS,
                    isc_median=correlations,
                    atlas_labels=labels_roi_dct,
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
            if correction == 'unc01':
                rsa_stats_imgs_liberal[model_id] = (rsa_img, rsa_thresh)

            if correction == "FDR" and sig_df.shape[0] > 0:
                rsa_stats_imgs_mvpa[model_id] = (rsa_img, rsa_thresh)
                rsa_tables_mvpa_fdr[model_id] = sig_df

                sig_df.columns = [
                    "ROI",
                    "Label",
                    "Spearman r",
                    "p-values",
                    "Coordinates (X,Y,Z)",
                ]
                fname = f"RSA_MVPA_{sim}_{cond}_table_{p_thresh_str}.csv"
                sig_df.to_csv(os.path.join(save_tables_shock_shss, fname), index=False)
                #save unc. stats img
                rsa_tables_mvpa_unc[model_id] = unc_img
                nib.save(unc_img, os.path.join(save_tables_shock_shss, f"RSA_MVPA_{sim}_{cond}_unc_stats_img.nii.gz"))
                print(f"Sig ROIs for {sim} – {cond}:", sig_df["Label"].to_list(), "\n")
#%%
# Glass brain with pruned p thresh
import warnings

for sim in results_dct.keys():  # similarity metric (e.g., 'euclidean', 'annak')
    for cond in results_dct[sim].keys():  # e.g., 'Analgesia', 'Hyperalgesia'
        model_id = sim + "_" + cond
        
        if model_id in rsa_stats_imgs_mvpa.keys():
            liberal_img = rsa_stats_imgs_liberal[model_id][0]
            fdr_img = rsa_stats_imgs_mvpa[model_id][0]
            unc_img = rsa_tables_mvpa_unc[model_id]
            display = plotting.plot_glass_brain(
                unc_img,
                display_mode="lyrz",
                colorbar=True,
                cmap="coolwarm",
                threshold=0,              # ensures transparent background
                plot_abs=False,
                vmax=np.max(np.abs(unc_img.get_fdata())),
                vmin=-np.max(np.abs(unc_img.get_fdata())),
                title=f"RSA MVPA {sim} – {cond})",
            )

            # Add liberal contours (gray, thin)
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning)
                display.add_contours(liberal_img, colors="gray", linewidths=0.7)

            # Add strict FDR contours (black, thick)
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning)
                display.add_contours(fdr_img, colors="black", linewidths=1.5)

#%%
import warnings

for sim in results_dct.keys():  # similarity metric (e.g., 'euclidean', 'annak')
    for cond in results_dct[sim].keys():  # e.g., 'Analgesia', 'Hyperalgesia'
        model_id = sim + "_" + cond
        if model_id in rsa_stats_imgs_mvpa.keys():
            liberal_img = rsa_stats_imgs_liberal[model_id][0]
            fdr_img = rsa_stats_imgs_mvpa[model_id][0]
            unc_img = rsa_tables_mvpa_unc[model_id]
            display = plotting.plot_stat_map(
                unc_img,
                colorbar=True,
                display_mode="mosaic",
                cmap="coolwarm",
                threshold=0,              # ensures transparent background
                vmax=np.max(np.abs(unc_img.get_fdata())),
                vmin=-np.max(np.abs(unc_img.get_fdata())),
                title=f"RSA MVPA {sim} – {cond})",
            )

            # Add liberal contours (gray, thin)
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning)
                display.add_contours(liberal_img, colors="gray", linewidths=0.7)

            # Add strict FDR contours (black, thick)
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning)
                display.add_contours(fdr_img, colors="black", linewidths=1.5)


#%%
# PLOT SIM MATRICES
def normalize_matrix(sim_matrix, mode="minmax"):
    if mode == "minmax":
        d = sim_matrix
        max_val = np.max(d)
        min_val = np.min(d)
        norm = 2 * (d - min_val) / (max_val - min_val) - 1
    else:
        # fallback to raw
        norm = sim_matrix
    return norm

Y_names = {
    "SHSS": "SHSS_score",
    "chge-pain": "total_chge_pain_hypAna",
}
from src.visu_utils import schaeffer_region_mapping
save_mat_shock_shss = os.path.join(save_path, RSA_MODEL_NAME, "pairwise_matrices")
os.makedirs(save_mat_shock_shss, exist_ok=True)

all_sim_mats = {}
for sim_model in results_dct.keys():
    for cond in results_dct[sim_model].keys():
        rsa_cond = sim_model + "_" + cond
        print(rsa_cond)

        if rsa_cond in rsa_tables_mvpa_fdr.keys(): # save sig conditions only
            sig_df = rsa_tables_mvpa_fdr[rsa_cond]
            sim_mats_ids = similarity_matrices_dct[sim_model][cond]  # mvpa similarity matrices dct
            sim_mats = {labels_roi_dct[id]: mat for id, mat in sim_mats_ids.items()}  # rename keys
            
            for sig_region in sig_df["Label"]:
                roi_id = sig_df[sig_df["Label"] == sig_region]["ROI"].values[0]
                if sig_region in schaeffer_region_mapping.keys():
                    anatomical_name = schaeffer_region_mapping[sig_region]["anatomical_label"]
                    region_id = schaeffer_region_mapping[sig_region]["region_index"]
                else:
                    anatomical_name = sig_region
                    region_id = roi_id
                # anatomical_name = schaeffer_region_mapping[sig_region]["anatomical_label"]
                # region_id = schaeffer_region_mapping[sig_region]["region_index"]
                mat = np.array(sim_mats[sig_region])
                ranked_indices = np.argsort(Y[Y_names[BEHAV_ID]].values)
                sim_mat = mat[np.ix_(ranked_indices, ranked_indices)]  # reorder
                np.fill_diagonal(sim_mat, 0)  # for visualization purposes
                vmax = np.max(np.abs(sim_mat))
                all_sim_mats[sig_region] = sim_mat
#%%      
# np.fill_diagonal(mat, 0)  # for visualization purposes
test_mat = [all_sim_mats['7Networks_RH_DorsAttn_Post_2'],
            all_sim_mats['7Networks_LH_DorsAttn_Post_9'],   
            all_sim_mats['7Networks_LH_Cont_PFCl_4'],
            all_sim_mats['7Networks_RH_Default_Temp_3'],
            all_sim_mats['7Networks_LH_DorsAttn_Post_8'],
]
fig_path = os.path.join(
    save_mat_shock_shss, f"{rsa_cond}_{sig_region}_pairwise-mat.png"
)
for mat in test_mat:
    visu_utils.plot_simmat_isc(
        simmat=mat,  # last one
        title=f"{region_id}:{anatomical_name}",
        x_label=None,  #' ranked subjects (SHSS)'
        colorbar_name=None,
        tick_fontsize=60,
        label_fontsize=70,
        # vmin=-vmax,

        # vmax=vmax,
        show_colorbar=True,
        # save_path=fig_path,
        ticks= None,
        dpi=300
    )

#%%
# PUBLICATION IS-RSA SHSS - PAIN CONTRAST
# View significant results and find coords to display
# Sequentially display interactive plots for each condition
views = []
for conditions in rsa_tables_mvpa_fdr.keys():
    img = rsa_stats_imgs_mvpa[conditions][0]
    display = view_img(img, title=f"RSA MVPA {conditions}", cmap="RdBu_r")
    views.append(display)
# %%
views[3]
#%%
# =====
# MAnual viewvinf for COORDS
reload(visu_utils)
from src.visu_utils import plot_single_roi
import atlasreader
from nilearn.image import math_img
pwd_scripts = os.path.join(project_dir, "scripts")
os.chdir(os.path.join(pwd_scripts, "atlas_views"))
print("Current working directory:", os.getcwd())
from atlasreader import create_output

roi_idx = 147
masked_region, _plot = plot_single_roi(
    atlas, roi_idx, id_labels_dct, title=f"roi idx {roi_idx}"
)  # 30 = Default_Temp_3 LH
_plot
#%%
create_output(masked_region, voxel_thresh=0.99, cluster_extent=5)
atlas_df = pd.read_csv("atlasreader_peaks.csv")
atlas_df
#%%
# PUBLICATION SLICES SHSS-PAIN CONTRAST: ANAAK / ANAlgesia
# --------------------------------
COORDS = {
    "x": [-28, -30, 54, 56, ],
    "y": [-34, -36],
}

SAVE_TO = os.path.join(save_plots_shock_shss, "Anak_ANA_shock_minus_N_ANA_shock")
os.makedirs(SAVE_TO, exist_ok=True)
IMG = rsa_stats_imgs_mvpa['annak_ANA_shock_minus_N_ANA_shock'][0]

cm = "RdBu_r"

visu_utils.save_slices(
    IMG, COORDS, SAVE_TO, img_id="ana_shss", cmap=cm
)
# save colorbar
fig_path = os.path.join(SAVE_TO, "colorbar.png")
cb_ab = visu_utils.save_colorbar(
    IMG,
    cmap=cm,
    outpath=fig_path,
    symmetric_cbar=True,  # false because we dont keep negative coeff
    offset=0,
    n_ticks=3,
    transparent=True,
)

#%%
# PUBLICATION SLICES SHSS-PAIN CONTRAST: ANAAK / HYPER
# --------------------------------
view_img(rsa_stats_imgs_mvpa['annak_HYPER_shock_minus_N_HYPER_shock'][0]
)
#%%
COORDS = {
    "x": [-44, -24, -48, -60],
    "y": [14, -36],
    "z": [-20, -16, -2]
}

SAVE_TO = os.path.join(save_plots_shock_shss, "Anak_HYPER_shock_minus_N_HYPER_shock")
os.makedirs(SAVE_TO, exist_ok=True)
IMG = rsa_stats_imgs_mvpa['annak_HYPER_shock_minus_N_HYPER_shock'][0]

cm = "RdBu_r"

visu_utils.save_slices(
    IMG, COORDS, SAVE_TO, img_id="ana_shss", cmap=cm
)
# save colorbar
fig_path = os.path.join(SAVE_TO, "colorbar.png")
cb_ab = visu_utils.save_colorbar(
    IMG,
    cmap=cm,
    outpath=fig_path,
    symmetric_cbar=True,  # false because we dont keep negative coeff
    offset=0,
    n_ticks=3,
    transparent=True,
)
#%%
# PUBLICATION: save slices HYPERALGESIA x 2 models 
# PUBLICATION SLICES SHSS-PAIN CONTRAST: EUCLIDEAN / ANAlgesia
# --------------------------------
view_img(rsa_stats_imgs_mvpa['euclidean_ANA_shock_minus_N_ANA_shock'][0])
rsa_tables_mvpa_fdr['euclidean_ANA_shock_minus_N_ANA_shock']
#%%
COORDS = {
    "x": [-28, -30, 54, 56, ],
    "y": [-34, -36],
}
CONTRAST = "Anak_HYPER_shock_minus_N_ANA_shock"
SAVE_TO = os.path.join(save_plots_shock_shss, CONTRAST)
os.makedirs(SAVE_TO, exist_ok=True)
IMG = rsa_stats_imgs_mvpa[CONTRAST][0]
IMG_ID = 'hyper_annak'
cm = "RdBu_r"

visu_utils.save_slices(
    IMG, COORDS, SAVE_TO, img_id=IMG_ID, cmap=cm
)
# save colorbar
fig_path = os.path.join(SAVE_TO, "colorbar.png")
cb_ab = visu_utils.save_colorbar(
    IMG,
    cmap=cm,
    outpath=fig_path,
    symmetric_cbar=True,  # false because we dont keep negative coeff
    offset=0,
    n_ticks=3,
    transparent=True,
)

#%%
# PUBLICATION SLICES SHSS-PAIN CONTRAST: EUCLIDEAN / HYPER
# --------------------------------
view_img(rsa_stats_imgs_mvpa['euclidean_HYPER_shock_minus_N_HYPER_shock'][0])
rsa_tables_mvpa_fdr['euclidean_HYPER_shock_minus_N_HYPER_shock']

#%%
COORDS = {
    "x": [-44, -24, -48, -60],
    "y": [14, -36],
    "z": [-20, -16, -2]
}

SAVE_TO = os.path.join(save_plots_shock_shss, "Anak_HYPER_shock_minus_N_HYPER_shock")
os.makedirs(SAVE_TO, exist_ok=True)
IMG = rsa_stats_imgs_mvpa['annak_HYPER_shock_minus_N_HYPER_shock'][0]

cm = "RdBu_r"

visu_utils.save_slices(
    IMG, COORDS, SAVE_TO, img_id="ana_shss", cmap=cm
)
# save colorbar
fig_path = os.path.join(SAVE_TO, "colorbar.png")
cb_ab = visu_utils.save_colorbar(
    IMG,
    cmap=cm,
    outpath=fig_path,
    symmetric_cbar=True,  # false because we dont keep negative coeff
    offset=0,
    n_ticks=3,
    transparent=True,
)
