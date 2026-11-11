# This script visualizes suggestibility ~ MVPA pain modulation with IS-RSA
# It then visualizes pain diff scores model ~ MVPA pain (PHG was main focus)
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

from glob import glob as glob
from nilearn.maskers import NiftiLabelsMasker
from nilearn.image import binarize_img
from nilearn.plotting import view_img
from nilearn.datasets import fetch_atlas_schaefer_2018
from nilearn.plotting import find_parcellation_cut_coords
from src import qc_utils, isc_utils

model_res = r"/data/rainville/dSutterlin/projects/ISC_hypnotic_suggestions/results/imaging/GLM/model3_final-isc_23subjects_nuis_nodrift_31-03-25"
project_dir = setup.project_dir
results_dir = setup["save_dir"]

# %% Load atlas: Schaefer + Tian 216 ROIs
from src.nilearn_helper import Atlas

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
#load updated BEHAV variables
from sklearn.preprocessing import StandardScaler
xlsx_preproc = os.path.join(project_dir, 'masks/Preprocessed_sugg_pain_behavioral.xlsx' )
behavioral_df = pd.read_excel(xlsx_preproc)  
scaler = StandardScaler(with_std=False) # keep variance, just mean center

behav_interest = {}
behav_interest['pain_diff_Ana'] = np.array(behavioral_df['pain_diff_Ana'], dtype='float')
behav_interest['pain_diff_Hyper'] = np.array(behavioral_df['pain_diff_Hyper'], dtype='float')
behav_interest['total_change_pain'] = np.array(behavioral_df['total_chge_pain_hypAna'], dtype='float')
behav_interest['SHSS_score'] = np.array(behavioral_df['SHSS_score'], dtype='float')
behav_interest['resid_pain_diff_Ana'] = np.array(behavioral_df['resid_pain_diff_Ana'], dtype='float')
behav_interest['resid_pain_diff_Hyper'] = np.array(behavioral_df['resid_pain_diff_Hyper'], dtype='float')

chge_ana = np.array(behavioral_df['pain_diff_Ana'], dtype='float')
chge_hyper = np.array(behavioral_df['pain_diff_Hyper'], dtype='float')
chge_ana[chge_ana > 0] = 0 #cap to 0 for no modulation
chge_hyper[chge_hyper < 0] = 0
absolute_pain_modulation = np.abs(chge_ana) + np.abs(chge_hyper)
sc_absolute_pain_modulation = scaler.fit_transform(absolute_pain_modulation.reshape(-1,1))

# %%
# ===========================
# MAIN CODE
from tqdm import tqdm
from nilearn.glm.thresholding import threshold_stats_img
reload(isc_utils)
mvpa_save_to = os.path.join(results_dir, "mvpa_similarity")
os.makedirs(mvpa_save_to, exist_ok=True)

n_perm_rsa = 10000

def z_score(x):
    return (x - np.mean(x)) / np.std(x)

behav_interest['pain_diff_Ana'] = z_score(behav_interest['pain_diff_Ana'])
# secondary pain variables
ANNAK_VERSION = 'mean'
annak_sim_matrices = {}
annak_sim_matrices['SHSS_score'] = isc_utils.compute_behav_similarity(behav_interest['SHSS_score'], metric="annak", annak_version=ANNAK_VERSION, vectorize=False)
annak_sim_matrices['total_change_pain'] = isc_utils.compute_behav_similarity(behav_interest['total_change_pain'], metric="annak", annak_version=ANNAK_VERSION, vectorize=False)
annak_sim_matrices['pain_diff_Ana'] = isc_utils.compute_behav_similarity(behav_interest['pain_diff_Ana'], metric="annak", annak_version=ANNAK_VERSION, vectorize=False)
annak_sim_matrices['pain_diff_Hyper'] = isc_utils.compute_behav_similarity(behav_interest['pain_diff_Hyper'], metric="annak", annak_version=ANNAK_VERSION, vectorize=False)

NN_sim_matrices = {}
NN_sim_matrices['SHSS_score'] = isc_utils.compute_behav_similarity(behav_interest['SHSS_score'], metric="euclidean", vectorize=False)
# NN_sim_matrices['total_change_pain'] = isc_utils.compute_behav_similarity(behav_interest['total_change_pain'], metric="euclidean", vectorize=False)
NN_sim_matrices['pain_diff_Ana'] = isc_utils.compute_behav_similarity(behav_interest['pain_diff_Ana'], metric="euclidean", vectorize=False)
NN_sim_matrices['pain_diff_Hyper'] = isc_utils.compute_behav_similarity(behav_interest['pain_diff_Hyper'], metric="euclidean", vectorize=False)


#%%visualization behavioral sim matrices
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(-1, 1))
sorted_annak = {}

for var, mat in NN_sim_matrices.items():
    print(var)
    ranked_indices = np.argsort(pd.Series(behav_interest[var].ravel()).values)

    ord_mat = mat[np.ix_(ranked_indices, ranked_indices)]
    ord_mat = scaler.fit_transform(ord_mat)
    visu_utils.plot_simmat_isc(
        simmat=ord_mat,
        title=None,
        x_label=None,  #' ranked subjects (SHSS)'
        colorbar_name=None,
        tick_fontsize=60,
        label_fontsize=70,
        vmin=-1,
        vmax=1,
        show_colorbar=False,
        ticks= None,
        dpi=1000)
    
for var, mat in annak_sim_matrices.items():
    print(var)
    ranked_indices = np.argsort(pd.Series(behav_interest[var].ravel()).values)

    ord_mat = mat[np.ix_(ranked_indices, ranked_indices)]
    sorted_annak[var] = ord_mat
    visu_utils.plot_simmat_isc(
        simmat=ord_mat,
        title=None, 
        x_label=None,  #' ranked subjects (SHSS)'
        colorbar_name=None,
        tick_fontsize=60,
        label_fontsize=70,
        vmin=-1,
        vmax=1,
        show_colorbar=False,
        ticks= None,
        dpi=1000)

ranked_shss_indices = np.argsort(pd.Series(behav_interest['SHSS_score'].ravel()).values)
# %%
# PUBLICATION
# FINAL MODEL Pain-specific ~ PAIN CONTRAST
# IS-RSA SHSS ~ ISC-PAIN
from src.mvpa_rsa_utils import compute_inter_subject_mvpa_similarity, build_subject_dict

results_dct_all = {}
similarity_matrices_dct_all = {}


# load specific models based on BEHAV var.
for BEHAV_ID in ['pain_diff_Ana', 'pain_diff_Hyper']:  

    NEURAL_SIM = 'dot' 
    effect_type = 'stat' #'effect_size' #'z_score' # stat = t maps
    save_path = f"/data/rainville/dSutterlin/projects/ISC_hypnotic_suggestions/results/imaging/RSA/IS-RSA-mvpa-{NEURAL_SIM}-{effect_type}_contrast-based_tian216_reproduced" #_{ANNAK_VERSION}-annak"
    pain_conditions = ["ANA_shock_minus_N_ANA_shock", "HYPER_shock_minus_N_HYPER_shock"]
   
    # load results
    load_path = os.path.join(
        save_path, f"IS-RSA-mvpa_pain-{NEURAL_SIM}_behav-{BEHAV_ID}_{n_perm_rsa}perm.pkl"
    )

    results_dct_all[BEHAV_ID] = isc_utils.load_pickle(load_path)
    similarity_matrices_dct_all[BEHAV_ID] = isc_utils.load_pickle(os.path.join(
        save_path, f"similarity_mat_IS-RSA-mvpa_pain-{NEURAL_SIM}_behav-{BEHAV_ID}_{n_perm_rsa}perm.pkl"
    ))
results_dct_ana = results_dct_all['pain_diff_Ana']
results_dct_hyper = results_dct_all['pain_diff_Hyper']

similarity_matrices_dct_ana = similarity_matrices_dct_all['pain_diff_Ana']
similarity_matrices_dct_hyper = similarity_matrices_dct_all['pain_diff_Hyper']

results_dct = {}

#%%

# Visualization 1 sample test
# Which regions have > 0 median neural SIM

from sklearn.preprocessing import MinMaxScaler
from brainiak.isc import bootstrap_isc
reload(isc_utils)
import warnings

issc_dfs = {}
issc_matrices = {}
issc_values = {}
issc_p_cond = {}
issc_dfs = {}
issc_ci_cond = {}
n_boot=10000

save_1sample_sim = os.path.join(save_path, 'one_sample_sim')
os.makedirs(save_1sample_sim, exist_ok=True)
save_tables_1sample_sim = os.path.join(save_1sample_sim, 'tables')
os.makedirs(save_tables_1sample_sim, exist_ok=True)
save_plots_1sample_sim = os.path.join(save_1sample_sim, 'plots')
os.makedirs(save_plots_1sample_sim, exist_ok=True)

#load results
for cond in pain_conditions:
    observed, ci, p_values, distribution = isc_utils.load_pickle(os.path.join(save_1sample_sim, f"issc_mvpa_pain_{cond}_{n_boot}boot.pkl"))
    issc_values[cond] = observed
    issc_p_cond[cond] = p_values
    issc_ci_cond[cond] = ci
    issc_dfs_data = []

    for region_id, region in labels_roi_dct.items():
        region_idx = region_id - 1
        correlation = observed[region_idx]
        p_value = p_values[region_idx]
        region_coords = coords[region_idx, :]
        issc_dfs_data.append(
            {
                "ROI": region_id,
                "Label": region,
                "r": round(correlation, 2),
                "p_value": f"{p_value:.4f}",
                "Coordinates": tuple(np.round(region_coords, 0).astype(int)),
            }
        )
    issc_dfs[cond] = pd.DataFrame(issc_dfs_data)

# Visualization 1 sample test
# Which regions have > 0 median neural SIM
reload(visu_utils)
one_sample_fdr_imgs = {}
one_sample_fdr_tables = {}

global_p_values = np.concatenate([issc_dfs[cond]["p_value"].values.astype(float) for cond in issc_dfs.keys()])
global_p_thresh = isc_utils.fdr(global_p_values, q=0.05)
print(f"Global FDR correction IS-RSA MVPA pain is : {global_p_thresh}"
      )

for cond in issc_values.keys():
    observed = issc_values[cond]
    p_values = issc_p_cond[cond]

    p_thresh = isc_utils.bonferroni(global_p_values, alpha=0.05)
    p_thresh_str = "Bonf_p" + str(round(p_thresh, 4))
    issc_img, isc_thresh, sig_df, unc_img = visu_utils.project_isc_to_brain(
                atlas_img=atlas,
                isc_median=observed,
                atlas_labels=id_labels_dct,
                roi_coords=coords,
                p_values=p_values,
                p_threshold=p_thresh,
                title=f'ISSC MVPA pain {cond} (FDR {round(p_thresh,4)})',
                coords_bool_mask=None,
                save_path=None,
                show=True,
                display_mode="x",
                cut_coords_plot=None
            )
    one_sample_fdr_imgs[cond] = (issc_img, isc_thresh)
    one_sample_fdr_tables[cond] = sig_df
    if sig_df.shape[0] > 0:
        sig_df.columns = [
            "ROI",
            "Label",
            "Spearman_r",
            "p_value",
            "Coordinates (X,Y,Z)",
        ]
        # CI array was saved as issc_ci_cond[cond]; transpose to get (n_rois, 2)
        ci_df = pd.DataFrame(issc_ci_cond[cond]).T
        mask = np.asarray(p_values) <= p_thresh
        cols = ci_df.columns[:2]
        selected_ci_rows = ci_df.loc[mask, cols]
        sig_ci = [np.round(tuple(row), 2) for row in selected_ci_rows.values]
        sig_df["CI"] = sig_ci


        # if the number of selected CIs matches sig_df rows, add as a column
        if len(sig_ci) == sig_df.shape[0]:
            sig_df["CI"] = sig_ci
        fname = f"IS-RSA_MVPA_pain_{cond}_table_{p_thresh_str}.csv"
        sig_df.to_csv(os.path.join(save_tables_1sample_sim, fname), index=False)
        print(f"Sig ROIs for IS-RSA MVPA pain {cond}:", sig_df["Label"].to_list(), "\n")
        view = view_img(issc_img, threshold=isc_thresh, title=f'ISSC MVPA pain {cond} ({p_thresh_str})')
        view.save_as_html(os.path.join(save_plots_1sample_sim, f"viewer_IS-RSA_MVPA_pain_{cond}_{p_thresh_str}{round(p_thresh,4)}.html"))

        unc_img.to_filename(os.path.join(save_1sample_sim, f"IS-RSA_MVPA_pain_{cond}_unc_stats_img.nii.gz"))


#%%
# plot glass brain from thresholed FDR img of 1 sample test
# POSITIVE ONLY
from nilearn.plotting import plot_glass_brain
from nilearn.image import new_img_like
all_rsa_correl = np.concatenate([issc_dfs[cond]['r'].values for cond in issc_dfs.keys()])
vmin, vmax = np.min(all_rsa_correl), np.max(all_rsa_correl)
fake_data = np.zeros_like(atlas.get_fdata())
all_rsa_correl[all_rsa_correl < 0] = 0
fake_data.flat[: len(all_rsa_correl)] = all_rsa_correl
synthetic_img = new_img_like(atlas, fake_data)
cmap = "Reds"

cbar = visu_utils.save_colorbar(
    img=synthetic_img,
    cmap=cmap,
    outpath=os.path.join(save_plots_1sample_sim, "similar_MVPA_colorbar.png"),
    symmetric_cbar=False,
    offset=None,
    n_ticks=3,
    transparent=True,
)
condition_names = {
    "ANA_shock_minus_N_ANA_shock": "Hypoalgesia ",
    "HYPER_shock_minus_N_HYPER_shock": "Hyperalgesia",
}

for i, (cond, (rsa_img, rsa_thresh)) in enumerate(one_sample_fdr_imgs.items()):
    # if i >= 1:
    #     break  # Stop after the first iteration
    print(cond)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.set_axis_off()

    title = f"{condition_names[cond]}"
    display = plot_glass_brain(
        rsa_img,
        threshold=rsa_thresh,
        colorbar=False,
        display_mode="lzr",
        plot_abs=False,
        cmap="Reds",
        title=None,
        axes=ax,
        vmax=np.max(all_rsa_correl),
        annotate=False,
    )
    y = 0.20
    ax.text(0.1, y, "L", transform=ax.transAxes, fontsize=20, ha="center")
    ax.text(0.9, y, "R", transform=ax.transAxes, fontsize=20, ha="center")

    # Add title
    fig.suptitle(title, fontsize=25, y=0.85)
    fig.savefig(
        os.path.join(save_plots_1sample_sim, f"glass_brain_{cond}_FDR05.png"), dpi=1000
    )
cbar

#%%

# from src.MEP_brain_vizu import plot_3D_brain_maps

# img_path =os.path.join(save_1sample_sim, f"IS-RSA_MVPA_pain_{cond}_unc_stats_img.nii.gz")

# plot_3D_brain_maps(
#     path_data=img_path,
#     path_output=save_plots_1sample_sim,
#     views=["lateral", "medial"],
#     thresh=isc_thresh,
#     extension="png",
# )
#%%
# IS-RSA pain contrast ~ Pain diff
# =================
# VISUALISE PAIN-contrast ~ pain diff

# Load Is-RSA results for BEHAV ~ MVPA pain patterns
# Compute FDR correction across all models/conditions
# Visualize and save FDR-corrected results

BEHAV_ID = 'pain_diff_Ana'
rsa_stats_imgs_fdr = {}
rsa_tables_mvpa_fdr = {}
rsa_stats_imgs_liberal = {}
rsa_stats_imgs_mvpa_unc = {}
RSA_MODEL_NAME = f"shock_contrast_{BEHAV_ID}"
save_tables_shock_shss = os.path.join(save_path, RSA_MODEL_NAME, "tables")
os.makedirs(save_tables_shock_shss, exist_ok=True)
save_plots_shock_shss = os.path.join(save_path, RSA_MODEL_NAME, "plots")
os.makedirs(save_plots_shock_shss, exist_ok=True)

if BEHAV_ID == 'pain_diff_Ana':
    results_dct = results_dct_ana
elif BEHAV_ID == 'pain_diff_Hyper':
    results_dct = results_dct_hyper

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
                    atlas_img=atlas,
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
            
            rsa_stats_imgs_mvpa_unc[model_id] = unc_img
            
            if correction == 'unc01':
                rsa_stats_imgs_liberal[model_id] = (rsa_img, rsa_thresh)

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
                sig_df.to_csv(os.path.join(save_tables_shock_shss, fname), index=False)
                #save unc. stats img
                nib.save(unc_img, os.path.join(save_tables_shock_shss, f"RSA_MVPA_{sim}_{cond}_unc_stats_img.nii.gz"))
                print(f"Sig ROIs for {sim} – {cond}:", sig_df["Label"].to_list(), "\n")
                view = view_img(rsa_img, threshold=rsa_thresh, title=f"RSA MVPA {sim} - {cond} ({p_thresh_str})")
                view.save_as_html(os.path.join(save_plots_shock_shss, f"viewer_RSA_{sim}_{cond}_{p_thresh_str}.html"))
#%%
# Glass brain with pruned p thresh
import warnings
views = []
for sim in results_dct.keys():  # similarity metric (e.g., 'euclidean', 'annak')
    for cond in results_dct[sim].keys():  # e.g., 'Analgesia', 'Hyperalgesia'
        model_id = sim + "_" + cond
        
for model_id in rsa_stats_imgs_mvpa_unc.keys():
        
        unc_img = rsa_stats_imgs_mvpa_unc[model_id]
        
        display = plotting.plot_glass_brain(
            unc_img,
            display_mode="lyrz",
            colorbar=True,
            cmap="coolwarm",
            threshold=0, # ensures transparent background: otherwise gray if =None (bug?)
            plot_abs=False,
            vmax=np.max(np.abs(unc_img.get_fdata())),
            vmin=-np.max(np.abs(unc_img.get_fdata())),
            title=f"RSA MVPA {model_id}",
        )

        if model_id in rsa_stats_imgs_liberal.keys():
            liberal_img = rsa_stats_imgs_liberal[model_id][0]
            # Add liberal contours (gray, thin)
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning)
                display.add_contours(liberal_img, colors="gray", linewidths=0.7)

        if model_id in rsa_stats_imgs_fdr.keys():
            liberal_img = rsa_stats_imgs_liberal[model_id][0]
            fdr_img = rsa_stats_imgs_fdr[model_id][0]

            # Add strict FDR contours (black, thick)
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning)
                display.add_contours(fdr_img, colors="black", linewidths=1.5)
            views.append(view_img(fdr_img, threshold=0, title=f"RSA MVPA {sim} – {cond}"))
        

#%%–5 p.m.
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
    "SHSS_score": "SHSS_score",
    "chge-pain": "total_chge_pain_hypAna",
    "pain_diff_Ana": "pain_diff_Ana",
    "pain_diff_Hyper": "pain_diff_Hyper",
}
reload(visu_utils)
from src.visu_utils import schaeffer_region_mapping

VISU_METRIC = 'dot'
save_sim_mat_shock_shss = os.path.join(save_path, RSA_MODEL_NAME, f"sig_{VISU_METRIC}_similarity_matrices")
os.makedirs(save_sim_mat_shock_shss, exist_ok=True)
#%%
# PUBLICATION : Plot neural sim matrices

from sklearn.preprocessing import MinMaxScaler

def rescale_matrix(mat, feature_range=(-1, 1)):

    scaler = MinMaxScaler(feature_range=feature_range)
    flat = mat.flatten().reshape(-1, 1)
    scaled = scaler.fit_transform(flat).reshape(mat.shape)

    return scaled

SHOW_MAT  = True
VISU_BEHAV = 'pain_diff_Ana'  # 'SHSS_score'  # 'pain_diff_Ana'  # 'pain_diff_Hyper'
all_sim_mats = {}
all_sim_mats_ranked = {}

for sim_model in results_dct_all[VISU_BEHAV].keys():

    for cond in results_dct[sim_model].keys():
        rsa_cond = sim_model + "_" + cond
        print(rsa_cond)

        if rsa_cond in rsa_tables_mvpa_fdr.keys(): # save sig conditions only
            sig_df = rsa_tables_mvpa_fdr[rsa_cond]
            sim_mats_ids = similarity_matrices_dct_all[VISU_BEHAV][sim_model][cond]  # mvpa similarity matrices dct
            sim_mats = {labels_roi_dct[id]: mat for id, mat in sim_mats_ids.items()}  # rename keys
            
            for sig_region in sig_df["Label"]:
                roi_id = sig_df[sig_df["Label"] == sig_region]["ROI"].values[0]
                if sig_region in schaeffer_region_mapping.keys():
                    anatomical_name = schaeffer_region_mapping[sig_region]["anatomical_label"]
                    region_id = schaeffer_region_mapping[sig_region]["region_index"]
                else:
                    anatomical_name = sig_region
                    region_id = roi_id
               
                mat = np.array(sim_mats[sig_region])
                ranked_indices = np.argsort(behav_interest[Y_names[VISU_BEHAV]].ravel())  # ranked by SHSS
                ranked_subjects = [subjects[i] for i in ranked_indices]
                sim_mat = mat[np.ix_(ranked_indices, ranked_indices)]  # reorder
                np.fill_diagonal(sim_mat, 0)  # for visualization purposes
                vmax = np.max(np.abs(sim_mat))

                all_sim_mats[sig_region] = mat
                all_sim_mats_ranked[sig_region] = sim_mat

                fig_path = os.path.join(
                    save_sim_mat_shock_shss, f"{rsa_cond}_{roi_id}_{sig_region}_pairwise-mat.png"
                )
                if SHOW_MAT:
                    visu_utils.plot_simmat_isc(
                        simmat=all_sim_mats_ranked[sig_region],  # last one
                        title=f"{anatomical_name}\n(rho = {sig_df[sig_df['Label'] == sig_region]['Spearman r'].values[0]:.2f})",
                        x_label=None,  #' ranked subjects (SHSS)'
                        colorbar_name=None,
                        tick_fontsize=60,
                        label_fontsize=70,
                        vmin=-vmax,
                        vmax=vmax,
                        show_colorbar=False,
                        # save_path=fig_path,
                        ticks= None,
                        dpi=400 # long to save, chge to 300 for quick
                    )


# %%
#%%
# PUBLICATION IS-RSA SHSS - PAIN CONTRAST
# View significant results and find coords to display
# Sequentially display interactive plots for each condition
views = []
for conditions in rsa_tables_mvpa_fdr.keys():
    img = rsa_stats_imgs_fdr[conditions][0]
    display = view_img(img, title=f"RSA MVPA {conditions}", cmap="RdBu_r")
    views.append(display)
# %%
views[2]


