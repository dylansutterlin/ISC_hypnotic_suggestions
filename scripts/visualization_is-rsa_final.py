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

# load all models with one behavioral variable
for NEURAL_SIM in ['dot']: #, 'dot']: #, 'cosine']: #, 'dot']: #['euclidean', 'dot']: #'cosine', 'pearson']:
    # BEHAV_ID = "SHSS"  # "SHSS" or "chge-pain"
    effect_type = 'stat' #'effect_size' #'z_score' # stat = t maps
    save_path = f"/data/rainville/dSutterlin/projects/ISC_hypnotic_suggestions/results/imaging/RSA/IS-RSA-mvpa-{NEURAL_SIM}-{effect_type}_contrast-based_tian216_reproduced" #_{ANNAK_VERSION}-annak"

    BEHAV_ID = "SHSS_score"  # "SHSS" or "chge-pain" or "pain_diff_Ana" or "pain_diff_Hyper" or "total_change_pain"
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
final_metric = 'dot'
results_dct = results_dct_all[final_metric]
similarity_matrices_dct = similarity_matrices_dct_all[final_metric]
save_path = f"/data/rainville/dSutterlin/projects/ISC_hypnotic_suggestions/results/imaging/RSA/IS-RSA-mvpa-{final_metric}-{effect_type}_contrast-based_tian216_reproduced" #_{ANNAK_VERSION}-annak"
#%%
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


#%%
# PUBLICATION: Supp mat.
# Visualization 1 sample test
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
# PUBLICATION plot glass brain from thresholed FDR img
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
# PUBLICATION: IS-RSA pain contrast ~ SHSS_score
# =================
# VISUALISE PAIN-contrast ~ SHSS

rsa_stats_imgs_fdr = {}
rsa_tables_mvpa_fdr = {}
rsa_stats_imgs_liberal = {}
rsa_stats_imgs_mvpa_unc = {}
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
    "SHSS_score": "SHSS_score",
    "chge-pain": "total_chge_pain_hypAna",
    "pain_diff_Ana": "pain_diff_Ana",
    "pain_diff_Hyper": "pain_diff_Hyper",
}
reload(visu_utils)
from src.visu_utils import schaeffer_region_mapping

VISU_METRIC = 'cosine'
save_sim_mat_shock_shss = os.path.join(save_path, RSA_MODEL_NAME, f"sig_{VISU_METRIC}_similarity_matrices")
os.makedirs(save_sim_mat_shock_shss, exist_ok=True)
#%%
from sklearn.preprocessing import MinMaxScaler
def rescale_matrix(mat, feature_range=(-1, 1)):
    scaler = MinMaxScaler(feature_range=feature_range)
    flat = mat.flatten().reshape(-1, 1)
    scaled = scaler.fit_transform(flat).reshape(mat.shape)
    return scaled

all_sim_mats = {}
all_sim_mats_ranked = {}
for sim_model in results_dct_all[VISU_METRIC].keys():

    for cond in results_dct[sim_model].keys():
        rsa_cond = sim_model + "_" + cond
        print(rsa_cond)

        if rsa_cond in rsa_tables_mvpa_fdr.keys(): # save sig conditions only
            sig_df = rsa_tables_mvpa_fdr[rsa_cond]
            sim_mats_ids = similarity_matrices_dct_all[VISU_METRIC][sim_model][cond]  # mvpa similarity matrices dct
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
                ranked_indices = np.argsort(behav_interest[Y_names[BEHAV_ID]].ravel())  # ranked by SHSS
                ranked_subjects = [subjects[i] for i in ranked_indices]
                sim_mat = mat[np.ix_(ranked_indices, ranked_indices)]  # reorder
                np.fill_diagonal(sim_mat, 0)  # for visualization purposes
                vmax = np.max(np.abs(sim_mat))

                all_sim_mats[sig_region] = mat
                all_sim_mats_ranked[sig_region] = sim_mat

                fig_path = os.path.join(
                    save_sim_mat_shock_shss, f"{rsa_cond}_{roi_id}_{sig_region}_pairwise-mat.png"
                )
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
                    save_path=fig_path,
                    ticks= None,
                    dpi=1000 # long to save, chge to 300 for quick
                )

                if sig_region == '7Networks_LH_Default_PHC_1':
                    phg_mat = mat
                    phg_mat_ranked = sim_mat


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
#%%
# =====
# MAnual viewving for COORDS
reload(visu_utils)
from src.visu_utils import plot_single_roi
import atlasreader
from nilearn.image import math_img
pwd_scripts = os.path.join(project_dir, "scripts")
os.chdir(os.path.join(pwd_scripts, "atlas_views"))
print("Current working directory:", os.getcwd())
from atlasreader import create_output

roi_idx = 158 # change roi to view best cut
masked_region, _plot = plot_single_roi(
    atlas, roi_idx, id_labels_dct, title=f"roi idx {roi_idx}"
)  # 30 = Default_Temp_3 LH
_plot
#%%
create_output(masked_region, voxel_thresh=0.99, cluster_extent=5)
atlas_df = pd.read_csv("atlasreader_peaks.csv")
atlas_df
#%%
# PUBLICATION SLICES SHSS-PAIN CONTRAST: ANALGESIA / ANNAK
# --------------------------------
COORDS = {
    "x": [-28, -30,-22 ],
    "y": [-34, -36],
}
CONTRAST = "annak_ANA_shock_minus_N_ANA_shock"
SAVE_TO = os.path.join(save_plots_shock_shss, CONTRAST)
os.makedirs(SAVE_TO, exist_ok=True)
IMG = rsa_stats_imgs_fdr['annak_ANA_shock_minus_N_ANA_shock'][0]

cm = "RdBu_r"

visu_utils.save_slices(IMG, COORDS, SAVE_TO, img_id="ana_shss", cmap=cm
)
# save colorbar
fig_path = os.path.join(SAVE_TO, "colorbar.png")
cb_ab = visu_utils.save_colorbar(
    IMG,
    cmap=cm,
    fontsize=25,
    outpath=fig_path,
    symmetric_cbar=True,  # false because we dont keep negative coeff
    offset=0,
    n_ticks=3,
    transparent=True,
)

#%%
# PUBLICATION SLICES SHSS-PAIN CONTRAST: EUCLIDEAN / ANAlgesia
# --------------------------------
view_img(rsa_stats_imgs_fdr['euclidean_ANA_shock_minus_N_ANA_shock'][0])
# rsa_tables_mvpa_fdr['euclidean_ANA_shock_minus_N_ANA_shock']
COORDS = {
    "x": [-28, -30,-46, 50, 56, ],
    "y": [2, 10, 12],
}

CONTRAST = "euclidean_ANA_shock_minus_N_ANA_shock"
SAVE_TO = os.path.join(save_plots_shock_shss, CONTRAST)
os.makedirs(SAVE_TO, exist_ok=True)
IMG = rsa_stats_imgs_fdr[CONTRAST][0]
IMG_ID = 'ana_euclidean'
cm = "RdBu_r"

visu_utils.save_slices(
    IMG, COORDS, SAVE_TO, img_id=IMG_ID, cmap=cm
)
# save colorbar
fig_path = os.path.join(SAVE_TO, "colorbar.png")
cb_ab = visu_utils.save_colorbar(
    IMG,
    cmap=cm,
    fontsize=25,
    outpath=fig_path,
    symmetric_cbar=True,  # false because we dont keep negative coeff
    offset=0,
    n_ticks=3,
    transparent=True,
)

#%%
# PUBLICATION SLICES SHSS-PAIN CONTRAST: ANAAK / HYPER
# --------------------------------
view_img(rsa_stats_imgs_fdr['annak_HYPER_shock_minus_N_HYPER_shock'][0]
)
#%%
COORDS = {
    "x": [14, -4, -48, -60],
    "y": [-10, 28],
    "z": [-20, -16, -2]
}
CONTRAST = 'annak_HYPER_shock_minus_N_HYPER_shock'
SAVE_TO = os.path.join(save_plots_shock_shss, CONTRAST)
os.makedirs(SAVE_TO, exist_ok=True)
IMG = rsa_stats_imgs_fdr[CONTRAST][0]
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
    fontsize=25,
    outpath=fig_path,
    symmetric_cbar=True,  # false because we dont keep negative coeff
    offset=0,
    n_ticks=3,
    transparent=True,
)

#%%
# PUBLICATION SLICES SHSS-PAIN CONTRAST: EUCLIDEAN / HYPER
# --------------------------------
view_img(rsa_stats_imgs_fdr['euclidean_HYPER_shock_minus_N_HYPER_shock'][0])
rsa_tables_mvpa_fdr['euclidean_HYPER_shock_minus_N_HYPER_shock']

#%%
COORDS = {
    "x": [6,8,10],
 }
CONTRAST = 'euclidean_HYPER_shock_minus_N_HYPER_shock'
SAVE_TO = os.path.join(save_plots_shock_shss, CONTRAST)
os.makedirs(SAVE_TO, exist_ok=True)
IMG = rsa_stats_imgs_fdr[CONTRAST][0]
IMG_ID = 'hyper_euclidean'
cm = "RdBu_r"

visu_utils.save_slices(
    IMG, COORDS, SAVE_TO, img_id=IMG_ID, cmap=cm
)
# save colorbar
fig_path = os.path.join(SAVE_TO, "colorbar.png")
cb_ab = visu_utils.save_colorbar(
    IMG,
    cmap=cm,
    fontsize=25,
    outpath=fig_path,
    symmetric_cbar=True,  # false because we dont keep negative coeff
    offset=0,
    n_ticks=3,
    transparent=True,
)

#%%
# plot second level maps per regions
# %%
# Visualize second-level contrast for pain
pain_conditions = ["ANA_shock_minus_N_ANA_shock", "HYPER_shock_minus_N_HYPER_shock"]

condition_names = {
    "ANA_sugg": "Analgesia",
    "N_ANA_sugg": "Neutral (Ana.)",
    "HYPER_sugg": "Hyperalgesia",
    "N_HYPER_sugg": "Neutral (Hyper.)",
}
# test = "/data/rainville/dSutterlin/projects/ISC_hypnotic_suggestions/results/imaging/GLM/model3_final-isc_23subjects_nuis_nodrift_31-03-25/second_level/ANA_shock_minus_N_ANA_shock_all_effects.pkl"
second_lev_dir = os.path.join(results_dir, "second_level")
t_imgs = {}
for cond in pain_conditions:

    pkl = isc_utils.load_pickle(os.path.join(second_lev_dir, cond + "_all_effects.pkl"))
    t_img = pkl["stat"]
    img = nib.load(os.path.join(second_lev_dir, "group_" + cond + ".nii.gz"))
    t_imgs[cond] = t_img

view_img(t_img)

#%%
from nilearn.image import new_img_like

# Create a single region MVPA per conditions
def mask_img_from_atlas(labels_roi_dct, atlas, stats_img, only_labels=None):

    roi_indices = list(labels_roi_dct.keys())
    atlas_data = atlas.get_fdata()
    if only_labels is not None:
        roi_indices = [k for k, v in labels_roi_dct.items() if v in only_labels]

    filt_imgs = {}

    for roi_idx in roi_indices:
        roi_mask = atlas_data == roi_idx

        new_img = new_img_like(stats_img, np.zeros(atlas.shape))
        new_img.get_fdata()[roi_mask] = stats_img.get_fdata()[roi_mask]
        filt_imgs[roi_idx] = new_img
        
    return filt_imgs

roi_pattern_dct_hyper = mask_img_from_atlas(labels_roi_dct, atlas, t_imgs['HYPER_shock_minus_N_HYPER_shock'])
roi_pattern_dct_ana = mask_img_from_atlas(labels_roi_dct, atlas, t_imgs['ANA_shock_minus_N_ANA_shock'])


#%%
def combine_roi_maps(roi_pattern_dct, roi_ids, stats_img):
    """
    Combine multiple region maps into one image.
    roi_pattern_dct: dict of region id -> region image (values at region, 0 elsewhere)
    roi_ids: list of region ids to combine
    stats_img: reference image for shape/affine
    Returns: combined image (nifti)
    """
    combined_data = np.zeros(stats_img.shape)
    for roi_id in roi_ids:
        roi_img = roi_pattern_dct[roi_id]
        roi_data = roi_img.get_fdata()
        combined_data += roi_data
    # Optionally, set nonzero to 1 if you want a binary mask
    # combined_data = (combined_data != 0).astype(float)
    return new_img_like(stats_img, combined_data)

all_ana_sig_rois = list(rsa_tables_mvpa_fdr['euclidean_ANA_shock_minus_N_ANA_shock']['ROI'].values) + list(
    rsa_tables_mvpa_fdr['annak_ANA_shock_minus_N_ANA_shock']['ROI'].values)

ana_sig_roi_map = combine_roi_maps(roi_pattern_dct_ana, all_ana_sig_rois, t_imgs['ANA_shock_minus_N_ANA_shock'])

all_hyper_sig_rois = list(rsa_tables_mvpa_fdr['euclidean_HYPER_shock_minus_N_HYPER_shock']['ROI'].values) + list(
    rsa_tables_mvpa_fdr['annak_HYPER_shock_minus_N_HYPER_shock']['ROI'].values)
hyper_sig_roi_map = combine_roi_maps(roi_pattern_dct_hyper, all_hyper_sig_rois, t_imgs['HYPER_shock_minus_N_HYPER_shock'])

#%%
# SUPPLEMENTARY MATERIAL: second level pattern in sig regions from IS-RSA
# # PUBLICATION SLICES SHSS-PAIN CONTRAST: ANALGESIA / ANNAK
# --------------------------------
COORDS = {
    "x": [-28, -30,-22, -46, 50,56 ],
    "y": [-34, -36, 2,10,12],
}


SAVE_TO = os.path.join(save_plots_shock_shss, 'ana_second_level_MVpatterns')
os.makedirs(SAVE_TO, exist_ok=True)
IMG = ana_sig_roi_map
ana_sig_roi_map.to_filename(os.path.join(SAVE_TO, 'ana_second_level_MVpatterns.nii.gz'))
cm = "RdBu_r"

visu_utils.save_slices(IMG, COORDS, SAVE_TO, img_id="ana_second_level", cmap=cm
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
COORDS = {
    "x": [6,8,10, 14, -4, -48, -60],
    "y": [-10, 28],
    "z": [-20, -16, -2]
}

SAVE_TO = os.path.join(save_plots_shock_shss, 'hyper_second_level_MVpatterns')
os.makedirs(SAVE_TO, exist_ok=True)
IMG = hyper_sig_roi_map
hyper_sig_roi_map.to_filename(os.path.join(SAVE_TO, 'hyper_second_level_MVpatterns.nii.gz'))


visu_utils.save_slices(
    IMG, COORDS, SAVE_TO, img_id="hyper_shss", cmap=cm
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

# %%
#==================================================
# PUBLICATION: Pairwise MVPA similarity in ROI for 1ST LEVEL contrasts
#==================================================
from src.mvpa_rsa_utils import compute_inter_subject_mvpa_similarity, build_subject_dict

# Load 1st level maps 
# Visualize second-level contrast for pain
pain_conditions = ["ANA_shock_minus_N_ANA_shock", "HYPER_shock_minus_N_HYPER_shock"]
imgs_type = 'stat'
first_level_dir = os.path.join(results_dir, "first_level")
first_level_t_imgs = {}
for cond in pain_conditions:
    imgs_dct = {}
    for sub in subjects:
        img_path = os.path.join(first_level_dir, cond, f"firstlev_{imgs_type}_{sub}.nii.gz")
        imgs_dct[sub] = img_path
    first_level_t_imgs[cond] = imgs_dct

# get mv pattern per roi for all subjects
# roi interest = 100 (phg) and 156 (amcc)
labels_roi_selected_ana = [id_labels_dct[100]]  # phg
labels_roi_selected_hyper = [id_labels_dct[156]]  # amcc

extracted_mv_pattern_per_roi_ana = {}
for sub in subjects:
    stat_img = nib.load(first_level_t_imgs['ANA_shock_minus_N_ANA_shock'][sub])
    extracted_mv_pattern_per_roi_ana[sub] = mask_img_from_atlas(labels_roi_dct,
                                                                    atlas,
                                                                    stat_img,
                                                        only_labels=labels_roi_selected_ana
                                                        )
extracted_mv_pattern_per_roi_hyper = {}
for sub in subjects:
    stat_img = nib.load(first_level_t_imgs['HYPER_shock_minus_N_HYPER_shock'][sub])
    extracted_mv_pattern_per_roi_hyper[sub] = mask_img_from_atlas(labels_roi_dct,
                                                                    atlas,
                                                                    stat_img,
                                                        only_labels=labels_roi_selected_hyper
                                                        )
#%%
# get similarity matrices per roi to find pairs of subject to plot MV patterns
phc_roi = id_labels_dct[100]
amcc_roi = id_labels_dct[156]

sim_mat_phc = similarity_matrices_dct_all['cosine']['annak']['ANA_shock_minus_N_ANA_shock'][100]
sim_mat_amcc = similarity_matrices_dct_all['cosine']['annak']['ANA_shock_minus_N_ANA_shock'][156]

# reorder sim mat by SHSS scores
ranked_indices = np.argsort(behav_interest['SHSS_score'].ravel())  # ranked by SHSS
ranked_subjects = [subjects[i] for i in ranked_indices]

sim_mat_phc_ranked = np.array(sim_mat_phc)[np.ix_(ranked_indices, ranked_indices)]  # reorder
sim_mat_amcc_ranked = np.array(sim_mat_amcc)[np.ix_(ranked_indices, ranked_indices)]  # reorder
sim_mat_phc_ranked = pd.DataFrame(sim_mat_phc_ranked, index=ranked_subjects, columns=ranked_subjects)
sim_mat_amcc_ranked = pd.DataFrame(sim_mat_amcc_ranked, index=ranked_subjects, columns=ranked_subjects)

# indices are selected to display high-high and low-low similarity pairs
# high-high or low-low pair refers to ranked beahvioral scores
# low-low pair are two subject with low SHSS scores...
# Similarity between the pair varies depedning on the model:
# Annak : low-low pair wil have low similarity (phg)
# Euclidean: low-low pair will have high similarity (amcc)

# phg_high_high_indices = (subjects.index('sub-27'),subjects.index('sub-20'))  # from sim mat inspection
# phg_low_low_indices = (subjects.index('sub-16'),subjects.index('sub-37'))
phg_high_high_subjects = ('sub-27','sub-20')
phg_low_low_subjects = ('sub-16','sub-37')

amcc_high_high_subjects = ('sub-28','sub-42')
amcc_low_low_subjects = ('sub-36','sub-16')
# amcc_high_high_indices = (subjects.index('sub-28'),subjects.index('sub-42'))  # from sim mat inspection
# amcc_low_low_indices = (subjects.index('sub-36'),subjects.index('sub-16'))

# get selected subjects maps
phg_maps_high_high = [img for sub, img in extracted_mv_pattern_per_roi_ana.items() if sub in phg_high_high_subjects]
phg_maps_low_low = [img for sub, img in extracted_mv_pattern_per_roi_ana.items() if sub in phg_low_low_subjects]

amcc_maps_high_high = [img for sub, img in extracted_mv_pattern_per_roi_hyper.items() if sub in amcc_high_high_subjects]
amcc_maps_low_low = [img for sub, img in extracted_mv_pattern_per_roi_hyper.items() if sub in amcc_low_low_subjects]
#%%
import numpy as np
from scipy.spatial.distance import cosine
from scipy.stats import pearsonr

def find_most_similar_slice(subject1_dct, subject2_dct, axis='z', similarity='pearson'):
    """
    Find the slice index where two masked PHG images are most similar.

    Args:
        subject1_dct, subject2_dct: masked_img_dct for each subject (1 ROI each)
        axis: axis along which to slice ('x', 'y', 'z')
        similarity: 'pearson' or 'cosine'

    Returns:
        slice_index: int, best matching slice index
    """
    img1 = list(subject1_dct.values())[0].get_fdata()
    img2 = list(subject2_dct.values())[0].get_fdata()
    ax = {'x': 0, 'y': 1, 'z': 2}[axis]

    n_slices = img1.shape[ax]
    best_sim = -np.inf
    best_idx = 0

    for i in range(n_slices):
        if ax == 0:
            s1, s2 = img1[i, :, :], img2[i, :, :]
        elif ax == 1:
            s1, s2 = img1[:, i, :], img2[:, i, :]
        else:
            s1, s2 = img1[:, :, i], img2[:, :, i]

        if np.all(s1 == 0) or np.all(s2 == 0):
            continue

        # Flatten and compute similarity (only on nonzero mask overlap)
        mask = (s1 != 0) & (s2 != 0)
        if np.sum(mask) < 10:
            continue

        vec1 = s1[mask].flatten()
        vec2 = s2[mask].flatten()

        try:
            if similarity == 'pearson':
                sim = pearsonr(vec1, vec2)[0]
            elif similarity == 'cosine':
                sim = 1 - cosine(vec1, vec2)
            elif similarity == 'dot':
                sim = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
            else:
                raise ValueError("Unsupported similarity metric.")
        except:
            sim = -np.inf

        if sim > best_sim:
            best_sim = sim
            best_idx = i

    return best_idx

def extract_consistent_slice(masked_img_dct, slice_index, axis='z', rotate=True):
    """
    Extract and optionally rotate a 2D slice from the masked image along a given axis.
    
    Parameters:
        masked_img_dct: dict containing {roi_id: Nifti1Image}
        slice_index: int, index of the slice to extract
        axis: str, 'x', 'y', or 'z'
        rotate: bool, apply 90° counterclockwise rotation (True by default)

    Returns:
        2D numpy array slice (cropped and correctly oriented)
    """
    img = list(masked_img_dct.values())[0]
    data = img.get_fdata()

    # Extract the appropriate slice
    if axis == 'x':
        slice_data = data[slice_index, :, :]
    elif axis == 'y':
        slice_data = data[:, slice_index, :]
    else:  # 'z'
        slice_data = data[:, :, slice_index]

    # Crop bounding box to remove zero-only rows/columns
    rows = np.any(slice_data != 0, axis=1)
    cols = np.any(slice_data != 0, axis=0)
    cropped = slice_data[np.ix_(rows, cols)]

    # Rotate for proper orientation
    if rotate:
        cropped = np.rot90(cropped)

    return cropped
def extract_representative_slice(masked_img_dct, axis='z'):
    """
    Extract a 2D slice from a masked image (e.g., PHG ROI) with the largest number of non-zero voxels.

    Args:
        masked_img_dct: dict of {roi_id: nib.Nifti1Image}
        axis: which axis to slice along: 'x', 'y', or 'z'

    Returns:
        2D numpy array representing the most informative slice.
    """
    img = list(masked_img_dct.values())[0]
    data = img.get_fdata()

    axis_map = {'x': 0, 'y': 1, 'z': 2}
    if axis not in axis_map:
        raise ValueError("Axis must be one of 'x', 'y', 'z'.")

    ax = axis_map[axis]
    n_slices = data.shape[ax]

    # Find the slice with the most non-zero voxels
    max_nonzero = -1
    best_slice = None
    for i in range(n_slices):
        if ax == 0:
            slice_data = data[i, :, :]
        elif ax == 1:
            slice_data = data[:, i, :]
        elif ax == 2:
            slice_data = data[:, :, i]

        nonzero_count = np.count_nonzero(slice_data)
        if nonzero_count > max_nonzero:
            max_nonzero = nonzero_count
            best_slice = slice_data

    # Optional: crop bounding box to remove outer zero rows/columns
    def crop_bounding_box(arr):
        rows = np.any(arr != 0, axis=1)
        cols = np.any(arr != 0, axis=0)
        return arr[np.ix_(rows, cols)]

    return crop_bounding_box(best_slice)

import matplotlib.pyplot as plt
import numpy as np

def plot_subject_dyad_heatmaps(data1, data2, title1='Subject A', title2='Subject B'):
    """Plot a 1x2 heatmap for two subjects' masked ROI slices (e.g. PHG), with colorbar."""
    
    # Set black background where values are 0
    data1_masked = np.ma.masked_where(data1 == 0, data1)
    data2_masked = np.ma.masked_where(data2 == 0, data2)
    
    vlim = np.max(np.abs([data1_masked, data2_masked]))

    fig, axs = plt.subplots(1, 2, figsize=(10, 5))

    for ax, data, title in zip(axs, [data1_masked, data2_masked], [title1, title2]):
        im = ax.imshow(data, cmap='coolwarm', vmin=-vlim, vmax=vlim)
        # ax.set_title(title, fontsize=14)
        ax.axis('off')

    # cbar = fig.colorbar(im, ax=axs.ravel().tolist(), orientation='vertical', pad=0.02)
    # cbar.set_label('t-values', fontsize=12)
    # cbar.set_ticks([-vlim, 0, vlim])
    # cbar.ax.tick_params(labelsize=10)


    plt.tight_layout()
    plt.show()

# Axis to slice along
AXIS_CUT = 'z'

# Determine best slice for high-similarity dyad (subjects 0 and 1)
best_sim_slice_idx = find_most_similar_slice(
    phg_maps_high_high[0], phg_maps_high_high[1], axis=AXIS_CUT, similarity='dot'
)

# Extract the same slice from all 4 PHG images
phg_slices = [
    extract_consistent_slice(img_dct, best_sim_slice_idx, axis=AXIS_CUT)
    for img_dct in list(phg_maps_high_high + phg_maps_low_low)
]

plot_subject_dyad_heatmaps(phg_slices[0], phg_slices[1],
    title1='High Sim – Subject 18', title2='High Sim – Subject 19')
plot_subject_dyad_heatmaps(phg_slices[2], phg_slices[3],
    title1='Low Sim – Subject 01', title2='Low Sim – Subject 02')

#%%
# Plot colorbar
reload(visu_utils)
tick_fontsize=75    
tick_labels = [np.round(np.min(phg_slices),2), "", np.round(-np.min(phg_slices),2)]
visu_utils.plot_isc_colorbar(
    vmin=np.min(phg_slices),
    vmax=-np.min(phg_slices),
    cmap=cmap,
    figsize=(2, 7),
    label=None,
    tick_fontsize=tick_fontsize,
    tick_labels=tick_labels,
    horizontal=False,)

visu_utils.plot_isc_colorbar(
    vmin=-1,
    vmax=1,
    cmap=cmap,
    figsize=(8,3),
    label=None,
    tick_fontsize=tick_fontsize,
    tick_labels=tick_labels,
    horizontal=True,)

#%%
# AMCC PAIRWISE SIMILARITY PLOTTING
AXIS_CUT = 'x'  # or 'z'
sim = 'pearson'

best_sim_slice_idx_amcc_high = find_most_similar_slice(
    amcc_maps_high_high[0], amcc_maps_high_high[1], axis=AXIS_CUT, similarity=sim
)
amcc_slices_high = [
    extract_consistent_slice(img_dct, best_sim_slice_idx_amcc_high, axis=AXIS_CUT)
    for img_dct in list(amcc_maps_high_high)
]

best_slice_idx_amcc_low = find_most_similar_slice(
    amcc_maps_low_low[0], amcc_maps_low_low[1], axis=AXIS_CUT, similarity=sim
)
amcc_slices_low = [
    extract_consistent_slice(img_dct, best_slice_idx_amcc_low, axis=AXIS_CUT)
    for img_dct in list(amcc_maps_low_low)
]


plot_subject_dyad_heatmaps(amcc_slices_high[0], amcc_slices_high[1],
                           title1='High Similarity – Subject 18',
                           title2='High Similarity – Subject 19')

plot_subject_dyad_heatmaps(amcc_slices_low[0], amcc_slices_low[1],
                           title1='Low Similarity – Subject 01',
                           title2='Low Similarity – Subject 02')

#%%
for NEURAL_SIM in ['dot']: #, 'dot']: #, 'cosine']: #, 'dot']: #['euclidean', 'dot']: #'cosine', 'pearson']:
    specific_re_run = False
    # NEURAL_SIM = 'euclidean'  # 'cosine' or 'dot' or 'euclidean'
    # BEHAV_ID = "SHSS"  # "SHSS" or "chge-pain"
    effect_type = 'stat' #'effect_size' #'z_score'
    save_path = f"/data/rainville/dSutterlin/projects/ISC_hypnotic_suggestions/results/imaging/RSA/IS-RSA-mvpa-{NEURAL_SIM}-{effect_type}_contrast-based_tian216" #_{ANNAK_VERSION}-annak"
    os.makedirs(save_path, exist_ok=True)
    ONLY_ON = [list(annak_sim_matrices.keys())[0]]

    BEHAV_ID = "pain_diff_Hyper"  # "SHSS" or "chge-pain" or "pain_diff_Ana" or "pain_diff_Hyper" or "total_change_pain"
    pain_conditions = ["ANA_shock_minus_N_ANA_shock", "HYPER_shock_minus_N_HYPER_shock"]
    if specific_re_run is True:
        
        print(f"Running IS-RSA for neural sim: {NEURAL_SIM}")
        for BEHAV_ID in ONLY_ON:
            print(f"Running IS-RSA for BEHAV: {BEHAV_ID}")


            if BEHAV_ID == "SHSS_score":
                sim_model = {
                    "euclidean": sim_behav_vec,
                    "annak": sim_behav_vec_annak,
                }
            if BEHAV_ID == "total_change_pain":
                sim_model = {
                    "euclidean": sim_behav_vec_pain,
                    "annak": sim_behav_vec_annak_pain,
                }
            if BEHAV_ID == "pain_diff_Ana":
                sim_model = {
                    "euclidean": isc_utils.compute_behav_similarity(
                        behav_interest['pain_diff_Ana'], metric="euclidean", vectorize=True
                    ),
                    "annak": isc_utils.compute_behav_similarity(
                        behav_interest['pain_diff_Ana'], metric="annak", annak_version=ANNAK_VERSION, vectorize=True
                    ),
                }
                pain_conditions = [pain_conditions[0]]
            if BEHAV_ID == "pain_diff_Hyper":
                sim_model = {
                    "euclidean": isc_utils.compute_behav_similarity(
                        behav_interest['pain_diff_Hyper'], metric="euclidean", vectorize=True
                    ),
                    "annak": isc_utils.compute_behav_similarity(
                        behav_interest['pain_diff_Hyper'], metric="annak", annak_version=ANNAK_VERSION, vectorize=True
                    ),
                }
                pain_conditions = [pain_conditions[1]]

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
                            pain_dict, atlas_img=altas, labels_roi_dct=labels_roi_dct,
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