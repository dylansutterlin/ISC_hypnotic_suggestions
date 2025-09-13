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


print("current working dir : ", os.getcwd())
# %%
from src import preproc_utils, visu_utils, qc_utils
import src.glm_utils as utils

from sklearn.utils import Bunch
from importlib import reload
from nilearn import datasets, image
from datetime import datetime

# %% [markdown]
## load data
preproc_model_name = r"model2_23subjects_zscore_sample_detrend_25-02-25"  # r'model2_3subjects_zscore_sample_detrend_25-02-25'
model_dir = rf"/data/rainville/dSutterlin/projects/ISC_hypnotic_suggestions/results/imaging/preproc_data/{preproc_model_name}"
model_name = "model3_23subjects_nuis_nodrift_{}".format(
    datetime.today().strftime("%d-%m-%y")
)

model_name = "model3_final-isc_23subjects_nuis_nodrift_31-03-25"  # final model, reproduced Desmarteaux et al., 2019 !! 31 mars

# load manually
# model_name = r'model3_23subjects_allnuis_nodrift_10-03-25'

# %%
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

save_glm = os.path.join(setup.project_dir, "results", "imaging", "GLM", model_name)
os.makedirs(save_glm, exist_ok=True)
setup.save_dir = save_glm

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
# ==========================
# mULTIVARIATE BEHAVIORAL
# ==========================
from sklearn.preprocessing import StandardScaler
from src import preproc_utils
import seaborn as sns
import statsmodels.api as sm

reload(preproc_utils)
xlsx_path = r"/data/rainville/dSutterlin/projects/ISC_hypnotic_suggestions/masks/Hypnosis_variables_20190114_pr_jc.xlsx"
subjects = list(setup["subjects"])
apm_subjects = ["APM" + subj[4:] for subj in subjects]
print(apm_subjects)

Y = preproc_utils.load_process_y(xlsx_path, subjects)

sugg_cols = [
    "SHSS_score",
    "Chge_hypnotic_depth",
    "Mental_relax_absChange",
    "Abs_diff_automaticity",
]

pain_cols = ["raw_change_HYPER", "raw_change_ANA"]

scaler = StandardScaler()
Y_sugg = scaler.fit_transform(np.array(Y[sugg_cols].values, dtype=float))
Y_pain = scaler.fit_transform(np.array(Y[pain_cols].values, dtype=float))


# Compute pairwise cosine similarity using the previously defined function
cosine_sim_sugg = isc_utils.compute_behav_similarity(
    Y_sugg, metric="cosine", vectorize=False
)
cosine_sim_pain = isc_utils.compute_behav_similarity(
    Y_pain, metric="cosine", vectorize=False
)

cosine_vec_sugg = isc_utils.compute_behav_similarity(
    Y_sugg, metric="cosine", vectorize=True
)
cosine_vec_pain = isc_utils.compute_behav_similarity(
    Y_pain, metric="cosine", vectorize=True
)


# ------------
import scipy.cluster.hierarchy as sch

# compute and apply clustering order
link = sch.linkage(cosine_sim_sugg, method="average")
order = sch.dendrogram(link, no_plot=True)["leaves"]
sim_ord = cosine_sim_sugg[np.ix_(order, order)]
labels_ord = Y.index[order]

# plot
plt.figure(figsize=(10, 8))
sns.heatmap(
    sim_ord,
    cmap="coolwarm",
    vmin=-1,
    vmax=1,
    xticklabels=labels_ord,
    yticklabels=labels_ord,
    square=True,
    cbar_kws={"label": "Cosine Similarity", "shrink": 0.8},
)
cbar = plt.gca().collections[0].colorbar
cbar.ax.tick_params(labelsize=15)
cbar.set_label("Cosine Similarity", fontsize=18)
plt.title(
    "Pairwise subject Cosine Similarity of hypnotic pattern responses", fontsize=22
)
plt.xticks(rotation=45, ha="right", fontsize=15)
plt.yticks(rotation=0, fontsize=15)
plt.tight_layout()
plt.show()
# ---------------------


VD = Y["total_chge_pain_hypAna"].apply(pd.to_numeric, errors="coerce")
X = pd.DataFrame(Y_sugg, index=Y.index).apply(pd.to_numeric, errors="coerce")
X = sm.add_constant(X)  # adds a column "const" for the intercept
X.columns = ["const"] + sugg_cols

model = sm.OLS(VD, X).fit()

print(model.summary())
coeffs = model.params
pvals = model.pvalues
print("\nCoefficients:\n", coeffs)
print("\nP-values:\n", pvals)

# 6) And some key statistics:
print(f"\nR² = {model.rsquared:.3f}, adj. R² = {model.rsquared_adj:.3f}")
print(f"F-statistic = {model.fvalue:.2f}, p(F) = {model.f_pvalue:.3g}")

# %%
# RSA
from sklearn.metrics.pairwise import cosine_similarity
from nilearn.image import load_img


def compute_inter_subject_mvpa_similarity(condition_dict, atlas_img, labels_roi_dct):
    """
    Computes inter-subject cosine similarity matrices (subject x subject) for each ROI
    based on multivoxel patterns from a single condition.

    Parameters
    ----------
    condition_dict : dict
        Dictionary mapping subject IDs to NIfTI image file paths for a single condition.
    atlas_img : Nifti1Image
        NIfTI image of the brain atlas (e.g., Schaefer parcellation).
    roi_indices : list of int
        List of ROI indices to include.

    Returns
    -------
    similarity_matrices : dict
        Keys are ROI indices, values are subject x subject cosine similarity matrices.
    vectorized_df : pd.DataFrame
        DataFrame where rows are pairwise subject combinations and columns are ROIs.
        Each cell is the cosine similarity between two subjects for that ROI.
    """

    atlas_data = np.round(atlas_img.get_fdata()).astype(int)
    subjects = sorted(condition_dict.keys())
    n_subjects = len(subjects)
    roi_indices = list(labels_roi_dct.keys())
    roi_labels = list(labels_roi_dct.values())

    # Load all data once
    condition_data_dict = {
        subj: load_img(condition_dict[subj]).get_fdata() for subj in subjects
    }

    similarity_matrices = {}
    similarity_vectors = []

    for roi_idx in roi_indices:
        roi_mask = atlas_data == roi_idx
        if not roi_mask.any():
            print("Region id not found in atlas, skipping..")
            continue

        subj_vectors = []

        for subj in subjects:
            vec = condition_data_dict[subj][roi_mask].flatten()

            if np.linalg.norm(vec) == 0:
                subj_vectors.append(np.full(np.sum(roi_mask), np.nan))
                print("region mask is all 0 ")
            else:
                subj_vectors.append(vec)

        subj_vectors = np.array(subj_vectors)

        valid_mask = ~np.isnan(subj_vectors).any(axis=1)
        valid_vectors = subj_vectors[valid_mask]
        valid_subjects = np.array(subjects)[valid_mask]

        if valid_vectors.shape[0] < 2:
            print("Too few subjects for a ROI, skipping..")
            continue

        sim_matrix = cosine_similarity(valid_vectors)
        similarity_matrices[roi_idx] = pd.DataFrame(
            sim_matrix, index=valid_subjects, columns=valid_subjects
        )

        # Extract upper triangle for this ROI
        upper_tri_indices = np.triu_indices(sim_matrix.shape[0], k=1)
        upper_tri_values = sim_matrix[upper_tri_indices]
        similarity_vectors.append(pd.Series(upper_tri_values, name=roi_idx))

    vectorized_df = pd.concat(similarity_vectors, axis=1)
    vectorized_df.columns.name = "ROIs"
    vectorized_df.columns = roi_labels

    return similarity_matrices, vectorized_df


def build_subject_dict(file_list):
    """Build a dict: {subject_id: filepath} from a list of NIfTI file paths."""
    subject_dict = {}
    for path in file_list:
        fname = os.path.basename(path)
        subj_id = fname.split("_")[-1].replace(
            ".nii.gz", ""
        )  # expects '..._sub-01.nii.gz'
        subject_dict[subj_id] = path
    return subject_dict


# %%
# Visualize second-level contrast for pain
conditions = [
    "ANA_shock_minus_N_ANA_shock",
    "HYPER_shock_minus_N_HYPER_shock",
    "N_ANA_shock_minus_N_HYPER_shock",
]
test = "/data/rainville/dSutterlin/projects/ISC_hypnotic_suggestions/results/imaging/GLM/model3_final-isc_23subjects_nuis_nodrift_31-03-25/second_level/ANA_shock_minus_N_ANA_shock_all_effects.pkl"
second_lev_dir = os.path.join(results_dir, "second_level")
z_imgs = {}
for cond in conditions:

    pkl = isc_utils.load_pickle(os.path.join(second_lev_dir, cond + "_all_effects.pkl"))
    z_img = pkl["stat"]
    img = nib.load(os.path.join(second_lev_dir, "group_" + cond + ".nii.gz"))
    z_imgs[cond] = z_img

view_img(z_img)

# Threshold and plot the second-level z-map using FDR correction
thresh_imgs = []
for cond, z_img in z_imgs.items():
    # FDR thresholding
    thresholded_img, threshold = threshold_stats_img(
        z_img,
        alpha=0.05,
        height_control="bonferroni",
        cluster_threshold=0,
        two_sided=True,
    )
    print(f"{cond}: FDR threshold = {threshold:.3f}")

    # Plot thresholded map
    plot_stat_map(
        thresholded_img,
        bg_img=mni_bg,
        display_mode="ortho",
        threshold=threshold,
        title=f"{cond} (FDR<0.05)",
        colorbar=True,
    )
    thresh_imgs.append(thresholded_img)
    plt.show()

# %%
# Atlas reader to make cluster
from atlasreader import create_output

create_output(thresholded_img, cluster_extent=5)

# %%
view_img(thresh_imgs[1])
# %%
# ===========================
# MAIN CODE
from tqdm import tqdm
from nilearn.glm.thresholding import threshold_stats_img

mvpa_save_to = os.path.join(results_dir, "mvpa_similarity")
os.makedirs(mvpa_save_to, exist_ok=True)

conditions = [
    "ANA_sugg_minus_N_ANA_sugg",
    "HYPER_sugg_minus_N_HYPER_sugg",
]  # , 'modulation_sugg', 'HYPER_sugg', 'ANA_sugg', 'neutral_sugg', 'ana_run_sugg', 'hyper_run_sugg']
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

save_path = os.path.join( project_dir, "results/imaging/RSA/mvpa_IS-RSA_sugg-pain_contrast-based_tian216")
os.makedirs(save_path, exist_ok=True)

# %%

compute_rsa_pain = False
if compute_rsa_pain is True:
    # SUGGEST MVPA IS-RSA
    # sim_model = {
    #     #"cosine": cosine_sim_sugg,
    #     "euclidean": sim_behav_vec,
    #     "annak": sim_behav_vec_annak,
    # }
    sim_model = {"euclidean": sim_behav_vec_pain, "annak": sim_behav_vec_annak_pain}

    sugg_conditions = {"ANA_sugg_minus_N_ANA_sugg", "HYPER_sugg_minus_N_HYPER_sugg"}
    shock_conditions = {
        "ANA_shock_minus_N_ANA_shock",
        "HYPER_shock_minus_N_HYPER_shock",
    }
    results_dct = {}

    for sim, behav_sim_vec_i in sim_model.items():
        print(f"Performing IS-RSA with {sim} similarity metric")
        results_dct[sim] = {}

        for cond in tqdm(shock_conditions):  # shock model!!
            print("Performing RSA on : ", cond)
            # load maps
            # all_shock_maps = glob(os.path.join(model_res, 'all_shock', 'firstlev_localizer_*.nii.gz'))
            sugg_maps = glob(
                os.path.join(
                    model_res, "first_level", cond, "firstlev_localizer_*.nii.gz"
                )
            )
            sugg_dict = build_subject_dict(sugg_maps)  # not sorted!!
            sugg_dict = {sub: sugg_dict[sub] for sub in subjects}

            # mvpa similarity
            similarity_matrices_dct, vec_similarity_df = (
                compute_inter_subject_mvpa_similarity(
                    sugg_dict, atlas_img=ATLAS, labels_roi_dct=labels_roi_dct
                )
            )

            # RSA computation + perm
            rsa_rows = []  # build df
            for i, roi in tqdm(
                enumerate(vec_similarity_df.columns),
                total=len(vec_similarity_df.columns),
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
                        "ROI": roi + 1,  # corrects for 0-indexing
                        "spearman_r": round(r, 3),
                        "p_values": round(p, 5),
                        "Coordinates": tuple(np.round(coords[i], 0).astype(int)),
                    }
                )

            results_dct[sim][cond] = pd.DataFrame(rsa_rows)  # .sort_values(
            #     by="spearman_r", ascending=False
            # )
            print(
                "Max r, mean and fdr",
                results_dct[sim][cond]["spearman_r"].max(),
                results_dct[sim][cond]["spearman_r"].mean(),
                isc_utils.fdr(results_dct[sim][cond]["p_values"].to_numpy()),
            )

    save_to = os.path.join(
        save_path, f"IS-RSA_mvpa-pain_behav-pain{n_perm_rsa}perm.pkl"
    )
    isc_utils.save_data(save_to, results_dct)
    print(f"Saved RSA results to {save_to}")

else:
    # load results
    load_path = os.path.join(
        save_path, f"IS-RSA_mvpa-pain_behav-pain{n_perm_rsa}perm.pkl"
    )
results_dct = isc_utils.load_pickle(load_path)

# %%
# VISUALISE
rsa_stats_imgs_mvpa = {}
rsa_tables_mvpa_fdr = {}
model_name = "shock_contrast_chge-pain"
save_tables_rsa = os.path.join(save_path, model_name, "tables", "rsa")
os.makedirs(save_tables_rsa, exist_ok=True)

all_p_values = []
for sim in results_dct.keys():  # similarity metric (e.g., 'euclidean', 'annak')
    for cond in results_dct[sim].keys():  # e.g., 'Analgesia', 'Hyperalgesia'

        model_id = sim + "_" + cond
        rsa_df = results_dct[sim][cond]
        p_values = rsa_df["p_values"].values
        all_p_values.extend(p_values)

p_thresh_global = isc_utils.fdr(np.array(all_p_values), q=0.05)
print("Global FDR threshold across all models:", p_thresh_global)

# Main loop to visualize results and threshold
for sim in results_dct.keys():  # similarity metric (e.g., 'euclidean', 'annak')
    for cond in results_dct[sim].keys():  # e.g., 'Analgesia', 'Hyperalgesia'

        model_id = sim + "_" + cond
        rsa_df = results_dct[sim][cond]
        # Ensure the dataframe rows are in the same order as the atlas labels

        correlations = rsa_df["spearman_r"].values
        p_values = rsa_df["p_values"].values
        region_id = rsa_df["ROI"].values

        for correction in ["unc01", "within_FDR", "FDR", "bonf"]:
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
                sig_df.to_csv(os.path.join(save_tables_rsa, fname), index=False)

                print(f"Sig ROIs for {sim} – {cond}:", sig_df["Label"].to_list(), "\n")
# =====================================
# Significant results for MVPA pain-activation ~ individual differences in change in pain ratings (global scores)
# %%
rsa_views = []
for model_id in rsa_stats_imgs_mvpa.keys():
    title = f"RSA MVPA {model_id}"
    rsa_views.append(
        view_img(rsa_stats_imgs_mvpa[model_id][0], title=title, cmap="RdBu_r")
    )

# %%
# CAN WE DECODE sugg condition from MVPA pattern in OFC?

from sklearn.model_selection import GroupKFold
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler

from nilearn.maskers import NiftiMasker
from nilearn.image import new_img_like

# %%
roi_to_test = ["7Networks_LH_Default_PHC_1"]
roi_to_test = ["7Networks_LH_Limbic_OFC_1"]
# hyper contrast roi_to_test = ['7Networks_LH_Limbic_OFC_1', '7Networks_LH_Default_PFC_3', 'PUT-rh'] #, '7Networks_LH_Default_PHC_1', '7Networks_RH_DorsAttn_Post_4', '7Networks_RH_SalVentAttn_TempOccPar_2']
# ana classif roi_to_test = ['7Networks_LH_Default_PHC_1', '7Networks_RH_DorsAttn_Post_4', '7Networks_RH_SalVentAttn_TempOccPar_2']

decoding_conditions = ["ANA_shock", "N_ANA_shock"]
decoding_conditions = ["HYPER_shock", "N_HYPER_shock"]  # , "ANA_shock", "N_ANA_shock"]

kernel = "rbf"
condition_maps = {}
roi_maskers = {}

for cond in tqdm(decoding_conditions):  # shock model!!
    # load maps
    # all_shock_maps = glob(os.path.join(model_res, 'all_shock', 'firstlev_localizer_*.nii.gz'))
    shock_maps = glob(
        os.path.join(model_res, "first_level", cond, "firstlev_localizer_*.nii.gz")
    )
    shock_dict = build_subject_dict(shock_maps)  # not sorted!!
    shock_dict = {sub: shock_dict[sub] for sub in subjects}
    condition_maps[cond] = shock_dict  # re-ordered dict

atlas_data = atlas.get_fdata()

# pre-load maps
all_imgs = {}  # Structure: all_imgs[cond][sub_id] = loaded NIfTI image

for cond in decoding_conditions:
    all_imgs[cond] = {}
    for sub_id, map_path in condition_maps[cond].items():
        all_imgs[cond][sub_id] = nib.load(map_path)

roi_maskers = {}

# Initialize masker for each ROI
for roi_label in roi_to_test:
    roi_idx = [k for k, v in labels_roi_dct.items() if v == roi_label][0]
    roi_mask = atlas_data == roi_idx
    mask_img = new_img_like(atlas, roi_mask)
    roi_maskers[roi_label] = NiftiMasker(mask_img=mask_img, standardize=False)

for roi_label in tqdm(roi_to_test):
    print(f"--------\n {roi_label}\n---------")
    roi_masker = roi_maskers[roi_label]

    X_all = []
    y_all = []
    subjects_all = []

    for cond in decoding_conditions:
        for sub_id, img in all_imgs[cond].items():
            vec = roi_masker.fit_transform(img)  # shape = [1, voxels]
            X_all.append(vec[0])
            y_all.append(cond)
            subjects_all.append(sub_id)

    X_all = np.vstack(X_all)
    le = LabelEncoder()
    y_encoded = le.fit_transform(y_all)

    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=0.80, svd_solver="full")),
            ("svm", SVC(kernel=kernel, C=1)),
        ]
    )

    gkf = GroupKFold(n_splits=5)
    scores = cross_val_score(
        pipeline, X_all, y_encoded, groups=np.array(subjects_all), cv=gkf
    )

    y_pred = cross_val_predict(pipeline, X_all, y_encoded, groups=subjects_all, cv=gkf)
    cm = confusion_matrix(y_encoded, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=le.classes_)
    disp.plot()

    print("Mean accuracy (PCA + SVM, GroupKFold-5):", np.mean(scores))

    # 2D PCA for visualization
    scaler_vis = StandardScaler()
    X_scaled = scaler_vis.fit_transform(X_all)
    pca_vis = PCA(n_components=2)
    X_2d = pca_vis.fit_transform(X_scaled)

    # Fit on full data for plot only
    clf_vis = SVC(kernel=kernel, C=1)
    clf_vis.fit(X_2d, y_encoded)

    # Plot decision surface
    x_min, x_max = X_2d[:, 0].min() - 1, X_2d[:, 0].max() + 1
    y_min, y_max = X_2d[:, 1].min() - 1, X_2d[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300), np.linspace(y_min, y_max, 300))
    Z = clf_vis.decision_function(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, Z, alpha=0.75, cmap=plt.cm.coolwarm)
    scatter = plt.scatter(
        X_2d[:, 0], X_2d[:, 1], c=y_encoded, cmap=plt.cm.coolwarm, edgecolors="k"
    )
    plt.title(f"Decision Surface – {roi_label}")
    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")
    plt.colorbar(scatter, ticks=range(len(le.classes_)), label="Condition")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# ================================
# %%
views = []
for model_id in rsa_stats_imgs_mvpa.keys():
    title = f"RSA MVPA {model_id}"
    view = view_img(rsa_stats_imgs_mvpa[model_id][0], title=title)
    views.append(view)
# %%
# PUBLICATION
# NEW MODEL SUGGESTION VAR ~ PAIN CONTRAST
# IS-RSA SHSS ~ ISC-PAIN
specific_re_run = False

save_path = f"/data/rainville/dSutterlin/projects/ISC_hypnotic_suggestions/results/imaging/RSA/mvpa_IS-RSA_sugg-pain_contrast-based_tian216"
os.makedirs(save_path, exist_ok=True)


if specific_re_run is True:
    sim_model = {
        "euclidean": sim_behav_vec,
        "annak": sim_behav_vec_annak,
    }

    pain_conditions = {"ANA_shock_minus_N_ANA_shock", "HYPER_shock_minus_N_HYPER_shock"}
    results_dct = {}

    for sim, behav_sim_vec_i in sim_model.items():
        print(f"Performing IS-RSA with {sim} similarity metric")
        results_dct[sim] = {}

        for cond in tqdm(pain_conditions):
            print("Performing RSA on : ", cond)
            # load maps
            # all_shock_maps = glob(os.path.join(model_res, 'all_shock', 'firstlev_localizer_*.nii.gz'))
            pain_maps = glob(
                os.path.join(
                    model_res, "first_level", cond, "firstlev_localizer_*.nii.gz"
                )
            )
            pain_dict = build_subject_dict(pain_maps)  # not sorted!!
            pain_dict = {sub: pain_dict[sub] for sub in subjects}

            # mvpa similarity
            similarity_matrices_dct, vec_similarity_df = (
                compute_inter_subject_mvpa_similarity(
                    pain_dict, atlas_img=ATLAS, labels_roi_dct=labels_roi_dct
                )
            )

            # RSA computation + perm
            rsa_rows = []  # build df
            # Use tqdm to show progress bar over ROI columns
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

            print(
                "Max r, mean and fdr",
                results_dct[sim][cond]["spearman_r"].max(),
                results_dct[sim][cond]["spearman_r"].mean(),
                isc_utils.fdr(results_dct[sim][cond]["p_values"].to_numpy()),
            )

    save_to = os.path.join(
        save_path, f"IS-RSA_mvpa-pain_behav-pain{n_perm_rsa}perm.pkl"
    )
    isc_utils.save_data(save_to, results_dct)
    print(f"Saved RSA results to {save_to}")

else:
    # load results

    load_path = os.path.join(
        save_path, f"IS-RSA_mvpa-pain_behav-pain{n_perm_rsa}perm.pkl"
    )
    results_dct = isc_utils.load_pickle(load_path)
# %%
# =================
# VISUALISE PAIN-contrast ~ SHSS
rsa_stats_imgs_mvpa = {}
rsa_tables_mvpa_fdr = {}
model_name = "shock_contrast_SHSS"
save_tables_shock_shss = os.path.join(save_path, model_name, "tables")
os.makedirs(save_tables_shock_shss, exist_ok=True)
save_plots_shock_shss = os.path.join(save_path, model_name, "plots")
os.makedirs(save_plots_shock_shss, exist_ok=True)

all_p_values = []
for sim in results_dct.keys():  # similarity metric (e.g., 'euclidean', 'annak')
    for cond in results_dct[sim].keys():  # e.g., 'Analgesia', 'Hyperalgesia'

        model_id = sim + "_" + cond
        rsa_df = results_dct[sim][cond]
        # Ensure the dataframe rows are in the same order as the atlas labels

        correlations = rsa_df["spearman_r"].values
        p_values = rsa_df["p_values"].values
        region_id = rsa_df["ROI"].values
        all_p_values.append(p_values)

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

        for correction in ["unc01", "within_FDR", "FDR", "bonf"]:
            show = False

            if correction == "unc01":
                p_thresh = 0.01
                p_thresh_str = "0.01"

            elif correction == "within_FDR":
                p_thresh = isc_utils.fdr(p_values, q=0.05)
                p_thresh_str = "within_FDR" + str(round(p_thresh, 4))

            elif correction == "FDR":
                p_thresh = global_fdr_thresh  # or load global value
                p_thresh_str = "FDR" + str(round(p_thresh, 4))
                show = True

            elif correction == "bonf":
                p_thresh = isc_utils.bonferroni(p_values, alpha=0.05)
                p_thresh_str = "Bonf" + str(round(p_thresh, 4))

            rsa_img, rsa_thresh, sig_df, stats_img = (
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

                print(f"Sig ROIs for {sim} – {cond}:", sig_df["Label"].to_list(), "\n")
#%%
# PUBLICATION IS-RSA SHSS - PAIN CONTRAST
# View significant results and find coords to display
shss_annak_Ana_df = rsa_tables_mvpa_fdr['annak_ANA_shock_minus_N_ANA_shock']
shss_annak_Ana_img = rsa_stats_imgs_mvpa['annak_ANA_shock_minus_N_ANA_shock'][0]
view_img(shss_annak_Ana_img, title="RSA MVPA annak_ANA_shock_minus_N_ANA_shock", cmap="RdBu_r")

#%%
# =====
# MAnual viewvinf for COORDS
import atlasreader
from nilearn.image import math_img
pwd_scripts = os.path.join(project_dir, "scripts")
os.chdir(os.path.join(pwd_scripts, "atlas_views"))
print("Current working directory:", os.getcwd())
from atlasreader import create_output

def filter_atlas(atlas_img, atlas_labels_dict, keep_ids):
    atlas_data = atlas_img.get_fdata()
    mask_data = np.isin(atlas_data, keep_ids) * atlas_data
    filtered_img = new_img_like(atlas_img, mask_data)

    # Build new labels dict for the kept IDs
    new_labels = {
        roi: label for roi, label in atlas_labels_dict.items() if roi in keep_ids
    }
    return filtered_img, new_labels


# Plot the filtered atlas with clear discrete colorbar
def plot_filtered_atlas(filtered_img, filtered_labels, cmap="tab20"):
    label_ids = list(filtered_labels.keys())
    label_names = list(filtered_labels.values())

    return plotting.view_img(
        filtered_img,
        cmap=cmap,
        colorbar=True,
        title="Filtered Atlas",
        symmetric_cmap=False,
    )


def plot_single_roi(atlas_img, roi_id, atlas_labels=None, title=None):

    mask_img = math_img(f"img == {roi_id}", img=atlas_img)

    title = f"ROI {roi_id}"
    if atlas_labels and roi_id in atlas_labels:
        title += f" {atlas_labels[roi_id]}"

    print(f"Displaying region {roi_id}: {title}")
    return mask_img, plotting.view_img(
        mask_img, cmap="Reds", title=title, colorbar=False
    )

roi_idx = 85
masked_region, _plot = plot_single_roi(
    atlas, roi_idx, id_labels_dct, title=f"roi idx {roi_idx}"
)  # 30 = Default_Temp_3 LH
_plot

create_output(masked_region, voxel_thresh=0.99, cluster_extent=5)
atlas_df = pd.read_csv("atlasreader_peaks.csv")
atlas_df
#%%
# PUBLICATION SLICES SHSS-PAIN CONTRAST: ANAAK
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

# %%
from nilearn.image import new_img_like, math_img
from nilearn.maskers import NiftiMasker
# DECODING SUGG CONDITION FROM MVPA PATTERN IN ROIS
roi_to_test = ["7Networks_RH_DorsAttn_Post_10", "7Networks_RH_SalVentAttn_Med_1"]
roi_to_test = [
    "7Networks_LH_Default_PHC_1"
]  # , "7Networks_LH_SalVentAttn_FrOperIns_4"]
# hyper contrast roi_to_test = ['7Networks_LH_Limbic_OFC_1', '7Networks_LH_Default_PFC_3', 'PUT-rh'] #, '7Networks_LH_Default_PHC_1', '7Networks_RH_DorsAttn_Post_4', '7Networks_RH_SalVentAttn_TempOccPar_2']
# ana classif roi_to_test = ['7Networks_LH_Default_PHC_1', '7Networks_RH_DorsAttn_Post_4', '7Networks_RH_SalVentAttn_TempOccPar_2']

decoding_conditions = ["HYPER_shock", "N_HYPER_shock", "ANA_shock", "N_ANA_shock"]
decoding_conditions = ["ANA_shock", "HYPER_shock"]

# REGESSION
# Assuming Y is a pandas DataFrame already sorted by subject
pain_cols = {
    "N_ANA": ["mean_VAS_Nana_int", "mean_VAS_Nana_UnP"],
    "ANA": ["mean_VAS_ana_int", "mean_VAS_ana_UnP"],
    "N_HYPER": ["mean_VAS_Nhyper_int", "mean_VAS_Nhyper_UnP"],
    "HYPER": ["mean_VAS_hyper_int", "mean_VAS_hyper_UnP"],
}

Y_pain_mean = pd.DataFrame()
for cond, cols in pain_cols.items():
    Y_pain_mean[cond] = Y[cols].mean(axis=1)

# Load MVP in each regions based on ROIs to test
condition_maps = {}
roi_maskers = {}

for cond in tqdm(decoding_conditions):  # shock model!!
    # load maps
    # all_shock_maps = glob(os.path.join(model_res, 'all_shock', 'firstlev_localizer_*.nii.gz'))
    shock_maps = glob(
        os.path.join(model_res, "first_level", cond, "firstlev_localizer_*.nii.gz")
    )
    shock_dict = build_subject_dict(shock_maps)  # not sorted!!
    shock_dict = {sub: shock_dict[sub] for sub in subjects}
    condition_maps[cond] = shock_dict  # re-ordered dict

atlas_data = atlas.get_fdata()

# pre-load maps
all_imgs = {}  # Structure: all_imgs[cond][sub_id] = loaded NIfTI image

for cond in decoding_conditions:
    all_imgs[cond] = {}
    for sub_id, map_path in condition_maps[cond].items():
        all_imgs[cond][sub_id] = nib.load(map_path)

roi_maskers = {}

# Initialize masker for each ROI
for roi_label in roi_to_test:
    roi_idx = [k for k, v in labels_roi_dct.items() if v == roi_label][0]
    roi_mask = atlas_data == roi_idx
    mask_img = new_img_like(atlas, roi_mask)
    roi_maskers[roi_label] = NiftiMasker(mask_img=mask_img, standardize=False)


# %%
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

for roi_label in tqdm(roi_to_test):
    print(f"--------\n {roi_label}\n---------")
    roi_masker = roi_maskers[roi_label]

    X_all = []
    y_all = []
    subjects_all = []

    for cond in decoding_conditions:
        key = cond[:-6]  # Matches keys in Y_pain_mean: e.g., 'ANA'
        for i, sub_id in enumerate(subjects):
            img = all_imgs[cond][sub_id]
            vec = roi_masker.fit_transform(img)  # shape = [1, voxels]
            X_all.append(vec[0])
            key_idx = Y_pain_mean.columns.get_loc(key)
            y_all.append(Y_pain_mean.iloc[i, key_idx])
            subjects_all.append(sub_id)

    X_all = np.vstack(X_all)
    y_all = np.array(y_all)
    subjects_all = np.array(subjects_all)

    gkf = GroupKFold(n_splits=5)
    fold_r2s = []
    fold_rmses = []
    y_pred_all = np.zeros_like(y_all)

    for fold_idx, (train_idx, test_idx) in enumerate(
        gkf.split(X_all, y_all, groups=subjects_all)
    ):
        X_train, X_test = X_all[train_idx], X_all[test_idx]
        y_train, y_test = y_all[train_idx], y_all[test_idx]

        pipeline = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("pca", PCA(n_components=10, svd_solver="full")),
                ("reg", RidgeCV(cv=5)),
            ]
        )

        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        y_pred_all[test_idx] = y_pred  # store full predictions

        fold_r2s.append(r2_score(y_test, y_pred))
        fold_rmses.append(mean_squared_error(y_test, y_pred, squared=False))

    # Print overall performance
    mean_r2 = r2_score(y_all, y_pred_all)
    mean_rmse = mean_squared_error(y_all, y_pred_all, squared=False)
    print(f"Mean R²: {mean_r2:.3f}, RMSE: {mean_rmse:.3f}")

    # Scatter plot (true vs. predicted)
    plt.figure(figsize=(6, 6))
    plt.scatter(y_all, y_pred_all, c="steelblue", edgecolor="k", alpha=0.7)
    plt.plot([y_all.min(), y_all.max()], [y_all.min(), y_all.max()], "r--")
    plt.xlabel("True Pain Ratings")
    plt.ylabel("Predicted Pain Ratings")
    plt.title(
        f"{roi_label} – Ridge Regression\nR² = {mean_r2:.2f}, RMSE = {mean_rmse:.2f}"
    )
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# %%
# CLASSIFICATION
# get pain scores for each conditions from Y
columns = []
kernel = "rbf"

# Initialize masker for each ROI
for roi_label in roi_to_test:
    roi_idx = [k for k, v in labels_roi_dct.items() if v == roi_label][0]
    roi_mask = atlas_data == roi_idx
    mask_img = new_img_like(atlas, roi_mask)
    roi_maskers[roi_label] = NiftiMasker(mask_img=mask_img, standardize=False)

for roi_label in tqdm(roi_to_test):
    print(f"--------\n {roi_label}\n---------")
    roi_masker = roi_maskers[roi_label]

    X_all = []
    y_all = []
    subjects_all = []

    for cond in decoding_conditions:
        for sub_id, img in all_imgs[cond].items():
            vec = roi_masker.fit_transform(img)  # shape = [1, voxels]
            X_all.append(vec[0])
            y_all.append(cond)
            subjects_all.append(sub_id)

    X_all = np.vstack(X_all)
    le = LabelEncoder()
    y_encoded = le.fit_transform(y_all)

    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=0.80, svd_solver="full")),
            ("svm", SVC(kernel=kernel, C=1)),
        ]
    )
    
    gkf = GroupKFold(n_splits=5)
    scores = cross_val_score(
        pipeline, X_all, y_encoded, groups=np.array(subjects_all), cv=gkf
    )

    y_pred = cross_val_predict(pipeline, X_all, y_encoded, groups=subjects_all, cv=gkf)
    cm = confusion_matrix(y_encoded, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=le.classes_)
    disp.plot()

    print("Mean accuracy (PCA + SVM, GroupKFold-5):", np.mean(scores))

    # 2D PCA for visualization
    scaler_vis = StandardScaler()
    X_scaled = scaler_vis.fit_transform(X_all)
    pca_vis = PCA(n_components=2)
    X_2d = pca_vis.fit_transform(X_scaled)

    # Fit on full data for plot only
    clf_vis = SVC(kernel=kernel, C=1)
    clf_vis.fit(X_2d, y_encoded)

    # Plot decision surface
    x_min, x_max = X_2d[:, 0].min() - 1, X_2d[:, 0].max() + 1
    y_min, y_max = X_2d[:, 1].min() - 1, X_2d[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300), np.linspace(y_min, y_max, 300))
    Z = clf_vis.decision_function(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, Z, alpha=0.75, cmap=plt.cm.coolwarm)
    scatter = plt.scatter(
        X_2d[:, 0], X_2d[:, 1], c=y_encoded, cmap=plt.cm.coolwarm, edgecolors="k"
    )
    plt.title(f"Decision Surface – {roi_label}")
    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")
    plt.colorbar(scatter, ticks=range(len(le.classes_)), label="Condition")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# %%
# IS-RSA SUGGESTION - PAIN
# SUGGEST MVPA IS-RSA
# goal : test regions that have consistent mvpa across suggestion to pain
if specific_re_run is False:
    sugg_pain_conditions = {
        "ANA_sugg_minus_N_ANA_sugg": "ANA_shock_minus_N_ANA_shock",
        "HYPER_sugg_minus_N_HYPER_sugg": "HYPER_shock_minus_N_HYPER_shock",
    }

    #     'modulation_sugg': 'modulation_shock',
    #     'HYPER_sugg': 'HYPER_shock',
    #    'ANA_sugg': 'ANA_shock',
    #    'neutral_sugg': 'neutral_shock',
    #    'ana_run_sugg' : 'ana_run_shock',
    #    'hyper_run_sugg': 'hyper_run_shock'}

    results_dct = {}

    for cond_sugg, cond_shock in tqdm(sugg_pain_conditions.items()):
        cond = cond_sugg

        print("Performing RSA on : ", cond_sugg, "with shock condition:", cond_shock)
        # load maps
        shock_maps = glob(
            os.path.join(
                model_res, "first_level", cond_shock, "firstlev_localizer_*.nii.gz"
            )
        )
        sugg_maps = glob(
            os.path.join(model_res, "first_level", cond, "firstlev_localizer_*.nii.gz")
        )

        def build_subject_dict(file_list):
            """Build a dict: {subject_id: filepath} from a list of NIfTI file paths."""
            subject_dict = {}
            for path in file_list:
                fname = os.path.basename(path)
                subj_id = fname.split("_")[-1].replace(
                    ".nii.gz", ""
                )  # expects '..._sub-01.nii.gz'
                subject_dict[subj_id] = path
            return subject_dict

        sugg_dict = build_subject_dict(sugg_maps)  # not sorted!!
        shock_dict = build_subject_dict(shock_maps)
        subjects = sorted(set(sugg_dict) & set(shock_dict))

        # mvpa similarity
        sugg_similarity_matrices_dct, sugg_vec_similarity_df = (
            compute_inter_subject_mvpa_similarity(
                sugg_dict, atlas_img=ATLAS, labels_roi_dct=labels_roi_dct
            )
        )
        pain_similarity_matrices_dct, pain_vec_similarity_df = (
            compute_inter_subject_mvpa_similarity(
                shock_dict, atlas_img=ATLAS, labels_roi_dct=labels_roi_dct
            )
        )

        # RSA computation + perm
        rsa_rows = []  # build df
        for i, roi in enumerate(sugg_vec_similarity_df.columns):

            mvpa_sim_vec_sugg = sugg_vec_similarity_df[roi].values
            mvpa_sim_vec_pain = pain_vec_similarity_df[roi].values
            r, p, dist = isc_utils.matrix_permutation(
                mvpa_sim_vec_sugg,
                mvpa_sim_vec_pain,
                n_permute=n_perm_rsa,
                metric="spearman",
                how="upper",
                tail=2,
                return_perms=True,
            )

            rsa_rows.append(
                {
                    "ROI": roi,
                    "spearman_r": r,
                    "p_values": round(p, 5),
                    "Coordinates": tuple(np.round(coords[i], 0).astype(int)),
                }
            )

        results_dct[cond] = pd.DataFrame(rsa_rows).sort_values(
            by="spearman_r", ascending=False
        )
        print(
            "Max r, mean and fdr",
            results_dct[cond]["spearman_r"].max(),
            results_dct[cond]["spearman_r"].mean(),
            isc_utils.fdr(results_dct[cond]["p_values"].to_numpy()),
        )

    save_to = os.path.join(
        save_path, f"IS-RSA_contrast-based_mvpa-sugg_mvpa-pain{n_perm_rsa}perm.pkl"
    )
    isc_utils.save_data(save_to, results_dct)
    print(f"Saved RSA results to {save_to}")

# %%
# CROSS MODEL IS-RSA ISC-sugg ~ MVPA-pain
# New, added on june 21 25

# conditions to fetch for isc mat sugg (key) and pain MVPA (value)
isc_mvpa_conditions = {
    "ana_run": "ANA_shock_minus_N_ANA_shock",
    "hyper_run": "HYPER_shock_minus_N_HYPER_shock",
}

# Load isc matrices for suggestions
isc_conditions = ["ana_run", "hyper_run"]
MODEL_ISC = "model2_sugg_23-sub_schafer_tian-200-2mm_mask-whole-brain_pairWise-True_preproc_reg-mvmnt-True-8"
n_perm_isc = 10000
isc_sugg_vecs = {}
for isc_cond in isc_conditions:

    if isc_cond in ["ANA", "HYPER", "NANA", "NHYPER"]:
        sugg_path = f"/data/rainville/dSutterlin/projects/ISC_hypnotic_suggestions/results/imaging/ISC/{MODEL_ISC}/{isc_cond}/isc_results_{isc_cond}_{n_perm_isc}boot_pairWiseTrue.pkl"
        # pain_path =   f'/data/rainville/dSutterlin/projects/ISC_hypnotic_suggestions/results/imaging/ISC/{model_pain}/{isc_cond}/isc_results_{isc_cond}_{n_perm}boot_pairWiseTrue.pkl'
    else:
        sugg_path = f"/data/rainville/dSutterlin/projects/ISC_hypnotic_suggestions/results/imaging/ISC/{MODEL_ISC}/concat_suggs_1samp_boot/isc_results_{isc_cond}_{n_perm_isc}boot_pairWiseTrue.pkl"
        # pain_path = f'/data/rainville/dSutterlin/projects/ISC_hypnotic_suggestions/results/imaging/ISC/{model_pain}/concat_suggs_1samp_boot/isc_results_{isc_cond}_{n_perm}boot_pairWiseTrue.pkl'

    isc_sugg_vec = pd.DataFrame(isc_utils.load_pickle(sugg_path)["isc"], columns=labels)
    print(isc_sugg_vec.shape)
    isc_sugg_vecs[isc_cond] = isc_sugg_vec

# run IS-RSA
results_dct = {}

for cond_sugg, cond_shock in tqdm(isc_mvpa_conditions.items()):

    print("Performing RSA on : ", cond_sugg, "with shock condition:", cond_shock)
    isc_vec = isc_sugg_vecs[cond_sugg]
    # load maps
    shock_maps = glob(
        os.path.join(
            model_res, "first_level", cond_shock, "firstlev_localizer_*.nii.gz"
        )
    )
    shock_dict = build_subject_dict(shock_maps)

    # mvpa similarity
    pain_similarity_matrices_dct, pain_vec_similarity_df = (
        compute_inter_subject_mvpa_similarity(
            shock_dict, atlas_img=ATLAS, labels_roi_dct=labels_roi_dct
        )
    )

    # RSA computation + perm
    rsa_rows = []  # build df
    for i, roi in enumerate(pain_vec_similarity_df.columns):

        isc_sim_vec_sugg = isc_vec[roi].values
        mvpa_sim_vec_pain = pain_vec_similarity_df[roi].values

        r, p, dist = isc_utils.matrix_permutation(
            isc_sim_vec_sugg,
            mvpa_sim_vec_pain,
            n_permute=n_perm_rsa,
            metric="spearman",
            how="upper",
            tail=2,
            return_perms=True,
        )

        rsa_rows.append(
            {
                "ROI": roi,
                "spearman_r": r,
                "p_values": round(p, 5),
                "Coordinates": tuple(np.round(coords[i], 0).astype(int)),
            }
        )

    results_dct[cond_sugg] = pd.DataFrame(rsa_rows).sort_values(
        by="spearman_r", ascending=False
    )
    print(
        "Max r, mean and fdr",
        results_dct[cond_sugg]["spearman_r"].max(),
        results_dct[cond_sugg]["spearman_r"].mean(),
        isc_utils.fdr(results_dct[cond_sugg]["p_values"].to_numpy()),
    )

save_to = os.path.join(
    save_path, f"IS-RSA_ISC-sugg-run_mvpa-pain-contrasts{n_perm_rsa}perm.pkl"
)
isc_utils.save_data(save_to, results_dct)
print(f"Saved RSA results to {save_to}")


# ANA_sugg_minus_N_ANA_sugg
# %%
# MVPA SUGG-PAIN for matched conditions (vs all shocks)

# results_dct = {}
# shock_cond = 'all_shock' # all shocks, to match with suggestion conditions

# for cond in tqdm(conditions):
#     print('Performing RSA on : ', cond)
#     # load maps
# %%
# VISUALIZE RSA
# from src import isc_utils
# from nilearn.glm.thresholding import threshold_stats_img

# rse_path = '/data/rainville/dSutterlin/projects/ISC_hypnotic_suggestions/results/imaging/RSA/mvpa_IS-RSA_sugg-pain_contrast-based/IS-RSA_mvpa_sugg_behav-sugg10000perm.pkl'
# res_path = '/data/rainville/dSutterlin/projects/ISC_hypnotic_suggestions/results/imaging/RSA/mvpa_IS-RSA_sugg-pain_contrast-based/IS-RSA_mvpa_sugg_mvpa-pain-matched-shocks10000perm.pkl'
# rsa_dict = isc_utils.load_pickle(res_path)


# views_mvpa = {}
# for cond in conditions:
#     print('cond', cond)
#     rsa_df = rsa_dict[cond].sort_index(ascending=True)

#     # === Prepare variables for projection ===
#     correlations = rsa_df['spearman_r'].values
#     p_values = rsa_df['p_values'].values
#     roi_labels = rsa_df['ROI'].values  # assumes label matches atlas
#     fdr_p = isc_utils.fdr(p_values, q=0.05)
#     print(f'FDR threshold: {fdr_p:.4f}')

#     title = f"Multivariate IS-RSA during {cond} (FDR<.05)"

#     # === Visualize with your existing function ===rsa_df
#     rsa_img, rsa_thresh, sig_labels, _= visu_utils.project_isc_to_brain_perm(
#         atlas_img=atlas,
#         isc_median=correlations,
#         atlas_labels=labels_roi_dct,
#         roi_coords = coords,
#         p_values=p_values,
#         p_threshold=fdr_p, #!!
#         title=title, #"RSA-ISC: Suggestion-Pain Similarity (FDR<.05)",
#         save_path=None,
#         show=True,
#         display_mode='x',
#         cut_coords_plot=None, #(-52, -40, 34),
#         color='Reds'
#     )

#     views_mvpa[cond] = plotting.view_img(rsa_img, threshold=rsa_thresh, title=f"RSA suggestion - pain similarity {cond}", colorbar=True,symmetric_cmap=False, cmap = 'Reds')


# # SUPP VISU MVPA PATTERNs that show sig RS
# #visu second level maps
# pain_cont_dict = isc_utils.load_pickle('/data/rainville/dSutterlin/projects/ISC_hypnotic_suggestions/results/imaging/GLM/model3_final-isc_23subjects_nuis_nodrift_31-03-25/second_level/ANA_shock_minus_N_ANA_shock_all_effects.pkl')
# sugg_cont_dict = isc_utils.load_pickle('/data/rainville/dSutterlin/projects/ISC_hypnotic_suggestions/results/imaging/GLM/model3_final-isc_23subjects_nuis_nodrift_31-03-25/second_level/ANA_sugg_minus_N_ANA_sugg_all_effects.pkl')

#     all_shock_maps = glob(os.path.join(model_res, 'first_level', shock_cond, 'firstlev_localizer_*.nii.gz'))
#     sugg_maps = glob(os.path.join(model_res,'first_level', cond, 'firstlev_localizer_*.nii.gz'))

#     def build_subject_dict(file_list):
#         """Build a dict: {subject_id: filepath} from a list of NIfTI file paths."""
#         subject_dict = {}
#         for path in file_list:
#             fname = os.path.basename(path)
#             subj_id = fname.split('_')[-1].replace('.nii.gz', '')  # expects '..._sub-01.nii.gz'
#             subject_dict[subj_id] = path
#         return subject_dict

#     sugg_dict = build_subject_dict(sugg_maps) # not sorted!!
#     shock_dict = build_subject_dict(all_shock_maps)
#     subjects = sorted(set(sugg_dict) & set(shock_dict))

#     # mvpa similarity
#     sugg_similarity_matrices_dct, sugg_vec_similarity_df = compute_inter_subject_mvpa_similarity(
#         sugg_dict, atlas_img=ATLAS, labels_roi_dct = labels_roi_dct
#     )
#     pain_similarity_matrices_dct, pain_vec_similarity_df = compute_inter_subject_mvpa_similarity(
#         shock_dict, atlas_img=ATLAS, labels_roi_dct = labels_roi_dct
#     )


#     # RSA computation + permrsa_df
#     rsa_rows = [] # build df
#     for roi_idx, roi in enumerate(sugg_vec_similarity_df.columns):

#         mvpa_sim_vec_sugg = sugg_vec_similarity_df[roi].values
#         mvpa_sim_vec_pain = pain_vec_similarity_df[roi].values
#         r, p, dist = isc_utils.matrix_permutation(mvpa_sim_vec_sugg, mvpa_sim_vec_pain, n_permute=n_perm_rsa, metric="spearman", how="upper", tail=1, return_perms=True)

#         rsa_rows.append({
#             'ROI': roi,
#             'spearman_r': r,
#             'p_values': round(p, 5),ANA_sugg_minus_N_ANA_sugg
#             'x': coords[roi_idx][0],
#             'y': coords[roi_idx][1],
#             'z': coords[roi_idx][2]
#         })

#     results_dct[cond] = pd.DataFrame(rsa_rows).sort_values(by='spearman_r', ascending=False)
#     print('Max r, mean and fdr', results_dct[cond]['spearman_r'].max(), results_dct[cond]['spearman_r'].mean(), isc_utils.fdr(results_dct[cond]['p_values'].to_numpy()))

# save_path = f'/data/rainville/dSutterlin/projects/ISC_hypnotic_suggestions/results/imaging/RSA/mvpa_IS-RSA_sugg-pain'
# os.makedirs(save_path, exist_ok=True)

# save_to = os.path.join(save_path, f'IS-RSA_mvpa_sugg_mvpa-pain{n_perm_rsa}perm.pkl')
# isc_utils.save_data(save_to, results_dct)
# print(f'Saved RSA results to {save_to}')


print("-----Done with MVPA IS-rsa!-----")

# # %%
# # VISUALIZE RSA
# from src import isc_utils
# from nilearn.glm.thresholding import threshold_stats_img

# rse_path = '/data/rainville/dSutterlin/projects/ISC_hypnotic_suggestions/results/imaging/RSA/mvpa_IS-RSA_sugg-pain_contrast-based/IS-RSA_mvpa_sugg_behav-sugg10000perm.pkl'
# res_path = '/data/rainville/dSutterlin/projects/ISC_hypnotic_suggestions/results/imaging/RSA/mvpa_IS-RSA_sugg-pain_contrast-based/IS-RSA_mvpa_sugg_mvpa-pain-matched-shocks10000perm.pkl'
# rsa_dict = isc_utils.load_pickle(res_path)


# views_mvpa = {}
# for cond in conditions:
#     print('cond', cond)
#     rsa_df = rsa_dict[cond].sort_index(ascending=True)

#     # === Prepare variables for projection ===
#     correlations = rsa_df['spearman_r'].values
#     p_values = rsa_df['p_values'].values
#     roi_labels = rsa_df['ROI'].values  # assumes label matches atlas
#     fdr_p = isc_utils.fdr(p_values, q=0.05)
#     print(f'FDR threshold: {fdr_p:.4f}')

#     title = f"Multivariate IS-RSA during {cond} (FDR<.05)"

#     # === Visualize with your existing function ===rsa_df
#     rsa_img, rsa_thresh, sig_labels, _= visu_utils.project_isc_to_brain_perm(
#         atlas_img=atlas,
#         isc_median=correlations,
#         atlas_labels=labels_roi_dct,
#         roi_coords = coords,
#         p_values=p_values,
#         p_threshold=fdr_p, #!!
#         title=title, #"RSA-ISC: Suggestion-Pain Similarity (FDR<.05)",
#         save_path=None,
#         show=True,
#         display_mode='x',
#         cut_coords_plot=None, #(-52, -40, 34),
#         color='Reds'
#     )

#     views_mvpa[cond] = plotting.view_img(rsa_img, threshold=rsa_thresh, title=f"RSA suggestion - pain similarity {cond}", colorbar=True,symmetric_cmap=False, cmap = 'Reds')


# # SUPP VISU MVPA PATTERNs that show sig RS
# #visu second level maps
# pain_cont_dict = isc_utils.load_pickle('/data/rainville/dSutterlin/projects/ISC_hypnotic_suggestions/results/imaging/GLM/model3_final-isc_23subjects_nuis_nodrift_31-03-25/second_level/ANA_shock_minus_N_ANA_shock_all_effects.pkl')
# sugg_cont_dict = isc_utils.load_pickle('/data/rainville/dSutterlin/projects/ISC_hypnotic_suggestions/results/imaging/GLM/model3_final-isc_23subjects_nuis_nodrift_31-03-25/second_level/ANA_sugg_minus_N_ANA_sugg_all_effects.pkl')

# plotting.view_img(sugg_cont_dict['z_score'], threshold=3, title='ANA suggestion - neutral suggestion', colorbar=True)
# plotting.view_img(pain_cont_dict['z_score'], threshold=3, title='ANA shock - neutral shock', colorbar=True)


# # %%
# #===========================
# # ISC - RSA!!
# #===========================
# reload(visu_utils)

# rsa_path = '/data/rainville/dSutterlin/projects/ISC_hypnotic_suggestions/results/imaging/RSA/2025-05-21_23'
# rsa_path = '/data/rainville/dSutterlin/projects/ISC_hypnotic_suggestions/results/imaging/RSA/tian2162025-05-22_23'
# # rsa_isc = isc_utils.load_pickle(os.path.join(save_path, f'rsa_isc_pain-behav_sugg-pain_{n_perm_rsa}perm.pkl'))
# # rsa_isc = isc_utils.load_pickle('/data/rainville/dSutterlin/projects/ISC_hypnotic_suggestions/results/imaging/RSA/rsa_pain-behav_sugg-pain_10000perm.pkl')
# rsa_isc_sugg = isc_utils.load_pickle(os.path.join(rsa_path, 'rsa_cosine-behav_isc-sugg5000perm.pkl' ))
# rsa_isc_pain = isc_utils.load_pickle(os.path.join(rsa_path, 'rsa_cosine-behav_isc-pain5000perm.pkl' ))

# #%%
# for key in rsa_isc_sugg.keys():

#     views={}
#     for cond in ['ANA']: #, 'HYPER', 'all_sugg', 'neutral']:

#         print('cond', cond)
#         rsa_df = rsa_isc_sugg[key][cond].sort_index(ascending=True) # to match the atlas labels

#         # === Prepare variables for projection ===
#         correlations = rsa_df['spearman_r'].values
#         p_values = rsa_df['p_values'].values
#         roi_labels = rsa_df['ROI'].values  # assumes label matches atlas
#         fdr_p = isc_utils.fdr(p_values, q=0.05)
#         print(f'FDR threshold: {fdr_p:.4f}')

#         title = f"ISC-RSA during {cond}"
#         # === Visualize with your existing function ===
#         rsa_img, rsa_thresh, sig_labels = visu_utils.project_isc_to_brain_perm(
#             atlas_img=atlas,
#             isc_median=correlations,
#             atlas_labels=id_labels_dct, #!!!!!!
#             roi_coords = coords,
#             p_values=p_values,
#             p_threshold=0.01, #!!
#             title=title, #"RSA-ISC: Suggestion-Pain Similarity (FDR<.05)",
#             save_path=None,
#             show=True,
#             display_mode='x',
#             cut_coords_plot=None, #(-52, -40, 34),
#             color='Reds'
#         )
#         if sig_labels.shape[0] > 0:
#             sig_labels.columns = ['Region', 'Spearman rho', 'p-value', 'Coordinatate (X,Y,Z)']

#         views[cond] = plotting.view_img(rsa_img, threshold=rsa_thresh, title=f"RSA suggestion - pain similarity {cond}", colorbar=True,symmetric_cmap=False, cmap = 'Reds')


# %%
