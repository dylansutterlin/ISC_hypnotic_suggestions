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
ANNAK_VERSION = 'mean'
annak_sim_matrices = {}
annak_sim_matrices['SHSS_score'] = isc_utils.compute_behav_similarity(y, metric="annak", annak_version=ANNAK_VERSION, vectorize=False)
annak_sim_matrices['total_change_pain'] = isc_utils.compute_behav_similarity(behav_interest['total_change_pain'], metric="annak", annak_version=ANNAK_VERSION, vectorize=False)
# annak_sim_matrices['resid_pain_diff_Ana'] = isc_utils.compute_behav_similarity(behav_interest['resid_pain_diff_Ana'], metric="annak", vectorize=False)
# annak_sim_matrices['resid_pain_diff_Hyper'] = isc_utils.compute_behav_similarity(behav_interest['resid_pain_diff_Hyper'], metric="annak", vectorize=False)
annak_sim_matrices['pain_diff_Ana'] = isc_utils.compute_behav_similarity(behav_interest['pain_diff_Ana'], metric="annak", annak_version=ANNAK_VERSION, vectorize=False)
annak_sim_matrices['pain_diff_Hyper'] = isc_utils.compute_behav_similarity(behav_interest['pain_diff_Hyper'], metric="annak", annak_version=ANNAK_VERSION, vectorize=False)
# # annak_sim_matrices['Ana_diff_global_neutral'] = isc_utils.compute_behav_similarity(behav_interest['Ana_diff_global_neutral'], metric="annak", vectorize=False)

NN_sim_matrices = {}
NN_sim_matrices['SHSS_score'] = isc_utils.compute_behav_similarity(y, metric="euclidean", vectorize=False)
# NN_sim_matrices['total_change_pain'] = isc_utils.compute_behav_similarity(behav_interest['total_change_pain'], metric="euclidean", vectorize=False)
NN_sim_matrices['pain_diff_Ana'] = isc_utils.compute_behav_similarity(behav_interest['pain_diff_Ana'], metric="euclidean", vectorize=False)
NN_sim_matrices['pain_diff_Hyper'] = isc_utils.compute_behav_similarity(behav_interest['pain_diff_Hyper'], metric="euclidean", vectorize=False)

# %%
# PUBLICATION
# FINAL MODEL Pain-specific ~ PAIN CONTRAST
# IS-RSA SHSS ~ ISC-PAIN
from src.mvpa_rsa_utils import compute_inter_subject_mvpa_similarity, build_subject_dict

for NEURAL_SIM in ['dot']: #, 'dot']: #, 'cosine']: #, 'dot']: #['euclidean', 'dot']: #'cosine', 'pearson']:
    specific_re_run = True
    # NEURAL_SIM = 'euclidean'  # 'cosine' or 'dot' or 'euclidean'
    # BEHAV_ID = "SHSS"  # "SHSS" or "chge-pain"
    effect_type = 'stat' #'effect_size' #'z_score'
    save_path = f"/data/rainville/dSutterlin/projects/ISC_hypnotic_suggestions/results/imaging/RSA/IS-RSA-mvpa-{NEURAL_SIM}-{effect_type}_contrast-based_tian216_reproduced" #_{ANNAK_VERSION}-annak"
    os.makedirs(save_path, exist_ok=True)
    ONLY_ON = [list(annak_sim_matrices.keys())][0]

    # BEHAV_ID = "pain_diff_Ana"  # "SHSS" or "chge-pain" or "pain_diff_Ana" or "pain_diff_Hyper" or "total_change_pain"
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
                    print(pd.DataFrame(rsa_rows).sort_values(by='p_values').head(10))

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

#%%
compute_median_test = True
# if compute_median_test:
from sklearn.preprocessing import MinMaxScaler
from brainiak.isc import bootstrap_isc
reload(isc_utils)
import warnings
issc_dfs = {}
issc_matrices = {}
issc_values = {}
issc_p_cond = {}
n_boot=10000

save_1sample_sim = os.path.join(save_path, 'one_sample_sim')
os.makedirs(save_1sample_sim, exist_ok=True)

#reformat sim matrices in columns (n_pairs x parcels)
for cond in pain_conditions:

    regions_dct = similarity_matrices_dct['euclidean'][cond] #same for both sim models
    column_dict = {}

    for region in tqdm(regions_dct.keys()):
        sim_mat = np.array(regions_dct[region])
        np.fill_diagonal(sim_mat, 0)
        vec_sim = isc_utils.isc_matrix_to_vector(sim_mat).T
        vec_sim = MinMaxScaler(feature_range=(-1, 1)).fit_transform(vec_sim.reshape(-1,1)).ravel()
        column_dict[region] = vec_sim

    issc_df = pd.DataFrame(column_dict)
    issc_dfs[cond] = issc_df
    
    observed, ci, p_values, distribution = bootstrap_isc(np.array(issc_df), summary_statistic='median', pairwise=True, n_bootstraps=n_boot, ci_percentile=95,side='right', random_state=None)

    #project sign median to brain
    issc_matrices[cond] = {
            region: isc_utils.vector_to_isc_matrix(issc_df.iloc[:, col_j], diag=1)
            for col_j, region in enumerate(issc_df.columns)
        }

    issc_values[cond] = observed
    issc_p_cond[cond] = p_values

    print(
        f"min p value: {np.min(p_values):.6f}, and max ISC : {np.max(observed):.6f}"
    )
    print(
        f"Within condition FDR thresh : {isc_utils.fdr(p_values, q=0.05):.6f}, and Bonferroni : {isc_utils.bonferroni(p_values, alpha=0.05):.6f}"
    )
    isc_utils.save_data(os.path.join(save_1sample_sim, f"issc_mvpa_pain_{cond}_{n_boot}boot.pkl"), (observed, ci, p_values, distribution))
