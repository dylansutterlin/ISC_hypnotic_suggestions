
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

from glob import glob as glob
from nilearn.maskers import NiftiLabelsMasker
from nilearn.image import binarize_img
from nilearn.plotting import view_img
from src import qc_utils, isc_utils

print("current working dir : ", os.getcwd())
# %% [markdown]
## load data
preproc_model_name = r'model2_23subjects_zscore_sample_detrend_25-02-25' #r'model2_3subjects_zscore_sample_detrend_25-02-25'
model_dir = rf"/data/rainville/dSutterlin/projects/ISC_hypnotic_suggestions/results/imaging/preproc_data/{preproc_model_name}"

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

# Load main model for GLM results
MODEL_NAME = "model3_final-isc_23subjects_nuis_nodrift_31-03-25"  # final model, reproduced Desmarteaux et al., 2019 !! 31 mars
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
#%%
# BEHAVIORAL DATA
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
#%%
# Similarity matrices for behavioral data
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


#%%
# Load IS-RSA results for ROIs
rsa_model_path = f"/data/rainville/dSutterlin/projects/ISC_hypnotic_suggestions/results/imaging/RSA/IS-RSA-mvpa-dot-stat_contrast-based_tian216_reproduced" 

rsa_tables_mvpa_fdr = {}
rsa_tables_mvpa_fdr['annak_ANA_shock_minus_N_ANA_shock'] = pd.read_csv(
    os.path.join(rsa_model_path, "shock_contrast_pain_diff_Ana/tables", "RSA_MVPA_annak_ANA_shock_minus_N_ANA_shock_table_FDR0.0002.csv")
) # only annalk sig.
rsa_tables_mvpa_fdr['euclidean_HYPER_shock_minus_N_HYPER_shock'] = pd.read_csv(
    os.path.join(rsa_model_path, "shock_contrast_pain_diff_Hyper/tables", "RSA_MVPA_euclidean_HYPER_shock_minus_N_HYPER_shock_table_FDR0.0006.csv")
)
rsa_tables_mvpa_fdr['annak_HYPER_shock_minus_N_HYPER_shock'] = pd.read_csv(
    os.path.join(rsa_model_path, "shock_contrast_pain_diff_Hyper/tables", "RSA_MVPA_annak_HYPER_shock_minus_N_HYPER_shock_table_FDR0.0006.csv")
)

#%%
#========================================
# Visualize second-level contrast for pain
from nilearn.glm.thresholding import threshold_stats_img

seond_lev_pain_conditions = ["ANA_shock_minus_N_ANA_shock",
                              "HYPER_shock_minus_N_HYPER_shock",
                              "neutral_shock"] 

condition_names = {
    "ANA_sugg": "Analgesia",
    "N_ANA_sugg": "Neutral (Ana.)",
    "HYPER_sugg": "Hyperalgesia",
    "N_HYPER_sugg": "Neutral (Hyper.)",
}
# test = "/data/rainville/dSutterlin/projects/ISC_hypnotic_suggestions/results/imaging/GLM/model3_final-isc_23subjects_nuis_nodrift_31-03-25/second_level/ANA_shock_minus_N_ANA_shock_all_effects.pkl"
second_lev_dir = os.path.join(results_dir, "second_level")
t_imgs = {}
for cond in seond_lev_pain_conditions:

    pkl = isc_utils.load_pickle(os.path.join(second_lev_dir, cond + "_all_effects.pkl"))
    t_img = pkl["z_score"]  # or 'effect_size' for cohen d
    img = nib.load(os.path.join(second_lev_dir, "group_" + cond + ".nii.gz"))
    t_imgs[cond] = t_img

hypo_stat_img = t_imgs['ANA_shock_minus_N_ANA_shock']
hyper_stat_img = t_imgs['HYPER_shock_minus_N_HYPER_shock']

corrected_hypo_modulation, thresh_hypo = threshold_stats_img(hypo_stat_img, alpha=0.05, height_control='fdr')
corrected_hyper_modulation, thresh_hyper = threshold_stats_img(hyper_stat_img, alpha=0.05, height_control='fdr')

view_img(corrected_hypo_modulation)
#%%
from nilearn.image import new_img_like

# Create a single region MVPA based on atlas
# Used e.g. to plot MV pattern in only one region
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

# all_ana_sig_rois = list(rsa_tables_mvpa_fdr['euclidean_ANA_shock_minus_N_ANA_shock']['ROI'].values) + list(
#     rsa_tables_mvpa_fdr['annak_ANA_shock_minus_N_ANA_shock']['ROI'].values)
all_ana_sig_rois = list(rsa_tables_mvpa_fdr['annak_ANA_shock_minus_N_ANA_shock']['ROI'].values)
ana_sig_roi_map = combine_roi_maps(roi_pattern_dct_ana, all_ana_sig_rois, t_imgs['ANA_shock_minus_N_ANA_shock'])

if 'HYPER' in rsa_tables_mvpa_fdr.keys():
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

SAVE_TO = os.path.join(save_glm_adhoc, 'ANA_shock_minus_N_ANA_shock_second_level_MVpatterns')
os.makedirs(SAVE_TO, exist_ok=True)
IMG = ana_sig_roi_map
ana_sig_roi_map.to_filename(os.path.join(SAVE_TO, 'ANA_shock_minus_N_ANA_shock_second_level_MVpatterns.nii.gz'))
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

SAVE_TO = os.path.join(save_glm_adhoc, 'HYPER_shock_minus_N_HYPER_shock_second_level_MVpatterns')
os.makedirs(SAVE_TO, exist_ok=True)
IMG = hyper_sig_roi_map
hyper_sig_roi_map.to_filename(os.path.join(SAVE_TO, 'HYPER_shock_minus_N_HYPER_shock_second_level_MVpatterns.nii.gz'))


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
# PUBLICATION: FIGURE 6. Pairwise MVPA similarity in ROI for 1ST LEVEL contrasts
#==================================================
from src.mvpa_rsa_utils import compute_inter_subject_mvpa_similarity, build_subject_dict
# Load similarity matrices for further visualization 
NEURAL_SIM = 'cosine'
n_perm_rsa = 10000
effect_type = 'stat' # t maps
save_path = f"/data/rainville/dSutterlin/projects/ISC_hypnotic_suggestions/results/imaging/RSA/IS-RSA-mvpa-{NEURAL_SIM}-{effect_type}_contrast-based_tian216_reproduced" #_{ANNAK_VERSION}-annak"

ana_name = 'pain_diff_Ana'
similarity_matrices_dct_ana = {}
similarity_matrices_dct_ana[NEURAL_SIM] = isc_utils.load_pickle(os.path.join(
    save_path, f"similarity_mat_IS-RSA-mvpa_pain-{NEURAL_SIM}_behav-{ana_name}_{n_perm_rsa}perm.pkl"
))
hyper_name = 'pain_diff_Hyper'
similarity_matrices_dct_hyper = {}
similarity_matrices_dct_hyper[NEURAL_SIM] = isc_utils.load_pickle(os.path.join(
    save_path, f"similarity_mat_IS-RSA-mvpa_pain-{NEURAL_SIM}_behav-{hyper_name}_{n_perm_rsa}perm.pkl"
))  

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
# Key ROI: PHG (100) and AMCC (156)
# PHG comes out in Ana while AMCC in Hyper
# get similarity matrices per roi to find pairs of subject to plot MV patterns
phc_roi = id_labels_dct[100]
amcc_roi = id_labels_dct[156]

sim_mat_phc = similarity_matrices_dct_ana['cosine']['annak']['ANA_shock_minus_N_ANA_shock'][100]
sim_mat_amcc = similarity_matrices_dct_hyper['cosine']['annak']['HYPER_shock_minus_N_HYPER_shock'][156]

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
phg_low_low_subjects = ('sub-36','sub-16')

amcc_high_high_subjects = ('sub-20','sub-29') #('sub-28','sub-42')
amcc_low_low_subjects = ('sub-26','sub-37')
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

def plot_subject_dyad_heatmaps(data1, data2, cmap="coolwarm", title1='Subject A', title2='Subject B'):
    """Plot a 1x2 heatmap for two subjects' masked ROI slices (e.g. PHG), with colorbar."""
    
    # Set black background where values are 0
    data1_masked = np.ma.masked_where(data1 == 0, data1)
    data2_masked = np.ma.masked_where(data2 == 0, data2)
    
    vlim = np.max(np.abs([data1_masked, data2_masked]))

    fig, axs = plt.subplots(1, 2, figsize=(10, 5))

    for ax, data, title in zip(axs, [data1_masked, data2_masked], [title1, title2]):
        im = ax.imshow(data, cmap=cmap, vmin=-vlim, vmax=vlim)
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
phg_cmap='coolwarm'
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
    title1='High Sim – Subject 18', title2='High Sim – Subject 19',
    cmap=phg_cmap)
plot_subject_dyad_heatmaps(phg_slices[2], phg_slices[3],
    title1='Low Sim – Subject 01', title2='Low Sim – Subject 02',
    cmap=phg_cmap)

#%%
cmap = "RdBu_r"
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
    figsize=(7, 2.1),
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

# best_slice_idx_amcc_low = find_most_similar_slice(
#     amcc_maps_low_low[0], amcc_maps_low_low[1], axis=AXIS_CUT, similarity=sim
# )
amcc_slices_low = [
    extract_consistent_slice(img_dct, best_sim_slice_idx_amcc_high, axis=AXIS_CUT)
    for img_dct in list(amcc_maps_low_low)
]


plot_subject_dyad_heatmaps(amcc_slices_high[0], amcc_slices_high[1],
                           title1='High Similarity – Subject 18',
                           title2='High Similarity – Subject 19')

plot_subject_dyad_heatmaps(amcc_slices_low[0], amcc_slices_low[1],
                           title1='Low Similarity – Subject 01',
                           title2='Low Similarity – Subject 02')
tick_labels = [np.round(np.min(amcc_slices_low),2), "", np.round(-np.min(amcc_slices_low),2)]
visu_utils.plot_isc_colorbar(
    vmin=-1,
    vmax=1,
    cmap=cmap,
    figsize=(7,2.1),
    label=None,
    tick_fontsize=tick_fontsize,
    tick_labels=tick_labels,
    horizontal=True,
)

#%%
# PUBLICATION: CORREL ~ mean beta weight in PHG vs SHSS score and pain modulation
reload(visu_utils)
color_conds = {"Ana": "#004197", "Hyper": "#df265e", "Neutral_hypo": "#a1a0a0","Neutral_hyper" : "#a1a0a0", "SHSS": "#76afcf"}
color_conds['Ana_sim'] = "#004197"
color_conds['PHG_pattern'] = "#7a9aab"
color_conds['PHG_pattern_sim'] = "#7a9aab"
color_conds['amcc_pattern'] = "#c86885"
phg_id = 100

# extract non 0 values patterns and get mean
mean_phg_weight = {}
for sub in extracted_mv_pattern_per_roi_ana.keys():
    img = extracted_mv_pattern_per_roi_ana[sub][phg_id]
    data = img.get_fdata()
    mean_val = np.mean(data[data != 0])
    mean_phg_weight[sub] = mean_val

# plot scatter
visu_utils.jointplot_brain_correl(y = -np.array(behav_interest['pain_diff_Ana'].ravel()),
           x = np.array(list(mean_phg_weight.values())).ravel(),
              y_label='Δ Hypoalgesia',
                x_label='Mean PHG t-value',
                density_color_y=color_conds['Ana'],
                density_color_x=color_conds['PHG_pattern'],
                plot_w = 12,
                plot_h = 10,
                text_legend="Subject",
           )

#%% PUBLICATION SCATTER SIMILARITY phg ~ pain diff sim
# plot scatter for similarity pair on pain ratings and similarity in region
reload(visu_utils)
ana_annak = annak_sim_matrices['pain_diff_Ana']
phg_simat = np.array(similarity_matrices_dct_ana['cosine']['annak']['ANA_shock_minus_N_ANA_shock'][phg_id])
np.fill_diagonal(phg_simat, np.nan)  # remove self-similarity

is_rsa_phg = 'IS-RSA = 0.37***'

reload(visu_utils)
visu_utils.jointplot_brain_correl(y=ana_annak, x=phg_simat,
                     input_correl_text = is_rsa_phg,
                     plot_w = 12,
                     plot_h = 10,
                     y_label = 'Δ Hypoalgesia sim.',
                     x_label = 'PHG pattern sim.',
                    density_color_y = color_conds['Ana'],
                    density_color_x = color_conds['PHG_pattern'],
                    text_legend="Pair"

                     )
#%%
# Plot AMCC
amcc_id = 156
mean_amcc_weight = {}
for sub in extracted_mv_pattern_per_roi_hyper.keys():
    img = extracted_mv_pattern_per_roi_hyper[sub][amcc_id]
    data = img.get_fdata()
    mean_val = np.mean(data[data != 0])
    mean_amcc_weight[sub] = mean_val

# plot scatter
visu_utils.jointplot_brain_correl(y = np.array(behav_interest['pain_diff_Hyper'].ravel()),
           x = np.array(list(mean_amcc_weight.values())).ravel(),
              y_label='Δ Hyperalgesia',
                x_label='Mean AMCC t-value',
                density_color_y=color_conds['Hyper'],
                density_color_x=color_conds['amcc_pattern'],
                plot_w = 12,
                plot_h = 10,
                text_legend="Subject",
           )

# amcc_sim = np.array(similarity_matrices_dct_hyper['cosine']['annak']['HYPER_shock_minus_N_HYPER_shock'][amcc_id])
# np.fill_diagonal(amcc_sim, np.nan)  # remove self-similarity
# visu_utils.jointplot_brain_correl(y=NN_sim_matrices['pain_diff_Hyper'],
#                                   x=amcc_sim,
#                      input_correl_text = 'rho = 0.29**',
#                      plot_w = 12,
#                      plot_h = 12,
#                      y_label = 'Δ Hyperalgesia sim.',
#                      x_label = 'AMCC pattern sim.',
#                     density_color_x = color_conds['Hyper'],
#                     density_color_y = color_conds['PHG_pattern']
# )


#%%
reload(visu_utils)

# plot ROI on surface plot
sugg_rois = [76, 187,189]
pain_rois = [100]
masked_region_sugg, _plot = visu_utils.plot_many_roi(
    atlas, sugg_rois, id_labels_dct, title=None
)  # 30 = Default_Temp_3 LH
_plot

masked_region_pain, _plot = visu_utils.plot_many_roi(
    atlas, pain_rois, id_labels_dct, title=None
)  # 30 = Default_Temp_3 LH
_plot
#%%
# FIGURE 6 PUBLICATION

cmaps_roi = []
for i, roi_img in enumerate([masked_region_sugg, masked_region_pain]):
    # if i >= 1:
    #     break  # Stop after the first iteration
    if i == 0: title = 'aSTG & pSTG'
    elif i == 1: title = 'Left PHG'
    print()
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.set_axis_off()

    title = title
    display = plot_glass_brain(
        roi_img,
        threshold=isc_thresh,
        colorbar=False,
        display_mode="lr",
        plot_abs=False,
        cmap="Greens",
        title=None,
        colorbar=True,
        axes=ax,
        vmax=None,
    )
    y = 0.20
    ax.text(0.1, y, "L", transform=ax.transAxes, fontsize=20, ha="center")
    ax.text(0.9, y, "R", transform=ax.transAxes, fontsize=20, ha="center")

    # Add title
    fig.suptitle(title, fontsize=30, y=0.85)
    # fig.savefig(
    #     os.path.join(save_plots_1sample, f"glass_brain_{cond}_FDR05.png"), dpi=1000
    # )
cbar

# plotting.plot_glass_brain(masked_region, title="Sugg ROIs", cmap='Greens', vmin=-1, vmax=1)  

# plotting.plot_glass_brain(masked_region_pain, title="Pain ROIs", cmap='Reds', vmin=-1, vmax=1)

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
