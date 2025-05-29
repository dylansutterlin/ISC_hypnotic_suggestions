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


print('current working dir : ', os.getcwd())
# %%
from src import preproc_utils, visu_utils, qc_utils
import src.glm_utils as utils

from sklearn.utils import Bunch
from importlib import reload
from nilearn import datasets,image
from datetime import datetime

# %% [markdown]
## load data
preproc_model_name = r'model2_23subjects_zscore_sample_detrend_25-02-25' #r'model2_3subjects_zscore_sample_detrend_25-02-25'
model_dir = rf'/data/rainville/dSutterlin/projects/ISC_hypnotic_suggestions/results/imaging/preproc_data/{preproc_model_name}'
model_name = "model3_23subjects_nuis_nodrift_{}".format(datetime.today().strftime("%d-%m-%y"))

model_name = 'model3_final-isc_23subjects_nuis_nodrift_31-03-25' #final model, reproduced Desmarteaux et al., 2019 !! 31 mars

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
single_img = image.index_img(ref_img, 0)
mask = utils.load_data_mask(ref_img)

mni_temp = datasets.load_mni152_template(resolution=1)
mni_bg = qc_utils.resamp_to_img_mask(mni_temp, mask)

save_glm = os.path.join(setup.project_dir, 'results', 'imaging', 'GLM', model_name)
os.makedirs(save_glm, exist_ok=True)
setup.save_dir = save_glm

# %%
#========================================
# Load 1st level maps for further analyses
#==========================================
from glob import glob as glob
from nilearn.maskers import NiftiLabelsMasker
from nilearn.image import binarize_img
from nilearn.plotting import view_img
from nilearn.datasets import fetch_atlas_schaefer_2018
from nilearn.plotting import find_parcellation_cut_coords
from src import qc_utils, isc_utils

model_res = r'/data/rainville/dSutterlin/projects/ISC_hypnotic_suggestions/results/imaging/GLM/model3_final-isc_23subjects_nuis_nodrift_31-03-25'
model_res = r'/data/rainville/dSutterlin/projects/ISC_hypnotic_suggestions/results/imaging/GLM/model3_final-isc_23subjects_nuis_nodrift_31-03-25'
project_dir = setup.project_dir
results_dir = setup['save_dir']


# LOAD ATLAS
#===========================================
#load Tian subcortical + combine with schaeffer 
tian_sub_cortical = nib.load(os.path.join(project_dir, 'masks/Tian2020_schaeffer200_subcortical16/Schaefer2018_200Parcels_17Networks_order_Tian_Subcortex_S1.dlabel.nii.gz'))
tian_data = tian_sub_cortical.get_fdata()

# Load the labels from a text file
labels_tian16 = os.path.join(project_dir, 'masks/Tian2020_schaeffer200_subcortical16/Schaefer2018_200Parcels_17Networks_order_Tian_Subcortex_S1_label.txt')
with open(labels_tian16, 'r') as f:
    tian16_labels = [line.strip() for line in f][::2][0:16] #!! getting only labels, only 16 !!


atlas_data = fetch_atlas_schaefer_2018(n_rois = 200, resolution_mm=2)
atlas = nib.load(atlas_data['maps'])
atlas_path = atlas_data['maps'] #os.path.join(project_dir,os.path.join(project_dir, 'masks', 'k50_2mm', '*.nii*'))
# labels_bytes = list(atlas_data['labels'])
full_labels = [str(label, 'utf-8') if isinstance(label, bytes) else str(label) for label in atlas_data['labels']]
roi_index = [full_labels.index(lbl)+1 for lbl in full_labels]
id_labels_dct = dict(zip(roi_index, full_labels))

#combined Tian + shaeffer
combined_data = atlas.get_fdata().copy()
combined_data[tian_data > 0] = tian_data[tian_data > 0] + 200  # Avoid index collision
combined_img = nib.Nifti1Image(combined_data, affine=atlas.affine, header=atlas.header)
nib.save(combined_img, os.path.join(project_dir, 'masks/Tian2020_schaeffer200_subcortical16/', 'combined_schaefer200_tian16.nii.gz'))

plotting.plot_roi(combined_img,bg_img=single_img, colorbar=True, display_mode = 'x', cut_coords=(-60,-50,0,50,60 ))

# final atlas variables
roi_index = np.unique(combined_data[combined_data > 0])
atlas_labels = full_labels + tian16_labels
labels_roi_dct = dict(zip(roi_index, atlas_labels))

ATLAS = combined_img
coords = find_parcellation_cut_coords(labels_img=ATLAS)

print(len(atlas_labels), len(roi_index), len(coords))
#%%
#==========================
# mULTIVARIATE BEHAVIORAL
#==========================
from sklearn.preprocessing import StandardScaler
from src import preproc_utils
import seaborn as sns
import statsmodels.api as sm

reload(preproc_utils)
xlsx_path = r'/data/rainville/dSutterlin/projects/ISC_hypnotic_suggestions/masks/Hypnosis_variables_20190114_pr_jc.xlsx'
subjects = list(setup['subjects'])
apm_subjects = ['APM' + subj[4:] for subj in subjects]
print(apm_subjects)

Y = preproc_utils.load_process_y_extended(xlsx_path, subjects)

sugg_cols = [
    "SHSS_score",
    "Chge_hypnotic_depth",
    "Mental_relax_absChange",
    "Abs_diff_automaticity"
]

pain_cols = ['raw_change_HYPER', 'raw_change_ANA']

scaler = StandardScaler()
Y_sugg = scaler.fit_transform(np.array(Y[sugg_cols].values, dtype=float))
Y_pain = scaler.fit_transform(np.array(Y[pain_cols].values, dtype=float))

plt.figure(figsize=(12, 6))
plt.imshow(Y_sugg)
plt.colorbar(label='scores')
plt.xticks(np.arange(len(sugg_cols)), sugg_cols, rotation=45, ha='right')
plt.yticks(np.arange(Y.shape[0]), Y.index)
plt.title('Hypnotic features response patterns')
plt.tight_layout()

plt.figure()
plt.imshow(Y_pain)
plt.colorbar(label='scores')
plt.xticks(np.arange(len(pain_cols)), pain_cols, rotation=45, ha='right')
plt.yticks(np.arange(Y.shape[0]), Y.index)
plt.title('Behavioral Features Heatmap')
plt.tight_layout()

# Compute pairwise cosine similarity using the previously defined function
cosine_sim_sugg = isc_utils.compute_behav_similarity(Y_sugg, metric='cosine', vectorize=False)
cosine_sim_pain = isc_utils.compute_behav_similarity(Y_pain, metric='cosine', vectorize=False)

cosine_vec_sugg = isc_utils.compute_behav_similarity(Y_sugg, metric='cosine', vectorize=True)
cosine_vec_pain = isc_utils.compute_behav_similarity(Y_pain, metric='cosine', vectorize=True)

# plot cosine_sim_sugg heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(
    cosine_sim_sugg,
    annot=False,
    fmt=".2f",
    cmap="coolwarm",
    vmin=-1,
    vmax=1,
    xticklabels=Y.index,
    yticklabels=Y.index,
    square=True,
    cbar_kws={'label': 'Cosine Similarity', 'shrink': 0.8}  # Adjust shrink to control size
)
cbar = plt.gca().collections[0].colorbar
cbar.ax.tick_params(labelsize=15)  # Increase the fontsize of the color bar
cbar.set_label('Cosine Similarity', fontsize=18)  # Increase the label size
plt.title('Pairwise subject Cosine Similarity of hypnotic pattern responses', fontsize=22)
plt.xticks(rotation=45, ha='right', fontsize=15)
plt.yticks(rotation=0, fontsize=15)
plt.tight_layout()
plt.show()
#------------
import scipy.cluster.hierarchy as sch

# compute and apply clustering order
link = sch.linkage(cosine_sim_sugg, method='average')
order = sch.dendrogram(link, no_plot=True)['leaves']
sim_ord = cosine_sim_sugg[np.ix_(order, order)]
labels_ord = Y.index[order]

# plot
plt.figure(figsize=(10,8))
sns.heatmap(sim_ord,
            cmap='coolwarm', vmin=-1, vmax=1,
            xticklabels=labels_ord, yticklabels=labels_ord,
            square=True, cbar_kws={'label':'Cosine Similarity','shrink':0.8})
cbar = plt.gca().collections[0].colorbar
cbar.ax.tick_params(labelsize=15)
cbar.set_label('Cosine Similarity', fontsize=18)
plt.title('Pairwise subject Cosine Similarity of hypnotic pattern responses', fontsize=22)
plt.xticks(rotation=45, ha='right', fontsize=15)
plt.yticks(rotation=0, fontsize=15)
plt.tight_layout()
plt.show()
#---------------------


#cov matrix Y_sugg
corr_matrix_sugg = np.corrcoef(Y_sugg, rowvar=False)
np.fill_diagonal(corr_matrix_sugg, 0)

plt.figure(figsize=(10, 8))
sns.heatmap(
    corr_matrix_sugg,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    vmin=-1,
    vmax=1,
    xticklabels=sugg_cols,
    yticklabels=sugg_cols,
    square=True,
    cbar_kws={'label': 'Correlation', 'shrink': 0.8}  # Adjust shrink to control size
)
cbar = plt.gca().collections[0].colorbar
cbar.ax.tick_params(labelsize=15)  # Increase the fontsize of the color bar
cbar.set_label('Correlation', fontsize=18)  # Increase the label size
plt.title('Correlation Matrix of hypnotic features', fontsize=22)
plt.xticks(rotation=45, ha='right', fontsize=15)
plt.yticks(rotation=0, fontsize=15)

# Remove text in diagonal elements
for i in range(len(sugg_cols)):
    plt.gca().texts[i * (len(sugg_cols) + 1)].set_text('')

plt.tight_layout()
plt.show()

VD = Y['total_chge_pain_hypAna'].apply(pd.to_numeric, errors='coerce')
X = pd.DataFrame(Y_sugg, index=Y.index).apply(pd.to_numeric, errors='coerce')
X = sm.add_constant(X)  # adds a column "const" for the intercept
X.columns = ['const'] + sugg_cols

model = sm.OLS(VD, X).fit()

print(model.summary())
coeffs = model.params
pvals  = model.pvalues
print("\nCoefficients:\n", coeffs)
print("\nP-values:\n", pvals)

# 6) And some key statistics:
print(f"\nR² = {model.rsquared:.3f}, adj. R² = {model.rsquared_adj:.3f}")
print(f"F-statistic = {model.fvalue:.2f}, p(F) = {model.f_pvalue:.3g}")

#%%
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
    atlas_data = atlas_img.get_fdata()
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
            continue

        subj_vectors = []

        for subj in subjects:
            vec = condition_data_dict[subj][roi_mask].flatten()

            if np.linalg.norm(vec) == 0:
                subj_vectors.append(np.full(np.sum(roi_mask), np.nan))
            else:
                subj_vectors.append(vec)

        subj_vectors = np.array(subj_vectors)
        valid_mask = ~np.isnan(subj_vectors).any(axis=1)
        valid_vectors = subj_vectors[valid_mask]
        valid_subjects = np.array(subjects)[valid_mask]

        if valid_vectors.shape[0] < 2:
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
    vectorized_df.columns.name = 'ROIs'
    vectorized_df.columns = roi_labels
    return similarity_matrices, vectorized_df

#%%
#===========================
# MAIN CODE
from tqdm import tqdm

mvpa_save_to = os.path.join(results_dir, 'mvpa_similarity')
os.makedirs(mvpa_save_to, exist_ok=True)

conditions = ['modulation_sugg', 'HYPER_sugg', 'ANA_sugg', 'neutral_sugg']
n_perm_rsa = 5000
results_dct = {}

#%%
for cond in tqdm(conditions):
    print('Performing RSA on : ', cond)
    # load maps 
    all_shock_maps = glob(os.path.join(model_res, 'all_shock', 'firstlev_localizer_*.nii.gz'))
    sugg_maps = glob(os.path.join(model_res,'first_level', cond, 'firstlev_localizer_*.nii.gz'))

    def build_subject_dict(file_list):
        """Build a dict: {subject_id: filepath} from a list of NIfTI file paths."""
        subject_dict = {}
        for path in file_list:
            fname = os.path.basename(path)
            subj_id = fname.split('_')[-1].replace('.nii.gz', '')  # expects '..._sub-01.nii.gz'
            subject_dict[subj_id] = path
        return subject_dict

    sugg_dict = build_subject_dict(sugg_maps) # not sorted!!
    shock_dict = build_subject_dict(all_shock_maps)
    subjects = sorted(set(sugg_dict) & set(shock_dict))

    # mvpa similarity
    similarity_matrices_dct, vec_similarity_df = compute_inter_subject_mvpa_similarity(
        sugg_dict, atlas_img=ATLAS, labels_roi_dct = labels_roi_dct
    )

    # RSA computation + perm
    rsa_rows = [] # build df 
    for roi_idx, roi in enumerate(vec_similarity_df.columns):

        mvpa_sim_vec = vec_similarity_df[roi].values
        r, p, dist = isc_utils.matrix_permutation(cosine_vec_sugg, mvpa_sim_vec, n_permute=n_perm_rsa, metric="spearman", how="upper", tail=1, return_perms=True)

        rsa_rows.append({
            'ROI': roi,
            'spearman_r': r,
            'p_values': round(p, 5),
            'x': coords[roi_idx][0],
            'y': coords[roi_idx][1],
            'z': coords[roi_idx][2]
        })

    results_dct[cond] = pd.DataFrame(rsa_rows).sort_values(by='spearman_r', ascending=False)
    print('Max r, mean and fdr', results_dct[cond]['spearman_r'].max(), results_dct[cond]['spearman_r'].mean(), isc_utils.fdr(results_dct[cond]['p_values'].to_numpy()))

save_path = f'/data/rainville/dSutterlin/projects/ISC_hypnotic_suggestions/results/imaging/RSA/mvpa_IS-RSA_sugg'
os.makedirs(save_path, exist_ok=True)

save_to = os.path.join(save_path, f'IS-RSA_mvpa_suggestion_tian216{n_perm_rsa}perm.pkl')
isc_utils.save_data(save_to, results_dct)
print(f'Saved RSA results to {save_to}')
print('-----Done with rsa!-----')

# %%
# VISUALIZE RSA
from src import isc_utils
res_path = r'/data/rainville/dSutterlin/projects/ISC_hypnotic_suggestions/results/imaging/RSA/mvpa_IS-RSA_sugg/IS-RSA_mvpa_suggestion_tian2165000perm.pkl'

rsa_dict = isc_utils.load_pickle(res_path)

views_mvpa = {}
for cond in conditions:
    print('cond', cond)
    rsa_df = rsa_dict[cond].sort_index(ascending=True) 

    # === Prepare variables for projection ===
    correlations = rsa_df['spearman_r'].values
    p_values = rsa_df['p_values'].values
    roi_labels = rsa_df['ROI'].values  # assumes label matches atlas
    fdr_p = isc_utils.fdr(p_values, q=0.05)
    print(f'FDR threshold: {fdr_p:.4f}')

    title = f"Multivariate IS-RSA during {cond}"

    # === Visualize with your existing function ===
    rsa_img, rsa_thresh, sig_labels = visu_utils.project_isc_to_brain_perm(
        atlas_img=atlas,
        isc_median=correlations,
        atlas_labels=labels_roi_dct,
        roi_coords = coords,
        p_values=p_values,
        p_threshold=fdr_p, #!!
        title=title, #"RSA-ISC: Suggestion-Pain Similarity (FDR<.05)",
        save_path=None,
        show=True,
        display_mode='x',
        cut_coords_plot=None, #(-52, -40, 34),
        color='Reds'
    )

    views_mvpa[cond] = plotting.view_img(rsa_img, threshold=rsa_thresh, title=f"RSA suggestion - pain similarity {cond}", colorbar=True,symmetric_cmap=False, cmap = 'Reds')



# %%
#===========================
# ISC - RSA!!
#===========================
reload(visu_utils)

rsa_path = '/data/rainville/dSutterlin/projects/ISC_hypnotic_suggestions/results/imaging/RSA/2025-05-21_23'
rsa_path = '/data/rainville/dSutterlin/projects/ISC_hypnotic_suggestions/results/imaging/RSA/tian2162025-05-22_23'
# rsa_isc = isc_utils.load_pickle(os.path.join(save_path, f'rsa_isc_pain-behav_sugg-pain_{n_perm_rsa}perm.pkl'))
# rsa_isc = isc_utils.load_pickle('/data/rainville/dSutterlin/projects/ISC_hypnotic_suggestions/results/imaging/RSA/rsa_pain-behav_sugg-pain_10000perm.pkl')
rsa_isc_sugg = isc_utils.load_pickle(os.path.join(rsa_path, 'rsa_cosine-behav_isc-sugg5000perm.pkl' ))
rsa_isc_pain = isc_utils.load_pickle(os.path.join(rsa_path, 'rsa_cosine-behav_isc-pain5000perm.pkl' ))

#%%
for key in rsa_isc_sugg.keys():

    views={}
    for cond in ['ANA']: #, 'HYPER', 'all_sugg', 'neutral']:

        print('cond', cond)
        rsa_df = rsa_isc_sugg[key][cond].sort_index(ascending=True) # to match the atlas labels

        # === Prepare variables for projection ===
        correlations = rsa_df['spearman_r'].values
        p_values = rsa_df['p_values'].values
        roi_labels = rsa_df['ROI'].values  # assumes label matches atlas
        fdr_p = isc_utils.fdr(p_values, q=0.05)
        print(f'FDR threshold: {fdr_p:.4f}')

        title = f"ISC-RSA during {cond}"
        # === Visualize with your existing function ===
        rsa_img, rsa_thresh, sig_labels = visu_utils.project_isc_to_brain_perm(
            atlas_img=atlas,
            isc_median=correlations,
            atlas_labels=id_labels_dct, #!!!!!!
            roi_coords = coords,
            p_values=p_values,
            p_threshold=0.01, #!!
            title=title, #"RSA-ISC: Suggestion-Pain Similarity (FDR<.05)",
            save_path=None,
            show=True,
            display_mode='x',
            cut_coords_plot=None, #(-52, -40, 34),
            color='Reds'
        )
        if sig_labels.shape[0] > 0:
            sig_labels.columns = ['Region', 'Spearman rho', 'p-value', 'Coordinatate (X,Y,Z)']

        views[cond] = plotting.view_img(rsa_img, threshold=rsa_thresh, title=f"RSA suggestion - pain similarity {cond}", colorbar=True,symmetric_cmap=False, cmap = 'Reds')

    
# %%
