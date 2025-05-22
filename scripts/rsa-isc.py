# %% RSA between suggestion and pain ISC structures
import os
from datetime import datetime
import numpy as np
import pandas as pd

from nilearn.datasets import fetch_atlas_schaefer_2018
from nilearn.maskers import NiftiLabelsMasker
from nilearn.plotting import find_parcellation_cut_coords

import nibabel as nib
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr
from src import isc_utils

import src.isc_utils as isc_utils
import src.visu_utils as visu_utils
from importlib import reload
reload(visu_utils)
reload(isc_utils)
#%%
# Config
model_names = {
    'model1_sugg': 'model1_sugg_23-sub_schafer-200-2mm_mask-whole-brain_pairWise-True_preproc_reg-mvmnt-True-8',
    'model1-6': 'model1_sugg_23-sub_schafer-200-2mm_mask-whole-brain_pairWise-True_preproc_reg-mvmnt-True-6',
    'model2_sugg': 'model2_sugg_23-sub_schafer-200-2mm_mask-whole-brain_pairWise-False_preproc_reg-mvmnt-True-8',
    'model2_shock_loo': 'model2_shock_23-sub_schafer-200-2mm_mask-whole-brain_pairWise-False_preproc_reg-mvmnt-True-8',
    'model3-shock': 'model3_shock_23-sub_schafer-200-2mm_mask-whole-brain_pairWise-True_preproc_reg-mvmnt-True-8',
    'model1_mean': 'model4-mean_sugg_23-sub_schafer-200-2mm_mask-whole-brain_pairWise-True_preproc_reg-mvmnt-True-8',
    'model5_isfc_sugg' :'model5-isfc_sugg_23-sub_schafer-200-2mm_mask-lanA800_pairWise-True_preproc_reg-mvmnt-True-8',
    'model5_isfc_shock' : 'model5-isfc_shock_23-sub_schafer-200-2mm_mask-lanA800_pairWise-True_preproc_reg-mvmnt-True-8',
    '9 avril ...' : ' ',
    'single_trial' : 'model_single-trial_sugg_23-sub_schafer-200-2mm_mask-lanA800_pairWise-True_preproc_reg-mvmnt-True-8',
    'single_trial_wb' : 'model_single-trial-wb_sugg_23-sub_schafer-200-2mm_mask-whole-brain_pairWise-True_preproc_reg-mvmnt-True-8',
    'model5_sugg_tian' : 'model5-with-subcort_sugg_23-sub_schafer-200-2mm_mask-whole-brain_pairWise-True_preproc_reg-mvmnt-True-8'
}

model_sugg = model_names['model1_sugg']
model_pain = model_names['model3-shock']

project_dir = "/data/rainville/dSutterlin/projects/ISC_hypnotic_suggestions"
results_dir = os.path.join(project_dir, f'results/imaging/ISC/{model_sugg}')
setup = isc_utils.load_json(os.path.join(results_dir, "setup_parameters.json"))
subjects = setup['subjects']

DATE = datetime.now().strftime("%Y-%m-%d_%H")
save_path = f'/data/rainville/dSutterlin/projects/ISC_hypnotic_suggestions/results/imaging/RSA/{DATE}'
os.makedirs(save_path, exist_ok=True)

conditions = ['ANA', 'HYPER', 'all_sugg', 'neutral']
sim_model = 'euclidean'
n_perm = setup['n_perm'] # to load isc results
n_perm_rsa = 5000
n_tail_rsa = 1 # hypothesize that only pos. associations are of interest

#%%
#================================
# BEHAHVIORAL DATA
#================================
import seaborn as sns

xlsx_path = r'/data/rainville/dSutterlin/projects/ISC_hypnotic_suggestions/masks/Hypnosis_variables_20190114_pr_jc.xlsx'
subjects = list(setup['subjects'])
apm_subjects = ['APM' + subj[4:] for subj in subjects]
print(apm_subjects)

# def load_process_y(xlsx_path, subjects):
'''Load behavioral variables from xlsx file and process them for further analysis
'''

# dependant variables
original_y = rawY = pd.read_excel(xlsx_path, sheet_name=0, index_col=1, header=2)
rawY = pd.read_excel(xlsx_path, sheet_name=0, index_col=1, header=2).iloc[
    2:, [4,5,6,7,8,9,10,11,12, 17, 18, 19, 38, 48, 65, 67]
]

columns_of_interest = [
    "SHSS_score",
    "VAS_Nana_Int",
    "VAS_Ana_Int",
    "VAS_Nhyper_Int",
    "VAS_Hyper_Int",
    "VAS_Nana_UnP",
    "VAS_Ana_UnP",
    "VAS_Nhyper_UnP",
    "VAS_Hyper_UnP",
    "raw_change_ANA",
    "raw_change_HYPER",
    "total_chge_pain_hypAna",
    "Chge_hypnotic_depth",
    "Mental_relax_absChange",
    "Automaticity_post_ind",
    "Abs_diff_automaticity"]

rawY.columns = columns_of_interest
cleanY = rawY.iloc[:-6, :]  # remove sub04, sub34 and last 6 rows
cutY = cleanY.drop(["APM04*", "APM34*"])

filledY = cutY.fillna(cutY.astype(float).mean()).astype(float)
filledY["SHSS_groups"] = pd.cut(
    filledY["SHSS_score"], bins=[0, 4, 8, 12], labels=["0", "1", "2"]
)  # encode 3 groups for SHSS scores

# bin_edges = np.linspace(min(data_column), max(data_column), 4) # 4 bins
filledY["auto_groups"] = pd.cut(
    filledY["Abs_diff_automaticity"],
    bins=np.linspace(
        min(filledY["Abs_diff_automaticity"]) - 1e-10,
        max(filledY["Abs_diff_automaticity"]) + 1e-10,
        4,
    ),
    labels=["0", "1", "2"],
)

# rename 'APM_XX_HH' to 'APMXX' format, for compatibility with Y.rows
subjects_rewritten = ["APM" + s.split("-")[1] for s in subjects]

# reorder to match subjects order
Y = pd.DataFrame(columns=filledY.columns)
for namei in subjects_rewritten: 
    row = filledY.loc[namei]
    Y.loc[namei] = row


corr_matrix = Y.corr()

plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', square=True,
            linewidths=0.5, cbar_kws={'label': 'Pearson r'})

plt.title('Correlation Matrix of Behavioral Variables', fontsize=14)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

#%%

y_conditions = ['SHSS_score', 'raw_change_ANA', 'raw_change_HYPER', 'total_chge_pain_hypAna']

def plot_simmat(simmat, sim_method = 'Eucledian', y_name = 'SHSS_score'):
    plt.figure(figsize=(10, 8))
    plt.imshow(simmat, vmin=0, vmax=1, cmap='coolwarm')
    plt.colorbar(label='Similarity')
    plt.title(f'{sim_method} {y_name} similarity matrix')
    plt.xlabel('Subject rank')
    plt.ylabel('Subject rank')
    plt.show()

for metric in ['euclidean', 'annak']:
    for y_name in ['SHSS_score']:

        y = Y[y_name].values
        y = (y - np.mean(y)) / np.std(y)
        sim_behav = isc_utils.compute_behav_similarity(y, metric=metric, vectorize=False)

        ranks = np.argsort(y)         # gives indices that sort behavior low → high
        sim_ranked = sim_behav[np.ix_(ranks, ranks)]

        plot_simmat(sim_ranked, sim_method = 'Eucledian', y_name = y_name)
   
#%%
#================================
# Multivariate behavioral similarity
#================================
from sklearn.preprocessing import StandardScaler

sugg_cols = [
    "SHSS_score",
    "Chge_hypnotic_depth",
    "Mental_relax_absChange",
    "Abs_diff_automaticity"
]

# pain_cols = [
#     "VAS_Nana_Int",
#     "VAS_Ana_Int",
#     "VAS_Nhyper_Int",
#     "VAS_Hyper_Int",
#     "VAS_Nana_UnP",
#     "VAS_Ana_UnP",
#     "VAS_Nhyper_UnP",
#     "VAS_Hyper_UnP"
# ]

pain_cols = ['raw_change_HYPER', 'raw_change_ANA']

scaler = StandardScaler()
Y_sugg = scaler.fit_transform(np.array(Y[sugg_cols].values, dtype=float))
Y_pain = scaler.fit_transform(np.array(Y[pain_cols].values, dtype=float))

plt.figure()
plt.imshow(Y_sugg)
plt.colorbar(label='scores')
plt.xticks(np.arange(len(sugg_cols)), sugg_cols, rotation=45, ha='right')
plt.yticks(np.arange(Y.shape[0]), Y.index)
plt.title('Behavioral Features Heatmap')
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

plt.figure()
plt.imshow(cosine_sim_sugg, cmap='coolwarm')
plt.colorbar(label='Cosine Similarity')
plt.title('Cosine Similarity: multivariate hypnonic scores')
plt.tight_layout()

plt.figure()
plt.imshow(cosine_sim_pain, cmap='coolwarm')
plt.colorbar(label='Cosine Similarity')
plt.title('Cosine Similarity: multivariate pain scores')
plt.tight_layout()

# test correlation between univariate and multivariate similarity

# suggestion
y = Y['SHSS_score'].values
y = (y - np.mean(y)) / np.std(y)
sim_behav_vec = isc_utils.compute_behav_similarity(y, metric='euclidean', vectorize=True)
cosine_vec_sugg = isc_utils.compute_behav_similarity(Y_sugg, metric='cosine', vectorize=True)
r, p = spearmanr(cosine_vec_sugg, sim_behav_vec)
print(f"Spearman correlation between uni - multivariate SUGG var.: {r:.4f}, p-value: {p:.4f}")

y = Y['total_chge_pain_hypAna'].values
y = (y - np.mean(y)) / np.std(y)
sim_behav_vec = isc_utils.compute_behav_similarity(y, metric='euclidean', vectorize=True)
cosine_vec_pain = isc_utils.compute_behav_similarity(Y_pain, metric='cosine', vectorize=True)

r, p = spearmanr(cosine_vec_pain, sim_behav_vec)
print(f"Spearman correlation between uni - multivariate PAIN var.: {r:.4f}, p-value: {p:.4f}")

#%%

# sns.clustermap(sim_sugg,
#                metric='euclidean',  
#                method='average',      
#                cmap='coolwarm',
#                xticklabels=True,
#                yticklabels=True)

#%%
#================================
#Atlas
#================================
SCHAEFER_ONLY = True

atlas = nib.load('/data/rainville/dSutterlin/projects/ISC_hypnotic_suggestions/masks/Tian2020_schaeffer200_subcortical16/combined_schaefer200_tian16_DSG.nii.gz')
id_labels_dct = isc_utils.load_json('/data/rainville/dSutterlin/projects/ISC_hypnotic_suggestions/masks/Tian2020_schaeffer200_subcortical16/roi_labels_dict_DSG.json')
#remove background 
id_labels_dct.pop('0')
labels = list(id_labels_dct.values())
roi_index = list(id_labels_dct.keys())

atlas_masker = NiftiLabelsMasker(labels_img=atlas, labels=labels, standardize=False)
atlas_masker.fit()

if SCHAEFER_ONLY:
    atlas_data = fetch_atlas_schaefer_2018(n_rois = 200, resolution_mm=2)
    atlas = nib.load(atlas_data['maps'])
    # atlas_path = atlas_data['maps'] #os.path.join(project_dir,os.path.join(project_dir, 'masks', 'k50_2mm', '*.nii*'))
    # labels_bytes = list(atlas_data['labels'])
    labels = [str(label, 'utf-8') if isinstance(label, bytes) else str(label) for label in atlas_data['labels']]
    roi_index = [i+1 for i in range(len(labels))] # no background, 0

atlas_masker = NiftiLabelsMasker(labels_img=atlas, standardize=False)
atlas_masker.fit()
coords = find_parcellation_cut_coords(labels_img=atlas)
atlas_labels = dict(zip(roi_index, labels))
labels = list(atlas_labels.values())

#%% VISU
from nilearn import plotting
reload(visu_utils)

rsa_path = '/data/rainville/dSutterlin/projects/ISC_hypnotic_suggestions/results/imaging/RSA/2025-05-21_23'
# rsa_isc = isc_utils.load_pickle(os.path.join(save_path, f'rsa_isc_pain-behav_sugg-pain_{n_perm_rsa}perm.pkl'))
rsa_isc = isc_utils.load_pickle('/data/rainville/dSutterlin/projects/ISC_hypnotic_suggestions/results/imaging/RSA/rsa_pain-behav_sugg-pain_10000perm.pkl')
rsa_isc_sugg = isc_utils.load_pickle(os.path.join(rsa_path, 'rsa_cosine-sugg-behav_isc-sugg5000perm.pkl' ))

views={}
for cond in ['ANA']: #['HYPER', 'ANA', 'all_sugg', 'neutral']:

    print('cond', cond)
    rsa_df = rsa_isc_sugg[cond].sort_index(ascending=True) # to match the atlas labels

    # === Prepare variables for projection ===
    correlations = rsa_df['spearman_r'].values
    p_values = rsa_df['p_values'].values
    roi_labels = rsa_df['ROI'].values  # assumes label matches atlas
    fdr_p = isc_utils.fdr(p_values, q=0.05)
    print(f'FDR threshold: {fdr_p:.4f}')

    # === Map ROI label names to atlas index ===
    label_to_index = {label: idx for idx, label in enumerate(labels)}
    roi_indices = [label_to_index[roi] for roi in roi_labels]

    # === Create full-length arrays aligned with atlas ===
    # full_r_values = np.zeros(len(labels))
    # full_p_values = np.ones(len(labels))

    # for idx, roi_idx in enumerate(roi_indices):
    #     full_r_values[roi_idx] = correlations[idx]
    #     full_p_values[roi_idx] = p_values[idx]

    unc_p = 0.01
    # === Visualize with your existing function ===
    rsa_img, rsa_thresh, sig_labels = visu_utils.project_isc_to_brain_perm(
        atlas_img=atlas,
        isc_median=correlations,
        atlas_labels=atlas_labels,
        roi_coords = coords,
        p_values=p_values,
        p_threshold=fdr_p, #!!
        title=None, #"RSA-ISC: Suggestion-Pain Similarity (FDR<.05)",
        save_path=None,
        show=True,
        display_mode='x',
        cut_coords_plot=None, #(-52, -40, 34),
        color='Reds'
    )
    
    views[cond] = plotting.view_img(rsa_img, threshold=rsa_thresh, title=f"RSA suggestion - pain similarity {cond}", colorbar=True,symmetric_cmap=False, cmap = 'Reds')

    
# %%
#=================================
# IS-RSA for ISC-SUGGESTION  & ISC-PAIN
rsa_per_cond = {}

for cond in conditions:
    print(f'Running RSA for condition: {cond}')
    # sugg_path = f'/data/rainville/dSutterlin/projects/ISC_hypnotic_suggestions/results/imaging/ISC/{model_sugg}/concat_suggs_1samp_boot/isc_results_{cond}_{n_perm}boot_pairWiseTrue.pkl'
    # pain_path = f'/data/rainville/dSutterlin/projects/ISC_hypnotic_suggestions/results/imaging/ISC/{model_pain}/concat_suggs_1samp_boot/isc_results_{cond}_{n_perm}boot_pairWiseTrue.pkl'

    if cond in ['ANA', 'HYPER']:
        sugg_path =   f'/data/rainville/dSutterlin/projects/ISC_hypnotic_suggestions/results/imaging/ISC/{model_sugg}/{cond}/isc_results_{cond}_{n_perm}boot_pairWiseTrue.pkl'
        pain_path =   f'/data/rainville/dSutterlin/projects/ISC_hypnotic_suggestions/results/imaging/ISC/{model_pain}/{cond}/isc_results_{cond}_{n_perm}boot_pairWiseTrue.pkl'
    else:
        sugg_path = f'/data/rainville/dSutterlin/projects/ISC_hypnotic_suggestions/results/imaging/ISC/{model_sugg}/concat_suggs_1samp_boot/isc_results_{cond}_{n_perm}boot_pairWiseTrue.pkl'
        pain_path = f'/data/rainville/dSutterlin/projects/ISC_hypnotic_suggestions/results/imaging/ISC/{model_pain}/concat_suggs_1samp_boot/isc_results_{cond}_{n_perm}boot_pairWiseTrue.pkl'

    isc_sugg = pd.DataFrame(isc_utils.load_pickle(sugg_path)['isc'], columns=labels)
    isc_pain = pd.DataFrame(isc_utils.load_pickle(pain_path)['isc'], columns=labels)

    assert isc_sugg.shape == isc_pain.shape

    rsa_rows = [] # build df 
    for roi_idx, roi in enumerate(isc_sugg.columns):
        sugg_vec = isc_sugg[roi].values
        pain_vec = isc_pain[roi].values
        r, p, dist = isc_utils.matrix_permutation(sugg_vec, pain_vec, n_permute=n_perm_rsa, metric="spearman", how="upper", tail=n_tail_rsa, return_perms=True)
    
        rsa_rows.append({
            'ROI': roi,
            'spearman_r': r,
            'p_values': round(p, 5),
            'x': coords[roi_idx][0],
            'y': coords[roi_idx][1],
            'z': coords[roi_idx][2]
        })

    rsa_per_cond[cond] = pd.DataFrame(rsa_rows).sort_values(by='spearman_r', ascending=False)
    print('Max r, mean and fdr', rsa_per_cond[cond]['spearman_r'].max(), rsa_per_cond[cond]['spearman_r'].mean(), isc_utils.fdr(rsa_per_cond[cond]['p_values'].to_numpy()))

save_to = os.path.join(save_path, f'rsa_isc-sugg_isc-pain_{n_perm_rsa}perm.pkl')
isc_utils.save_data(save_to, rsa_per_cond)
print(f'Saved RSA results to {save_to}')
print('-----Done with rsa!-----')

# #%%         
# # RSA ISFC
# reload(isc_utils)
# isfc_rsa_per_cond = {}

# isfc_model_sugg = model_names['model5_isfc_sugg'] 
# isfc_model_pain = model_names['model5_isfc_shock']
# n_perm = 5000 # for loading

# for cond in conditions:
#     print(f'Running ISFC RSA for condition: {cond}')
#     sugg_path = f'/data/rainville/dSutterlin/projects/ISC_hypnotic_suggestions/results/imaging/ISC/{model_sugg}/concat_suggs_1samp_boot/isc_results_{cond}_{n_perm}boot_pairWiseTrue.pkl'
#     pain_path = f'/data/rainville/dSutterlin/projects/ISC_hypnotic_suggestions/results/imaging/ISC/{model_pain}/concat_suggs_1samp_boot/isc_results_{cond}_{n_perm}boot_pairWiseTrue.pkl'

#     isfc_sugg = pd.DataFrame(isc_utils.load_pickle(sugg_path)['isc'], columns=labels)
#     isfc_pain = pd.DataFrame(isc_utils.load_pickle(pain_path)['isc'], columns=labels)

#     assert isc_sugg.shape == isfc_pain.shape

#     rsa_rows = [] # build df 
#     for roi_idx, roi in enumerate(isc_sugg.columns):
#         sugg_vec = isfc_sugg[roi].values
#         pain_vec = isfc_pain[roi].values
#         r, p, dist = isc_utils.matrix_permutation(sugg_vec, pain_vec, n_permute=n_perm_rsa, metric="spearman", how="upper", tail=n_tail_rsa, return_perms=True)
        
#         rsa_rows.append({
#             'ROI': roi,
#             'spearman_r': r,
#             'p_values': round(p, 5),
#             'x': coords[roi_idx][0],
#             'y': coords[roi_idx][1],
#             'z': coords[roi_idx][2]
#         })

#     isfc_rsa_per_cond[cond] = pd.DataFrame(rsa_rows).sort_values(by='spearman_r', ascending=False)
#     print('Max r, mean and fdr', isfc_rsa_per_cond[cond]['spearman_r'].max(), isfc_rsa_per_cond[cond]['spearman_r'].mean(), isc_utils.fdr(isfc_rsa_per_cond[cond]['p_values'].to_numpy()))

# save_to = os.path.join(save_path, f'rsa_isfc_sugg-pain_{n_perm_rsa}perm.pkl')
# isc_utils.save_data(save_to, isfc_rsa_per_cond)
# print(f'Saved RSA results to {save_to}')
# print('-----Done with ISFC rsa!-----')

# %%
reload(isc_utils)
from tqdm import tqdm
#=================================
# IS-RSA for SUGGESTION
#=================================

# check if all cond are in isc_results
save_to = os.path.join(save_path, f'rsa_cosine-sugg-behav_isc-sugg{n_perm_rsa}perm.pkl')

rsa_behav_per_cond = {}
# conditions = ['ANA', 'HYPER', 'all_sugg'] # 'neutral']
for cond in conditions:
    if cond in ['ANA', 'HYPER']:
        pain_path =   f'/data/rainville/dSutterlin/projects/ISC_hypnotic_suggestions/results/imaging/ISC/{model_sugg}/{cond}/isc_results_{cond}_{n_perm}boot_pairWiseTrue.pkl'
    else:
        pain_path = f'/data/rainville/dSutterlin/projects/ISC_hypnotic_suggestions/results/imaging/ISC/{model_sugg}/concat_suggs_1samp_boot/isc_results_{cond}_{n_perm}boot_pairWiseTrue.pkl'

    comparison = {'ANA' : 'raw_change_ANA',
                  'HYPER' : 'raw_change_HYPER',
                  'modulation' : 'total_chge_pain_hypAna',
                  'neutral' : 'total_chge_pain_hypAna',
                  'all_sugg' : 'total_chge_pain_hypAna'
        }   
    
    vec_behav_sim = cosine_sim_sugg #specify if sugg or pain
    isc_pairwise = pd.DataFrame(isc_utils.load_pickle(pain_path)['isc'], columns=labels)

    rsa_rows = [] # build df 
    for roi_idx, roi in tqdm(enumerate(isc_pairwise.columns), total=len(isc_pairwise.columns)):
        isc_roi = isc_pairwise[roi].values
        r, p, dist = isc_utils.matrix_permutation(vec_behav_sim, isc_roi, n_permute=n_perm_rsa, metric="spearman", how="upper", tail=n_tail_rsa, return_perms=True)
        
        rsa_rows.append({
            'ROI': roi,
            'spearman_r': r,
            'p_values': round(p, 5),
            'x': coords[roi_idx][0],
            'y': coords[roi_idx][1],
            'z': coords[roi_idx][2]
        })

    rsa_behav_per_cond[cond] = pd.DataFrame(rsa_rows).sort_values(by='spearman_r', ascending=False)
    print('Max r, mean and fdr', rsa_behav_per_cond[cond]['spearman_r'].max(), rsa_behav_per_cond[cond]['spearman_r'].mean(), isc_utils.fdr(rsa_behav_per_cond[cond]['p_values'].to_numpy()))

isc_utils.save_data(save_to, rsa_behav_per_cond)
print(f'Saved RSA results to {save_to}')


#%%
#=================================
# IS-RSA for PAIN
#=================================

# check if all cond are in isc_results
save_to = os.path.join(save_path, f'rsa_cosine-pain-behav_isc-pain{n_perm_rsa}perm.pkl')

rsa_behav_per_cond = {}
conditions = ['ANA', 'HYPER', 'modulation', 'all_sugg', 'neutral']
for cond in conditions:
    if cond in ['ANA', 'HYPER']:
        pain_path =   f'/data/rainville/dSutterlin/projects/ISC_hypnotic_suggestions/results/imaging/ISC/{model_pain}/{cond}/isc_results_{cond}_{n_perm}boot_pairWiseTrue.pkl'
    else:
        pain_path = f'/data/rainville/dSutterlin/projects/ISC_hypnotic_suggestions/results/imaging/ISC/{model_pain}/concat_suggs_1samp_boot/isc_results_{cond}_{n_perm}boot_pairWiseTrue.pkl'

    vec_behav_sim = cosine_vec_pain #specify if sugg or pain
    # Behav similarity
    # y_name = comparison[cond]
    # y = behav_df[y_name].values
    # y = (y - np.mean(y)) / np.std(y)
    # vec_behav = isc_utils.compute_behav_similarity(y, metric='euclidean', vectorize=True)
    
    isc_pairwise = pd.DataFrame(isc_utils.load_pickle(pain_path)['isc'], columns=labels)

    rsa_rows = [] # build df 
    for roi_idx, roi in enumerate(isc_pairwise.columns):
        isc_roi = isc_pairwise[roi].values
        r, p, dist = isc_utils.matrix_permutation(vec_behav_sim, isc_roi, n_permute=n_perm_rsa, metric="spearman", how="upper", tail=n_tail_rsa, return_perms=True)
        
        rsa_rows.append({
            'ROI': roi,
            'spearman_r': r,
            'p_values': round(p, 5),
            'x': coords[roi_idx][0],
            'y': coords[roi_idx][1],
            'z': coords[roi_idx][2]
        })

    rsa_behav_per_cond[cond] = pd.DataFrame(rsa_rows).sort_values(by='spearman_r', ascending=False)
    print('Max r, mean and fdr', rsa_behav_per_cond[cond]['spearman_r'].max(), rsa_behav_per_cond[cond]['spearman_r'].mean(), isc_utils.fdr(rsa_behav_per_cond[cond]['p_values'].to_numpy()))

isc_utils.save_data(save_to, rsa_behav_per_cond)
print(f'Saved RSA results to {save_to}')


#=================================
# IS-RSA for SUGGESTION(behav) to PAIN(ISC)
#=================================

# check if all cond are in isc_results
save_to = os.path.join(save_path, f'rsa_cosine-sugg-behav_isc-pain{n_perm_rsa}perm.pkl')

rsa_behav_per_cond = {}
# conditions = ['ANA', 'HYPER', 'all_sugg'] # 'neutral']
for cond in conditions:
    if cond in ['ANA', 'HYPER']:
        pain_path =   f'/data/rainville/dSutterlin/projects/ISC_hypnotic_suggestions/results/imaging/ISC/{model_pain}/{cond}/isc_results_{cond}_{n_perm}boot_pairWiseTrue.pkl'
    else:
        pain_path = f'/data/rainville/dSutterlin/projects/ISC_hypnotic_suggestions/results/imaging/ISC/{model_pain}/concat_suggs_1samp_boot/isc_results_{cond}_{n_perm}boot_pairWiseTrue.pkl'


    vec_behav_sim = cosine_sim_sugg #specify if sugg or pain
    isc_pairwise = pd.DataFrame(isc_utils.load_pickle(pain_path)['isc'], columns=labels)

    rsa_rows = [] # build df 
    for roi_idx, roi in tqdm(enumerate(isc_pairwise.columns), total=len(isc_pairwise.columns)):
        isc_roi = isc_pairwise[roi].values
        r, p, dist = isc_utils.matrix_permutation(vec_behav_sim, isc_roi, n_permute=n_perm_rsa, metric="spearman", how="upper", tail=n_tail_rsa, return_perms=True)
        
        rsa_rows.append({
            'ROI': roi,
            'spearman_r': r,
            'p_values': round(p, 5),
            'x': coords[roi_idx][0],
            'y': coords[roi_idx][1],
            'z': coords[roi_idx][2]
        })

    rsa_behav_per_cond[cond] = pd.DataFrame(rsa_rows).sort_values(by='spearman_r', ascending=False)
    print('Max r, mean and fdr', rsa_behav_per_cond[cond]['spearman_r'].max(), rsa_behav_per_cond[cond]['spearman_r'].mean(), isc_utils.fdr(rsa_behav_per_cond[cond]['p_values'].to_numpy()))

isc_utils.save_data(save_to, rsa_behav_per_cond)
# print(f'Saved RSA results to {save_to}')


#%%
print(f'Saved all RSA results to {save_path}')
print('Done with all RSA!')
# %%
