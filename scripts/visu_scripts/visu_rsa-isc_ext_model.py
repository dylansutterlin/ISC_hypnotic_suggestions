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
from tqdm import tqdm
import shutil
import sys


from src import isc_utils
import src.isc_utils as isc_utils
import src.visu_utils as visu_utils
from importlib import reload
reload(visu_utils)
reload(isc_utils)
#%%
# Config
model_names = {
    'model1_sugg_200': 'model1_sugg_23-sub_schafer-200-2mm_mask-whole-brain_pairWise-True_preproc_reg-mvmnt-True-8',
    'model1_sugg_200_run' : 'model1-ext-conds_sugg_23-sub_schafer-200-2mm_mask-whole-brain_pairWise-True_preproc_reg-mvmnt-True-8',
    'model1-6': 'model1_sugg_23-sub_schafer-200-2mm_mask-whole-brain_pairWise-True_preproc_reg-mvmnt-True-6',
    'model2_sugg_loo': 'model2_sugg_23-sub_schafer-200-2mm_mask-whole-brain_pairWise-False_preproc_reg-mvmnt-True-8',
    'model2_shock_loo': 'model2_shock_23-sub_schafer-200-2mm_mask-whole-brain_pairWise-False_preproc_reg-mvmnt-True-8',
    'model3_shock_200': 'model3_shock_23-sub_schafer-200-2mm_mask-whole-brain_pairWise-True_preproc_reg-mvmnt-True-8',
    'model3_shock_200_run' : 'model2-ext-conds_shock_23-sub_schafer-200-2mm_mask-whole-brain_pairWise-True_preproc_reg-mvmnt-True-8',
    'model1_mean': 'model4-mean_sugg_23-sub_schafer-200-2mm_mask-whole-brain_pairWise-True_preproc_reg-mvmnt-True-8',
    'model5_isfc_sugg' :'model5-isfc_sugg_23-sub_schafer-200-2mm_mask-lanA800_pairWise-True_preproc_reg-mvmnt-True-8',
    'model5_isfc_shock' : 'model5-isfc_shock_23-sub_schafer-200-2mm_mask-lanA800_pairWise-True_preproc_reg-mvmnt-True-8',
    '9 avril ...' : ' ',
    'single_trial' : 'model_single-trial_sugg_23-sub_schafer-200-2mm_mask-lanA800_pairWise-True_preproc_reg-mvmnt-True-8',
    'single_trial_wb' : 'model_single-trial-wb_sugg_23-sub_schafer-200-2mm_mask-whole-brain_pairWise-True_preproc_reg-mvmnt-True-8',
    'model5_sugg_tian' : 'model5-with-subcort_sugg_23-sub_schafer-200-2mm_mask-whole-brain_pairWise-True_preproc_reg-mvmnt-True-8',
    'model1_single-trial': 'model1_single-trial-wb_sugg_23-sub_schafer_tian-200-2mm_mask-whole-brain_pairWise-True_preproc_reg-mvmnt-True-8',
    'model2_sugg' : 'model2_sugg_23-sub_schafer_tian-200-2mm_mask-whole-brain_pairWise-True_preproc_reg-mvmnt-True-8',
    'model3_shock' : 'model3_shock_23-sub_schafer_tian-200-2mm_mask-whole-brain_pairWise-True_preproc_reg-mvmnt-True-8',
}


model_sugg = model_names['model1_sugg_200_run']
model_pain = model_names['model3_shock_200_run']



project_dir = "/data/rainville/dSutterlin/projects/ISC_hypnotic_suggestions"
results_dir = os.path.join(project_dir, f'results/imaging/ISC/{model_sugg}')
setup = isc_utils.load_json(os.path.join(results_dir, "setup_parameters.json"))
subjects = setup['subjects']

DATE = datetime.now().strftime("%Y-%m-%d_%H")
save_path = f'/data/rainville/dSutterlin/projects/ISC_hypnotic_suggestions/results/imaging/RSA/isc-RSA_ext_conds_sugg-pain_2tails'
os.makedirs(save_path, exist_ok=True)

# conditions = ['NANA', 'ANA', 'HYPER', 'all_sugg', 'neutral', 'ana_run', 'hyper_run']
conditions = ['ANA', 'HYPER', 'ana_run', 'hyper_run']

sim_model = 'euclidean'
n_perm = setup['n_perm'] # to load isc results
n_perm_rsa = 10000
n_tail_rsa = 2 # hypothesize that only pos. associations are of interest
n_jobs = 32 


#%%
#================================
# BEHAHVIORAL DATA
#================================
import seaborn as sns
import src.preproc_utils as preproc_utils
reload(preproc_utils)

xlsx_path = r'/data/rainville/dSutterlin/projects/ISC_hypnotic_suggestions/masks/Hypnosis_variables_20190114_pr_jc.xlsx'
subjects = list(setup['subjects'])
apm_subjects = ['APM' + subj[4:] for subj in subjects]
print(apm_subjects)

Y = preproc_utils.load_process_y(xlsx_path, subjects)

#cov matrix Y_sugg
corr_matrix_sugg = np.corrcoef(np.array(Y).astype(float), rowvar=False)
np.fill_diagonal(corr_matrix_sugg, 0)

plt.figure(figsize=(10, 8))
sns.heatmap(
    corr_matrix_sugg,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    vmin=-1,
    vmax=1,
    xticklabels=Y.columns,
    yticklabels=Y.columns,
    square=True,
    cbar_kws={'label': 'Correlation', 'shrink': 0.8}  # Adjust shrink to control size
)
cbar = plt.gca().collections[0].colorbar
cbar.ax.tick_params(labelsize=15)  # Increase the fontsize of the color bar
cbar.set_label('Correlation', fontsize=18)  # Increase the label size
plt.title('Correlation Matrix of hypnotic features', fontsize=22)
plt.xticks(rotation=45, ha='right', fontsize=15)
plt.yticks(rotation=0, fontsize=15)


#%%

y_conditions = ['SHSS_score', 'raw_change_ANA', 'raw_change_HYPER', 'total_chge_pain_hypAna']

def plot_simmat(simmat, title,title_id = 'SHSS', y_name = 'SHSS_score', cmap= 'coolwarm'):
    plt.figure(figsize=(10, 8))
    plt.imshow(simmat, vmin= np.min(simmat), vmax=np.max(simmat), cmap=cmap)
    plt.colorbar(label='Similarity')
    plt.title(title, fontsize=24)
    plt.xlabel(f'Subject ranked on {y_name}', fontsize=20)
    plt.ylabel(f'Subject ranked on {y_name}', fontsize=20)
    plt.show()

for metric in ['euclidean', 'annak']:
    for y_name in ['SHSS_score', 'total_chge_pain_hypAna']:

        y = Y[y_name].values
        y = (y - np.mean(y)) / np.std(y)
        sim_behav = isc_utils.compute_behav_similarity(y, metric=metric, vectorize=False)

        ranks = np.argsort(y)         # gives indices that sort behavior low → high
        sim_ranked = sim_behav[np.ix_(ranks, ranks)]

        title = f'{metric}-based {y_name} similarity matrix'
        plot_simmat(sim_ranked, title = title, y_name = y_name)
   
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

def plot_behavioral_values(Y, cols_names, index_names, cbar_label = 'scores', title='Behavioral Features Heatmap', cmap = 'viridis'):
    plt.figure()
    plt.imshow(Y)
    plt.colorbar(label=cbar_label, cmap = cmap)
    plt.xticks(np.arange(len(cols_names)), cols_names, rotation=45, ha='right')
    plt.yticks(np.arange(Y.shape[0]), index_names)
    plt.title(title)
    # plt.tight_layout()

plot_behavioral_values(Y_sugg, sugg_cols, Y.index, title = 'Hypnotic response patterns') 
plot_behavioral_values(Y_pain, pain_cols, Y.index, title= 'Pain modulation response patterns')
#%%

# MULTIVARIATE cosine behavioral similarity
# Compute pairwise cosine similarity using the previously defined function
cosine_sim_sugg = isc_utils.compute_behav_similarity(Y_sugg, metric='cosine', vectorize=False)
cosine_vec_sugg = isc_utils.compute_behav_similarity(Y_sugg, metric='cosine', vectorize=True)

cosine_sim_pain = isc_utils.compute_behav_similarity(Y_pain, metric='cosine', vectorize=False)
cosine_vec_pain = isc_utils.compute_behav_similarity(Y_pain, metric='cosine', vectorize=True)

plot_simmat(cosine_sim_sugg, title = 'Cosine', title_id = 'SUGG', y_name = 'SHSS_score', cmap= 'coolwarm')
plot_simmat(cosine_sim_pain, title = 'Cosine', title_id = 'PAIN', y_name = 'raw_change_HYPER', cmap= 'coolwarm')


# UNIVARIATE pairwise similarities : NN & AnnaK 
y = Y['SHSS_score'].values
y = (y - np.mean(y)) / np.std(y)
sim_behav_vec = isc_utils.compute_behav_similarity(y, metric='euclidean', vectorize=True)
sim_behav_vec_annak = isc_utils.compute_behav_similarity(y, metric='annak', vectorize=True)

r, p = spearmanr(cosine_vec_sugg, sim_behav_vec)
print(f"Spearman correlation between uni - multivariate SUGG var.: {r:.4f}, p-value: {p:.4f}")

# pain
y = Y['total_chge_pain_hypAna'].values
y = (y - np.mean(y)) / np.std(y)
sim_behav_pain_vec = isc_utils.compute_behav_similarity(y, metric='euclidean', vectorize=True)
sim_behav_pain_vec_annak = isc_utils.compute_behav_similarity(y, metric='annak', vectorize=True)

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
    atlas_labels = dict(zip(roi_index, labels))

else:
    atlas_masker = NiftiLabelsMasker(labels_img=atlas,atlas_labels = labels, standardize=False)
    atlas_masker.fit()
    atlas_labels = dict(zip(roi_index, labels))
    labels = list(atlas_labels.values())

coords = find_parcellation_cut_coords(labels_img=atlas)

df_labels_coords = pd.DataFrame(atlas_labels.items(), columns=['index', 'region'])
df_labels_coords['coords'] = [coords[i] for i in range(len(coords))]

#%%
# plot specific ROIs to find anatomical regions

from nilearn.image import math_img

# plot only selected roi to validate the location aMCC _ SMA
def plot_roi_by_label(label_name, atlas, df_labels_coords):
    try:
        row = df_labels_coords[df_labels_coords['region'] == label_name].iloc[0]
        region_value = int(row['index'])

        mask_img = math_img("img == %d" % region_value, img=atlas)

        print(f"Found region: {label_name} (Index {region_value})")
        print(f"Coordinates: {row['coords']}")

        return plotting.view_img(mask_img, threshold=0.5, title=f"{label_name}")

    
    except IndexError:
        print(f"Region label '{label_name}' not found in atlas.")
   
#$$
sig_ana_run_rois = [20, 28, 30, 52, 54, 76, 77, 122, 131, 132, 157, 158, 173, 187, 189]

for roi_idx in sig_ana_run_rois:
    roi_name = labels[roi_idx]
    # sma_name =
    view_amcc = plot_roi_by_label(roi_name, atlas, df_labels_coords)
    # view_sma = plot_roi_by_label(amcc_name, atlas, df_labels_coords)
    display(view_amcc)

#%% VISU ISC
#===========
from nilearn import plotting
reload(visu_utils)

# rsa_isc = isc_utils.load_pickle('/data/rainville/dSutterlin/projects/ISC_hypnotic_suggestions/results/imaging/RSA/isc-RSA_ext_conds_sugg-pain/rsa_isc-sugg_isc-pain_10000perm.pkl')
rsa_isc = isc_utils.load_pickle('/data/rainville/dSutterlin/projects/ISC_hypnotic_suggestions/results/imaging/RSA/isc-RSA_ext_conds_sugg-pain/rsa_cosine-behav_isc-sugg10000perm.pkl')
rsa_isc = isc_utils.load_pickle('/data/rainville/dSutterlin/projects/ISC_hypnotic_suggestions/results/imaging/RSA/2025-05-21_23/rsa_isc-sugg_isc-pain_5000perm.pkl')
rsa_isc = isc_utils.load_pickle('/data/rainville/dSutterlin/projects/ISC_hypnotic_suggestions/results/imaging/RSA/isc-RSA_ext_conds_sugg-pain/rsa_isc-sugg_isc-pain_10000perm.pkl')

views={}
tables = {}
for cond in ['HYPER', 'ANA', 'ana_run', 'hyper_run']: #, 'ANA', 'all_sugg', 'neutral']:

    print('cond', cond)
    rsa_df = rsa_isc[cond].sort_index(ascending=True) # to match the atlas labels

    # === Prepare variables for projection ===
    correlations = rsa_df['spearman_r'].values
    p_values = rsa_df['p_values'].values
    roi_labels = rsa_df['ROI'].values  # assumes label matches atlas
    fdr_p = isc_utils.fdr(p_values, q=0.05)
    print(f'FDR threshold: {fdr_p:.4f}')

    unc_p = 0.01
    # === Visualize with your existing function ===
    rsa_img, rsa_thresh, sig_labels, _ = visu_utils.project_isc_to_brain_perm(
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
    tables[cond] = sig_labels
    views[cond] = plotting.view_img(rsa_img,bg_img = bg_mni, threshold=rsa_thresh, title=f"RSA suggestion - pain similarity {cond}", colorbar=True,symmetric_cmap=False, cmap = 'Reds')

#%%
reload(visu_utils)
from nilearn import datasets, plotting
bg_mni =  datasets.load_mni152_template(resolution=1)

#VISU UNIVARIATE SHSS for run effect
load_dir = '/data/rainville/dSutterlin/projects/ISC_hypnotic_suggestions/results/imaging/RSA/isc-RSA_ext_conds_sugg-pain_2tails'
RESULT_FILE = 'rsa_SHSS-behav_isc-sugg10000perm.pkl' 

rsa_isc = isc_utils.load_pickle(os.path.join(load_dir, RESULT_FILE))
output_dir = os.path.join(load_dir, 'post-hoc_VISU-tables')
os.makedirs(output_dir, exist_ok=True)

n_perm = 10000

# isc for visu
isc_pairwise = {}
for cond in ['ana_run', 'hyper_run']:
    isc_path= f'/data/rainville/dSutterlin/projects/ISC_hypnotic_suggestions/results/imaging/ISC/{model_sugg}/concat_suggs_1samp_boot/isc_results_{cond}_{n_perm}boot_pairWiseTrue.pkl'
    isc_pairwise[cond] = pd.DataFrame(isc_utils.load_pickle(isc_path)['isc'], columns=labels)

y = Y['SHSS_score'].values
y = (y - np.mean(y)) / np.std(y)
sim_mat_behav_matrix = isc_utils.compute_behav_similarity(y, metric='euclidean', vectorize=False)
sim_mat_behav_annak = isc_utils.compute_behav_similarity(y, metric='annak', vectorize=False)


views={}
tables = {'euclidian': {}, 'annak': {}}
rsa_dfs = {'euclidian': {}, 'annak': {}}

model = 'annak'

stats_imgs = []
for cond in ['ana_run', 'hyper_run'] : #, 'HYPER', 'ANA']: #, 'all_sugg', 'neutral']:

    print('cond', cond)
    rsa_df = rsa_isc[model][cond].sort_index(ascending=True) # to match the atlas labels
    rsa_dfs[model][cond] = rsa_df

    # === Prepare variables for projection ===
    correlations = rsa_df['spearman_r'].values
    p_values = rsa_df['p_values'].values
    roi_labels = rsa_df['ROI'].values  # assumes label matches atlas
    fdr_p = isc_utils.fdr(p_values, q=0.05)
    print(f'FDR threshold: {fdr_p:.4f}')

    unc_p = 0.01
    title = f"IS-RSA : Hypnotic susceptibility ~ {cond} (FDR<.05)"

    if cond == 'hyper_run':
        cut_coords_plot = (-60, -56, -46, -42)
    elif cond == 'ana_run':
        cut_coords_plot = (-54,0, 56) 
    else:
        cut_coords_plot = None
    
    title = f"IS-RSA : Hypnotic susceptibility ~ {cond} (FDR<.05)"
    # === Visualize with your existing function ===
    rsa_img, rsa_thresh, sig_labels, stats_img = visu_utils.project_isc_to_brain_perm(
        atlas_img=atlas,
        isc_median=correlations,
        atlas_labels=atlas_labels,
        roi_coords = coords,
        p_values=p_values,
        p_threshold=fdr_p, #!!
        title=title, #"RSA-ISC: Suggestion-Pain Similarity (FDR<.05)",
        save_path=None,
        show=True,
        display_mode='x',
        cut_coords_plot=cut_coords_plot, #(-52, -40, 34),
        color='coolwarm'
    )
    stats_imgs.append(stats_img)

    if sig_labels.shape[0] > 0:
        # sig_labels = pd.DataFrame(sig_labels, columns=['ROI', 'Label', 'spearman_r', 'p_values', 'Coordinates (X,Y,Z)'])
        sig_labels.columns = ['ROI', 'Label', 'spearman_r', 'p_values', 'Coordinates (X,Y,Z)']
        sig_labels.to_csv(os.path.join(output_dir, f'{RESULT_FILE[:-13]}_{model}_{cond}_table-FDR.csv'), index=False)


    tables[model][cond] = sig_labels
    views[cond] = plotting.view_img(rsa_img, threshold=rsa_thresh, title=f"RSA suggestion - pain similarity {cond}", colorbar=True,symmetric_cmap=False, cmap = 'coolwarm')

stats_imgs[0].to_filename(os.path.join(output_dir, f'{RESULT_FILE[:-13]}_{model}_ana_run.nii.gz'))
# stats_imgs[1].to_filename(os.path.join(output_dir, f'{RESULT_FILE[:-13]}_{model}_hyper_run.nii.gz'))
#%%
model = 'euclidian'
tables_nn = {}
for cond in ['ana_run', 'hyper_run'] : #['HYPER', 'ANA', 'all_sugg', 'neutral']:

    print('cond', cond)
    rsa_df = rsa_isc[model][cond].sort_index(ascending=True) # to match the atlas labels
    rsa_dfs[model][cond] = rsa_df

    # === Prepare variables for projection ===
    correlations = rsa_df['spearman_r'].values
    p_values = rsa_df['p_values'].values
    roi_labels = rsa_df['ROI'].values  # assumes label matches atlas
    fdr_p = isc_utils.fdr(p_values, q=0.05)
    print(f'FDR threshold: {fdr_p:.4f}')

    unc_p = 0.01
    # === Visualize with your existing function ===


    rsa_img, rsa_thresh, sig_labels, _ = visu_utils.project_isc_to_brain_perm(
        atlas_img=atlas,
        isc_median=correlations,
        atlas_labels=atlas_labels,
        roi_coords = coords,
        p_values=p_values,
        p_threshold=unc_p, #!!
        title=None, #"RSA-ISC: Suggestion-Pain Similarity (FDR<.05)",
        save_path=None,
        show=True,
        display_mode='x',
        cut_coords_plot=None, #(-52, -40, 34),
        color='Reds'
    )
    if sig_labels.shape[0] > 0:
        sig_labels.columns = ['ROI', 'Label', 'spearman_r', 'p_values', 'Coordinates (X,Y,Z)']
    
    tables_nn[cond] = sig_labels
    views[cond] = plotting.view_img(rsa_img, threshold=rsa_thresh, title=f"RSA suggestion - pain similarity {cond}", colorbar=True,symmetric_cmap=False, cmap = 'Reds')

#%%
y_conditions = ['SHSS_score', 'raw_change_ANA', 'raw_change_HYPER', 'total_chge_pain_hypAna']

for metric in ['euclidean', 'annak']:
    for y_name in ['SHSS_score']:

        y = Y[y_name].values
        y = (y - np.mean(y)) / np.std(y)
        sim_behav = isc_utils.compute_behav_similarity(y, metric=metric, vectorize=False)

        ranks = np.argsort(y)         # gives indices that sort behavior low → high
        sim_ranked = sim_behav[np.ix_(ranks, ranks)]

        plot_simmat(sim_ranked, title = metric, y_name = y_name)

#%%
# Visualize ISC matrix in ROI
reload(visu_utils)
reload(isc_utils)
cond = 'hyper_run'
cond = 'ana_run'
cmap = 'coolwarm' 

sig_roi = tables['annak'][cond]


def plot_simmat(simmat, title,title_id = 'SHSS', y_name = 'SHSS_score', cmap= 'coolwarm', ):
    vmin = -np.max(simmat)
    plt.figure(figsize=(10, 8))
    plt.imshow(simmat, vmin= vmin, vmax=np.max(simmat), cmap=cmap)
    plt.colorbar(label='Similarity')
    plt.title(title, fontsize=24)
    plt.xlabel(f'Subject ranked on {y_name}', fontsize=20)
    plt.ylabel(f'Subject ranked on {y_name}', fontsize=20)
    plt.show()


# rsa_dfs['annak']
roi_ranked_mat = {}
for roi_idx, roi in zip(sig_roi['ROI'], sig_roi['Label']):
    
    isc_pairwise_roi = isc_pairwise[cond][roi]
    isc_mat = isc_utils.vector_to_isc_matrix(isc_pairwise_roi, diag=0)
    
    ranked_isc_mat = isc_mat[np.ix_(ranks, ranks)]
    roi_ranked_mat[roi] = ranked_isc_mat

    plot_simmat(ranked_isc_mat, title = f'ISC {roi}',title_id = roi, y_name = 'SHSS', cmap = cmap)

ranked_behav_mat_annak = sim_mat_behav_annak[np.ix_(ranks, ranks)]

plot_simmat(ranked_behav_mat_annak, title = 'Annak', y_name = 'SHSS_score', cmap = cmap)


#%%
#plot NN ~ Annak correlation values
reload(visu_utils)

cond = 'ana_run'
cond = 'hyper_run'
vec_rsa_nn = rsa_dfs['euclidian'][cond] 
vec_rsa_annak = rsa_dfs['annak'][cond] 

labels_map, yeo7_net = visu_utils.yeo_networks_from_schaeffer(labels)
visu_utils.plot_scatter_legend(vec_rsa_nn['spearman_r'], vec_rsa_annak['spearman_r'], grp_id=yeo7_net, var_name=['Eucledian-based ISC ', 'AnnaK-based ISC'], title=f'Eucledian vs Annak-based IS-RSA per region ', save_path=None)

# pair ttest 
from scipy.stats import ttest_rel
t_stat, p_val = ttest_rel(vec_rsa_nn['spearman_r'], vec_rsa_annak['spearman_r'])
print(f'T-test between Eucledian and AnnaK-based IS-RSA: t-statistic = {t_stat:.4f}, p-value = {p_val:.4f}')

#non-parametric test
from scipy.stats import wilcoxon
w_stat, p_val_w = wilcoxon(vec_rsa_nn['spearman_r'], vec_rsa_annak['spearman_r'])
print(f'Wilcoxon test between Eucledian and AnnaK-based IS-RSA: W-statistic = {w_stat:.4f}, p-value = {p_val_w:.4f}')

#%%
reload(visu_utils)

for roi_idx, roi in zip(sig_roi['ROI'], sig_roi['Label']):
    
    isc_pairwise_roi = isc_pairwise[cond][roi]
    isc_mat = isc_utils.vector_to_isc_matrix(isc_pairwise_roi, diag=0)
    
    ranked_isc_mat = isc_mat[np.ix_(ranks, ranks)]
    roi_ranked_mat[roi] = ranked_isc_mat

    median_per_row = np.median(ranked_isc_mat, axis=1)
    ranked_shss = np.sort(y)
    
    visu_utils.jointplot(Y['SHSS_score'], median_per_row,
                            x_label='SHSS score (ranked)', y_label=f'ISC {roi} (ranked)',
                            title=f'ISC {roi} vs SHSS score')
#%%
sig_ana_run_rois = [20, 28, 30, 52, 54, 76, 77, 122, 131, 132, 157, 158, 173, 187, 189]

for roi_idx in sig_ana_run_rois:

roi_name = atlas_labels[roi_idx] #adjusted for roi_index
# sma_name =
view_amcc = plot_roi_by_label(roi_name, atlas, df_labels_coords)
# view_sma = plot_roi_by_label(amcc_name, atlas, df_labels_coords)
display(view_amcc)

