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

model_sugg = model_names['model2_sugg']
model_pain = model_names['model3_shock']

project_dir = "/data/rainville/dSutterlin/projects/ISC_hypnotic_suggestions"
results_dir = os.path.join(project_dir, f'results/imaging/ISC/{model_sugg}')
setup = isc_utils.load_json(os.path.join(results_dir, "setup_parameters.json"))
subjects = setup['subjects']

MODEL_PATH = os.path.join(project_dir, 'results/imaging/RSA/isc-RSA_ext_conds_sugg-pain_tian216_2tails')
output_dir = os.path.join(MODEL_PATH, 'post-hoc_VISU-tables')
os.makedirs(output_dir, exist_ok=True)
output_figures = os.path.join(MODEL_PATH, 'post-hoc_figures')
os.makedirs(output_figures, exist_ok=True)

load_dir = '/data/rainville/dSutterlin/projects/ISC_hypnotic_suggestions/results/imaging/RSA/isc-RSA_ext_conds_sugg-pain_2tails'

# conditions = ['NANA', 'ANA', 'HYPER', 'all_sugg', 'neutral', 'ana_run', 'hyper_run']
conditions = ['ANA', 'HYPER', 'ana_run', 'hyper_run']

# sim_model = 'euclidean'
# n_perm = setup['n_perm'] # to load isc results
# n_perm_rsa = 10000
# n_tail_rsa = 2 # hypothesize that only pos. associations are of interest
# n_jobs = 32 


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

def z_score(x):
    x = np.array(x)
    return (x - np.mean(x)) / np.std(x)

#%%
reload(isc_utils)
# Invert if needed, then rescale to [0, 1]
def rescale_to_unit(x):
    x = np.array(x)
    x = -x  # Optional: only for analgesia if more negative = more modulation
    return (x - x.min()) / (x.max() - x.min())

Y['inverse_change_ANA'] = rescale_to_unit(Y['raw_change_ANA'])

y_conditions = ['SHSS_score', 'raw_change_ANA', 'raw_change_HYPER', 'total_chge_pain_hypAna', 'inverse_change_ANA']

def plot_simmat_behav(simmat, title,title_id = 'SHSS', y_name = 'SHSS_score', cmap= 'RdBu_r'):
    plt.figure(figsize=(12, 10))
    plt.imshow(simmat, vmin= -np.max(simmat), vmax=np.max(simmat), cmap=cmap)
    plt.colorbar(label='Similarity')
    plt.title(title, fontsize=24)
    plt.xlabel(f'Subject ranked on {y_name}', fontsize=20)
    plt.ylabel(f'Subject ranked on {y_name}', fontsize=20)
    plt.show()

def plot_hist(data, title='Histogram', xlabel='Values', ylabel='Frequency'):
    plt.figure(figsize=(10, 6))
    sns.histplot(data, bins=30, kde=True)
    plt.title(title, fontsize=20)
    plt.xlabel(xlabel, fontsize=16)
    plt.ylabel(ylabel, fontsize=16)
    plt.grid(True)
    plt.show()

#%%
unranked_simmat ={}
unranked_simmat_vec = {}
ranked_simmat = {}
ranked_simmat_vec = {}

for metric in ['euclidean', 'annak']:
    unranked_simmat[metric] = {}
    unranked_simmat_vec[metric] = {}
    ranked_simmat[metric] = {}
    ranked_simmat_vec[metric] = {}
    for y_name in y_conditions:
        
        y_native = np.array(Y[y_name].values, dtype=float)
        y = z_score(y_native)
        sim_behav = isc_utils.compute_behav_similarity(y, metric=metric, vectorize=False)
        sim_behav_vec = isc_utils.compute_behav_similarity(y, metric=metric, vectorize=True)
        
        unranked_simmat[metric][y_name] = sim_behav
        unranked_simmat_vec[metric][y_name] = sim_behav_vec

        if y_name == 'SHSS_score':
            shss_ranks = np.argsort(y)         # gives indices that sort behavior low → high
        
        sim_ranked = sim_behav[np.ix_(shss_ranks, shss_ranks)]
        ranked_simmat[metric][y_name] = sim_ranked
        ranked_simmat_vec[metric][y_name] = sim_behav_vec[shss_ranks]

        title = f'{metric}-based {y_name} similarity matrix'
        plot_simmat_behav(sim_ranked, title = title, y_name = y_name)
        plot_hist(y_native, title=f'Histogram of {y_name}', xlabel=y_name, ylabel='Frequency')

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

plot_simmat_behav(cosine_sim_sugg, title = 'Cosine', title_id = 'SUGG', y_name = 'SHSS_score', cmap= 'coolwarm')
plot_simmat_behav(cosine_sim_pain, title = 'Cosine', title_id = 'PAIN', y_name = 'raw_change_HYPER', cmap= 'coolwarm')


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
#LOOK AT BEHAVIORAL DATA AND SCATTER PLOTs
#----------------------

#jointplot beteween SHSS and pain columns
all_pain_var = ['raw_change_HYPER', 'raw_change_ANA', 'total_chge_pain_hypAna']
x_names = {'raw_change_HYPER': 'Hyperalgesia > Neutral Pain',
           'raw_change_ANA': 'Analgesia > Neutral Pain',
           'total_chge_pain_hypAna': 'Total Change in Pain'}
for var in all_pain_var:
    x_name = x_names[var]
    visu_utils.jointplot(np.array(Y[var]).astype(float),np.array(Y['SHSS_score']).astype(float), 
                        x_label=x_name, y_label='SHSS score',
                        title=None,
                        )
#%% 
#RSA between pain and SHSS simmat 
for model in ['euclidean', 'annak']:
    print(f"Model: {model}")
    shss_simmat_vec= unranked_simmat_vec[model]['SHSS_score']
    pain_simmat_vec = unranked_simmat_vec[model]['total_chge_pain_hypAna']

    r, p, dist = isc_utils.matrix_permutation(shss_simmat_vec, pain_simmat_vec, n_permute=10000, metric="spearman", how="upper", tail=1,n_jobs = 32, return_perms=True)
    print(f"Spearman correlation between SHSS and pain similarity matrices: {r:.4f}, p-value: {p:.4f}")

#%% K MEAN
from sklearn.cluster import KMeans

# Vectorize the total_chge_pain_hypAna column as a numpy array and z-score it
vec_zscore_change_pain = z_score(np.array(Y['total_chge_pain_hypAna'].values, dtype=float))[shss_ranks]
pain_scores = vec_zscore_change_pain.reshape(-1, 1)
pain_scores = StandardScaler().fit_transform(pain_scores)

# Cluster into 2 groups: low vs high pain modulation
kmeans = KMeans(n_clusters=3, random_state=42, n_init='auto')
clusters = kmeans.fit_predict(pain_scores)  # array of 0s and 1s

kmeans_indices = np.argsort(clusters)  # Group 0 first, then 1
shss_sim_sorted = unranked_simmat['annak']['SHSS_score'][np.ix_(shss_ranks, shss_ranks)]

cmap = 'RdBu_r'  # Choose a colormap
plt.figure(figsize=(10, 8))
plt.imshow(shss_sim_sorted, vmin= -np.max(shss_sim_sorted), vmax=np.max(shss_sim_sorted), cmap=cmap,aspect='auto', extent=[0, shss_sim_sorted.shape[1], shss_sim_sorted.shape[0], 0])
plt.colorbar(label='Similarity')

n_group0 = np.sum(clusters == 0)
plt.axhline(n_group0, color='gray', linewidth=4)
plt.axvline(n_group0, color='gray', linewidth=4)
n_group1 = np.sum(clusters == 1)
plt.axhline(n_group0 + n_group1, color='white', linewidth=5)
plt.axvline(n_group0 + n_group1, color='white', linewidth=5)

plt.title('SHSS Similarity Matrix (Sorted by Pain Modulation Clusters)', fontsize=14)
plt.xlabel('Subjects')
plt.ylabel('Subjects')
plt.tight_layout()
plt.show()

#%%
from matplotlib import cm
np.random.seed(42)  # For reproducibility

isc_matrix = np.array([
    [1.00, 0.65, 0.28, 0.15, -0.05, 0.05],
    [0.65, 1.00, 0.65, 0.85, 0.15, 0.22],
    [0.28, 0.65, 1.00, 0.75, 0.45, 0.70],
    [0.15, 0.85, 0.75, 1.00, 0.55, 0.35],
    [-0.05, 0.15, 0.45, 0.55, 1.00, 0.65],
    [0.05, 0.22, 0.70, 0.35, 0.65, 1.00]
])

# Plot the noisy ISC matrix
fig, ax = plt.subplots(figsize=(6, 6))
heatmap = ax.imshow(isc_matrix, cmap='RdBu_r', vmin=-1, vmax=1)
ax.axis('off')
plt.tight_layout()
plt.show()
plt.close()

#plot only the colorbar 
fig, ax = plt.subplots(figsize=(2, 6))
norm = plt.Normalize(-1, 1)
cbar = plt.colorbar(cm.ScalarMappable(norm=norm, cmap='RdBu_r'), cax=ax, orientation='vertical')

# Set custom ticks and format
cbar.set_ticks([-1, 0, 1])
cbar.ax.set_yticklabels(['-1', '0', '1'], fontsize=35)
# cbar.set_label('Similarity', rotation=270, labelpad=25, fontsize=55)

plt.tight_layout()
plt.show()

#%%
#================================
# SIMILARITY MATRICES - SAVE FIGURES / FINAL
#================================

reload(visu_utils)
cmap = 'RdBu_r' 

shss_simmat_annak = ranked_simmat['annak']['SHSS_score']
shss_simmat_euclidean = ranked_simmat['euclidean']['SHSS_score']
# Plot the similarity matrices
save_img = os.path.join(output_figures, 'SHSS_simmat_annak.png')
visu_utils.plot_simmat_behav(shss_simmat_annak,x_label = 'Ranked Subjects',colorbar_name = 'Similarity (Pairwise mean)',  title=None, cmap=cmap, save_path=save_img)
# plot_simmat_behav(shss_simmat_euclidean, title='Euclidean-based SHSS similarity matrix', y_name='SHSS_score', cmap='coolwarm')

save_img = os.path.join(output_figures, 'SHSS_simmat_euclidean.png')
visu_utils.plot_simmat_behav(shss_simmat_euclidean, x_label = 'Ranked Subjects',colorbar_name = 'Similarity (Euclidean)', title=None, cmap=cmap, save_path  =save_img)

#%% 
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
SCHAEFER_ONLY = False

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
# VISU ISC + RSA 
#---------------
reload(visu_utils)
from nilearn import datasets, plotting
bg_mni =  datasets.load_mni152_template(resolution=1)

#VISU UNIVARIATE SHSS for run effect
RESULT_FILE = 'rsa_SHSS-behav_isc-sugg10000perm.pkl'
rsa_isc = isc_utils.load_pickle(os.path.join(MODEL_PATH, RESULT_FILE))
n_perm = 10000
model = 'annak'

# Load ISC for visu sim matrices
isc_pairwise = {}
for cond in ['ana_run', 'hyper_run']:
    isc_path= f'/data/rainville/dSutterlin/projects/ISC_hypnotic_suggestions/results/imaging/ISC/{model_sugg}/concat_suggs_1samp_boot/isc_results_{cond}_{n_perm}boot_pairWiseTrue.pkl'
    isc_pairwise[cond] = pd.DataFrame(isc_utils.load_pickle(isc_path)['isc'], columns=labels)

isc_pairwise_pain = {}
for cond in ['ana_run', 'hyper_run', 'ANA', 'HYPER']:
    if cond in ['ANA', 'HYPER', 'NANA', 'NHYPER']:
        pain_path =   f'/data/rainville/dSutterlin/projects/ISC_hypnotic_suggestions/results/imaging/ISC/{model_pain}/{cond}/isc_results_{cond}_{n_perm}boot_pairWiseTrue.pkl'
    else:
        pain_path = f'/data/rainville/dSutterlin/projects/ISC_hypnotic_suggestions/results/imaging/ISC/{model_pain}/concat_suggs_1samp_boot/isc_results_{cond}_{n_perm}boot_pairWiseTrue.pkl'
    isc_pairwise_pain[cond] = pd.DataFrame(isc_utils.load_pickle(pain_path)['isc'], columns=labels)

y = z_score(Y['SHSS_score'].values)
sim_mat_behav_matrix = isc_utils.compute_behav_similarity(y, metric='euclidean', vectorize=False)
sim_mat_behav_annak = isc_utils.compute_behav_similarity(y, metric='annak', vectorize=False)

views={}
tables = {'euclidian': {}, 'annak': {}}
rsa_dfs = {'euclidian': {}, 'annak': {}}
stats_imgs_sugg = {}

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

    
    title = f"IS-RSA : Hypnotic susceptibility ~ {cond} (FDR<.05)"

    if cond == 'hyper_run':
        cut_coords_plot = [-56]
    elif cond == 'ana_run':
        cut_coords_plot = (-54,0, 56) 
    else:
        cut_coords_plot = None
    
    title = f"IS-RSA : Hypnotic susceptibility ~ {cond} (FDR<.05)"
    title = None
    # === Visualize with your existing function ===
    rsa_img, rsa_thresh, sig_labels, _ = visu_utils.project_isc_to_brain_perm(
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
        color=cmap
    )
    stats_imgs_sugg[cond] = (rsa_img, rsa_thresh)

    if sig_labels.shape[0] > 0:
        # sig_labels = pd.DataFrame(sig_labels, columns=['ROI', 'Label', 'spearman_r', 'p_values', 'Coordinates (X,Y,Z)'])
        sig_labels.columns = ['ROI', 'Label', 'spearman_r', 'p_values', 'Coordinates (X,Y,Z)']
        sig_labels.to_csv(os.path.join(output_dir, f'{RESULT_FILE[:-13]}_{model}_{cond}_table-FDR.csv'), index=False)


    tables[model][cond] = sig_labels
    views[cond] = plotting.view_img(rsa_img, threshold=rsa_thresh, title=f"RSA suggestion - pain similarity {cond}", colorbar=True,symmetric_cmap=False, cmap = 'coolwarm')

# stats_imgs[0].to_filename(os.path.join(output_dir, f'{RESULT_FILE[:-13]}_{model}_ana_run.nii.gz'))
# stats_imgs[1].to_filename(os.path.join(output_dir, f'{RESULT_FILE[:-13]}_{model}_hyper_run.nii.gz'))

#%%
#SAVE BRAIN SLICES
reload(visu_utils)

#ANA_RUN RSA
COORDS = {
    'x': [-54,0,56]
}
map, thresh = stats_imgs_sugg['ana_run']
visu_utils.save_slices(map, 
                  COORDS,
                  output_figures,
                  img_id = 'ana_run_isc-sugg_shss_annak_FDR05',
                  cmap = cmap
)
# save colorbar
fig_path = os.path.join(output_figures, 'ana_run_isc-sugg_shss_annak_colorbar.png')
cb_ab = visu_utils.save_colorbar(map,
              cmap=cmap,
              outpath=fig_path,
              symmetric_cbar=True, # false because we dont keep negative coeff
              offset=0,
              n_ticks=5,
              transparent=True)
# HYPER_RUN RSA
COORDS = {
    'x': [-56],
    'z' : [20]
}
map, thresh = stats_imgs_sugg['hyper_run']
visu_utils.save_slices(map, 
                  COORDS,
                  output_figures,
                  img_id = 'hyper_run_isc-sugg_shss_annak_FDR05',
                  cmap = cmap
)
# save colorbar
fig_path = os.path.join(output_figures, 'hyper_run_isc-sugg_shss_annak_colorbar.png')
cb_ab = visu_utils.save_colorbar(map,
                cmap=cmap,
                outpath=fig_path,
                symmetric_cbar=True, # false because we dont keep negative coeff
                offset=0,
                n_ticks=5,
                transparent=True)
#%%
#============================
# Visualize ISC matrix in ROI
reload(visu_utils)
reload(isc_utils)
save_isc_mat = os.path.join(output_figures, 'ISC_matrices_sugg')
os.makedirs(save_isc_mat, exist_ok=True)

cmap = 'RdBu_r' 

for cond in ['ana_run', 'hyper_run']:

    sig_roi = tables['annak'][cond]

    # rsa_dfs['annak']
    roi_ranked_mat = {}
    for roi_idx, roi in zip(sig_roi['ROI'], sig_roi['Label']):
        
        isc_pairwise_roi = isc_pairwise[cond][roi]
        isc_mat = isc_utils.vector_to_isc_matrix(isc_pairwise_roi, diag=0)
        
        ranked_isc_mat = isc_mat[np.ix_(shss_ranks, shss_ranks)]
        roi_ranked_mat[roi] = ranked_isc_mat

        save_file_as = os.path.join(save_isc_mat, f'isc_mat_{cond}_{roi}.png')
        visu_utils.plot_simmat_isc(ranked_isc_mat, title = None, title_id = roi, x_label = 'Ranked Subjects', colorbar_name = 'Pearson r', cmap = cmap,
                                save_path=save_file_as,
                                dpi = 1000)
        
#%%
#===================
# PLOT IS-RSA ISC-PAIN ~ SHSS behav (univaritate)
#===================

#VISU UNIVARIATE SHSS for run effect
RESULT_FILE = 'rsa_SHSS-behav_isc-pain10000perm.pkl' # PAIN model
rsa_isc = isc_utils.load_pickle(os.path.join(MODEL_PATH, RESULT_FILE))
model = 'annak' # Euclidean N.S.

isc_pairwise_pain = {}
for cond in ['ana_run', 'hyper_run', 'ANA', 'HYPER']:
    if cond in ['ANA', 'HYPER', 'NANA', 'NHYPER']:
        pain_path =   f'/data/rainville/dSutterlin/projects/ISC_hypnotic_suggestions/results/imaging/ISC/{model_pain}/{cond}/isc_results_{cond}_{n_perm}boot_pairWiseTrue.pkl'
    else:
        pain_path = f'/data/rainville/dSutterlin/projects/ISC_hypnotic_suggestions/results/imaging/ISC/{model_pain}/concat_suggs_1samp_boot/isc_results_{cond}_{n_perm}boot_pairWiseTrue.pkl'
    isc_pairwise_pain[cond] = pd.DataFrame(isc_utils.load_pickle(pain_path)['isc'], columns=labels)

y = z_score(Y['SHSS_score'].values)
sim_mat_behav_matrix = isc_utils.compute_behav_similarity(y, metric='euclidean', vectorize=False)
sim_mat_behav_annak = isc_utils.compute_behav_similarity(y, metric='annak', vectorize=False)

views={}
tables = {'euclidian': {}, 'annak': {}}
rsa_dfs = {'euclidian': {}, 'annak': {}}
stats_imgs = {}

for cond in ['ana_run', 'hyper_run', 'ANA', 'HYPER'] : #, 'HYPER', 'ANA']: #, 'all_sugg', 'neutral']:

    print('cond', cond)
    rsa_df = rsa_isc[model][cond].sort_index(ascending=True) # to match the atlas labels
    rsa_dfs[model][cond] = rsa_df

    # === Prepare variables for projection ===
    correlations = rsa_df['spearman_r'].values
    p_values = rsa_df['p_values'].values
    roi_labels = rsa_df['ROI'].values  # assumes label matches atlas
    fdr_p = isc_utils.fdr(p_values, q=0.05)
    print(f'FDR threshold: {fdr_p:.4f}')

    
    title = f"IS-RSA : Hypnotic susceptibility ~ {cond} (FDR<.05)"

    title = None
    # === Visualize with your existing function ===
    rsa_img, rsa_thresh, sig_labels, _ = visu_utils.project_isc_to_brain_perm(
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
        cut_coords_plot=None, #(-52, -40, 34),
        color=cmap
    )
    stats_imgs[cond] = (rsa_img, rsa_thresh)

    if sig_labels.shape[0] > 0:
        # sig_labels = pd.DataFrame(sig_labels, columns=['ROI', 'Label', 'spearman_r', 'p_values', 'Coordinates (X,Y,Z)'])
        sig_labels.columns = ['ROI', 'Label', 'spearman_r', 'p_values', 'Coordinates (X,Y,Z)']
        sig_labels.to_csv(os.path.join(output_dir, f'{RESULT_FILE[:-13]}_{model}_{cond}_table-FDR.csv'), index=False)

    tables[model][cond] = sig_labels
    views[cond] = plotting.view_img(rsa_img, threshold=rsa_thresh, title=f"RSA suggestion - pain similarity {cond}", colorbar=True,symmetric_cmap=False, cmap = 'coolwarm')

# ana_run, ana FDR
# Hyper run, hyper  N.S.
#%%
# SAVE SCLICES IS-RSA PAIN-SHSS
COORDS = { # ANA : 'x': [-10,-22,-42]
    'x': [-6, ], #Ana_run,
    'y': [-66],
    'z' : [6]
}
map, thresh = stats_imgs['ana_run']
visu_utils.save_slices(map, 
                  COORDS,
                  output_figures,
                  img_id = 'ana_run_isc-PAIN_shss_annak_FDR05',
                  cmap = cmap
)
# save colorbar
fig_path = os.path.join(output_figures, 'ana_run_isc-PAIN_shss_annak_colorbar.png')
cb_ab = visu_utils.save_colorbar(map,
              cmap=cmap,
              outpath=fig_path,
              symmetric_cbar=True, # false because we dont keep negative coeff
              offset=0,
              n_ticks=5,
              transparent=True)
# HYPER_RUN RSA


#%%
# VISUALIZE ISC (PAIN) per FDR regions
COND = 'hyper_run'
sig_roi = tables['annak'][COND]

# rsa_dfs['annak']
roi_ranked_mat = {}
for roi_idx, roi in zip(sig_roi['ROI'], sig_roi['Label']):
    
    isc_pairwise_roi = isc_pairwise[COND][roi]
    isc_mat = isc_utils.vector_to_isc_matrix(isc_pairwise_roi, diag=0)
    
    ranked_isc_mat = isc_mat[np.ix_(shss_ranks, shss_ranks)]
    roi_ranked_mat[roi] = ranked_isc_mat

    save_file_as = os.path.join(save_isc_mat, f'isc_mat_{COND}_{roi}.png')
    visu_utils.plot_simmat_isc(ranked_isc_mat, title = None, title_id = roi, x_label = 'Ranked Subjects', colorbar_name = 'Pearson r', cmap = cmap,
                               save_path=save_file_as)

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

