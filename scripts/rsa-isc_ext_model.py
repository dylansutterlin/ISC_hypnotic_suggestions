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
import sys
import shutil

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

# Save script copy in results folder --> saved at the end if all ran well!
current_script_path = os.path.abspath(sys.argv[0])
script_name = os.path.basename(current_script_path)
copy_name = f"ran_with_{script_name}"
destination_path = os.path.join(save_path, copy_name)

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
    

coords = find_parcellation_cut_coords(labels_img=atlas)

df_labels_coords = pd.DataFrame(atlas_labels.items(), columns=['index', 'region'])
df_labels_coords['coords'] = [coords[i] for i in range(len(coords))]

#%%
# plot specific ROIs 

#%%
# #%% VISU
# from nilearn import plotting
# reload(visu_utils)

# rsa_path = '/data/rainville/dSutterlin/projects/ISC_hypnotic_suggestions/results/imaging/RSA/2025-05-21_23'
# # rsa_isc = isc_utils.load_pickle(os.path.join(save_path, f'rsa_isc_pain-behav_sugg-pain_{n_perm_rsa}perm.pkl'))
# rsa_isc = isc_utils.load_pickle('/data/rainville/dSutterlin/projects/ISC_hypnotic_suggestions/results/imaging/RSA/rsa_pain-behav_sugg-pain_10000perm.pkl')
# rsa_isc_sugg = isc_utils.load_pickle(os.path.join(rsa_path, 'rsa_cosine-sugg-behav_isc-sugg5000perm.pkl' ))

# rsa_isc_sugg = isc_utils.load_pickle('/data/rainville/dSutterlin/projects/ISC_hypnotic_suggestions/results/imaging/RSA/isc-RSA_ext_conds_sugg-pain/rsa_cosine-behav_isc-sugg10000perm.pkl')

# views={}
# for cond in ['HYPER', 'ANA']: #,'all_sugg', 'neutral']:

#     print('cond', cond)
#     rsa_df = rsa_isc_sugg['sugg'][cond].sort_index(ascending=True) # to match the atlas labels

#     # === Prepare variables for projection ===
#     correlations = rsa_df['spearman_r'].values
#     p_values = rsa_df['p_values'].values
#     roi_labels = rsa_df['ROI'].values  # assumes label matches atlas
#     fdr_p = isc_utils.fdr(p_values, q=0.05)
#     print(f'FDR threshold: {fdr_p:.4f}')


#     unc_p = 0.01
#     # === Visualize with your existing function ===
#     rsa_img, rsa_thresh, sig_labels = visu_utils.project_isc_to_brain_perm(
#         atlas_img=atlas,
#         isc_median=correlations,
#         atlas_labels=atlas_labels,
#         roi_coords = coords,
#         p_values=p_values,
#         p_threshold=fdr_p, #!!
#         title=None, #"RSA-ISC: Suggestion-Pain Similarity (FDR<.05)",
#         save_path=None,
#         show=True,
#         display_mode='x',
#         cut_coords_plot=None, #(-52, -40, 34),
#         color='Reds'
#     )
    
#     views[cond] = plotting.view_img(rsa_img, threshold=rsa_thresh, title=f"RSA suggestion - pain similarity {cond}", colorbar=True,symmetric_cmap=False, cmap = 'Reds')

    
# %%
#=================================
# IS-RSA for ISC-SUGGESTION  & ISC-PAIN

do_brain_sugg_with_pain = True
if do_brain_sugg_with_pain:

    # model_sugg = model_names['model1_sugg_200_run']
    # model_pain = model_names['model3_shock_200_run']

    rsa_per_cond = {}
    for cond in conditions:
        print(f'Running RSA for condition: {cond}')
        # sugg_path = f'/data/rainville/dSutterlin/projects/ISC_hypnotic_suggestions/results/imaging/ISC/{model_sugg}/concat_suggs_1samp_boot/isc_results_{cond}_{n_perm}boot_pairWiseTrue.pkl'
        # pain_path = f'/data/rainville/dSutterlin/projects/ISC_hypnotic_suggestions/results/imaging/ISC/{model_pain}/concat_suggs_1samp_boot/isc_results_{cond}_{n_perm}boot_pairWiseTrue.pkl'

        if cond in ['ANA', 'HYPER', 'NANA', 'NHYPER']:
            sugg_path =   f'/data/rainville/dSutterlin/projects/ISC_hypnotic_suggestions/results/imaging/ISC/{model_sugg}/{cond}/isc_results_{cond}_{n_perm}boot_pairWiseTrue.pkl'
            pain_path =   f'/data/rainville/dSutterlin/projects/ISC_hypnotic_suggestions/results/imaging/ISC/{model_pain}/{cond}/isc_results_{cond}_{n_perm}boot_pairWiseTrue.pkl'
        else:
            sugg_path = f'/data/rainville/dSutterlin/projects/ISC_hypnotic_suggestions/results/imaging/ISC/{model_sugg}/concat_suggs_1samp_boot/isc_results_{cond}_{n_perm}boot_pairWiseTrue.pkl'
            pain_path = f'/data/rainville/dSutterlin/projects/ISC_hypnotic_suggestions/results/imaging/ISC/{model_pain}/concat_suggs_1samp_boot/isc_results_{cond}_{n_perm}boot_pairWiseTrue.pkl'

        isc_sugg = pd.DataFrame(isc_utils.load_pickle(sugg_path)['isc'], columns=labels)
        isc_pain = pd.DataFrame(isc_utils.load_pickle(pain_path)['isc'], columns=labels)

        assert isc_sugg.shape == isc_pain.shape

        rsa_rows = [] # build df 
        for roi_idx, roi in tqdm(enumerate(isc_sugg.columns), total=len(isc_sugg.columns)):            
            sugg_vec = isc_sugg[roi].values
            pain_vec = isc_pain[roi].values
            r, p, dist = isc_utils.matrix_permutation(sugg_vec, pain_vec, n_permute=n_perm_rsa, metric="spearman", how="upper", tail=n_tail_rsa,n_jobs = n_jobs, return_perms=True)
        
            rsa_rows.append({
                'ROI' : roi_idx,
                'Labels': roi,
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

# #%% VISU
# from nilearn import plotting
# reload(visu_utils)

# # rsa_isc = isc_utils.load_pickle('/data/rainville/dSutterlin/projects/ISC_hypnotic_suggestions/results/imaging/RSA/isc-RSA_ext_conds_sugg-pain/rsa_isc-sugg_isc-pain_10000perm.pkl')
# rsa_isc = isc_utils.load_pickle('/data/rainville/dSutterlin/projects/ISC_hypnotic_suggestions/results/imaging/RSA/isc-RSA_ext_conds_sugg-pain/rsa_cosine-behav_isc-sugg10000perm.pkl')
# rsa_isc = isc_utils.load_pickle('/data/rainville/dSutterlin/projects/ISC_hypnotic_suggestions/results/imaging/RSA/2025-05-21_23/rsa_isc-sugg_isc-pain_5000perm.pkl')
# rsa_isc = isc_utils.load_pickle('/data/rainville/dSutterlin/projects/ISC_hypnotic_suggestions/results/imaging/RSA/isc-RSA_ext_conds_sugg-pain/rsa_isc-sugg_isc-pain_10000perm.pkl')

# views={}
# tables = {}
# for cond in ['HYPER', 'ANA', 'ana_run', 'hyper_run']: #, 'ANA', 'all_sugg', 'neutral']:

#     print('cond', cond)
#     rsa_df = rsa_isc[cond].sort_index(ascending=True) # to match the atlas labels

#     # === Prepare variables for projection ===
#     correlations = rsa_df['spearman_r'].values
#     p_values = rsa_df['p_values'].values
#     roi_labels = rsa_df['ROI'].values  # assumes label matches atlas
#     fdr_p = isc_utils.fdr(p_values, q=0.05)
#     print(f'FDR threshold: {fdr_p:.4f}')

#     unc_p = 0.01
#     # === Visualize with your existing function ===
#     rsa_img, rsa_thresh, sig_labels, _ = visu_utils.project_isc_to_brain_perm(
#         atlas_img=atlas,
#         isc_median=correlations,
#         atlas_labels=atlas_labels,
#         roi_coords = coords,
#         p_values=p_values,
#         p_threshold=fdr_p, #!!
#         title=None, #"RSA-ISC: Suggestion-Pain Similarity (FDR<.05)",
#         save_path=None,
#         show=True,
#         display_mode='x',
#         cut_coords_plot=None, #(-52, -40, 34),
#         color='Reds'
#     )
#     tables[cond] = sig_labels
#     views[cond] = plotting.view_img(rsa_img,bg_img = bg_mni, threshold=rsa_thresh, title=f"RSA suggestion - pain similarity {cond}", colorbar=True,symmetric_cmap=False, cmap = 'Reds')

# %%

Specific_test = False

if Specific_test is False:
    reload(isc_utils)
    from tqdm import tqdm
    #=================================
    # IS-RSA for SUGGESTION (behav suggestion + behav pain)
    #=================================

    # check if all cond are in isc_results
    save_to = os.path.join(save_path, f'rsa_cosine-behav_isc-sugg{n_perm_rsa}perm.pkl')
    pairwise_behav = {'sugg': cosine_vec_sugg, 'pain': cosine_vec_pain}
    rsa_behav_per_cond = {}

    for domain in pairwise_behav.keys():
        rsa_behav_per_cond[domain] = {}
        # conditions = ['ANA', 'HYPER', 'all_sugg'] # 'neutral']
        for cond in conditions:
            if cond in ['ANA', 'HYPER', 'NANA', 'NHYPER']:
                pain_path =   f'/data/rainville/dSutterlin/projects/ISC_hypnotic_suggestions/results/imaging/ISC/{model_sugg}/{cond}/isc_results_{cond}_{n_perm}boot_pairWiseTrue.pkl'
            else:
                pain_path = f'/data/rainville/dSutterlin/projects/ISC_hypnotic_suggestions/results/imaging/ISC/{model_sugg}/concat_suggs_1samp_boot/isc_results_{cond}_{n_perm}boot_pairWiseTrue.pkl'

            vec_behav_sim = pairwise_behav[domain] #specify if sugg or pain
            isc_pairwise = pd.DataFrame(isc_utils.load_pickle(pain_path)['isc'], columns=labels)

            rsa_rows = [] # build df 
            for roi_idx, roi in tqdm(enumerate(isc_pairwise.columns), total=len(isc_pairwise.columns)):
                isc_roi = isc_pairwise[roi].values
                r, p, dist = isc_utils.matrix_permutation(vec_behav_sim, isc_roi, n_permute=n_perm_rsa, metric="spearman", how="upper", tail=n_tail_rsa, n_jobs = n_jobs, return_perms=True)
                
                rsa_rows.append({
                    'ROI' : roi_idx,
                    'Labels': roi,
                    'spearman_r': r,
                    'p_values': round(p, 5),
                    'x': coords[roi_idx][0],
                    'y': coords[roi_idx][1],
                    'z': coords[roi_idx][2]
                })

            rsa_behav_per_cond[domain][cond] = pd.DataFrame(rsa_rows).sort_values(by='spearman_r', ascending=False)
            print(f'Condition: {cond}')
            print('Max r, mean and fdr', rsa_behav_per_cond[domain][cond]['spearman_r'].max(), rsa_behav_per_cond[domain][cond]['spearman_r'].mean(), isc_utils.fdr(rsa_behav_per_cond[domain][cond]['p_values'].to_numpy()))

    isc_utils.save_data(save_to, rsa_behav_per_cond)
    print(f'Saved RSA results to {save_to}')



    #%%
    reload(isc_utils)
    from tqdm import tqdm
    #=================================
    # IS-RSA for SUGGESTION (UNIVARIATE- SHSS) behav suggestion + behav pain)
    #=================================

    # check if all cond are in isc_results
    save_to = os.path.join(save_path, f'rsa_SHSS-behav_isc-sugg{n_perm_rsa}perm.pkl')
    pairwise_behav = {'euclidian': sim_behav_vec, 'annak': sim_behav_vec_annak}
    rsa_behav_per_cond = {}

    for domain in pairwise_behav.keys():
        rsa_behav_per_cond[domain] = {}
        # conditions = ['ANA', 'HYPER', 'all_sugg'] # 'neutral']
        for cond in conditions:
            if cond in ['ANA', 'HYPER', 'NANA', 'NHYPER']:
                pain_path =   f'/data/rainville/dSutterlin/projects/ISC_hypnotic_suggestions/results/imaging/ISC/{model_sugg}/{cond}/isc_results_{cond}_{n_perm}boot_pairWiseTrue.pkl'
            else:
                pain_path = f'/data/rainville/dSutterlin/projects/ISC_hypnotic_suggestions/results/imaging/ISC/{model_sugg}/concat_suggs_1samp_boot/isc_results_{cond}_{n_perm}boot_pairWiseTrue.pkl'

            vec_behav_sim = pairwise_behav[domain] #specify if sugg or pain
            isc_pairwise = pd.DataFrame(isc_utils.load_pickle(pain_path)['isc'], columns=labels)

            rsa_rows = [] # build df 
            for roi_idx, roi in tqdm(enumerate(isc_pairwise.columns), total=len(isc_pairwise.columns)):
                isc_roi = isc_pairwise[roi].values
                r, p, dist = isc_utils.matrix_permutation(vec_behav_sim, isc_roi, n_permute=n_perm_rsa, metric="spearman", how="upper", tail=n_tail_rsa,n_jobs = n_jobs, return_perms=True)
                
                rsa_rows.append({
                    'ROI' : roi_idx,
                    'Labels': roi,
                    'spearman_r': r,
                    'p_values': round(p, 5),
                    'x': coords[roi_idx][0],
                    'y': coords[roi_idx][1],
                    'z': coords[roi_idx][2]
                })

            rsa_behav_per_cond[domain][cond] = pd.DataFrame(rsa_rows).sort_values(by='spearman_r', ascending=False)
            print(f'Condition: {cond}')
            print('Max r, mean and fdr', rsa_behav_per_cond[domain][cond]['spearman_r'].max(), rsa_behav_per_cond[domain][cond]['spearman_r'].mean(), isc_utils.fdr(rsa_behav_per_cond[domain][cond]['p_values'].to_numpy()))

    isc_utils.save_data(save_to, rsa_behav_per_cond)
    print(f'Saved RSA results to {save_to}')

#%%
#=================================
# IS-RSA for PAIN UNIVARIATE
#=================================
print('======================================')
print('Running RSA for PAIN univariate (SHSS)')

# check if all cond are in isc_results
save_to = os.path.join(save_path, f'rsa_SHSS-behav_isc-pain{n_perm_rsa}perm.pkl')
pairwise_behav = {'euclidian': sim_behav_vec, 'annak': sim_behav_vec_annak}
rsa_behav_per_cond = {}

for domain in pairwise_behav.keys():
    rsa_behav_per_cond[domain] = {}
    print(f'Running RSA for domain: {domain}')

    for cond in conditions:
        if cond in ['ANA', 'HYPER', 'NANA', 'NHYPER']:
            pain_path =   f'/data/rainville/dSutterlin/projects/ISC_hypnotic_suggestions/results/imaging/ISC/{model_pain}/{cond}/isc_results_{cond}_{n_perm}boot_pairWiseTrue.pkl'
        else:
            pain_path = f'/data/rainville/dSutterlin/projects/ISC_hypnotic_suggestions/results/imaging/ISC/{model_pain}/concat_suggs_1samp_boot/isc_results_{cond}_{n_perm}boot_pairWiseTrue.pkl'

        vec_behav_sim = pairwise_behav[domain] #specify if sugg or pain
        isc_pairwise = pd.DataFrame(isc_utils.load_pickle(pain_path)['isc'], columns=labels)

        rsa_rows = [] # build df 
        for roi_idx, roi in enumerate(isc_pairwise.columns):
            isc_roi = isc_pairwise[roi].values
            r, p, dist = isc_utils.matrix_permutation(vec_behav_sim, isc_roi, n_permute=n_perm_rsa, metric="spearman", how="upper", tail=n_tail_rsa, n_jobs = n_jobs, return_perms=True)
            
            rsa_rows.append({
                'ROI' : roi_idx,
                'Labels': roi,
                'spearman_r': r,
                'p_values': round(p, 5),
                'x': coords[roi_idx][0],
                'y': coords[roi_idx][1],
                'z': coords[roi_idx][2]
            })

        rsa_behav_per_cond[domain][cond] = pd.DataFrame(rsa_rows).sort_values(by='spearman_r', ascending=False)
        print(f'Condition: {cond}')
        print('Max r, mean and fdr', rsa_behav_per_cond[domain][cond]['spearman_r'].max(), rsa_behav_per_cond[domain][cond]['spearman_r'].mean(), isc_utils.fdr(rsa_behav_per_cond[domain][cond]['p_values'].to_numpy()))

isc_utils.save_data(save_to, rsa_behav_per_cond)
print(f'Saved RSA results to {save_to}')

#=================================
# IS-RSA for PAIN UNIVARIATE
#=================================
print('======================================')
print('Running RSA for PAIN univariate (Change PAIN)')

# check if all cond are in isc_results
save_to = os.path.join(save_path, f'rsa_SHSS-behav_isc-pain{n_perm_rsa}perm.pkl')
pairwise_behav = {'euclidian': sim_behav_pain_vec, 'annak': sim_behav_pain_vec_annak}
rsa_behav_per_cond = {}

for domain in pairwise_behav.keys():
    rsa_behav_per_cond[domain] = {}
    print(f'Running RSA for domain: {domain}')

    for cond in conditions:
        if cond in ['ANA', 'HYPER', 'NANA', 'NHYPER']:
            pain_path =   f'/data/rainville/dSutterlin/projects/ISC_hypnotic_suggestions/results/imaging/ISC/{model_pain}/{cond}/isc_results_{cond}_{n_perm}boot_pairWiseTrue.pkl'
        else:
            pain_path = f'/data/rainville/dSutterlin/projects/ISC_hypnotic_suggestions/results/imaging/ISC/{model_pain}/concat_suggs_1samp_boot/isc_results_{cond}_{n_perm}boot_pairWiseTrue.pkl'

        vec_behav_sim = pairwise_behav[domain] #specify if sugg or pain
        isc_pairwise = pd.DataFrame(isc_utils.load_pickle(pain_path)['isc'], columns=labels)

        rsa_rows = [] # build df 
        for roi_idx, roi in enumerate(isc_pairwise.columns):
            isc_roi = isc_pairwise[roi].values
            r, p, dist = isc_utils.matrix_permutation(vec_behav_sim, isc_roi, n_permute=n_perm_rsa, metric="spearman", how="upper", tail=n_tail_rsa, n_jobs = n_jobs, return_perms=True)
            
            rsa_rows.append({
                'ROI' : roi_idx,
                'Labels': roi,
                'spearman_r': r,
                'p_values': round(p, 5),
                'x': coords[roi_idx][0],
                'y': coords[roi_idx][1],
                'z': coords[roi_idx][2]
            })

        rsa_behav_per_cond[domain][cond] = pd.DataFrame(rsa_rows).sort_values(by='spearman_r', ascending=False)
        print(f'Condition: {cond}')
        print('Max r, mean and fdr', rsa_behav_per_cond[domain][cond]['spearman_r'].max(), rsa_behav_per_cond[domain][cond]['spearman_r'].mean(), isc_utils.fdr(rsa_behav_per_cond[domain][cond]['p_values'].to_numpy()))

isc_utils.save_data(save_to, rsa_behav_per_cond)
print(f'Saved RSA results to {save_to}')


#%%
#=================================
# IS-RSA for PAIN COSINE suggestion behavrioal
#=================================
print('======================================')
print('Running RSA for PAIN ~ cosine (hypnosis)')

# check if all cond are in isc_results
save_to = os.path.join(save_path, f'rsa_cosine-behav_isc-pain{n_perm_rsa}perm.pkl')
pairwise_behav = {'cosine' : cosine_vec_pain}
rsa_behav_per_cond = {}


for domain in pairwise_behav.keys():
    rsa_behav_per_cond[domain] = {}

    for cond in conditions:
        if cond in ['ANA', 'HYPER', 'NANA', 'NHYPER']:
            pain_path =   f'/data/rainville/dSutterlin/projects/ISC_hypnotic_suggestions/results/imaging/ISC/{model_pain}/{cond}/isc_results_{cond}_{n_perm}boot_pairWiseTrue.pkl'
        else:
            pain_path = f'/data/rainville/dSutterlin/projects/ISC_hypnotic_suggestions/results/imaging/ISC/{model_pain}/concat_suggs_1samp_boot/isc_results_{cond}_{n_perm}boot_pairWiseTrue.pkl'

        vec_behav_sim = pairwise_behav[domain] #specify if sugg or pain
        isc_pairwise = pd.DataFrame(isc_utils.load_pickle(pain_path)['isc'], columns=labels)

        rsa_rows = [] # build df 
        for roi_idx, roi in enumerate(isc_pairwise.columns):
            isc_roi = isc_pairwise[roi].values
            r, p, dist = isc_utils.matrix_permutation(vec_behav_sim, isc_roi, n_permute=n_perm_rsa, metric="spearman", how="upper", tail=n_tail_rsa, n_jobs = n_jobs, return_perms=True)
            
            rsa_rows.append({
                'ROI' : roi_idx,
                'Labels': roi,
                'spearman_r': r,
                'p_values': round(p, 5),
                'x': coords[roi_idx][0],
                'y': coords[roi_idx][1],
                'z': coords[roi_idx][2]
            })

        rsa_behav_per_cond[domain][cond] = pd.DataFrame(rsa_rows).sort_values(by='spearman_r', ascending=False)
        print(f'Condition: {cond}')
        print('Max r, mean and fdr', rsa_behav_per_cond[domain][cond]['spearman_r'].max(), rsa_behav_per_cond[domain][cond]['spearman_r'].mean(), isc_utils.fdr(rsa_behav_per_cond[domain][cond]['p_values'].to_numpy()))

isc_utils.save_data(save_to, rsa_behav_per_cond)
print(f'Saved RSA results to {save_to}')


#%%

try:
    shutil.copy(current_script_path, destination_path)
    print(f"Script copied to: {destination_path}")
except Exception as e:
    print(f"Failed to copy script: {e}")


print(f'Saved all RSA results to {save_path}')
print('Done with all RSA!')
# %%
