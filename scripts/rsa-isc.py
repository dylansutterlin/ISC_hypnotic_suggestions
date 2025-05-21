# %% RSA between suggestion and pain ISC structures
import os
import numpy as np
import pandas as pd

from nilearn.datasets import fetch_atlas_schaefer_2018
from nilearn.maskers import NiftiLabelsMasker
from nilearn.plotting import find_parcellation_cut_coords
import nibabel as nib

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

save_path = '/data/rainville/dSutterlin/projects/ISC_hypnotic_suggestions/results/imaging/RSA'

conditions = ['ANA', 'HYPER'] #all_sugg', 'modulation'] # 'neutral']
sim_model = 'euclidean'
n_perm = setup['n_perm']
n_perm_rsa = 10000
y_name = 'SHSS_score'

#Behavioral 
behav_df = pd.read_csv(
    os.path.join(setup['project_dir'], f'results/behavioral_data_cleaned.csv'),
    index_col=0
)
behav_df.index.name = 'subjects'
behav_df = behav_df.sort_index()

# Behav similarity
y = behav_df[y_name].values
y = (y - np.mean(y)) / np.std(y)
sim_behav = isc_utils.compute_behav_similarity(y, metric=sim_model)

#Atlas
atlas_data = fetch_atlas_schaefer_2018(n_rois = 200, resolution_mm=2)
atlas = nib.load(atlas_data['maps'])
atlas_path = atlas_data['maps'] #os.path.join(project_dir,os.path.join(project_dir, 'masks', 'k50_2mm', '*.nii*'))
# labels_bytes = list(atlas_data['labels'])
labels = [str(label, 'utf-8') if isinstance(label, bytes) else str(label) for label in atlas_data['labels']]
atlas_masker = NiftiLabelsMasker(labels_img=atlas_path, standardize=False)
atlas_masker.fit()

coords = find_parcellation_cut_coords(labels_img=atlas)

#%% VISU
from nilearn import plotting
rsa_path = '/data/rainville/dSutterlin/projects/ISC_hypnotic_suggestions/results/imaging/RSA/'
# rsa_isc = isc_utils.load_pickle(os.path.join(save_path, f'rsa_isc_pain-behav_sugg-pain_{n_perm_rsa}perm.pkl'))
rsa_isc = isc_utils.load_pickle('/data/rainville/dSutterlin/projects/ISC_hypnotic_suggestions/results/imaging/RSA/rsa_pain-behav_sugg-pain_10000perm.pkl')
rsa_isc = isc_utils.load_pickle(os.path.join(rsa_path, 'rsa_pain-behav_sugg-pain_10000perm.pkl' ))
views={}
for cond in ['HYPER']:
    print('cond', cond)
    rsa_df = rsa_isc[cond]

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
        atlas_path=atlas_path,
        isc_median=correlations,
        atlas_labels=labels,
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
# RSA
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
        r, p, dist = isc_utils.matrix_permutation(sugg_vec, pain_vec, n_permute=n_perm_rsa, metric="spearman", how="upper", tail=2, return_perms=True)
    
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


save_to = os.path.join(save_path, f'rsa_isc_sugg-pain_{n_perm_rsa}perm.pkl')
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
#         r, p, dist = isc_utils.matrix_permutation(sugg_vec, pain_vec, n_permute=n_perm_rsa, metric="spearman", how="upper", tail=2, return_perms=True)
        
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
#Behavrioral RSA

    # check if all cond are in isc_results
rsa_behav_per_cond = {}
conditions = ['ANA', 'HYPER', 'modulation', 'all_sugg'] # 'neutral']
for cond in conditions:
    if cond in ['ANA', 'HYPER']:
        pain_path =   f'/data/rainville/dSutterlin/projects/ISC_hypnotic_suggestions/results/imaging/ISC/{model_pain}/{cond}/isc_results_{cond}_{n_perm}boot_pairWiseTrue.pkl'
    else:
        pain_path = f'/data/rainville/dSutterlin/projects/ISC_hypnotic_suggestions/results/imaging/ISC/{model_pain}/concat_suggs_1samp_boot/isc_results_{cond}_{n_perm}boot_pairWiseTrue.pkl'

    comparison = {'ANA' : 'raw_change_ANA',
                  'HYPER' : 'raw_change_HYPER',
                  'modulation' : 'total_chge_pain_hypAna',
                  'neutral' : 'total_chge_pain_hypAna',
                  'all_sugg' : 'total_chge_pain_hypAna'
        }   
    # Behav similarity

    y_name = comparison[cond]
    y = behav_df[y_name].values
    y = (y - np.mean(y)) / np.std(y)
    vec_behav = isc_utils.compute_behav_similarity(y, metric='euclidean', vectorize=True)

    isc_pain = pd.DataFrame(isc_utils.load_pickle(pain_path)['isc'], columns=labels)

    rsa_rows = [] # build df 
    for roi_idx, roi in enumerate(isc_pain.columns):
        pain_vec = isc_pain[roi].values
        r, p, dist = isc_utils.matrix_permutation(vec_behav, pain_vec, n_permute=n_perm_rsa, metric="spearman", how="upper", tail=2, return_perms=True)
        
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

save_to = os.path.join(save_path, f'rsa_pain-behav_sugg-pain_{n_perm_rsa}perm.pkl')
isc_utils.save_data(save_to, rsa_behav_per_cond)
print(f'Saved RSA results to {save_to}')


#%%
print('Done with all RSA!')
# %%
