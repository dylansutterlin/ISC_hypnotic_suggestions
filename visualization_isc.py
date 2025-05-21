# %%
from nilearn.image import math_img
from nilearn.plotting import plot_stat_map
import os
import seaborn as sns
from nilearn import plotting
from nilearn.plotting import view_img 

import time
from importlib import reload
import nibabel as nib
import numpy as np
import pandas as pd
import json
from nilearn.image import concat_imgs
from brainiak.isc import isc, bootstrap_isc, permutation_isc, compute_summary_statistic, phaseshift_isc
from nilearn.maskers import MultiNiftiMapsMasker, MultiNiftiMasker
from nilearn.datasets import fetch_atlas_schaefer_2018
from sklearn.utils import Bunch
from nilearn.plotting import view_img_on_surf
from nilearn.maskers import NiftiLabelsMasker

from nilearn.plotting import find_parcellation_cut_coords

import src.isc_utils as isc_utils
import src.visu_utils as visu_utils
reload(visu_utils)
reload(isc_utils)


# %% Load the data
model_names = {
    'model1_sugg': 'model1-ext-conds_sugg_23-sub_schafer-200-2mm_mask-whole-brain_pairWise-True_preproc_reg-mvmnt-True-8',
    'model1-6': 'model1_sugg_23-sub_schafer-200-2mm_mask-whole-brain_pairWise-True_preproc_reg-mvmnt-True-6',
    'model2_sugg-sensaas': 'model1-LNG_sugg_23-sub_SENSAAS_mask-whole-brain_pairWise-True_preproc_reg-mvmnt-True-8',
    'model2_shock_loo': 'model2_shock_23-sub_schafer-200-2mm_mask-whole-brain_pairWise-False_preproc_reg-mvmnt-True-8',
    'model3-shock': 'model3_shock_23-sub_schafer-200-2mm_mask-whole-brain_pairWise-True_preproc_reg-mvmnt-True-8',
    'model1_mean': 'model4-mean_sugg_23-sub_schafer-200-2mm_mask-whole-brain_pairWise-True_preproc_reg-mvmnt-True-8',
    'model5_isfc_sugg' :'model5-isfc_sugg_23-sub_schafer-200-2mm_mask-lanA800_pairWise-True_preproc_reg-mvmnt-True-8',
    'model5_isfc_shock' : 'model5-isfc_shock_23-sub_schafer-200-2mm_mask-lanA800_pairWise-True_preproc_reg-mvmnt-True-8',
    '9 avril ...' : ' ',
    'single_trial' : 'model_single-trial_sugg_23-sub_schafer-200-2mm_mask-lanA800_pairWise-True_preproc_reg-mvmnt-True-8',
    'single_trial_wb' : 'model_single-trial-wb_sugg_23-sub_schafer-200-2mm_mask-whole-brain_pairWise-True_preproc_reg-mvmnt-True-8',
    'model5_sugg' : 'model5-with-subcort_sugg_23-sub_schafer-200-2mm_mask-whole-brain_pairWise-True_preproc_reg-mvmnt-True-8'
}


model_is = 'model3-shock'
model_is = 'single_trial_wb'
model_is = 'model2_sugg-sensaas'
model_is = 'model1_sugg'
model_is = 'model5_sugg'

ref_preproc_models = { 'model2' : model_names['model1_sugg']}
ref_model_name = ref_preproc_models['model2']

project_dir = "/data/rainville/dSutterlin/projects/ISC_hypnotic_suggestions"

color_isc = 'Reds' 
# base_path = "/data/rainville/dSutterlin/projects/ISC_hypnotic_suggestions/results/data/test_data_sugg_3sub"
preproc_model_data = '23subjects_zscore_sample_detrend_FWHM6_low-pass428_10-12-24/suggestion_blocks_concat_4D_23sub'
base_path = os.path.join(project_dir, 'results/imaging/preproc_data', preproc_model_data)
model_name = model_names[model_is]
# preproc_model_name =  model_names['model1_sugg'] #'model3_shock_23-sub_schafer-200-2mm_mask-whole-brain_pairWise-True_preproc_reg-mvmnt-True-8'

results_dir = os.path.join(project_dir, f'results/imaging/ISC/{model_name}')
setup = isc_utils.load_json(os.path.join(results_dir, "setup_parameters.json"))

#create visualization dir.
save_visu = os.path.join(results_dir, 'visualization')
os.makedirs(save_visu, exist_ok=True)

parcel_name = setup['atlas_name']
do_pairWise = setup['do_pairwise']
n_boot = setup['n_boot']

post_hoc_dir = os.path.join(results_dir, 'post_hoc_results')
os.makedirs(post_hoc_dir, exist_ok=True)

# %% 
# all_results_paths = utils.load_json(os.path.join(results_dir, "result_paths.json"))
# atlas_name = 'Difumo256' # change to setup['atlas_name']
# n_sub = setup['n_sub']
shaeffer_only = False
if shaeffer_only:

    atlas_data = fetch_atlas_schaefer_2018(n_rois = 200, resolution_mm=2)
    atlas = nib.load(atlas_data['maps'])
    atlas_path = atlas_data['maps'] #os.path.join(project_dir,os.path.join(project_dir, 'masks', 'k50_2mm', '*.nii*'))
    # labels_bytes = list(atlas_data['labels'])
    full_labels = [str(label, 'utf-8') if isinstance(label, bytes) else str(label) for label in atlas_data['labels']]
    roi_index = [full_labels.index(lbl)+1 for lbl in full_labels]
    id_labels_dct = dict(zip(roi_index, full_labels))

#---------------
#up to date scheaffer + subcortical regions
  
atlas = nib.load('/data/rainville/dSutterlin/projects/ISC_hypnotic_suggestions/masks/Tian2020_schaeffer200_subcortical16/combined_schaefer200_tian16_DSG.nii.gz')
id_labels_dct = isc_utils.load_json('/data/rainville/dSutterlin/projects/ISC_hypnotic_suggestions/masks/Tian2020_schaeffer200_subcortical16/roi_labels_dict_DSG.json')
#remove background 
id_labels_dct.pop('0')
full_labels = list(id_labels_dct.values())
roi_index = list(id_labels_dct.keys())

atlas_masker = NiftiLabelsMasker(labels_img=atlas, labels=full_labels, standardize=False)
atlas_masker.fit()

# # Sensaas 
# # condition_target = ['all_sugg', 'A'
# atlas = nib.load('/data/rainville/dSutterlin/projects/ISC_hypnotic_suggestions/masks/sensaas/SENSAAS_MNI_ICBM_152_2mm.nii')
# atlas_data = pd.read_csv('/data/rainville/dSutterlin/projects/ISC_hypnotic_suggestions/masks/sensaas/SENSAAS_description.csv')
# # atlas = qc_utils.resamp_to_img_mask(atlas_native, ref_img)
# # view_img(atlas, threshold=0.5, title='SENSAAS atlas')
# #sensaas ids and labels
# roi_index = atlas_data['Index'].values
# atlas_data['annot_abbreviation'] = atlas_data['Abbreviation'] + '_' + atlas_data['Hemisphere']
# full_labels = atlas_data['annot_abbreviation'].values
# id_labels_dct = dict(zip(atlas_data['Index'], atlas_data['annot_abbreviation']))

if 'SENSAAS' in setup['atlas_name']:
    roi_coords = list(zip(
    atlas_data['Xmm'].astype(float),
    atlas_data['Ymm'].astype(float),
    atlas_data['Zmm'].astype(float)
    ))
else:
    roi_coords = find_parcellation_cut_coords(labels_img=atlas)


#%%
from nilearn.plotting import plot_roi
# Plot the atlas with labels
plot_roi(atlas, title='SENSAAS Atlas', colorbar=True, cmap='Paired')
#%%

isc_results_roi = {}

behav_df = pd.read_csv(
    os.path.join(setup['project_dir'], f'results/behavioral_data_cleaned.csv'),
    index_col=0
)
behav_df.index.name = 'subjects'
behav_df = behav_df.sort_index()

#%%

xlsx_path = r'/data/rainville/dSutterlin/projects/ISC_hypnotic_suggestions/masks/Hypnosis_variables_20190114_pr_jc.xlsx'
subjects = list(setup['subjects'])
apm_subjects = ['APM' + subj[4:] for subj in subjects]
print(apm_subjects)

# def load_process_y(xlsx_path, subjects):
'''Load behavioral variables from xlsx file and process them for further analysis
'''

# dependant variables
original_y = rawY = pd.read_excel(xlsx_path, sheet_name=0, index_col=1, header=2)

# Display the full width of the dataframe
pd.set_option('display.max_columns', None)
original_y

#%%
rawY = pd.read_excel(xlsx_path, sheet_name=0, index_col=1, header=2).iloc[
    2:, [4,5,6,7,8,9,10,11,12, 17, 18, 19, 38, 48, 65, 67]
]
columns_of_interest = [
    "SHSS_score",
    "raw_change_ANA",
    "raw_change_HYPER",
    "total_chge_pain_hypAna",
    "Chge_hypnotic_depth",
    "Mental_relax_absChange",
    "Automaticity_post_ind",
    "Abs_diff_automaticity",
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

#%%
import matplotlib.pyplot as plt
behav_corr_df = Y
corr_matrix = behav_corr_df.corr()

plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', square=True,
            linewidths=0.5, cbar_kws={'label': 'Pearson r'})

plt.title('Correlation Matrix of Behavioral Variables', fontsize=14)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()


# %%
# Bootstrap per condition visualization
# =====================================
reload(visu_utils)
reload(isc_utils)
result_key = 'isc_results'
conditions = ['NANA', 'ANA', 'NHYPER', 'HYPER'] if 'single-trial' not in model_name else ['N_ANA1_instrbk_1', 'N_HYPER1_instrbk_1']
all_conditions = setup['conditions'] + setup['combined_conditions'] #['HYPER', 'ANA', 'NANA', 'NHYPER']
# conditions = setup['conditions']

#%%
# %%
    # cond = conditions[0]
reload(visu_utils)

iscs_cond = {}
views = {}
surf = {}
interactive_views = []
sig_dfs_one_sample = {}
for cond in ['ANA'] : #setup['conditions']: #'NHYPER']: #conditions:
    print('----------------------------------------------------')
    print(cond)

    #masker =utils.load_pickle(os.path.join(results_dir, cond, f'maskers_{atlas_name}_{cond}_{n_sub}sub.pkl'))
    #isc_bootstrap = utils.load_pickle(all_results_paths[result_key][cond])
    maskers = isc_utils.load_pickle(f'/data/rainville/dSutterlin/projects/ISC_hypnotic_suggestions/results/imaging/ISC/{model_name}/{cond}/maskers_{parcel_name}_{cond}_23sub.pkl')
    atlas = maskers[0].labels_img # adjust number of ROI based on mask

    labels = list(maskers[0].region_names_.values())
    if len(labels) != len(full_labels):
        labels_indices = [full_labels.index(lbl) for lbl in labels]
        print(labels_indices)
    else:
        labels_indices = list(range(len(labels)))
        # used for coords 
    
    isc_bootstrap = isc_utils.load_pickle(f'/data/rainville/dSutterlin/projects/ISC_hypnotic_suggestions/results/imaging/ISC/{model_name}/{cond}/isc_results_{cond}_{n_boot}boot_pairWise{do_pairWise}.pkl')
    isc_rois = pd.DataFrame(isc_bootstrap['isc'], columns=labels)
    isc_results_roi[cond] = isc_rois
    
    isc_median = isc_bootstrap['observed']
    iscs_cond[cond] = isc_median
    # print(isc_median, np.mean(isc_rois))
    ci = isc_bootstrap['confidence_intervals']
    p_values = isc_bootstrap['p_values']
    dist = isc_bootstrap['distribution']
    n_boot = setup['n_boot']
    fdr_p = isc_utils.fdr(p_values, q=0.05)
    bonf_p = isc_utils.bonferroni(p_values, alpha=0.05)
    print(f'FDR thresh : {fdr_p}')
    unc_p = 0.01
    # fdr_p = 0.01

    # sig_mask, sig_df = visu_utils.plot_isc_median_with_significance(
    # isc_median=isc_median,
    # p_values=p_values,
    # atlas = atlas,
    # atlas_labels=labels,
    # p_threshold=fdr_p, #adjusted for HYPER !! (unc.)
    # save_path=None,
    # coords_bool_mask = labels_indices,
    # show=False, # False!
    # fdr_correction=False
    # )  # get FDR table

    # if 'HYPER' in cond: #lower to display
    #     fdr_p = unc_p
    #     title_view = f'ISC_{cond}_(unc. p<.01)'
    # else : title_view = f'ISC_{cond}_(FDR<.05)'
    
    title_view = f'ISC_{cond}_(FDR<.05)'

    # to brain plot 
    reload(visu_utils)
    if labels == None:
        labels = [f'ROI_{i}' for i in range(isc_rois.shape[1])]


    #x = -48, -2, 6, 50
    cut_coords = (5) #(-48, -2, 6, 50)
    isc_img, isc_thresh, sig_df = visu_utils.project_isc_to_brain(
        atlas_img=atlas,
        isc_median=isc_median,
        atlas_labels=id_labels_dct,
        roi_coords = roi_coords,
        p_values=p_values,
        p_threshold=fdr_p,
        title = title_view,
        coords_bool_mask = labels_indices,
        save_path=None,
        show=True,
        display_mode='x',
        cut_coords_plot=cut_coords,
    )

    p_mask = p_values < fdr_p
    sig_labels = [labels[i] for i in range(len(labels)) if p_mask[i]]
    print('Sig ROIs :', sig_labels)
    
    sig_dfs_one_sample[cond] = sig_df #sig_df.sort_values(by='ISC', ascending=False)
    
    # views[cond] = plotting.view_img(isc_img, threshold=0,symmetric_cmap=False, title=title_view)
    surf[cond] = view_img_on_surf(isc_img, threshold=None, surf_mesh='fsaverage')

    # bar plots 
    reload(visu_utils)

    interactive_view = view_img(
        isc_img,
        threshold=0,
        symmetric_cmap=False,
        title=title_view,
        cmap = 'Reds'
    )
    interactive_views.append(interactive_view)
    interactive_view.save_as_html(os.path.join(clean_save_to, f'{title_view}.html'))


#frontal = PFC
 # %%
#for voxel wise case, ignored otherwise
masker = isc_utils.load_pickle('/data/rainville/dSutterlin/projects/ISC_hypnotic_suggestions/results/imaging/ISC/{model_name}/HYPER/maskers_voxelWise_lanA800_HYPER_23sub.pkl')

#%%
# Combined conditions
# ================
from src import visu_utils
reload(visu_utils)
reload(isc_utils)

result_key = 'isc_results'
interactive_views = {}
# all_conditions = ['all_sugg', 'modulation'] #'neutral']
combined_conditions = ['all_sugg', 'ana_run', 'hyper_run']#setup['combined_conditions'] #['all_sugg', 'modulation', 'neutral']
# n_scans = [468, 200, 268] #[438, 188, 250]
views = {}
sig_df_conditions = {}
for i, cond in enumerate(combined_conditions):
    print('----------------------------------------------------')
    print(cond)
    # scans = n_scans[i]
    #masker =utils.load_pickle(os.path.join(results_dir, cond, f'maskers_{atlas_name}_{cond}_{n_sub}sub.pkl'))
    #isc_bootstrap = utils.load_pickle(all_results_paths[result_key][cond])
    # masker = utils.load_pickle(f'/data/rainville/dSutterlin/projects/ISC_hypnotic_suggestions/results/imaging/ISC/{model_name}/HYPER/maskers_schafer100_2mm_HYPER_23sub.pkl')
    isc_bootstrap = isc_utils.load_pickle(f'/data/rainville/dSutterlin/projects/ISC_hypnotic_suggestions/results/imaging/ISC/{model_name}/concat_suggs_1samp_boot/isc_results_{cond}_{n_boot}boot_pairWise{do_pairWise}.pkl')
    isc_rois = pd.DataFrame(isc_bootstrap['isc'], columns=labels)
    isc_results_roi[cond] = isc_rois

    isc_median = isc_bootstrap['observed']
    ci = isc_bootstrap['confidence_intervals']
    p_values = isc_bootstrap['p_values']
    dist = isc_bootstrap['distribution']
    n_boot = setup['n_boot']
    fdr_p = isc_utils.fdr(p_values, q=0.05)
    bonf_p = isc_utils.bonferroni(p_values, alpha=0.05)
    print(f'FDR thresh : {fdr_p}')
    # to brain plot 
    reload(visu_utils)

    # sig_mask, _df = visu_utils.plot_isc_median_with_significance(
    #     isc_median=isc_median,
    #     p_values=p_values,
    #     atlas = atlas,
    #     atlas_labels=labels,
    #     p_threshold=fdr_p,
    #     save_path=None,
    #     coords_bool_mask = labels_indices,
    #     show=True,
    #     fdr_correction=False
    # )

    if 'shock' in model_name:
        cond = cond+'_shock'
    if cond == 'all_sugg':
        cond_name_title = 'All suggestions' 
    else : 
        cond_name_title = cond

    if 'hyper' in cond:
        title_view = f'ISC {cond_name_title} (Unc. p < 0.01)'
        fdr_p = 0.01
        save_name = f'ISC_{cond}_unc01'
        print('!! REDUCING p-value threshold to 0.01 !! ')
    else:
        title_view = f'ISC {cond_name_title} (FDR<.05)'
        save_name = f'ISC_{cond}_FDR05'

    if 'voxelWise' in parcel_name:
        isc_median[~sig_mask] = 0
        isc_img = masker[0].inverse_transform(isc_median)
        isc_thresh = 0
    else:
        # isc_img, isc_thresh, sig_df = visu_utils.project_isc_to_brain(
        #     atlas_path=atlas_path,
        #     isc_median=isc_median,
        #     atlas_labels=labels,
        #     p_values=p_values,
        #     p_threshold=fdr_p,
        #     title = title_view,
        #     coords_bool_mask = labels_indices,
        #     save_path=None,
        #     show=True
        # )
        cut_coords = (5) #(-48, -2, 6, 50)

        isc_img, isc_thresh, sig_df = visu_utils.project_isc_to_brain(
            atlas_img=atlas,
            isc_median=isc_median,
            atlas_labels=id_labels_dct,
            roi_coords = roi_coords,
            p_values=p_values,
            p_threshold=fdr_p,
            title = title_view,
            coords_bool_mask = labels_indices,
            save_path=None,
            show=True,
            display_mode='x',
            cut_coords_plot=cut_coords,
        )
        
    sig_df_conditions[cond] = sig_df
    sig_dfs_one_sample[cond] = sig_df.sort_values(by='ISC', ascending=False)

    vmax = isc_median.max()
    views[cond] = view_img_on_surf(isc_img, threshold=isc_thresh,vmax = vmax,cmap = color_isc, symmetric_cmap = False,title = title_view, surf_mesh='fsaverage')
    reload(visu_utils)
    p_mask = p_values < fdr_p
    sig_labels = [labels[i] for i in range(len(labels)) if p_mask[i]]
    print('Sig ROIs :', sig_labels)
 
    interactive_view = view_img(
            isc_img,
            threshold=isc_thresh,
            title=title_view,
            symmetric_cmap=False,
            cmap = color_isc
                            )
    interactive_views[cond] = interactive_view
    # interactive_view.save_as_html(os.path.join(clean_save_to, f'{title_view}.html'))
    views[cond].save_as_html(os.path.join(clean_save_to, f'{save_name}.html'))

#%%
reload(visu_utils)
#rename conditions with full names
conditions_full_names = {cond : full for cond, full in zip(conditions, ['Neutral (Analg.)', 'Analgesia', 'Neutral (Hyperalg.)', 'Hyperalgesia'])} #, 'Neutral (Hyper)', 'N_Ana', 'N_Hyper'])}
# CAREFUL FOR NAME ORDER / CONDITIONS + full name
visu_utils.plot_median_isc_dots(sig_dfs_one_sample,conditions_full_names=conditions_full_names, title="Median ISC across significant ROIs")

#%% save tables of interest


# %%
# Pie plot of sig regions in df
#------------------------------
reload(visu_utils)

# all_sugg pie-chart
cond_pie ='all_sugg'
sig_rois_for_pie = sig_df_conditions[cond_pie]['ROI']
sig_networks = [label.split('_')[2] for label in sig_rois_for_pie]

visu_utils.plot_network_radar(sig_networks, title = f'Sig ISC during {cond_pie}')


#%%
import matplotlib.pyplot as plt
roi_index = 0
plt.hist(dist[:, roi_index], bins=50, alpha=0.7, label="Bootstrap Distribution")
plt.axvline(isc_median[roi_index], color='red', linestyle='--', label="Observed Median")
plt.xlabel("ISC Values")
plt.ylabel("Frequency")
plt.title(f"Bootstrap Distribution for ROI {roi_index}")
plt.legend()
plt.show()

shifted_dist = dist[:, roi_index] - np.median(dist[:, roi_index])
plt.hist(shifted_dist, bins=50, alpha=0.7, label="Shifted Bootstrap Distribution")
plt.axvline(isc_median[roi_index], color='red', linestyle='--', label="Observed Median")
plt.xlabel("Shifted ISC Values")
plt.ylabel("Frequency")
plt.title(f"Shifted Bootstrap Distribution for ROI {roi_index}")
plt.legend()
plt.show()

# %%

#%%
#=====================================
# # Bootstrap per combined conditions
# result_key = 'isc_combined_results'
# folder = 'concat_suggs_1samp_boot'
# #folder = 'concat_suggs_1samp_boot'
# combined_conditions = ['all_sugg', 'modulation', 'neutral']
# conditions = ['Hyper', 'Ana', 'NHyper', 'NAna']
# cond = combined_conditions[1]
# n_scans = 191

# #for cond in conditions:
# masker =utils.load_pickle(os.path.join(results_dir, conditions[0], f'maskers_{atlas_name}_{conditions[0]}_{n_sub}sub.pkl'))
# file = f"isc_results_{cond}_{n_scans}TRs_{setup['n_perm']}boot_pairWiseTrue.pkl"
# isc_bootstrap = utils.load_pickle(os.path.join(results_dir, folder, file))

# isc_rois = isc_bootstrap['isc']
# isc_median = isc_bootstrap['observed']
# ci = isc_bootstrap['confidence_intervals']
# p_values = isc_bootstrap['p_values']
# dist = isc_bootstrap['distribution']
# n_boot = setup['n_boot']

# fdr_p = utils.fdr(p_values, q=0.05)

#%%

# %%
# ===========================
# Difference (Permutation) per condition visualization
result_key = 'cond_contrast_permutation'
# conditions = ['Hyper', 'Ana', 'NHyper', 'NAna']
contrasts = ['ana_run-hyper_run', 'Ana-Hyper'] #, 'NAna-NHyper']
n_perm = setup['n_perm']
# contrasts = ['Ana-N_Ana', 'Hyper-N_Hyper', 'N_ANA1_instrbk_1-N_HYPER1_instrbk_1']

reload(visu_utils)
# n_scans = [100, 100, 134] #[94, 94, 125]
views = {}
sig_rois = {}
interactive_views=[]
for i, cont in enumerate(contrasts):
    # scans = n_scans[i]
    # masker =utils.load_pickle(os.path.join(results_dir, cond, f'maskers_{atlas_name}_{cond}_{n_sub}sub.pkl'))
    # file = f"isc_results_{contrast}_{n_scans}TRs_{setup['n_perm']}perm_pairWiseTrue.pkl"
    #isc_bootstrap = utils.load_pickle(all_results_paths[result_key][cond])
    
    file = f'/data/rainville/dSutterlin/projects/ISC_hypnotic_suggestions/results/imaging/ISC/{model_name}/cond_contrast_permutation/isc_results_{cont}_{n_perm}perm_pairWise{do_pairWise}.pkl'
    # isc_contrast = utils.load_pickle(os.path.join(results_dir, result_key, file))
    isc_contrast = isc_utils.load_pickle(file)

    grouped_isc = isc_contrast['grouped_isc']        # Grouped ISC values
    observed_diff = isc_contrast['observed_diff']    # Observed differences in ISC
    p_values = isc_contrast['p_value']               # P-values for the ISC contrasts
    distribution = isc_contrast['distribution']      
    fdr_p = isc_utils.fdr(p_values, q=0.05)
    # unc_p = 0.05

    reload(visu_utils)
    print(f'FDR thresh : {fdr_p}')
    diff_img, diff_thresh, sig_labels = visu_utils.project_isc_to_brain_perm(
        atlas_img=atlas,
        isc_median=observed_diff,
        atlas_labels=id_labels_dct,
        roi_coords = roi_coords,
        p_values=p_values,
        p_threshold=fdr_p,
        title = f"Difference in ISC between {cont}",
        save_path=None,
        show=True,
        color='Reds'
    )
    p_mask= p_values < fdr_p
    # sig_labels = [labels[i] for i in range(len(labels)) if p_mask[i]]
    print('Sig ROIs :', sig_labels)
 
    views[cont] = view_img_on_surf(diff_img, threshold=diff_thresh, surf_mesh='fsaverage')
    sig_rois[cont] = sig_labels


    interactive_view = view_img(
            diff_img,
            threshold=diff_thresh,
            title=f"Median ISC {cont} (FDR<.05)",
            cmap='Reds'
        )
    interactive_views.append(interactive_view)
    interactive_view.save_as_html(os.path.join(clean_save_to, f'{cont}_all_subjs.html'))

#%%
sig_rois['Ana-Hyper'].sort_values(by='Difference', ascending=False)
# %%
# INTERACTION ish : for LOO!
#-----------------
from matplotlib import pyplot as plt
from src import visu_utils
reload(visu_utils)
isc_shss_dct = {}

X_pheno = behav_df
behav_ls = ['SHSS_score', 'Abs_diff_automaticity','total_chge_pain_hypAna']
# roi_focus = [b'7Networks_LH_SomMot_2', b'7Networks_RH_SomMot_1', b'7Networks_RH_Default_Temp_4', b'7Networks_RH_Default_Temp_5']
# roi_focus_indices = [labels.index(roi) for roi in roi_focus]
top5_sig_regions = sig_df_conditions['modulation'].nlargest(5, 'ISC_median')
roi_focus = top5_sig_regions['ROI'].values

ana_loo = isc_results_roi['ANA'][roi_focus]
hyper_loo = isc_results_roi['HYPER'][roi_focus]
modulation_loo = isc_results_roi['modulation'][roi_focus]
all_sugg_loo = isc_results_roi['all_sugg'][roi_focus]
isc_diff_focus = ana_loo - hyper_loo

y = X_pheno[behav_ls[0]].values


# data_for_plot = pd.DataFrame({
#     'SHSS': y,
#     'ISC_diff': isc_diff_focus,
# })
for i in range(len(roi_focus)):
    x = isc_diff_focus.iloc[:, i]
    visu_utils.jointplot(x, y, x_label='Ana-Hyper ISC',title = list(top5_sig_regions['ROI'])[i], y_label='SHSS')


# %%


#%%
#=======================
# Difference for high vs low SHSS
result_key = 'cond_contrast_permutation'
# conditions = ['Hyper', 'Ana', 'NHyper', 'NAna']
contrasts = ['Hyper-Ana', 'Ana-Hyper', 'NHyper-NAna']

reload(visu_utils)
n_scans = [94, 94, 125]
views = {}
sig_rois = {}
surf_views = {}
cb_names =['counterbalance_H1', 'counterbalance_H2']
group_names = ['high_shss', 'low_shss'] if 'single_trial' in model_name else ['high_shss', 'low_shss', 'counterbalance_H1', 'counterbalance_H2']
# for shss_grp, n_sub in zip(group_names, [11, 12]):
for shss_grp, n_sub in zip(group_names, [11, 12]):
    print(f"Doing {shss_grp} with {n_sub} subjects")
    views[shss_grp] = []
    surf_views[shss_grp] = []

    for cont in contrasts: #'Hyper-Ana'  
        # masker =utils.load_pickle(os.path.join(results_dir, cond, f'maskers_{atlas_name}_{cond}_{n_sub}sub.pkl'))
        # file = f"isc_results_{contrast}_{n_scans}TRs_{setup['n_perm']}perm_pairWiseTrue.pkl"
        #isc_bootstrap = utils.load_pickle(all_results_paths[result_key][cond])
        file = os.path.join(setup['project_dir'], f'results/imaging/ISC/{model_name}/group_perm_{shss_grp}/isc_results_{n_sub}sub_{cont}_5000perm_pairWise{do_pairWise}.pkl')
        # isc_contrast = utils.load_pickle(os.path.join(results_dir, result_key, file))
        isc_contrast = isc_utils.load_pickle(file)

        grouped_isc = isc_contrast['grouped_isc']        
        observed_diff = isc_contrast['observed_diff']    # Observed differences in ISC
        p_values = isc_contrast['p_value']               # P-values for the ISC contrasts
        distribution = isc_contrast['distribution']      
        fdr_p = isc_utils.fdr(p_values, q=0.05)
        unc_p = 0.001
        
        print('FDR thresh : ', fdr_p)
        # reload(visu_utils)
        diff_img, diff_thresh, sig_labels = visu_utils.project_isc_to_brain_perm(
            atlas_path=atlas_path,
            isc_median=observed_diff,
            atlas_labels=labels,
            p_values=p_values,
            p_threshold=unc_p, # UNC
            title = f"{shss_grp} ({n_sub} subj.) : Difference in ISC between {cont} (unc. p = {unc_p})",
            save_path=None,
            show=True
        )
        view_title = f'{shss_grp}_{cont}_(unc_p<.01)'
        #  sig_labels = [labels[i] for i in range(len(labels)) if p_mask[i]]
        print('Sig ROIs :', sig_labels)
 
        surf_views[shss_grp].append(view_img_on_surf(diff_img, threshold=diff_thresh, surf_mesh='fsaverage'))
        sig_rois[shss_grp] = sig_labels

        view = view_img(
                diff_img,
                title=view_title,
            )
        views[shss_grp].append(view)
        view.save_as_html(os.path.join(clean_save_to, f'{view_title}.html'))

# %%
# # INTERACTION ish
# #-----------------
# isc_shss_dct = {}
# for shss_grp, n_sub in zip(['high_shss', 'low_shss'], [11, 12]):
#     for cont in ['Ana-Hyper']: #'Hyper-Ana'  
#         # masker =utils.load_pickle(os.path.join(results_dir, cond, f'maskers_{atlas_name}_{cond}_{n_sub}sub.pkl'))
#         # file = f"isc_results_{contrast}_{n_scans}TRs_{setup['n_perm']}perm_pairWiseTrue.pkl"
#         #isc_bootstrap = utils.load_pickle(all_results_paths[result_key][cond])
#         file = os.path.join(setup['project_dir'], f'results/imaging/ISC/{model_name}/group_perm_{shss_grp}/isc_results_{n_sub}sub_{cont}_5000perm_pairWise{do_pairWise}.pkl')
#         # isc_contrast = utils.load_pickle(os.path.join(results_dir, result_key, file))
#         isc_contrast = isc_utils.load_pickle(file)
#         isc_shss_dct[shss_grp] = isc_contrast

# diff_low = isc_shss_dct['low_shss']['observed_diff']
# diff_high = isc_shss_dct['high_shss']['observed_diff']

# %%

#%%
coords = sig_rois['high_shss']['Coordinates'][0]
diff_thresh = float(sig_rois['high_shss']['Difference'])
plot_stat_map(
    diff_img,
    threshold=None,
    title="Diff SHSS High ",
    display_mode="z",
    cut_coords=None,
    colorbar=True
    )


#%%
#=======================
# Group difference with behavioral

# %%
# ===========================
# grouped isc with behavioral
reload(isc_utils)
# QUESTION : does ISC differ as a function of SHHS score (median split)?
result_key = 'group_permutation_results'
conditions = ['Hyper', 'Ana', 'NHyper', 'NAna']
conditions = ['ana_run'] #'all_sugg', 'ANA'] #, 'neutral', 'modulation'] #, 'HYPER', 'ANA']

behav_ls = [
    'Chge_hypnotic_depth_median_grp', 'SHSS_score_median_grp', 'raw_change_HYPER_median_grp',
    'raw_change_ANA_median_grp', 'total_chge_pain_hypAna_median_grp',
    'Mental_relax_absChange_median_grp', 'Abs_diff_automaticity_median_grp'
]
behav_ls = ['raw_change_HYPER_median_grp', 'raw_change_ANA_median_grp', 'total_chge_pain_hypAna_median_grp']
behav_ls = ['Chge_hypnotic_depth_median_grp', 'Abs_diff_automaticity_median_grp','Mental_relax_absChange_median_grp' ] #, 'Abs_diff_automaticity_median_grp'] #, 'SHSS_score_median_grp', 'total_chge_pain_hypAna_median_grp'] #, 'SHSS_score', 'Abs_diff_automaticity', 'total_chge_pain_hypAna']

sig_rois_cond = {}
views = {}
for cond in conditions:
    #masker =utils.load_pickle(os.path.join(results_dir, conditions[0], f'maskers_{atlas_name}_{conditions[0]}_{n_sub}sub.pkl'))
    # isc_group = utils.load_pickle(all_results_paths[result_key][cond])
    isc_group = isc_utils.load_pickle(os.path.join(setup['project_dir'], f'results/imaging/ISC/{model_name}/behavioral_group_permutation/{cond}_group_permutation_results_{n_perm}perm.pkl'))
    # behav_ls = list(isc_group.keys())
    sig_rois_cond[cond] = {}
    views[cond] = {}
    for y in behav_ls:
    
        print(f"Condition: {cond}, Behavior: {y}")
        grouped_isc = isc_group[y]['grouped_isc']        # Grouped ISC values
        observed_diff = isc_group[y]['observed_diff']    # Observed differences in ISC
        p_values = isc_group[y]['p_value']               # P-values for the ISC contrasts
        distribution = isc_group[y]['distribution'] 

        fdr_p = isc_utils.fdr(p_values, q=0.05)
        # unc_p = 0.01
        print(f'FDR thresh : {fdr_p}')
        reload(visu_utils)

        cut_coords = (5)
        diff_img, diff_thresh, sig_labels = visu_utils.project_isc_to_brain_perm(
            atlas_img=atlas,
            isc_median=observed_diff,
            atlas_labels=id_labels_dct,
            p_values=p_values,
            p_threshold=fdr_p,
            title = None,# f"Difference in ISC for {cond} high SHSS > low SHSS (p unc. < {unc_p})",
            save_path=None,
            show=True,
            display_mode='x',
            cut_coords_plot=cut_coords,
        )
        sig_rois_cond[cond][y] = sig_labels
        print(sig_labels)
        views[cond][y] = plotting.view_img(diff_img, threshold=diff_thresh, title=f"grp diff {cond} {y}")
        # view_img_on_surf(diff_img, threshold=diff_thresh, surf_mesh='fsaverage')

    # SHSS : x = 14, 52
# %%
#rename conditions with full names
conditions_full_names = {cond : full for cond, full in zip(conditions, ['Neutral (Analg.)', 'Analgesia', 'Neutral (Hyperalg.)', 'Hyperalgesia'])} #, 'Neutral (Hyper)', 'N_Ana', 'N_Hyper'])}
# CAREFUL FOR NAME ORDER / CONDITIONS + full name
visu_utils.plot_median_isc_dots(sig_dfs_one_sample,conditions_full_names=conditions_full_names, title="Median ISC across significant ROIs")

# %%
# LOO RSA ish
#-------------
reload(isc_utils)
from scipy.stats import spearmanr
heatmaps = {}
cond = 'ANA'
y_name = 'SHSS_score'
simil_model = 'euclidean'

for i, simil_model in enumerate([simil_model]):
    # Load ISC bootstrap data
    isc_bootstrap = isc_utils.load_pickle(
        os.path.join(project_dir, f'results/imaging/ISC/{model_name}/{cond}/isc_results_{cond}_5000boot_pairWise{do_pairWise}.pkl')
    )
    isc_rois = pd.DataFrame(isc_bootstrap['isc'], columns=labels)

    y = np.array(behav_df[y_name])
    sim_behav = isc_utils.compute_behav_similarity_LOO(y, metric=simil_model)

    for roi in roi_focus:
        rsa_vec = isc_diff_focus[roi]

        # 2. Compute Spearman correlation between behavioral LOO similarity and ISC modulation
        rho, p_val = spearmanr(sim_behav, rsa_vec)

        print(f"Spearman correlation between behavior LOO similarity and ISC modulation:")
        print(f"rho = {rho:.3f}, p = {p_val:.4f}")

        # 3. Optional: visualize the relationship with a scatter plot
        plot_df = pd.DataFrame({
            'Behavior_LOO_Similarity': sim_behav,
            'ISC_Modulation': mod_isc_subject
        })

        sns.set(style='whitegrid')
        plt.figure(figsize=(6, 5))
        ax = sns.regplot(data=plot_df, x='Behavior_LOO_Similarity', y='ISC_Modulation', 
                        scatter_kws={'s': 60, 'alpha': 0.8}, line_kws={'color': 'black'})
        plt.title(f'Spearman rho = {rho:.2f}, p = {p_val:.3f}')
        plt.xlabel('Behavioral Similarity (LOO)')
        plt.ylabel('ISC Modulation (ANA - HYPER)')
        plt.tight_layout()
        plt.show()


# %%
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from scipy.stats import ttest_1samp
from scipy.stats import ttest_rel
reload(visu_utils
       )
# ===========================
# ISC-RSA
behav_df = pd.read_csv(f'/data/rainville/dSutterlin/projects/ISC_hypnotic_suggestions/results/behavioral_data_cleaned.csv')
X_pheno = behav_df
result_key = 'rsa_isc_results'
conditions = ['all_sugg', 'modulation'] #['HYPER','ANA', 'NHYPER', 'NANA'] # ['Hyper', 'Ana', 'NHyper', 'NAna']
# conditions = ['ANA']
#conditions = ['Hyper', 'Ana', 'NHyper', 'NAna', 'all_sugg', 'modulation', 'neutral']
behav_ls = ['total_chge_pain_hypAna']
models =['euclidean', 'annak']

rsa_dict_2ttest = {}

# compare NN and AnnaK models using 2sample t tests
# tests differences ISC-RSA correlation between two models for each var.
n_test_p = []
for y_name in behav_ls:
    print(f'====Processing {y_name} across conditions======')
    rsa_dict_2ttest[y_name] = {}

    for cond in conditions:
        if cond == 'all_sugg' or cond == 'modulation':
            p = f'/data/rainville/dSutterlin/projects/ISC_hypnotic_suggestions/results/imaging/ISC/{model_name}/concat_suggs_1samp_boot/isc_results_{cond}_5000boot_pairWise{do_pairWise}.pkl'
        else:
            p = f'/data/rainville/dSutterlin/projects/ISC_hypnotic_suggestions/results/imaging/ISC/{model_name}/{cond}/isc_results_{cond}_5000boot_pairWise{do_pairWise}.pkl'

        isc_bootstrap = isc_utils.load_pickle(p)
        isc_rois = pd.DataFrame(isc_bootstrap['isc'], columns=labels)
        rsa_dict_2ttest[y_name][cond] = {}

        for simil_model in models:
            # Load RSA data
            rsa_df = pd.read_csv(
                f'/data/rainville/dSutterlin/projects/ISC_hypnotic_suggestions/results/imaging/ISC/{model_name}/rsa_isc_results_{simil_model}/rsa-isc_{cond}/{y_name}_rsa_isc_{simil_model}simil_{n_perm_rsa}perm_pvalues.csv'
            )
            p_values = np.array(rsa_df['p_value'])
            fdr_p = isc_utils.fdr(p_values, q=0.05)

            rsa_dict_2ttest[y_name][cond][simil_model] = {
                'correlation': rsa_df['correlation'],
                'p_values': p_values,
                'fdr_p': fdr_p
            }

        # Perform paired t-test between the two models
        correl1 = rsa_dict_2ttest[y_name][cond][models[0]]['correlation']
        correl2 = rsa_dict_2ttest[y_name][cond][models[1]]['correlation']

        t_stat, p_value = ttest_rel(correl1, correl2)
        n_test_p.append(p_value) # for fdr correction across Ttest
        # Store results
        t_dict = {'t': t_stat, 'p': p_value, 'df': len(correl1) - 1}
        rsa_dict_2ttest[y_name][cond]['t_test'] = t_dict

        labels_map, yeo7_net = visu_utils.yeo_networks_from_schaeffer(labels)
        visu_utils.plot_scatter_legend(correl1, correl2, grp_id=yeo7_net,var_name = models, title = f'{y_name} {cond} RSA-ISC per ROI', save_path=None)
        
        # Print results
        if t_stat > 0 and p_value <= fdr_p:
            print(f'{models[0]} is better than {models[1]} in {cond}')
            print(f"t({t_dict['df']}) = {t_dict['t']:.2f}, p = {t_dict['p']:.4f}")
        elif t_stat < 0 and p_value < 0.05/12:
            print(f'{models[1]} is better than {models[0]} in {cond}')
            print(f"t({t_dict['df']}) = {t_dict['t']:.2f}, p = {t_dict['p']:.4f}")
        else:
            print(f'No significant difference between {models[0]} and {models[1]} in {cond}')
            print(f"t({t_dict['df']}) = {t_dict['t']:.2f}, p = {t_dict['p']:.4f}")

# %%

# %% 
y_name = 'total_chge_pain_hypAna'
best_model = 'annak' #'euclidean'
rsa_views= {}
for cond in conditions:
    correl = rsa_dict_2ttest[y_name][cond][best_model]['correlation']
    p_values = rsa_dict_2ttest[y_name][cond][best_model]['p_values']
    fdr_p = rsa_dict_2ttest[y_name][cond][best_model]['fdr_p']
    p_unc = 0.01
    print('FDR thresh : ', fdr_p)

#     sig_mask, sig_df = visu_utils.plot_isc_median_with_significance(
#     isc_median=correl,
#     p_values=p_values,
#     atlas=atlas,
#     atlas_labels=labels,
#     p_threshold=p_unc,
#     save_path=None,
#     show=True,
#     fdr_correction=False
# )   
        # === Visualize with your existing function ===
    rsa_img, rsa_thresh, sig_labels = visu_utils.project_isc_to_brain_perm(
        atlas_path=atlas_path,
        isc_median=correl,
        atlas_labels=labels,
        p_values=p_values,
        p_threshold=p_unc,
        title="RSA-ISC: Suggestion-Pain Similarity (FDR<.05)",
        save_path=None,
        show=True,
        display_mode='x',
        cut_coords_plot=None,
        color='Reds'
    )
    
    rsa_views[cond] = plotting.view_img(rsa_img, threshold=rsa_thresh, title=f"RSA suggestion - pain similarity {cond}", colorbar=True,symmetric_cmap=False, cmap = 'Reds')




#%%
y_name = 'Abs_diff_automaticity'
best_model = 'annak' #'euclidean'
print(f'====Processing {y_name} across conditions======')
for cond in conditions:
    correl = rsa_dict_2ttest[y_name][cond][best_model]['correlation']
    p_values = rsa_dict_2ttest[y_name][cond][best_model]['p_values']
    fdr_p = rsa_dict_2ttest[y_name][cond][best_model]['fdr_p']
    p_unc = 0.01
    print('FDR thresh : ', fdr_p)
    sig_mask, sig_df = visu_utils.plot_isc_median_with_significance(
    isc_median=correl,
    p_values=p_values,
    atlas_labels=labels,
    p_threshold=p_unc,
    save_path=None,
    show=True,
    fdr_correction=False
)
    print(sig_df)

# Take home is that annak is better in Hyper related conditions while NN in Ana

# %%

#%%
reload(visu_utils)
fdr_test_p = isc_utils.fdr(np.array(n_test_p), q=0.05)

#%%
heatmaps = {}

for i, simil_model in enumerate(models):
    # Load ISC bootstrap data
    isc_bootstrap = isc_utils.load_pickle(
        os.path.join(project_dir, f'results/imaging/ISC/{model_name}/{cond}/isc_results_{cond}_5000boot_pairWiseTrue.pkl'
    )
    isc_rois = pd.DataFrame(isc_bootstrap['isc'], columns=labels)

    # Compute similarity matrix
    y = np.array(behav_df[y_name])
    sim_behav = isc_utils.compute_behav_similarity(y, metric=simil_model)

    # Normalize similarity matrix for better visualization
    sim_behav_norm = (sim_behav - sim_behav.min()) / (sim_behav.max() - sim_behav.min())
    heatmaps[simil_model] = sim_behav_norm

    if i == 1:  # Generate plots at the second iteration
        # Plot Heatmaps
        fig, axs = plt.subplots(1, 2, figsize=(12, 6))

        for ax, (model, matrix) in zip(axs, heatmaps.items()):
            sns.heatmap(
                matrix, ax=ax, cmap='viridis', square=True,
                cbar=True, xticklabels=False, yticklabels=False
            )
            ax.set_title(f"{model.capitalize()} Model Similarity")

        plt.tight_layout()
        plt.show()

        # Plot histograms of correlations
        plt.figure(figsize=(8, 6))
        for model, matrix in heatmaps.items():
            corr_values = matrix[np.triu_indices(matrix.shape[0], k=1)]
            plt.hist(corr_values, bins=20, alpha=0.6, label=f"{model.capitalize()} Model")

        plt.title("Correlation Distribution")
        plt.xlabel("Similarity Value")
        plt.ylabel("Frequency")
        plt.legend()
        plt.show()
#%%
reload(visu_utils)


# Plot the similarity matrix and RSA correlation histogram
visu_utils.plot_similarity_and_histogram(
    similarity_matrix=sim_behav,
    correlations=correl,
    p_values=p_values,
    atlas_labels=atlas_labels,
    behav_name=y_name,
    save_path=None
)

# %%

# ===========================
# supplementary ISC-RSA with other behavioral
X_pheno = pd.read_csv(f'/data/rainville/dSutterlin/projects/ISC_hypnotic_suggestions/results/behavioral_data_cleaned.csv')
behav_ls = ['mean_VAS_Hyper', 'mean_VAS_Ana', 'mean_VAS_NHyper', 'mean_VAS_Nana']
conditions = ['Hyper', 'Ana', 'NHyper', 'NAna']
models =['euclidean', 'annak']
atlas_labels = labels
rsa_dict_2ttest = {}
n_perm_rsa = 10000
save_cond_rsa = '/data/rainville/dSutterlin/projects/ISC_hypnotic_suggestions/results/imaging/ISC/model5_jeni_lvlpreproc-23sub_schafer100_2mm/rsa_isc_results_euclidean/supp_analyses'

# compare NN and AnnaK models using 2sample t tests
# tests differences ISC-RSA correlation between two models for each var.
n_test_p = []
for y_name, cond in zip(behav_ls, conditions):
    print(f'====Processing {y_name} across conditions======')
    key_name = f'{y_name}-{cond}'
    rsa_dict_2ttest[key_name] = {}
    isc_bootstrap = isc_utils.load_pickle(
        f'/data/rainville/dSutterlin/projects/ISC_hypnotic_suggestions/results/imaging/ISC/model5_jeni_lvlpreproc-23sub_schafer100_2mm/{cond}/isc_results_{cond}_5000boot_pairWiseTrue.pkl'
    )
    isc_pairwise = isc_bootstrap['isc']
    isc_rois = pd.DataFrame(isc_bootstrap['isc'], columns=labels)
    
    values_rsa_perm = {}
    # Load RSA data
    y = np.array(X_pheno[y_name])
    y = (y - np.mean(y)) / np.std(y)
    for simil_model in models:
        sim_behav = isc_utils.compute_behav_similarity(y, metric = simil_model)
        df_subjectwise_rsa = pd.DataFrame(index=range(isc_pairwise.shape[0]), columns=atlas_labels)

        for col_j in range(isc_pairwise.shape[1]): # j ROIs
            if atlas_name == 'voxelWise':
                roi_name = f'voxel_{col_j}'
            else:
                roi_name = atlas_labels[col_j]

            isc_roi_vec = isc_pairwise[:, col_j]
            rsa_results = isc_utils.matrix_permutation(sim_behav, isc_roi_vec, n_permute=n_perm_rsa, metric="spearman", how="upper", tail=1, return_perms = True)
            values_rsa_perm[roi_name] = {'correlation': rsa_results['correlation'], 'p_value': rsa_results['p']}
            #distribution_rsa_perm[roi_name] = rsa_results['perm_dist']

        rsa_df = pd.DataFrame.from_dict(values_rsa_perm, orient='index')
        csv_path = os.path.join(save_cond_rsa, f'isc_rsa_{n_perm_rsa}perm_{y_name}_{simil_model}simil_pvalues.csv')
        rsa_df.to_csv(csv_path)
        
        p_values = np.array(rsa_df['p_value'])
        fdr_p = isc_utils.fdr(p_values, q=0.05)

        rsa_dict_2ttest[key_name][simil_model] = {
            'correlation': rsa_df['correlation'],
            'p_values': p_values,
            'fdr_p': fdr_p
        }

    # Perform paired t-test between the two models
    correl1 = rsa_dict_2ttest[key_name][models[0]]['correlation']
    correl2 = rsa_dict_2ttest[key_name][models[1]]['correlation']

    t_stat, p_value = ttest_rel(correl1, correl2)
    n_test_p.append(p_value) # for fdr correction across Ttest
    # Store results
    t_dict = {'t': t_stat, 'p': p_value, 'df': len(correl1) - 1}
    rsa_dict_2ttest[key_name]['t_test'] = t_dict

    labels_map, yeo7_net = visu_utils.yeo_networks_from_schaeffer(labels)
    visu_utils.plot_scatter_legend(correl1, correl2, grp_id=yeo7_net,var_name = models, title = f'{y_name} {cond} RSA-ISC per ROI', save_path=None)
    
    if t_stat > 0 and p_value <= 0.01:
        print(f'{models[0]} is better than {models[1]} in {cond}')
        print(f"t({t_dict['df']}) = {t_dict['t']:.2f}, p = {t_dict['p']:.4f}")
    elif t_stat < 0 and p_value < 0.01:
        print(f'{models[1]} is better than {models[0]} in {cond}')
        print(f"t({t_dict['df']}) = {t_dict['t']:.2f}, p = {t_dict['p']:.4f}")
    else:
        print(f'No significant difference between {models[0]} and {models[1]} in {cond}')
        print(f"t({t_dict['df']}) = {t_dict['t']:.2f}, p = {t_dict['p']:.4f}")

# %%

rsa_isc_pkl = f'results/imaging/RSA/rsa_isc_sugg-pain_10000perm.pkl'
rsa_isfc_pkl = '/data/rainville/dSutterlin/projects/ISC_hypnotic_suggestions/results/imaging/RSA/rsa_isfc_sugg-pain_10000perm.pkl'

rsa_isc = isc_utils.load_pickle(os.path.join(project_dir,rsa_isc_pkl))
rsa_isfc = isc_utils.load_pickle(rsa_isfc_pkl)

cond = 'all_sugg'
rsa_isc_cond = rsa_isfc[cond]

correl = rsa_isc_cond['spearman_r'].to_numpy()
p_values = rsa_isc_cond['p_values'].to_numpy()
fdr_p = isc_utils.fdr(p_values, q=0.05)
print('FDR thresh : ', fdr_p)

sig_mask, sig_df = visu_utils.plot_isc_median_with_significance(
isc_median=correl,
p_values=p_values,
atlas = atlas,
atlas_labels=labels,
p_threshold=fdr_p,
save_path=None,
show=True,
fdr_correction=False
)

# %%
