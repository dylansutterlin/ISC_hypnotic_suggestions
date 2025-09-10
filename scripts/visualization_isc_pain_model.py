# %%
from nilearn.image import math_img
from nilearn.plotting import plot_stat_map
import os
import seaborn as sns
from nilearn import plotting
from nilearn.plotting import view_img 

import time
import matplotlib.pyplot as plt
from importlib import reload
import nibabel as nib
import numpy as np
import pandas as pd
from nilearn.image import concat_imgs
from brainiak.isc import isc, bootstrap_isc, permutation_isc, compute_summary_statistic, phaseshift_isc
from nilearn.maskers import MultiNiftiMapsMasker, MultiNiftiMasker
from nilearn.datasets import fetch_atlas_schaefer_2018
from sklearn.utils import Bunch
from nilearn.plotting import view_img_on_surf
from nilearn.maskers import NiftiLabelsMasker
from nilearn.plotting import plot_glass_brain

from nilearn.plotting import find_parcellation_cut_coords

import src.isc_utils as isc_utils
import src.visu_utils as visu_utils
reload(visu_utils)
reload(isc_utils)


# %% Load the data
model_names = {
    'model1_sugg_200': 'model1_sugg_23-sub_schafer-200-2mm_mask-whole-brain_pairWise-True_preproc_reg-mvmnt-True-8',
    'model1_sugg_run' : 'model1-runs_sugg_23-sub_schafer_tian-200-2mm_mask-whole-brain_pairWise-True_preproc_reg-mvmnt-True-8',
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
    'single_trial_language_mask' : 'model_single-trial_sugg_23-sub_schafer-200-2mm_mask-lanA800_pairWise-True_preproc_reg-mvmnt-True-8',
    'single_trial_whole-brain' : 'model_single-trial-wb_sugg_23-sub_schafer-200-2mm_mask-whole-brain_pairWise-True_preproc_reg-mvmnt-True-8',
    'model5_sugg_tian' : 'model5-with-subcort_sugg_23-sub_schafer-200-2mm_mask-whole-brain_pairWise-True_preproc_reg-mvmnt-True-8',
    'model1_single-trial': 'model1_single-trial-wb_sugg_23-sub_schafer_tian-200-2mm_mask-whole-brain_pairWise-True_preproc_reg-mvmnt-True-8',
    'model2_sugg' : 'model2_sugg_23-sub_schafer_tian-200-2mm_mask-whole-brain_pairWise-True_preproc_reg-mvmnt-True-8',
    'model3_shock' : 'model3_shock_23-sub_schafer_tian-200-2mm_mask-whole-brain_pairWise-True_preproc_reg-mvmnt-True-8',
}

model_is = 'model3_shock'

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
save_visu = os.path.join(results_dir, 'post-hoc_VISU-tables')
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
else:
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

from nilearn import datasets, plotting
bg_mni =  datasets.load_mni152_template(resolution=1)

#%%
# PUBLICATION surface plot for method section
from nilearn.plotting import plot_roi
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from nibabel.freesurfer import read_annot
from nilearn import datasets, plotting, surface

base_cmap = plt.get_cmap('tab20')
colors = base_cmap.colors * 11  # Repeat to cover 220+ parcels
colors = np.array(colors)

# plot_roi(atlas, bg_img =bg_mni, title='SENSAAS Atlas', display_mode = 'x', cut_coords = [0, -40, 45,48, 50,55, 60], view_type = 'continuous', colorbar=True, cmap=custom_cmap, linewidths=3)
shuffle_idx = np.argsort(np.sin(np.linspace(0, np.pi, len(colors))))  # or use np.random.permutation for random
shuffled_colors = colors[shuffle_idx]
shuffled_cmap = ListedColormap(shuffled_colors)

# Use fsaverage surface
fsaverage = datasets.fetch_surf_fsaverage(mesh='fsaverage6')
# Downloaded from shaeffer github repo
annot_path = '/home/dsutterlin/Downloads/rh.Schaefer2018_200Parcels_7Networks_order.annot'

# Load as GIFTI label texture
labels, ctab, names = read_annot(annot_path)

fig = plt.figure(figsize=(8, 6))
ax = plt.gca()

im = plotting.plot_surf_stat_map(
    fsaverage.infl_right,
    stat_map=labels,
    hemi='right',
    bg_on_data=True,
    alpha=0.9,
    colorbar=False,
    title= None,
    cmap=shuffled_cmap,
    output_file = os.path.join(post_hoc_dir, "schaefer200_surface.png")
)

#%%
from src import preproc_utils

isc_results_roi = {}

behav_df = pd.read_csv(
    os.path.join(setup['project_dir'], f'results/behavioral_data_cleaned.csv'),
    index_col=0
)
behav_df.index.name = 'subjects'
behav_df = behav_df.sort_index()

xlsx_path = r'/data/rainville/dSutterlin/projects/ISC_hypnotic_suggestions/masks/Hypnosis_variables_20190114_pr_jc.xlsx'
subjects = list(setup['subjects'])
apm_subjects = ['APM' + subj[4:] for subj in subjects]
Y = preproc_utils.load_process_y(xlsx_path, subjects)

# %%
# Bootstrap per condition visualization
# =====================================
reload(visu_utils)
reload(isc_utils)
result_key = 'isc_results'
conditions = ['NANA', 'ANA', 'NHYPER', 'HYPER'] if 'single-trial' not in model_name else ['N_ANA1_instrbk_1', 'N_HYPER1_instrbk_1']
all_conditions = setup['conditions'] + setup['combined_conditions'] #['HYPER', 'ANA', 'NANA', 'NHYPER']
conditions = setup['conditions']


#%%
# Visualizatino ISC 1-sample using global FDR threshold (based on x 4 conditions)
# Note : Excludes subsequent context grouping (ana_run + hyper_run)
# From threshold computing because it dilutes pvals to do FDR: .006 (6 conds) vs .003 (4 conds) 
# Context grouping is basically redundant with the condition ISCs
#======================
reload(visu_utils)

isc_matrices = {}
iscs_cond = {}
isc_p_cond = {}

save_plots_1sample = os.path.join(save_visu, 'plots_1sample_isc')
os.makedirs(save_plots_1sample, exist_ok=True)
save_tables_1sample = os.path.join(save_visu, 'tables_1sample_isc')
os.makedirs(save_tables_1sample, exist_ok=True)

# Load ISC results
for cond in setup['conditions']: #'NHYPER']: #conditions:
    print('----------------------------------------------------')
    print(cond)

    maskers = isc_utils.load_pickle(os.path.join(results_dir, f'{cond}/maskers_{parcel_name}_{cond}_23sub.pkl'))
    atlas = maskers[0].labels_img # adjust number of ROI based on mask
    labels = list(maskers[0].region_names_.values())
    if labels == None:
        labels = [f'ROI_{i}' for i in range(isc_rois.shape[1])]

    # accounts for a case where we masked the brain, e.g. with LAN800
    #labels are ok, but need index to access proper coords
    if len(labels) != len(full_labels):
        labels_indices = [full_labels.index(lbl) for lbl in labels]
        print(labels_indices)
    else:
        labels_indices = list(range(len(labels)))
        # used for coords 
    
    # load result variables
    isc_bootstrap = isc_utils.load_pickle(os.path.join(results_dir, f'{cond}/isc_results_{cond}_{n_boot}boot_pairWise{do_pairWise}.pkl'))
    isc_rois = pd.DataFrame(isc_bootstrap['isc'], columns=labels)
    isc_results_roi[cond] = isc_rois
    isc_median = isc_bootstrap['observed']
    ci = isc_bootstrap['confidence_intervals']
    p_values = isc_bootstrap['p_values']
    dist = isc_bootstrap['distribution']
    n_boot = setup['n_boot']

    isc_matrices[cond] = {region : isc_utils.vector_to_isc_matrix(isc_rois.iloc[:,col_j],diag=0) for col_j, region in enumerate(isc_rois.columns)}

    #stack for brain plotting/cond
    iscs_cond[cond] = isc_median
    isc_p_cond[cond] = p_values

    print(f"min p value: {np.min(p_values):.6f}, and max ISC : {np.max(isc_median):.6f}")
    print(f'Within condition FDR thresh : {isc_utils.fdr(p_values, q=0.05):.6f}, and Bonferroni : {isc_utils.bonferroni(p_values, alpha=0.05):.6f}')

# FDR thresh across all conditions since
all_ps = []
all_iscs = []
for cond in setup['conditions']:
    p = isc_p_cond[cond]
    all_ps.extend(p)
    all_iscs.extend(iscs_cond[cond]) # used for global scale/colorbar in plots
fdr_all_1samp = isc_utils.fdr(np.array(all_ps), q=0.05)
print('\nFDR across all conditions : ', fdr_all_1samp)

#-------------
# Project ISc results to brain maps 

isc_imgs = {} #used further to produce brain plots of each conditions
sig_isc_tables = {}
sig_isc_matrices = {}

# Generate stats tables for each conditions and save
for correction in ['FDR']: #['unc_01', 'FDR', 'bonf']: # between condition FDR / within bonferroni
    for cond in setup['conditions']: #'NHYPER']: #conditions:
        print('-----------------------------------------------')
        print(cond)

        maskers = isc_utils.load_pickle(os.path.join(results_dir, f'{cond}/maskers_{parcel_name}_{cond}_23sub.pkl'))
        atlas = maskers[0].labels_img # adjust number of ROI based on mask
        
        isc_median = iscs_cond[cond]
        p_values = isc_p_cond[cond]
        show = False

        if correction == 'unc_01':
            p_thresh = 0.01
            p_thresh_str = '0.01'
        elif correction == 'FDR':
            # p_thresh = isc_utils.fdr(p_values, q=0.05)
            p_thresh = fdr_all_1samp # all conditions!
            p_thresh_str = 'FDR'+ str(round(p_thresh, 4))
            show = True
        elif correction == 'bonf':
            p_thresh = isc_utils.bonferroni(p_values, alpha=0.05)
            p_thresh_str = 'Bonf'+ str(round(p_thresh, 4))
        
        title_view = f'ISC_{cond}_{correction}'
        file_id = f'ISC_{cond}_FDR05'

        #x = -48, -2, 6, 50
        cut_coords = (5) #(-48, -2, 6, 50)
        isc_img, isc_thresh, sig_df = visu_utils.project_isc_to_brain(
            atlas_img=atlas,
            isc_median=isc_median,
            atlas_labels=id_labels_dct,
            roi_coords = roi_coords,
            p_values=p_values,
            p_threshold=p_thresh, #modif in loop
            title = title_view,
            coords_bool_mask = labels_indices,
            save_path=None,
            show=show,
            display_mode='x',
            cut_coords_plot=cut_coords,
        )

        isc_imgs[cond] = (isc_img, isc_thresh)
        p_mask = p_values < p_thresh
        sig_labels = [labels[i] for i in range(len(labels)) if p_mask[i]]
        sig_isc_tables[cond] = sig_df #sig_df.sort_values(by='ISC', ascending=False)
        print('Sig ROIs :', sig_labels)
        
        if correction == 'FDR': # stack isc matrices for sig. regions
            sig_isc_matrices[cond] = {}
            for sig_region_name in sig_df['Label']:
                sig_isc_matrices[cond][sig_region_name] = isc_matrices[cond][sig_region_name]


        #save sign dfs as csv
        if sig_df.shape[0] > 0:
            # sig_labels = pd.DataFrame(sig_labels, columns=['ROI', 'Label', 'spearman_r', 'p_values', 'Coordinates (X,Y,Z)'])
            sig_df.columns = ['Region id', 'Label', 'ISC', 'P-value', 'Coordinates (X,Y,Z)']
            sig_df.to_csv(os.path.join(save_tables_1sample, f'ISC_{cond}_table_{correction}_thresh{round(p_thresh,4)}.csv'), index=False)

#%%
# ----------
# PUBLICATION GLASS BRAINS per conditions
# Save glass brain to post-hoc plot directory )

# compute color bar across all values/conditions (all_iscs)
from matplotlib import cm
from nilearn.image import new_img_like

vmin, vmax = np.min(all_iscs), np.max(all_iscs)
fake_data = np.zeros_like(atlas.get_fdata())
fake_data.flat[:len(all_iscs)] = all_iscs
synthetic_img = new_img_like(atlas, fake_data)
cmap = cm.get_cmap("Reds")

cbar = visu_utils.save_colorbar(
    img=synthetic_img,
    cmap=cmap,
    outpath=os.path.join(save_plots_1sample, "shared_isc_colorbar.png"),
    symmetric_cbar=False,
    offset=None,
    n_ticks=3,
    transparent=True
)

condition_names = {'HYPER' : 'Hyperalgesia', 'ANA' : 'Analgesia', 'NHYPER' : 'Control Hyperalgesia', 'NANA' : 'Control Analgesia'}

for i, (cond, (isc_img, isc_thresh)) in enumerate(isc_imgs.items()):
    # if i >= 1:
    #     break  # Stop after the first iteration
    print(cond)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.set_axis_off()

    title = f"{condition_names[cond]} (FDR q < .05)"
    display = plot_glass_brain(
        isc_img,
        threshold=isc_thresh,
        colorbar=False,
        display_mode='lr',
        plot_abs=False,
        cmap='Reds',
        title=None,
        axes=ax,
        vmax= np.max(all_iscs)
    )
    y = 0.20
    ax.text(0.1, y, 'L', transform=ax.transAxes,
            fontsize=20, ha='center')
    ax.text(0.9, y, 'R', transform=ax.transAxes,
            fontsize=20, ha='center')

    # Add title
    fig.suptitle(title, fontsize=25, y=0.85)
    fig.savefig(os.path.join(save_plots_1sample, f'glass_brain_{cond}_FDR05.png'), dpi=1000)
cbar
#%%
# PUBLICATION violin plots of median ISC across all contrasts conditions
reload(visu_utils)
# violin plots displaying median SIc diff per region
delta_dict = {cond: isc for cond, isc in iscs_cond.items()}
delta_dict.update({'ANA1': iscs_cond['ANA'], 'NANA1': iscs_cond['NANA']})
delta_dict.update({'HYPER1': iscs_cond['HYPER'], 'NHYPER1': iscs_cond['NHYPER']})
condition_labels = ['Hyper.', 'Ana.', 'Cont. Hyper', 'Cont. Ana', 'Ana', 'Cont. Ana', 'Hyper', 'Cont. Hyper.']
condition_labels = {
    'HYPER1': 'Hyper.',
    'ANA1': 'Ana.',
    'NHYPER1': 'Cont. Hyper.',
    'NANA1': 'Cont. Ana.',
    'ANA': 'Ana.',
    'NANA': 'Cont. Ana.',
    'HYPER': 'Hyper.',
    'NHYPER': 'Cont. Hyper.'
}
fig = visu_utils.plot_delta_condition_violin(delta_dict, min_y = np.min(all_iscs), max_y = np.max(all_iscs)+0.05,
                                             condition_labels = condition_labels, dot_size=12, lab_rotation=75,
                                             title = 'Change in ISC per region across conditions',
                                             ylabel = 'Median ISC')

#%%
#========================
# PLOT ISC matrices per conditions (SUPP MATERIAL) 
from sklearn.preprocessing import StandardScaler
reload(visu_utils)
from src.visu_utils import schaeffer_region_mapping

region_mapping = {}
consistent_isc_roi = ['7Networks_LH_Default_Temp_3', '7Networks_RH_Default_Temp_3']
# Concatenate matrices across all conditions and selected ROIs to find global color scale
concat_all = []

for region in consistent_isc_roi:
    for cond in conditions:
        mat = sig_isc_matrices[cond][region]
        concat_all.append(mat)

concat_mat = np.concatenate(concat_all, axis=0)
vmin, vmax = np.min(concat_mat), np.max(concat_mat)
print(f"Global ISC min: {vmin:.2f}, max: {vmax:.2f}")

# SHSS similarity matrices and rank 
#----------
shss = StandardScaler().fit_transform(Y['SHSS_score'].values.reshape(-1, 1)).flatten()
sim_mat_behav_matrix = isc_utils.compute_behav_similarity(
    shss, metric="euclidean", vectorize=False
)
sim_mat_behav_annak = isc_utils.compute_behav_similarity(
    shss, metric="annak", vectorize=False
)

sorted_indices = np.argsort(shss)
def reorder_matrix(mat, indices):
    return mat[np.ix_(indices, indices)]

# Plot RANKED IS CMATRICES : 1SAMPLE ISC
#-----------------------------------------
for region in consistent_isc_roi:
    anat_name = schaeffer_region_mapping[region]['anatomical_label']
    for cond in conditions:
        isc_mat = sig_isc_matrices[cond][region]    
        reordered_mat = reorder_matrix(isc_mat, sorted_indices)

        title = f"{cond} – {anat_name}"
        visu_utils.plot_simmat_isc(
            simmat=reordered_mat,
            title=title,
            x_label = 'ranked subjects (SHSS)',
            vmin=-vmax,
            vmax=vmax,
            show_colorbar=False,
            save_path=None,  # or a path if saving,
            tick_fontsize=50,
            label_fontsize=50,)

visu_utils.plot_isc_colorbar(
    vmin=-vmax,
    vmax=vmax,cmap="RdBu_r",
    label="Pairwise r",
    save_path=None,  # or PNG, PDF
    label_fontsize=40,
    tick_fontsize=30,
    dpi=600,
)

#%%

#%%
# get isc matrix for each condition
region_name = consistent_isc_roi[0]
ls_mat = []
for cond in conditions:
    mat = sig_isc_matrices[cond][region_name]
    ls_mat.append(mat)

    concat_mat = np.concatenate(ls_mat, axis=0)
    print(concat_mat.shape)
    min, max = np.min(concat_mat), np.max(concat_mat)
    
 # %%
#for voxel wise case, ignored otherwise
masker = isc_utils.load_pickle('/data/rainville/dSutterlin/projects/ISC_hypnotic_suggestions/results/imaging/ISC/{model_name}/HYPER/maskers_voxelWise_lanA800_HYPER_23sub.pkl')

# #%%
# reload(visu_utils)
# #rename conditions with full names
# conditions_full_names = {cond : full for cond, full in zip(conditions, ['Neutral (Analg.)', 'Analgesia', 'Neutral (Hyperalg.)', 'Hyperalgesia'])} #, 'Neutral (Hyper)', 'N_Ana', 'N_Hyper'])}
# # CAREFUL FOR NAME ORDER / CONDITIONS + full name
# visu_utils.plot_median_isc_dots(sig_isc_tables,conditions_full_names=conditions_full_names, title="Median ISC across significant ROIs")

# # Pie plot of sig regions in df
# #------------------------------
# # all_sugg pie-chart
# cond_pie = 'ANA'
# sig_rois_for_pie = list(sig_isc_tables[cond_pie]['Label'])
# sig_networks = [label.split('_')[2] if len(label.split('_')) > 2 else 'subcortical' for label in sig_rois_for_pie]

# visu_utils.plot_network_radar(sig_networks, title = f'Sig ISC during {cond_pie}')
# # # all_sugg pie-chart
# # cond_pie ='hyper_run'
# # sig_rois_for_pie = list(sig_df_conditions[cond_pie]['Label'])
# # sig_networks = [label.split('_')[2] if len(label.split('_')) > 2 else 'subcortical' for label in sig_rois_for_pie]
# visu_utils.plot_network_radar(sig_networks, title = f'Sig ISC during {cond_pie}')
#%%
# BLOCK MODELS : (ANA > HYPER, etc. run1>run2)
# ==========================
# Difference (Permutation) per condition visualization
model_name = model_names['model2_sugg']
results_dir = os.path.join(project_dir, f'results/imaging/ISC/{model_name}')
setup = isc_utils.load_json(os.path.join(results_dir, "setup_parameters.json"))

#create contrast output dir.
save_plots_contrasts = os.path.join(save_visu, 'plots_contrasts_isc')
os.makedirs(save_plots_contrasts, exist_ok=True)
save_tables_contrasts = os.path.join(save_visu, 'tables_contrasts_isc')
os.makedirs(save_tables_contrasts, exist_ok=True)

parcel_name = setup['atlas_name']
do_pairWise = setup['do_pairwise']
n_boot = setup['n_boot']

result_key = 'cond_contrast_permutation'
contrasts = ['ana_run-hyper_run','Ana-Hyper', 'NAna-NHyper'] if 'single-trial' not in model_name else ['Ana-N_Ana', 'Hyper-N_Hyper']
n_perm = setup['n_perm']

sig_rois = {}
interactive_views=[]

delta_iscs = {}
delta_p_values = {}

for i, cont in enumerate(contrasts):
    print(f'-------\n{cont}')
    file = os.path.join(results_dir, f'cond_contrast_permutation/isc_results_{cont}_{n_perm}perm_pairWise{do_pairWise}.pkl')
    isc_contrast = isc_utils.load_pickle(file)

    grouped_isc = isc_contrast['grouped_isc']        # Grouped ISC values
    distribution = isc_contrast['distribution']   
    delta_iscs[cont] = isc_contrast['observed_diff']    # Observed differences in ISC
    delta_p_values[cont] = isc_contrast['p_value']               # P-values for the ISC contrasts

    print(f"min p value: {np.min(delta_p_values[cont]):.6f}, and max ISC diff. (delta) : {np.max(delta_iscs[cont]):.6f}")
    print(f'Within condition FDR thresh : {isc_utils.fdr(delta_p_values[cont], q=0.05):.6f}, and Bonferroni : {isc_utils.bonferroni(delta_p_values[cont], alpha=0.05):.6f}')

# FDR thresh across all contrasts 
all_delta_p = []
all_delta_iscs = []
for cont in contrasts:
    all_delta_p.extend(delta_p_values[cont])
    all_delta_iscs.extend(delta_iscs[cont]) # used for global scale/colorbar in plots
fdr_all_block = isc_utils.fdr(np.array(all_delta_p), q=0.05)
print('\nFDR across all block (3) contrasts : ', fdr_all_block)

# %%
reload(visu_utils)
# ===========================
# SINGLE TRIAL MODEL : (Modulation - Neutre)
# Difference (Permutation) per condition visualization
model_name_single_trial = model_names['model1_single-trial']
results_dir = os.path.join(project_dir, f'results/imaging/ISC/{model_name_single_trial}')
setup = isc_utils.load_json(os.path.join(results_dir, "setup_parameters.json"))

#create visualization dir.
save_plots_contrasts = os.path.join(save_visu, 'plots_contrasts_isc')
os.makedirs(save_plots_contrasts, exist_ok=True)
save_tables_contrasts = os.path.join(save_visu, 'tables_contrasts_isc')
os.makedirs(save_tables_contrasts, exist_ok=True)

parcel_name = setup['atlas_name']
do_pairWise = setup['do_pairwise']
n_boot = setup['n_boot']

result_key = 'cond_contrast_permutation'
contrasts_trials = ['Ana-Hyper', 'NAna-NHyper'] if 'single-trial' not in model_name_single_trial else ['Ana-N_Ana', 'Hyper-N_Hyper']
n_perm = setup['n_perm']

# contrast_imgs_txt = {}

for i, cont in enumerate(contrasts_trials):
    print(f'----------\n {cont}')
    file = os.path.join(results_dir, f'cond_contrast_permutation/isc_results_{cont}_{n_perm}perm_pairWise{do_pairWise}.pkl')
    isc_contrast = isc_utils.load_pickle(file)

    grouped_isc = isc_contrast['grouped_isc']        # Grouped ISC values
    distribution = isc_contrast['distribution']      
    delta_iscs[cont] = isc_contrast['observed_diff']    # Observed differences in ISC
    delta_p_values[cont] = isc_contrast['p_value']               # P-values for the ISC contrasts

    print(f"min p value: {np.min(delta_p_values[cont]):.6f}, and max ISC diff. (delta) : {np.max(delta_iscs[cont]):.6f}")
    print(f'Within condition FDR thresh : {isc_utils.fdr(delta_p_values[cont], q=0.05):.6f}, and Bonferroni : {isc_utils.bonferroni(delta_p_values[cont], alpha=0.05):.6f}')

#%%
# compute gloabl FDR thresh across all (5) contrasts conditions
# p < 0.001* with 5 conditinns/ vs p < 0.003 with the 3 conditions from block model
for cont in contrasts_trials:
    p = delta_p_values[cont]
    all_delta_p.extend(p)
fdr_all = isc_utils.fdr(np.array(all_delta_p), q=0.05)
print(f'fdr_all: {fdr_all:.6f}')

#%%
#-------
# Create CONTRAST stats images + tables for BLOCK MODEL & TRIAL model
# Note : within FDR thresh is always around 0.001, similar to the global FDR (5 cond.)
contrast_imgs= {}
contrast_fdr_df = {}

for correction in ['unc_01', 'within_FDR', 'FDR', 'bonf']: # between condition FDR / within bonferroni
    print(f'-----{correction}')
    for i, cont in enumerate(contrasts+contrasts_trials):

        observed_diff = delta_iscs[cont]
        p_values = delta_p_values[cont]

        show = False
        save_fig = None
        if correction == 'unc_01':
            p_thresh = 0.01
            p_thresh_str = '0.01'
        elif correction == 'within_FDR':
            p_thresh = isc_utils.fdr(p_values, q=0.05)
            p_thresh_str = 'within_FDR'+ str(round(p_thresh, 4))
        elif correction == 'FDR':
            # p_thresh = isc_utils.fdr(p_values, q=0.05)
            p_thresh = fdr_all # all contrasts!
            p_thresh_str = 'FDR'+ str(round(p_thresh, 4))
            show = True
            save_fig = save_plots_contrasts + f'/ISC_contrast_{cont}_{p_thresh_str}.png'
        elif correction == 'bonf':
            p_thresh = isc_utils.bonferroni(p_values, alpha=0.05)
            p_thresh_str = 'Bonf'+ str(round(p_thresh, 4))
            show=True

        print(f'Correction thresh : {p_thresh}')

        if cont == 'NAna-NHyper':
            observed_diff = -observed_diff
            # !! Reverse contrast since the name was opposite of difference
            # NAna has stronger ISC than Hyper (~ .20 max)

        diff_img, diff_thresh, sig_df, _ = visu_utils.project_isc_to_brain_perm(
            atlas_img=atlas,
            isc_median=observed_diff,
            atlas_labels=id_labels_dct,
            roi_coords = roi_coords,
            p_values=p_values,
            p_threshold=p_thresh,
            title = f"Difference in ISC between {cont} ({correction})",
            save_path=False,
            show=show,
            save_fig_as = save_fig,
            display_mode='x',
            color='Reds'
        )
        contrast_imgs[cont] = (diff_img, diff_thresh)

        if correction == 'within_FDR': # save within contrast threshold.
            contrast_fdr_df[cont] = sig_df

        if sig_df.shape[0] > 0:
            # sig_labels = pd.DataFrame(sig_labels, columns=['ROI', 'Label', 'spearman_r', 'p_values', 'Coordinates (X,Y,Z)'])
            sig_df.columns = ['Region id', 'Label', 'Delta_r', 'p_values', 'Coordinates (X,Y,Z)']
            sig_df.to_csv(os.path.join(save_tables_contrasts, f'ISC_{cont}_table_{p_thresh_str}.csv'), index=False)

            print('Sig ROIs : ', sig_df['Label'].to_list())

#%%
# Load 1 sample ISC for runs/contexts : Hyper & Analgesia-context
# Will be used for plotting region ISC matrices
# stack contrast imgs in `contrast_imgs[cont]`

# Reload BLOCK model
model_name = model_names['model2_sugg']
results_dir = os.path.join(project_dir, f'results/imaging/ISC/{model_name}')
setup = isc_utils.load_json(os.path.join(results_dir, "setup_parameters.json"))

# Have been overwrite by trial model code !
save_plots_contrasts = os.path.join(save_visu, 'plots_contrasts_isc')
save_tables_contrasts = os.path.join(save_visu, 'tables_contrasts_isc')

result_key = 'isc_results'
interactive_views = {}
combined_conditions = ['ana_run','hyper_run'] # 'neutral']#setup['combined_conditions'] #['all_sugg', 'modulation', 'neutral']

for i, cond in enumerate(combined_conditions):
    print(f'Fetching 1 sample ISC values from BLOCK model, for contextual effect.\n {cond}')

    isc_bootstrap = isc_utils.load_pickle(os.path.join(results_dir, f'concat_suggs_1samp_boot/isc_results_{cond}_{n_boot}boot_pairWise{do_pairWise}.pkl'))
    isc_rois = pd.DataFrame(isc_bootstrap['isc'], columns=full_labels)
    isc_matrices[cond] = {region : isc_utils.vector_to_isc_matrix(isc_rois.iloc[:,col_j],diag=0) for col_j, region in enumerate(isc_rois.columns)}

    isc_median = isc_bootstrap['observed']
    ci = isc_bootstrap['confidence_intervals']
    p_values = isc_bootstrap['p_values']
    dist = isc_bootstrap['distribution']
    n_boot = setup['n_boot']

    iscs_cond[cond] = isc_median # one sample variables (vs. delta_iscs for contrasts)
    isc_p_cond[cond] = p_values

# build brain images and save tables
for correction in ['FDR']: #['unc_01', 'within_FDR', 'FDR', 'bonf']: 
        
    for cond in combined_conditions:
        print(cond)

        isc_median = iscs_cond[cond]
        p_values = isc_p_cond[cond]
        show = False
        save_fig_as  =None 

        if correction == 'unc_01':
            p_thresh = 0.01
            p_thresh_str = '0.01'
        elif correction == 'FDR':
            p_thresh = fdr_all_1samp # based on 4 conditions, .003. See note above
            p_thresh_str = 'FDR'+ str(round(p_thresh, 4))
            show = True
            save_fig_as = os.path.join(save_plots_1sample, f'ISC_stats_map_{cond}_FDR05.png')

        elif correction == 'bonf':
            p_thresh = isc_utils.bonferroni(p_values, alpha=0.05)
            p_thresh_str = 'bonf'+ str(round(p_thresh, 4))
        elif correction == 'within_FDR':
            p_thresh = isc_utils.fdr(p_values, q=0.05)
            p_thresh_str = 'within_FDR'+ str(round(p_thresh, 4))
            print('Within FDR p-threshold : ', p_thresh_str)
        
        cut_coords = (5) #(-48, -2, 6, 50)
        isc_img, isc_thresh, sig_df = visu_utils.project_isc_to_brain(
            atlas_img=atlas,
            isc_median=isc_median,
            atlas_labels=id_labels_dct,
            roi_coords = roi_coords,
            p_values=p_values,
            p_threshold=p_thresh,
            title = f'ISC {cond} - {p_thresh_str}',
            coords_bool_mask = labels_indices,
            save_path=None,
            show=show,
            save_fig_as=save_fig_as,
            display_mode='x',
            cut_coords_plot=cut_coords,
        )
        
        isc_imgs[cond] = (isc_img, isc_thresh)

        if correction == 'FDR': # stack isc matrices for sig. regions
            for sig_region_name in sig_df['Label']:
                sig_isc_matrices[cond][sig_region_name] = isc_matrices[cond][sig_region_name]
        
            sig_isc_matrices[cond] = {}
            sig_isc_tables[cond] = sig_df

        if sig_df.shape[0] > 0:
            # sig_labels = pd.DataFrame(sig_labels, columns=['ROI', 'Label', 'spearman_r', 'p_values', 'Coordinates (X,Y,Z)'])
            sig_df.columns = ['Region id', 'Label', 'Pearson_r', 'p_values', 'Coordinates (X,Y,Z)']
            sig_df.to_csv(os.path.join(save_tables_1sample, f'ISC_{cond}_table_{p_thresh_str}.csv'), index=False)

            sig_df.sort_values(by='Pearson_r', ascending=False)
            print('Sig ROIs : ', sig_df['Label'].to_list(), '\n')

#%%
# ================
# PUBLICATION ANATOMICAL brain plots for CONTRAST contextual effect (FDR global)

contrast_ana_hyper_run, delta_thresh = contrast_imgs['ana_run-hyper_run']
max_diff = np.max(np.abs(contrast_ana_hyper_run.get_fdata()))

view_img(contrast_ana_hyper_run, threshold=delta_thresh, title='Ana-Hyper (FDR q < .05)', cmap='coolwarm', vmax=max_diff)
plotting.plot_stat_map(contrast_ana_hyper_run, threshold=delta_thresh, title='Ana-Hyper (FDR q < .05)', cmap='Reds', vmax=max_diff, 
                       display_mode = 'mosaic')
#%%
# PUBLICATION SLICES 
COORDS = {
    'x': [54, 42, -48, -60],
    'y' : [-22, 6],
    'z': [42, 58],

}

SAVE_TO = os.path.join(save_plots_contrasts, 'ana_run-hyper_run')
os.makedirs(SAVE_TO, exist_ok=True)

cm = 'Reds'

visu_utils.save_slices(contrast_ana_hyper_run, 
                  COORDS,
                  SAVE_TO,
                  img_id = 'run_contrast',
                  cmap = 'Reds'
                  )
# save colorbar
fig_path = os.path.join(SAVE_TO, 'colorbar.png')
cb_ab = visu_utils.save_colorbar(contrast_ana_hyper_run,
              cmap='Reds',
              outpath=fig_path,
              symmetric_cbar=False, # false because we dont keep negative coeff
              offset=0,
              n_ticks=3,
              transparent=True)

#%%
# Plot ISC MATRICES for PUBLICATION
#------------------------------
# Ana-context &  Hyper-context only
min_pairwise_isc = 0
max_pairwise_isc = 0
for region, isc_mat in sig_isc_matrices[cond].items():
    min = np.min(isc_mat)
    max = np.max(isc_mat)
    if max > max_pairwise_isc: max_pairwise_isc = max
    if min < min_pairwise_isc: min_pairwise_isc = min

print(min_pairwise_isc, max_pairwise_isc)

#%%
reload(visu_utils)
for cond in combined_conditions:
    for region, isc_mat in sig_isc_matrices[cond].items():

        visu_utils.plot_simmat_isc(
            simmat=isc_mat,
            title=f"{region}",

            vmin=-max_pairwise_isc,
            vmax=max_pairwise_isc,
            show_colorbar=True,
            save_path=None
        )

#%%
# Find Atlas region of interest
def filter_atlas(atlas_img, atlas_labels_dict, keep_ids):
    atlas_data = atlas_img.get_fdata()
    mask_data = np.isin(atlas_data, keep_ids) * atlas_data
    filtered_img = new_img_like(atlas_img, mask_data)
    
    # Build new labels dict for the kept IDs
    new_labels = {roi: label for roi, label in atlas_labels_dict.items() if roi in keep_ids}
    return filtered_img, new_labels

# Plot the filtered atlas with clear discrete colorbar
def plot_filtered_atlas(filtered_img, filtered_labels, cmap='tab20'):
    label_ids = list(filtered_labels.keys())
    label_names = list(filtered_labels.values())

    return plotting.view_img(
        filtered_img,
        cmap=cmap,
        colorbar=True,
        title="Filtered Atlas",
        symmetric_cmap=False,
        
    )
def plot_single_roi(atlas_img, roi_id, atlas_labels=None):

    mask_img = math_img(f"img == {roi_id}", img=atlas_img)

    title = f"ROI {roi_id}"
    if atlas_labels and roi_id in atlas_labels:
        title += f" {atlas_labels[roi_id]}"

    print(f"Displaying region {roi_id}: {title}")
    return plotting.view_img(mask_img, cmap='Reds')

#%%
plot_single_roi(atlas, 189, id_labels_dct) # 30 = Default_Temp_3 LH
plot_filtered_atlas(*filter_atlas(atlas, id_labels_dct, []), cmap='tab20')

#%%
# IS-RSA SHSS~ISC
# ===============
# Load results and save tables

load_dir = "/data/rainville/dSutterlin/projects/ISC_hypnotic_suggestions/results/imaging/RSA/isc-RSA_ext_conds_sugg-pain_tian216_2tails"
RESULT_FILE = "rsa_SHSS-behav_isc-sugg10000perm.pkl"
rsa_isc_sugg = isc_utils.load_pickle(os.path.join(load_dir, RESULT_FILE))

RESULT_FILE = "rsa_SHSS-behav_isc-pain10000perm.pkl"
rsa_isc_pain = isc_utils.load_pickle(os.path.join(load_dir, RESULT_FILE))

save_rsa = os.path.join(load_dir, "post-hoc_VISU-tables_NEW")
os.makedirs(save_rsa, exist_ok=True)

save_tables_rsa = os.path.join(save_rsa, 'tables_rsa')
os.makedirs(save_tables_rsa, exist_ok=True)
save_plots_rsa = os.path.join(save_rsa, 'plots_rsa')
os.makedirs(save_plots_rsa, exist_ok=True)

n_perm = 10000

rsa_tables_fdr = {} # changed for `euclidean`` in original script
rsa_dfs = {}
models = ['annak', 'euclidian']
model = "annak"

rsa_stats_imgs = {}
rsa_p_values = []

# Load rsa results and find FDR thresh
for model in models:
    for cond in ["ana_run", "hyper_run"]:  # , 'HYPER', 'ANA']: #, 'all_sugg', 'neutral']:

        model_cond_key = f"{model}_{cond}"

        print("cond", model_cond_key)
        rsa_df = rsa_isc_sugg[model][cond].sort_index(
            ascending=True
        )  # to match the atlas labels
        rsa_dfs[model_cond_key] = rsa_df

        # === Prepare variables for projection ===
        correlations = rsa_df["spearman_r"].values
        p_values = rsa_df["p_values"].values
        rsa_p_values.extend(p_values)

        # !! Shift ROI id in df by 1, since starts at 0 and ends at 215 (atlas is 216)
        rsa_df['ROI'] = rsa_df['ROI'] + 1 # fixed in rsa-isc_ext_model.py, rm if re-run !!
        fdr_p = isc_utils.fdr(p_values, q=0.05)
        print(f"Within FDR threshold: {fdr_p:.4f}")
        print(f'Min p values : {p_values.min():.4f}')
        print(f'max spearman: {correlations.max():.4f}')

# compute across condition FDR thresh
fdr_all_rsa = isc_utils.fdr(np.array(rsa_p_values), q=0.05)
print(f"Across conditions FDR threshold: {fdr_all_rsa:.4f}")

# Project RSA values to brain and save tables
for correction in ['unc01', 'within_FDR', 'FDR', 'bonf']:
    for rsa_cond in rsa_dfs.keys():  # ['ana_run', 'hyper_run']: #, 'HYPER', 'ANA']: #, 'all_sugg', 'neutral']:

        rsa_df = rsa_dfs[rsa_cond]
        correlations = rsa_df["spearman_r"].values
        p_values = rsa_df["p_values"].values
        region_id = rsa_df["ROI"].values

        show = False
        if correction == 'unc01':
            p_thresh = 0.01
            p_thresh_str = '0.01'
        if correction == 'within_FDR':
            p_thresh = isc_utils.fdr(p_values, q = 0.05)
            p_thresh_str = 'within_FDR' + str(round(p_thresh, 4))
        if correction == 'FDR':
            p_thresh = fdr_all_rsa
            p_thresh_str = 'FDR'+ str(round(p_thresh, 4))
            show = True
        if correction == 'bonf':
            p_thresh = isc_utils.bonferroni(p_values, alpha=0.05)
            p_thresh_str = 'Bonf' + str(round(p_thresh, 4))
        
        rsa_img, rsa_thresh, sig_df, stats_img = visu_utils.project_isc_to_brain_perm(
            atlas_img=atlas,
            isc_median=correlations,
            atlas_labels=id_labels_dct,
            roi_coords=roi_coords,
            p_values=p_values,
            p_threshold=p_thresh,  #!!
            title=f'IS-RSA SHSS~{rsa_cond} ({p_thresh_str})',  # "RSA-ISC: Suggestion-Pain Similarity (FDR<.05)",
            save_path=None,
            show=show,
            display_mode="x",
            cut_coords_plot=(5),  # (-52, -40, 34),
            color="coolwarm",
        )

        if correction == 'FDR' and sig_df.shape[0] > 0:
            
            rsa_stats_imgs[rsa_cond] = (rsa_img, rsa_thresh)
            rsa_tables_fdr[rsa_cond] = sig_df
        
            sig_df.columns = ['ROI', 'Label', 'Spearman r', 'p-values', 'Coordinates (X,Y,Z)']
            sig_df.to_csv(os.path.join(save_tables_rsa, f'RSA_{rsa_cond}_table_{p_thresh_str}.csv'), index=False)

            print('Sig ROIs : ', sig_df['Label'].to_list(), '\n')

#%%
views_rsa = []
for cond_rsa in rsa_dfs.keys():
    view = view_img(rsa_stats_imgs[cond_rsa][0], threshold=rsa_stats_imgs[cond_rsa][1], title=f'IS-RSA SHSS~{cond_rsa} (FDR<.05)')
    views_rsa.append(view)

#%% Find which region to plot ISC matrix
plot_single_roi(atlas, 187)
keep_ids = list(rsa_tables_fdr['annak_ana_run']['ROI'].astype(float))
filt_atlas_img, filt_labels = filter_atlas(atlas, id_labels_dct, keep_ids)
plot_filtered_atlas(filt_atlas_img, filt_labels)

#%%
# SAVE ISC matrices for sig. regions in RSA
#-------------------------------------------

# Fetch ISC mat from sgnificant IS-RSA regions across conditions
# Save and plot
save_sig_rsa_matrices = os.path.join(save_plots_rsa, 'isc_mat_FDR_rsa')
os.makedirs(save_sig_rsa_matrices, exist_ok=True)

for rsa_cond in rsa_tables_fdr.keys():  # ['ana_run', 'hyper_run']: #, 'HYPER', 'ANA']: #, 'all_sugg', 'neutral']:
    print(rsa_cond)
    sig_df = rsa_tables_fdr[rsa_cond]
    
    if 'ana_run' in rsa_cond: # to get ISC mat key
        sugg_cond = 'ana_run'
    elif 'hyper_run' in rsa_cond:
        sugg_cond = 'hyper_run'
    
    for sig_region in sig_df['Label']:

        anatomical_name = schaeffer_region_mapping[sig_region]['anatomical_label']
        region_id = schaeffer_region_mapping[sig_region]['region_index']
        isc_mat = isc_matrices[sugg_cond][sig_region] # get from 1 sample ISC dct
        ranked_indices = np.argsort(Y['SHSS_score'].values)
        isc_mat = isc_mat[np.ix_(ranked_indices, ranked_indices)] # reorder

        vmax = np.max(np.abs(isc_mat))
        
        fig_path = os.path.join(save_sig_rsa_matrices, f'{rsa_cond}_{sig_region}_id{region_id}.png')
        visu_utils.plot_simmat_isc(
            simmat= isc_mat,
            x_label = None,#' ranked subjects (SHSS)'
            colorbar_name = None, 
            tick_fontsize=60,
            label_fontsize=70,
            title=f"{anatomical_name}",
            vmin=-vmax,
            vmax=vmax,
            show_colorbar=True,
            save_path=fig_path
        )


#%%
# PUBLICATION BRAIN PLOTS for IS-RSA SHSS~ISC
#========================
cmap = 'RdBu_r'

# ANA-RUN
#--------
COORDS = {
    'x': [-54, 0, 8, 56, 64],
    'y' : [-32, -26, -4, 6],
    'z': [42, 58],

}

SAVE_TO = os.path.join(save_plots_rsa, 'annak_SHSS-ISC_ana_run') #!!!
os.makedirs(SAVE_TO, exist_ok=True)
IMG = rsa_stats_imgs['annak_ana_run'][0]
IMG_ID = 'FDR_ana_run'

visu_utils.save_slices(IMG, 
                  COORDS,
                  SAVE_TO,
                  img_id = IMG_ID,
                  cmap = cmap
                  )
# save colorbar
fig_path = os.path.join(SAVE_TO, 'colorbar.png')
cb_ab = visu_utils.save_colorbar(IMG,
              cmap=cmap,
              outpath=fig_path,
              symmetric_cbar=False, # false because we dont keep negative coeff
              offset=0,
              n_ticks=3,
              transparent=True)
#%%
# HYPER-RUN
#----------
views_rsa[1]
COORDS = {
    'x': [-54, -48, -44],
    'y' : [-40],
    'z': [24],

}

SAVE_TO = os.path.join(save_plots_rsa, 'annak_SHSS-ISC_hyper_run') #!!!
os.makedirs(SAVE_TO, exist_ok=True)
IMG = rsa_stats_imgs['annak_hyper_run'][0] #!!
IMG_ID = 'FDR_hyper_run' #!!

visu_utils.save_slices(IMG, 
                  COORDS,
                  SAVE_TO,
                  img_id = IMG_ID,
                  cmap = cmap
                  )
# save colorbar
fig_path = os.path.join(SAVE_TO, 'colorbar.png')
cb_ab = visu_utils.save_colorbar(IMG,
              cmap= cmap,
              outpath=fig_path,
              symmetric_cbar=False, # false because we dont keep negative coeff
              offset=0,
              n_ticks=3,
              transparent=True)

#%%
# ANA-RUN EUCLIDEWAN MODEL
#----------
views_rsa[2]
COORDS = {
    'x': [32, 36],
    'y' : [36],
    'z': [50],

}

SAVE_TO = os.path.join(save_plots_rsa, 'euclidean_SHSS-ISC_ana_run') #!!!
os.makedirs(SAVE_TO, exist_ok=True)
IMG = rsa_stats_imgs['euclidian_ana_run'][0]
IMG_ID = 'FDR_ana_run'

visu_utils.save_slices(IMG, 
                  COORDS,
                  SAVE_TO,
                  img_id = IMG_ID,
                  cmap = cmap
                  )
# save colorbar
fig_path = os.path.join(SAVE_TO, 'colorbar.png')
cb_ab = visu_utils.save_colorbar(IMG,
              cmap= cmap,
              outpath=fig_path,
              symmetric_cbar=False, # false because we dont keep negative coeff
              offset=0,
              n_ticks=3,
              transparent=True)

#%%
#====================================================================
    roi_labels = rsa_df["ROI"].values  # assumes label matches atlas
    unc_p = 0.01
    title = f"IS-RSA : Hypnotic susceptibility ~ {cond} (FDR<.05)"

    if cond == "hyper_run":
        cut_coords_plot = (-60, -56, -46, -42)
    elif cond == "ana_run":
        cut_coords_plot = (-54, 0, 56)
    else:
        cut_coords_plot = None

    title = f"IS-RSA : Hypnotic susceptibility ~ {cond} (FDR<.05)"
    # === Visualize with your existing function ===
    

    if sig_labels.shape[0] > 0:
        # sig_labels = pd.DataFrame(sig_labels, columns=['ROI', 'Label', 'spearman_r', 'p_values', 'Coordinates (X,Y,Z)'])
        sig_labels.columns = [
            "ROI",
            "Label",
            "spearman_r",
            "p_values",
            "Coordinates (X,Y,Z)",
        ]
        sig_labels.to_csv(
            os.path.join(
                output_dir, f"{RESULT_FILE[:-13]}_{model}_{cond}_table-FDR.csv"
            ),
            index=False,
        )

    tables[model][cond] = sig_labels
    views[cond] = plotting.view_img(
        rsa_img,
        threshold=rsa_thresh,
        title=f"RSA suggestion - pain similarity {cond}",
        colorbar=True,
        symmetric_cmap=False,
        cmap="coolwarm",
    )

stats_imgs[0].to_filename(
    os.path.join(output_dir, f"{RESULT_FILE[:-13]}_{model}_ana_run.nii.gz")
) 
#%%
display_mode = 'z'
cut_coords = None
im = plot_stat_map(
    contrast_ana_hyper_run,
    bg_img=bg_mni,
    title=None,
    threshold=delta_thresh,
    vmax=max_diff,
    black_bg=False,
    colorbar=True,
    display_mode=display_mode,
    cut_coords=cut_coords,
    cmap='Reds',
    draw_cross=False,
)
plt.show()
#%%
if save_fig_as is not None:
plt.savefig(save_fig_as, dpi=1000, bbox_inches="tight")
print(f"ISC projection plot saved to {save_fig_as}")

#%%
    vmax = isc_median.max()
    views[cond] = view_img_on_surf(isc_img, threshold=isc_thresh,vmax = vmax,cmap = color_isc, symmetric_cmap = False,title = title_view, surf_mesh='fsaverage')

    interactive_view = view_img(
            isc_img,
            threshold=isc_thresh,
            title=title_view,
            symmetric_cmap=False,
            cmap = color_isc
                            )
    interactive_views[cond] = interactive_view
    # interactive_view.save_as_html(os.path.join(clean_save_to, f'{title_view}.html'))
    # views[cond].save_as_html(os.path.join(clean_save_to, f'{save_name}.html'))

from src import visu_utils
reload(visu_utils)
reload(isc_utils)
                
#%%
    # save tables !
    if sig_df.shape[0] > 0:
        # sig_labels = pd.DataFrame(sig_labels, columns=['ROI', 'Label', 'spearman_r', 'p_values', 'Coordinates (X,Y,Z)'])
        sig_df.columns = ['Region id', 'Label', 'Pearson_r', 'p_values', 'Coordinates (X,Y,Z)']
        sig_df.to_csv(os.path.join(save_visu, f'contrast_ISC_{cont}_table-FDR.csv'), index=False)

        sig_df_conditions_contrast[cont] = sig_df
        sig_dfs_contrasts[cont] = sig_df.sort_values(by='Pearson_r', ascending=False)
        print('Sig ROIs : ', sig_df['Label'].to_list())


    views[cont] = view_img_on_surf(diff_img, threshold=diff_thresh, surf_mesh='fsaverage')
    sig_rois[cont] = sig_df

#%%
#SAVE CONTRAST PLOTS 
save_plots_contrasts = os.path.join(post_hoc_dir, 'brain_plots_contrasts')
os.makedirs(save_plots_contrasts, exist_ok=True)

cont_img, thresh_cont = contrast_imgs['ana_run-hyper_run']
cont_name = 'ana_run-hyper_run'
glass_fig = plot_glass_brain(
    cont_img,
    threshold=thresh_cont,
    colorbar= True,
    display_mode='lyr',
    plot_abs=False,

    cmap='Reds',
    title=None
)

glass_fig._colorbar_ax.tick_params(labelsize=20)

# Shift colorbar right
pos = glass_fig._colorbar_ax.get_position()
new_x0 = pos.x0 + 0.02
glass_fig._colorbar_ax.set_position([new_x0, pos.y0, pos.width, pos.height])

# Round tick labels to 2 decimal places
ticks = glass_fig._colorbar_ax.get_yticks()
glass_fig._colorbar_ax.set_yticklabels([f'{tick:.2f}' for tick in ticks])

for name, ax in glass_fig.axes.items():
    if name in ['left', 'right']:
        if name == 'l':
            ax.annotate('L', xy=(0, 1.02), xycoords='axes fraction',
                        fontsize=25, ha='center', va='bottom')
        elif name == 'r':
            ax.annotate('R', xy=(1, 1.02), xycoords='axes fraction',
                        fontsize=25, ha='center', va='bottom')
#glass_fig.annotate(size = 18)
glass_fig.savefig(os.path.join(save_plots_contrasts, f'glass_brain_{cont_name}_FDR05.png'), dpi=1000)


#%%
import matplotlib.pyplot as plt
import matplotlib.colorbar as cb
import matplotlib.cm as cm
import numpy as np

img_data = diff_img.get_fdata()

cmap = cm.get_cmap("Reds")
norm = plt.Normalize(vmin=np.min(img_data), vmax=np.max(img_data))  # Adjust based on your stat range

# Create figure just for colorbar
fig, ax = plt.subplots(figsize=(2, 8))
cb1 = cb.ColorbarBase(ax, cmap=cmap, norm=norm, orientation='vertical')
cb1.set_label('ISC Value', fontsize=20)
cb1.ax.tick_params(labelsize=16)
fig.savefig("colorbar_only.png", dpi=1000, bbox_inches="tight")


#%%
# Save at high resolution
fig.savefig("high_res_surface_plot.png", dpi=1000, bbox_inches="tight")

# Display
show()


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


#%%
#=======================
# Difference for high vs low SHSS
result_key = 'cond_contrast_permutation'
# conditions = ['Hyper', 'Ana', 'NHyper', 'NAna']
contrasts = ['Hyper-Ana', 'Ana-Hyper', 'NHyper-NAna',] if 'single-trial' not in model_name else ['Ana-N_Ana', 'Hyper-N_Hyper']

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
        diff_img, diff_thresh, sig_df = visu_utils.project_isc_to_brain_perm(
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
        #  sig_df = [labels[i] for i in range(len(labels)) if p_mask[i]]
        print('Sig ROIs :', sig_df)
 
        surf_views[shss_grp].append(view_img_on_surf(diff_img, threshold=diff_thresh, surf_mesh='fsaverage'))
        sig_rois[shss_grp] = sig_df

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
conditions = ['ANA', 'HYPER'] # 'ana_run', 'hyper_run']
n_perm = 10000
behav_ls = [
    'Chge_hypnotic_depth_median_grp', 'SHSS_score_median_grp', 'raw_change_HYPER_median_grp',
    'raw_change_ANA_median_grp', 'total_chge_pain_hypAna_median_grp',
    'Mental_relax_absChange_median_grp', 'Abs_diff_automaticity_median_grp'
]
behav_ls = ['raw_change_HYPER_median_grp', 'raw_change_ANA_median_grp', 'total_chge_pain_hypAna_median_grp']
behav_ls = ['Chge_hypnotic_depth_median_grp', 'Abs_diff_automaticity_median_grp','Mental_relax_absChange_median_grp' ] #, 'Abs_diff_automaticity_median_grp'] #, 'SHSS_score_median_grp', 'total_chge_pain_hypAna_median_grp'] #, 'SHSS_score', 'Abs_diff_automaticity', 'total_chge_pain_hypAna']
behav_ls = ['total_chge_pain_hypAna_median_grp']

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
        unc_p = 0.05
        print(f'FDR thresh : {fdr_p}')
        reload(visu_utils)

        title = f"{cond} diff for {y} (FDR < 0.05)"
        cut_coords = (5)
        diff_img, diff_thresh, sig_df = visu_utils.project_isc_to_brain_perm(
            atlas_img=atlas,
            isc_median=observed_diff,
            atlas_labels=id_labels_dct,
            roi_coords = roi_coords,
            p_values=p_values,
            p_threshold=fdr_p,
            title = title,# f"Difference in ISC for {cond} high SHSS > low SHSS (p unc. < {unc_p})",
            save_path=None,
            show=True,
            display_mode='x',
            cut_coords_plot=cut_coords,
        )
        sig_rois_cond[cond][y] = sig_df
        print(sig_df)
        views[cond][y] = plotting.view_img(diff_img,  title=f"grp diff {cond} {y}")
        # view_img_on_surf(diff_img, threshold=diff_thresh, surf_mesh='fsaverage')

#%%
# combine unc. p < 0.05 table for Ana and Hyper to get ROI that interact with SHSS

ana_unc05_shss = sig_rois_cond['ANA']['SHSS_score_median_grp']
hyper_unc05_shss = sig_rois_cond['HYPER']['SHSS_score_median_grp']

unc05_ana_hyper_grp_contrast = pd.concat([ana_unc05_shss, hyper_unc05_shss], axis=0).sort_values(by='ROI', ascending=True)

save_to = '/data/rainville/dSutterlin/projects/ISC_hypnotic_suggestions/results/imaging/msc_output'
unc05_ana_hyper_grp_contrast.to_csv(os.path.join(save_to, 'shss_median-split_ana_concat_hyper_unc05.csv'), index=False)


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
        ))
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
