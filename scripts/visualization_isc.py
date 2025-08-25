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

model_is = 'model3-shock'
model_is = 'model2_sugg-sensaas'
model_is = 'model5_sugg'
model_is = 'model1_single-trial'
model_is = 'model2_sugg'

# model_is = 'model3_shock_200'
# model_is = 'model1_sugg_200_run'
# model_is = 'model3_shock_200_run'
# model_is = 'model1_sugg_200'
# model_is = 'model1_sugg_run'


ref_preproc_models = { 'model2' : model_names['model1_sugg_200']}
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

# Plot the surface atlas using Nilearn
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
reload(visu_utils)

iscs_cond = {}
isc_p_cond = {}
isc_imgs = {} #used further to produce brain plots of each conditions
views = {}
surf = {}
interactive_views = []
sig_dfs_one_sample = {}

save_plots_1sample = os.path.join(save_visu, 'plots_1sample_isc')
os.makedirs(save_plots_1sample, exist_ok=True)
save_tables_1sample = os.path.join(save_visu, 'tables_1sample_isc')
os.makedirs(save_tables_1sample, exist_ok=True)

for correction in ['unc_01', 'FDR']: #VAriables are stored for FDR
    # Generate stats tables for each conditions and save
    for cond in setup['conditions']: #'NHYPER']: #conditions:
        print('----------------------------------------------------')
        print(cond)

        maskers = isc_utils.load_pickle(f'/data/rainville/dSutterlin/projects/ISC_hypnotic_suggestions/results/imaging/ISC/{model_name}/{cond}/maskers_{parcel_name}_{cond}_23sub.pkl')
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
        
        isc_bootstrap = isc_utils.load_pickle(os.path.join(project_dir, f'results/imaging/ISC/{model_name}/{cond}/isc_results_{cond}_{n_boot}boot_pairWise{do_pairWise}.pkl'))
        isc_rois = pd.DataFrame(isc_bootstrap['isc'], columns=labels)
        isc_results_roi[cond] = isc_rois
        isc_median = isc_bootstrap['observed']
        # print(isc_median, np.mean(isc_rois))
        ci = isc_bootstrap['confidence_intervals']
        p_values = isc_bootstrap['p_values']
        dist = isc_bootstrap['distribution']
        n_boot = setup['n_boot']
        
        iscs_cond[cond] = isc_median
        isc_p_cond[cond] = p_values
        if correction == 'unc_01':
            fdr_p = 0.001
        elif correction == 'FDR':
            fdr_p = isc_utils.fdr(p_values, q=0.05)

        bonf_p = isc_utils.bonferroni(p_values, alpha=0.05)
        print(f'FDR thresh : {isc_utils.fdr(p_values, q=0.05)}')
        
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
            p_threshold=fdr_p, #modif in loop
            title = title_view,
            coords_bool_mask = labels_indices,
            save_path=None,
            show=True,
            display_mode='x',
            cut_coords_plot=cut_coords,
        )

        isc_imgs[cond] = (isc_img, isc_thresh)
        p_mask = p_values < fdr_p
        sig_labels = [labels[i] for i in range(len(labels)) if p_mask[i]]
        sig_dfs_one_sample[cond] = sig_df #sig_df.sort_values(by='ISC', ascending=False)
        print('Sig ROIs :', sig_labels)
        
        #save sign dfs as csv
        if sig_df.shape[0] > 0:
            # sig_labels = pd.DataFrame(sig_labels, columns=['ROI', 'Label', 'spearman_r', 'p_values', 'Coordinates (X,Y,Z)'])
            sig_df.columns = ['Region id', 'Label', 'ISC', 'P-value', 'Coordinates (X,Y,Z)']
            sig_df.to_csv(os.path.join(save_tables_1sample, f'ISC_{cond}_table_{correction}_thresh{round(fdr_p,4)}.csv'), index=False)

#%% Plot ISC distribution per condition
for cond in setup['conditions']:
    isc_values = iscs_cond[cond]
    p_values = isc_p_cond[cond]
    # histogram with distribution on the same plot. Use the same scale for all conditions
    # Add a lengend witht a distribution per conditions
    # Add each condition median in a vertical line with similar color as the distribution
from scipy.stats import gaussian_kde

fig, axes = plt.subplots(2, 2, figsize=(10, 6), sharex=True, sharey=True)
axes = axes.flatten()

x_vals = np.linspace(-0.2, 0.5, 500)

for ax, cond in zip(axes, conditions):
    values = iscs_cond[cond].astype(float)
    kde = gaussian_kde(values)
    ax.plot(x_vals, kde(x_vals), color=colors[cond], label=cond)
    ax.fill_between(x_vals, kde(x_vals), alpha=0.3, color=colors[cond])
    median_val = np.median(values)
    ax.axvline(median_val, color=colors[cond], linestyle='--')
    ax.set_title(cond)
    ax.set_xlabel('ISC')
    ax.set_ylabel('Density')

# Global legend
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', ncol=4)

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()


#%%
# FDR thresh across all conditions
all_ps = []
for cond in setup['conditions']:
    p = isc_p_cond[cond]
    all_ps.extend(p)
fdr_all = isc_utils.fdr(np.array(all_ps), q=0.05)
print('FDR across all conditions : ', fdr_all)

# Re-comute with this threshold
correction = fdr_all
for cond in setup['conditions']:

        isc_bootstrap = isc_utils.load_pickle(os.path.join(project_dir, f'results/imaging/ISC/{model_name}/{cond}/isc_results_{cond}_{n_boot}boot_pairWise{do_pairWise}.pkl'))
        isc_rois = pd.DataFrame(isc_bootstrap['isc'], columns=labels)
        isc_median = isc_bootstrap['observed']
        p_values = isc_bootstrap['p_values']
        dist = isc_bootstrap['distribution']
        
        title_view = f'ISC_{cond}_{correction}'

        #x = -48, -2, 6, 50
        cut_coords = (5) #(-48, -2, 6, 50)
        isc_img, isc_thresh, sig_df = visu_utils.project_isc_to_brain(
            atlas_img=atlas,
            isc_median=isc_median,
            atlas_labels=id_labels_dct,
            roi_coords = roi_coords,
            p_values=p_values,
            p_threshold=correction, #modif in loop
            title = title_view,
            coords_bool_mask = labels_indices,
            save_path=None,
            show=True,
            display_mode='x',
            cut_coords_plot=cut_coords,
        )
        global_fdr_img = (isc_img, isc_thresh)
        p_mask = p_values < correction
        sig_labels = [labels[i] for i in range(len(labels)) if p_mask[i]]
        sig_dfs_one_sample[cond] = sig_df #sig_df.sort_values(by='ISC', ascending=False)
        print('Sig ROIs :', sig_labels)


#%%

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

#%%

#%%
# BRain PLOTS
# SAve glass brain to post-hoc plot directory )
for cond, (isc_img, isc_thresh) in isc_imgs.items(): 

    print(cond)

    glass_fig = plot_glass_brain(
        isc_img,
        threshold=isc_thresh,
        colorbar= False if cond == 'HYPER' else True,
        display_mode='lyr',
        plot_abs=False,
    
        cmap='Reds',
        title=None
    )

    # Adjust colorbar tick size
    # if glass_fig._colorbar_ax is not None:
    if cond == 'HYPER': 
        continue
    else:
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
    glass_fig.savefig(os.path.join(save_visu, f'glass_brain_{cond}_FDR05.png'), dpi=1000)


    # interactive_view.save_as_html(os.path.join(clean_save_to, f'{title_view}.html'))
#%%

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
combined_conditions = ['ana_run', 'hyper_run'] # 'neutral']#setup['combined_conditions'] #['all_sugg', 'modulation', 'neutral']
# n_scans = [468, 200, 268] #[438, 188, 250]
views = {}
sig_df_conditions = {}
isc_matrices = {}
for i, cond in enumerate(combined_conditions):
    print('----------------------------------------------------')
    print(cond)
    # scans = n_scans[i]
    #masker =utils.load_pickle(os.path.join(results_dir, cond, f'maskers_{atlas_name}_{cond}_{n_sub}sub.pkl'))
    #isc_bootstrap = utils.load_pickle(all_results_paths[result_key][cond])
    # masker = utils.load_pickle(f'/data/rainville/dSutterlin/projects/ISC_hypnotic_suggestions/results/imaging/ISC/{model_name}/HYPER/maskers_schafer100_2mm_HYPER_23sub.pkl')
    isc_bootstrap = isc_utils.load_pickle(f'/data/rainville/dSutterlin/projects/ISC_hypnotic_suggestions/results/imaging/ISC/{model_name}/concat_suggs_1samp_boot/isc_results_{cond}_{n_boot}boot_pairWise{do_pairWise}.pkl')
    isc_rois = pd.DataFrame(isc_bootstrap['isc'], columns=full_labels)
    isc_matrices[cond] = {region : isc_utils.vector_to_isc_matrix(isc_rois.iloc[:,col_j],diag=0) for col_j, region in enumerate(isc_rois.columns)}
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
    
    if sig_df.shape[0] > 0:
        # sig_labels = pd.DataFrame(sig_labels, columns=['ROI', 'Label', 'spearman_r', 'p_values', 'Coordinates (X,Y,Z)'])
        sig_df.columns = ['Region id', 'Label', 'Pearson_r', 'p_values', 'Coordinates (X,Y,Z)']
        sig_df.to_csv(os.path.join(save_visu, f'ISC_{cond}_table-FDR.csv'), index=False)

        sig_df_conditions[cond] = sig_df
        sig_dfs_one_sample[cond] = sig_df.sort_values(by='Pearson_r', ascending=False)
        print('Sig ROIs : ', sig_df['Label'].to_list())

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

#%%
reload(visu_utils)
#rename conditions with full names
conditions_full_names = {cond : full for cond, full in zip(conditions, ['Neutral (Analg.)', 'Analgesia', 'Neutral (Hyperalg.)', 'Hyperalgesia'])} #, 'Neutral (Hyper)', 'N_Ana', 'N_Hyper'])}
# CAREFUL FOR NAME ORDER / CONDITIONS + full name
visu_utils.plot_median_isc_dots(sig_dfs_one_sample,conditions_full_names=conditions_full_names, title="Median ISC across significant ROIs")


# %%
# Pie plot of sig regions in df
#------------------------------
reload(visu_utils)

# all_sugg pie-chart
cond_pie ='ana_run'
sig_rois_for_pie = list(sig_df_conditions[cond_pie]['Label'])
sig_networks = [label.split('_')[2] if len(label.split('_')) > 2 else 'subcortical' for label in sig_rois_for_pie]

visu_utils.plot_network_radar(sig_networks, title = f'Sig ISC during {cond_pie}')

# # all_sugg pie-chart
# cond_pie ='hyper_run'
# sig_rois_for_pie = list(sig_df_conditions[cond_pie]['Label'])
# sig_networks = [label.split('_')[2] if len(label.split('_')) > 2 else 'subcortical' for label in sig_rois_for_pie]

# visu_utils.plot_network_radar(sig_networks, title = f'Sig ISC during {cond_pie}')
#%%
# ==========================
# BLOCK MODELS : (ANA > HYPER, etc. run1>run2)
# Difference (Permutation) per condition visualization
model_name = model_names['model2_sugg']
results_dir = os.path.join(project_dir, f'results/imaging/ISC/{model_name}')
setup = isc_utils.load_json(os.path.join(results_dir, "setup_parameters.json"))

#create visualization dir.
save_visu = os.path.join(results_dir, 'post-hoc_VISU-tables')
os.makedirs(save_visu, exist_ok=True)

parcel_name = setup['atlas_name']
do_pairWise = setup['do_pairwise']
n_boot = setup['n_boot']

result_key = 'cond_contrast_permutation'
contrasts = ['ana_run-hyper_run','Ana-Hyper', 'NAna-NHyper'] if 'single-trial' not in model_name else ['Ana-N_Ana', 'Hyper-N_Hyper']
n_perm = setup['n_perm']

sig_df_conditions_contrast = {}
sig_dfs_contrasts = {}
sig_rois = {}
interactive_views=[]
contrast_imgs= {}
p_values_contrasts_block = {}
for i, cont in enumerate(contrasts):

    file = f'/data/rainville/dSutterlin/projects/ISC_hypnotic_suggestions/results/imaging/ISC/{model_name}/cond_contrast_permutation/isc_results_{cont}_{n_perm}perm_pairWise{do_pairWise}.pkl'
    
    isc_contrast = isc_utils.load_pickle(file)

    grouped_isc = isc_contrast['grouped_isc']        # Grouped ISC values
    observed_diff = isc_contrast['observed_diff']    # Observed differences in ISC
    p_values = isc_contrast['p_value']               # P-values for the ISC contrasts
    distribution = isc_contrast['distribution']      
    fdr_p = isc_utils.fdr(p_values, q=0.05)
    fdr_p = fdr_all_cont
    p_values_contrasts_block[cont] = p_values
    # unc_p = 0.05

    print(f'FDR thresh : {fdr_p}')
    diff_img, diff_thresh, sig_df, _ = visu_utils.project_isc_to_brain_perm(
        atlas_img=atlas,
        isc_median=observed_diff,
        atlas_labels=id_labels_dct,
        roi_coords = roi_coords,
        p_values=p_values,
        p_threshold=fdr_p,
        title = f"Difference in ISC between {cont}",
        save_path=None,
        show=True,
         display_mode='x',
        color='coolwarm'
    )
    contrast_imgs[cont] = (diff_img, diff_thresh)

# %%
reload(visu_utils)
reload(visu_utils)
# ===========================
# SINGLE TRIAL MODEL : (Modulation - Neutre)
# Difference (Permutation) per condition visualization
model_name_single_trial = model_names['model1_single-trial']
results_dir = os.path.join(project_dir, f'results/imaging/ISC/{model_name_single_trial}')
setup = isc_utils.load_json(os.path.join(results_dir, "setup_parameters.json"))

#create visualization dir.
save_visu = os.path.join(results_dir, 'post-hoc_VISU-tables')
os.makedirs(save_visu, exist_ok=True)

parcel_name = setup['atlas_name']
do_pairWise = setup['do_pairwise']
n_boot = setup['n_boot']

result_key = 'cond_contrast_permutation'
contrasts = ['Ana-Hyper', 'NAna-NHyper'] if 'single-trial' not in model_name_single_trial else ['Ana-N_Ana', 'Hyper-N_Hyper']
n_perm = setup['n_perm']


sig_dfs_contrasts = {}
sig_rois = {}
contrast_imgs_txt = {}
p_values_contrasts_txt = {}
for i, cont in enumerate(contrasts):

    file = f'/data/rainville/dSutterlin/projects/ISC_hypnotic_suggestions/results/imaging/ISC/{model_name_single_trial}/cond_contrast_permutation/isc_results_{cont}_{n_perm}perm_pairWise{do_pairWise}.pkl'
    
    isc_contrast = isc_utils.load_pickle(file)

    grouped_isc = isc_contrast['grouped_isc']        # Grouped ISC values
    observed_diff = isc_contrast['observed_diff']    # Observed differences in ISC
    p_values = isc_contrast['p_value']               # P-values for the ISC contrasts
    distribution = isc_contrast['distribution']      
    fdr_p = isc_utils.fdr(p_values, q=0.05)
    fdr_p = fdr_all_cont
    p_values_contrasts_txt[cont] = p_values
    # unc_p = 0.05

    print(f'FDR thresh : {fdr_p}')
    diff_img, diff_thresh, sig_df, _ = visu_utils.project_isc_to_brain_perm(
        atlas_img=atlas,
        isc_median=observed_diff,
        atlas_labels=id_labels_dct,
        roi_coords = roi_coords,
        p_values=p_values,
        p_threshold=fdr_p,
        title = f"Difference in ISC between {cont}",
        save_path=None,
        show=True,
         display_mode='x',
        color='coolwarm'
    )
    contrast_imgs_txt[cont] = (diff_img, diff_thresh)

#%%
# global FDR across 4 contrasts
all_ps = []
for cont in ['Ana-Hyper', 'NAna-NHyper']:
    p = p_values_contrasts_block[cont]
    all_ps.extend(p)
for cont in ['Ana-N_Ana', 'Hyper-N_Hyper']:
    p = p_values_contrasts_txt[cont]
    all_ps.extend(p)
fdr_all_cont = isc_utils.fdr(np.array(all_ps), q=0.05)
print('FDR across all contrasts : ', fdr_all)
                
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
