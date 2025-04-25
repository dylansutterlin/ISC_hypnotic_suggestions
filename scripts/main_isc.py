

# %%
import sys
import os
import time
import json
import glob
from datetime import datetime
import importlib
from importlib import reload

import numpy as np
import pandas as pd
import nibabel as nib
from sklearn.utils import Bunch
from nilearn.image import concat_imgs, resample_img, new_img_like, binarize_img, resample_to_img, index_img
from nilearn.maskers import MultiNiftiMapsMasker, MultiNiftiMasker, NiftiMasker, NiftiLabelsMasker
from nilearn.datasets import fetch_atlas_schaefer_2018
from nilearn import plotting
from nilearn.plotting import plot_carpet, plot_roi, plot_stat_map
from nilearn import datasets
from nilearn.image import math_img, binarize_img
from nilearn.plotting import plot_roi


from brainiak.isc import isc, bootstrap_isc, permutation_isc, compute_summary_statistic, phaseshift_isc

import src.isc_utils as isc_utils
import src.visu_utils as visu_utils
import src.qc_utils as qc_utils

reload(isc_utils)

# %%
# Detect if running in an interactive environment (VSCode, Jupyter, etc.)
def is_interactive():
    return hasattr(sys, 'ps1') or "ipykernel" in sys.modules

# Use CLI arguments if running as a script, otherwise set a default setup
if not is_interactive():
    import argparse
    parser = argparse.ArgumentParser(description="Run ISC analysis with different setups")
    parser.add_argument("-s", "--setup", type=str, required=True, help="Name of the setup module (e.g., atlas_setup)")
    args = parser.parse_args()
    setup_name = args.setup.replace('/', '.').replace('.py', '')
    # setup_name = 'setups.debug_setup'
    # breakpoint()
else:
    print("Running in interactive mode. Using default setup: `debug_setup`")
    setup_name = "setups.debug_setup"  # Default setup when running cells manually

print("Current Working Directory:", os.getcwd())

if "setups" not in sys.modules:
    sys.path.append(os.path.abspath("setups"))

setup_module = importlib.import_module(f"{setup_name}")
importlib.reload(setup_module)
setup = setup_module.init()

print(f"Loaded setup: {setup_name}")
# print(f"Running model: {setup.model_name} with {setup.n_sub} subjects")
print(f"Results will be saved in: {setup.results_dir}")

# %%
project_dir = setup.project_dir
base_path = setup.base_path
behav_path = setup.behav_path
exclude_sub = setup.exclude_sub
keep_n_sub = setup.keep_n_sub
model_is = setup.model_is
reg_conf = setup.reg_conf
keep_n_confouds =  setup.keep_n_conf

conditions = setup.conditions
transform_imgs = setup.transform_imgs
do_isc_analyses = setup.do_isc_analyses

do_shss_split = setup.do_shss_split
do_rsa = setup.do_rsa

n_boot = setup.n_boot
do_pairwise = setup.do_pairwise
n_perm = setup.n_perm
n_perm_rsa = setup.n_perm_rsa

atlas_name = setup.atlas_name
apply_mask = setup.apply_mask

model_name = setup.model_name
results_dir = setup.results_dir

# behav data
reload(isc_utils)
isc_data_df = isc_utils.load_isc_data(base_path, folder_is='subject', model = model_is) # !! change params if change base_dir
if 'single-trial' in setup.model_id:
    print('Loading single trial data')
    model_is = 'instrbk'
    isc_data_df = isc_utils.load_single_trial_isc_data(base_path)

isc_data_df = isc_data_df.sort_values(by='subject')

# exclude subjects
isc_data_df = isc_data_df[~isc_data_df['subject'].isin(exclude_sub)]
isc_data_df = isc_data_df.sort_values(by='subject')
subjects = isc_data_df['subject'].unique()
n_sub = len(subjects)

# confounds !! unsorted!
dct_conf_unsorted = isc_utils.load_confounds(base_path)
confounds_ls = isc_utils.filter_and_rename_confounds(dct_conf_unsorted, subjects, model_is)
#isc_conf = isc_utils.check_confounds_isc(confounds_ls, conditions, subjects,show=True)

sub_check = {}
# Select a subset of subjects
if keep_n_sub is not None:
    isc_data_df = isc_data_df[isc_data_df['subject'].isin(isc_data_df['subject'].unique()[:keep_n_sub])]
    subjects = isc_data_df['subject'].unique()
    n_sub = len(subjects)
setup.subjects = subjects
setup.n_sub = n_sub

# create save dir and save params
setup.check_and_create_results_dir() 
set     
print('Results directory at :' , results_dir)

# save_setup = os.path.join(setup.results_dir, "setup_parameters.json")
# with open(save_setup, 'w') as fp:
#     json.dump(dict(setup), fp, indent=4)

result_paths = {
    "isc_results": {},
    "isc_combined_results": {},
    "condition_contrast_results": {},
    "group_permutation_results": {},
    "rsa_isc_results": {},
    "setup_parameters": os.path.join(setup.results_dir, 'setup_parameters.json')
}

# %%
reload(isc_utils)
# get dict will all cond files from all subjects
print("Subjects:", subjects)
sub_check['init'] = subjects

tasks_ordered = sorted(isc_data_df['task'].unique())
# code conditions for all_sugg, (ANA+Hyper), (Neutral_H + Neutral_A)
task_to_test = conditions #[tasks_ordered, tasks_ordered[0:2], tasks_ordered[2:4]]
subject_file_dict = {}
# organize files as dict{condX : {sub1 : [file1, file2, file3]
# Does nothing where comparing all tasks by itself
for i, cond in enumerate(conditions):
    subject_file_dict[cond] = isc_utils.get_files_for_condition_combination(subjects, task_to_test[i], isc_data_df)

print("Conditions:", conditions)
print("Condition combinations:", task_to_test)
sub_check['cond_files_sub'] = list(subject_file_dict[conditions[0]].keys())

# save subjects paths
for cond in conditions:
    cond_folder = os.path.join(results_dir, cond)
    if not os.path.exists(cond_folder):
        os.makedirs(cond_folder)
    save_csv = pd.DataFrame.from_dict(subject_file_dict[cond], orient='index', columns=['func_path'])
    save_csv.to_csv(os.path.join(cond_folder, f'setup_func_path_{cond}.csv'))
    isc_utils.save_data(os.path.join(cond_folder, f'setup_func_path_{cond}.pkl'), subject_file_dict[cond])

# %%
# ======
# Init masks
voxel_masker = Bunch(name="voxel_wise")
ref_img = index_img(nib.load(subject_file_dict[conditions[0]][subjects[0]]), 0)
data_shape = ref_img.shape[0:3] #3D
data_affine = ref_img.affine
prob_threshold = setup.prob_threshold

if apply_mask == 'lanA800':
    print(f'Loading language mask lan800 prob={prob_threshold}')
    full_mask = nib.load(os.path.join(project_dir, 'masks/lipkin2022_lanA800', 'LanA_n806.nii'))
    mask_native = binarize_img(full_mask, threshold=prob_threshold)
    mask_path = os.path.join(results_dir, f'bin_lanA800_{prob_threshold}thresh.nii.gz')

elif apply_mask == 'outside_lanA800':
    print(f'Creating inverse mask: regions outside LanA800 with prob threshold = {prob_threshold}')
    full_mask = nib.load(os.path.join(project_dir, 'masks/lipkin2022_lanA800', 'LanA_n806.nii'))
    mask_native = math_img('img < @thresh', img=full_mask, thresh=prob_threshold)
    mask_path = os.path.join(results_dir, f'outside_lanA800_{prob_threshold}thresh.nii.gz')

elif apply_mask == None or apply_mask == 'whole-brain':
    print('Loading whole brain MNI template as mask')
    mask_native = datasets.load_mni152_brain_mask()
    mask_native = nib.load('/data/rainville/Hypnosis_ISC/masks/brainmask_91-109-91.nii')
    mask_path = os.path.join(results_dir, 'mni_mask.nii.gz')

else:
    raise ValueError(f"Unknown apply_mask setting: {apply_mask}")

resamp_mask = qc_utils.resamp_to_img_mask(mask_native, ref_img)
resamp_mask.to_filename(mask_path)
plot_roi(resamp_mask, title='mask', display_mode='ortho', draw_cross=False, output_file=os.path.join(results_dir, 'resamp_mask_used_isc.png'))

masker_params_dict = {
    "standardize": 'zscore_sample',
    "mask_img": mask_path,
    "target_shape": None,
    "target_affine": None,
    "detrend": False,
    "low_pass": None,
    "high_pass": None,  # 1/428 sec.
    "t_r": 3,
    "smoothing_fwhm": None,
    "standardize_confounds": True,
    "verbose": 5,
    "high_variance_confounds": False,
    "mask_strategy": "whole-brain-template",  # ignore for atlas maskers
}

masker_params_path = os.path.join(setup.results_dir, "masker_params.json")
isc_utils.save_json(masker_params_path, masker_params_dict)

#save plot of mask
# mask_masker = NiftiMasker(mask_img=mask,mask_strategy='whole-brain-template').fit()
# mask_masker.fit_transform(resamp_mask)
voxel_masker.obj = NiftiMasker(**masker_params_dict)
#mask_img=mask,standardize=True, smoothing_fwhm=None, target_shape = None, target_affine = None, mask_strategy='whole-brain-template', verbose=5)

# load difumo atlas
#atlas_name = 'voxelWise' #Difumo256' # !!!!!!! 'Difumo256'
if  'voxelWise' in atlas_name or atlas_name == 'Difumo256':
    atlas_path = os.path.join(project_dir, 'masks/DiFuMo256/3mm/maps.nii.gz')
    atlas_dict_path = os.path.join(project_dir, 'masks/DiFuMo256/labels_256_dictionary.csv')
    atlas_native = nib.load(atlas_path)
    atlas = qc_utils.resamp_to_img_mask(atlas_native, ref_img)
    # atlas_df = pd.read_csv(atlas_dict_path)
    atlas_labels = None

elif 'schafer' in atlas_name : #atlas_name == f'schafer-{n_rois}-2mm':

    n_rois = int(atlas_name.split('-')[1])
    resolution = int(atlas_name.split('-')[2].replace('mm', ''))
    atlas_data = fetch_atlas_schaefer_2018(n_rois = n_rois, resolution_mm=resolution)

    atlas_native = nib.load(atlas_data['maps'])
    atlas = qc_utils.resamp_to_img_mask(atlas_native, ref_img)
    atlas_path = atlas_data['maps'] #os.path.join(project_dir,os.path.join(project_dir, 'masks', 'k50_2mm', '*.nii*'))
    # atlas_labels = list(atlas_data['labels'])
    # atlas_labels.insert(0, 'background')
    atlas_labels = [str(label, 'utf-8') if isinstance(label, bytes) else str(label) for label in atlas_data['labels']]
    atlas_labels.insert(0, 'background')

# elif atlas_name == 'lanA800':
#     atlas = nib.load(os.path.join(project_dir, 'masks/lipkin2022_lanA800', 'LanA_n806.nii'))
#     atlas_labels = None

    print('atlas loaded with N ROI : ', atlas.shape)

 # extract sphere signal
roi_coords = {
"amcc": (-2, 20, 32),
"rPO": (54, -28, 26),
"lPHG": (-20, -26, -14),
}
sphere_radius = 10
roi_folder = f"sphere_{sphere_radius}mm_{len(roi_coords)}ROIS_isc"

# %% load behavioral data
#==============================================
# append ../func path
# reload(utils)
APM_subjects = ['APM' + sub[4:] for sub in subjects] # make APMXX format instead of subXX
print(APM_subjects)
sub_check['APM_behav'] = APM_subjects
phenotype =pd.read_csv(behav_path, index_col=0)
phenotype.head()
# y_interest = ['SHSS_score', 'total_chge_pain_hypAna', 'Abs_diff_automaticity', ]
y_interest = [
    'Chge_hypnotic_depth', 'SHSS_score', 'raw_change_HYPER',
    'raw_change_ANA', 'total_chge_pain_hypAna',
    'Mental_relax_absChange', 'Abs_diff_automaticity'
]
X_pheno, group_labels_df, sub_check= isc_utils.load_process_behav(phenotype, y_interest, setup, sub_check)
#X_pheno includes group labels_df!! 
setup.group_labels = group_labels_df

# %%
# extract timseries from atlas
if transform_imgs == True:

    start_total_time = time.time()
    print(f'will fit and trasnform images for {n_sub} subjects')

    transformed_data_per_cond = {}
    fitted_maskers = {}
    transformed_sphere_per_roi = {}

    if atlas_name =='voxelWise' or atlas_name == 'voxelWise_lanA800':
        masker = voxel_masker.obj
    elif atlas_name == 'Difumo256':
        masker = MultiNiftiMapsMasker(maps_img=atlas, standardize=False, memory='nilearn_cache', verbose=5, n_jobs= 1)
    elif 'schafer' in atlas_name : # == f'schafer{n_rois}_2mm':
        masker = NiftiLabelsMasker(labels_img=atlas, labels = atlas_labels, mask_img=resamp_mask,resampling_target='data', standardize=True,high_variance_confounds=False,standardize_confounds=True,keep_masked_labels=False, memory='nilearn_cache', verbose=5, n_jobs= 1)

    #masker = MultiNiftiMapsMasker(maps_img=atlas, standardize=False, memory='nilearn_cache', verbose=5, n_jobs= 1)
    #masker = NiftiMasker(verbose=5)
    #masker.fit()

	# extract time series for each subject and condition/task
    for cond in conditions:
        start_cond_time = time.time()
        condition_files = subject_file_dict[cond]
        concatenated_subjects = {sub : concat_imgs(sub_files) for sub, sub_files in condition_files.items()}
        print('Imgs shape : ', [img.shape for _, img in  concatenated_subjects.items()])
        
        #qc_utils.assert_same_affine(concatenated_subjects)

	    #print(f'fitting images for condition : {cond} with shape {concatenated_subjects[subjects[0]][0].shape}')
        if 'voxelWise' in atlas_name or 'schafer' in atlas_name: # == f'schafer{n_rois}_2mm':
            # ls_voxel_wise = [masker.fit_transform(concatenated_subjects[sub],
            #                                        confounds= confounds_ls[i][cond])
            #                                          for i, sub in enumerate(subjects)
            # ]
            print(f'Fitting {atlas_name} masker for {cond}')
            ls_voxel_wise = []
            fitted_maskers[cond] = []
            
            for i, sub in enumerate(subjects):

                if reg_conf == True:
                    conf = confounds_ls[i][cond][:, :keep_n_confouds]
                else: conf = None

                transformed_data = masker.fit_transform(concatenated_subjects[sub], confounds= conf)
                ls_voxel_wise.append(transformed_data)
                fitted_maskers[cond].append(masker)  # Store the fitted masker for this subject

            print('ls_voxel_wise shape : ', [img.shape for img in ls_voxel_wise])
            transformed_data_per_cond[cond] = ls_voxel_wise
            
        if atlas_name == 'Difumo256':
            transformed_data_per_cond[cond] = masker.fit_transform(concatenated_subjects.values())
            fitted_maskers[cond] = masker

        # transformed_data_per_cond[cond] = masker.fit_transform(concatenated_subjects.values())
        # fitted_maskers[cond] = masker

        # sphere masker
        #transformed_sphere_per_roi[cond] = utils.extract_save_sphere(concatenated_subjects, cond, results_dir, roi_coords, sphere_radius=sphere_radius)
        end_cond_time = time.time()
        print(f"Condition {cond} completed in {end_cond_time - start_cond_time:.2f} seconds")

    end_total_time = time.time()
    print(f"Total time for data extraction: {end_total_time - start_total_time:.2f} seconds")

	# save transformed data and masker
    for cond in conditions:

        data = transformed_data_per_cond[cond]
        cond_folder = os.path.join(results_dir, cond)

        if not os.path.exists(cond_folder):
            os.makedirs(cond_folder)

        save_path = os.path.join(cond_folder, f'transformed_data_{atlas_name}_{cond}_{n_sub}sub.npz')
        data_3d = np.stack(data, axis=-1) # ensure TRs x ROI x subjs
        np.savez_compressed(save_path, data_3d)
        #utils.save_data(s3ave_path, data)

        masker_path = os.path.join(cond_folder, f'maskers_{atlas_name}_{cond}_{n_sub}sub.pkl')
        isc_utils.save_data(masker_path, fitted_maskers[cond])
        print(f'Transformed timseries and maskers saved to {cond_folder}')

    print('====Done with data extraction====')
 
else:   
    # account for case we load data from diff directory
    if setup.pre_computed != False:
        pre_computed_ts_dir = os.path.join(setup.project_dir, setup.results_branch, setup.pre_computed)

        if not os.path.isdir(pre_computed_ts_dir):
            raise OSError(f"Pre-computed directory {pre_computed_ts_dir} does not exist.")
        print(f'!! Loading existing data and fitted maskers from : {pre_computed_ts_dir}')

    else : 
        pre_computed_ts_dir = setup.results_dir
    
# Loading data
    print(f'Loading existing data and fitted maskers')
    transformed_data_per_cond = {}
    fitted_maskers = {}
    transformed_sphere_per_roi = {}

    for cond in setup.conditions:
        
        load_path = os.path.join(pre_computed_ts_dir, cond)

        transformed_path =  os.path.join(load_path, f'transformed_data_{atlas_name}_{cond}_{n_sub}sub.npz')
        transformed_data_per_cond[cond] = np.load(transformed_path)['arr_0'] #utils.load_pickle(os.path.join(load_path, transformed_path))
        fitted_mask_file = f'maskers_{atlas_name}_{cond}_{n_sub}sub.pkl'
        fitted_maskers[cond] = isc_utils.load_pickle(os.path.join(load_path, fitted_mask_file))

        #load_roi = os.path.join(load_path, roi_folder)
        #transformed_sphere_per_roi[cond] =utils.load_pickle(os.path.join(load_roi, f"{len(roi_coords)}ROIs_timeseries.pkl")) 

    print(f'Loading existing data and fitted maskers from : {load_path}')

# if 'schafer' not in atlas_name: # != f'schafer{n_rois}_2mm':
    # Assess 0 variance subjects

sub_to_remove = isc_utils.identify_zero_variance_subjects(transformed_data_per_cond, subjects)
print(f"Subjects with 0 variance will be removed : {sub_to_remove}")
# transformed_data_per_cond, fitted_masker, subjects = isc_utils.remove_subjects(transformed_data_per_cond, fitted_maskers, subjects, sub_to_remove)
setup.kept_subjects = subjects
n_sub= len(subjects)

isc_utils.plot_flat_rois(
    transformed_data_per_cond=transformed_data_per_cond,
    subjects=setup.kept_subjects,
    roi_names=atlas_labels,  # make sure they're decoded if needed
    condition_names=setup.conditions,
    save_dir=os.path.join(setup.results_dir, "QC_flat_rois")
)

#%%
if do_isc_analyses:
    # bootstrap_conditions = ['Hyper', 'Ana', 'NHyper', 'NAna', 'all_sugg', 'modulation', 'neutral']

    # ISC with ROI/voxels : bootstrap per condition (from chen et al. 2016)
    reload(isc_utils)
    isc_results = {}
    start_time = time.time()

    # assert that data is shape TRs x ROI x subjects and not ls of TRxROI
    transformed_formated = {}
    for cond, data in transformed_data_per_cond.items():

        if isinstance(data, list):
            data_3d = np.stack(data, axis=-1)
        elif isinstance(data, np.ndarray):
            data_3d = data
        assert data_3d.shape[-1] == n_sub, f"Data shape {data_3d.shape} does not match number of subjects {n_sub}"
        transformed_formated[cond] = data_3d

    transformed_data_per_cond = transformed_formated # replace with 3d arrays
    # breakpoint()    
    
    for cond, data in transformed_data_per_cond.items():
        print(f'Performing ISC for condition: {cond}')
        start_cond_time = time.time()
        # Convert list of 2D arrays to 3D array (n_TRs, n_voxels, n_subjects)
        print(data.shape)

        isc_results[cond] = isc_utils.isc_1sample(data, pairwise=do_pairwise,n_boot = n_boot, summary_statistic=None)

        # save results
        save_path = os.path.join(results_dir, cond, f"isc_results_{cond}_{n_boot}boot_pairWise{do_pairwise}.pkl")
        isc_utils.save_data(save_path, isc_results[cond])
        result_paths["isc_results"][cond] = save_path

        end_cond_time = time.time()
        print(f" {cond} done in {end_cond_time - start_cond_time:.2f} sec")
        print(f"ISC results saved for {cond} at {save_path}")

    # %%

    # Combined condition bootstrap
    print('====================================')

    combined_conditions = setup.combined_conditions
    task_to_test = setup.how_to_combine_conds #list of list of cond to include
    
    if 'instrbk' in setup.model_id:
        task_to_test = [conditions]
        combined_conditions = ['all_sugg']

    concat_cond_save = os.path.join(results_dir, 'concat_suggs_1samp_boot')
    os.makedirs(concat_cond_save, exist_ok=True)

    for i, comb_cond in enumerate(combined_conditions):
        start_cond_time = time.time()
        print(f'Performing bootstraped ISC for combined condition: {comb_cond}')
        print(f'--Conditions to combine: {task_to_test[i]}')

        combined_data = np.concatenate([transformed_data_per_cond[task]for task in task_to_test[i]], axis=0)

        transformed_data_per_cond[comb_cond] = combined_data
        n_scans = combined_data.shape[0] #  concat along TR axis for each 3d task array
        # n_sub = combined_data.shape[-1]
        # isc_combined = isc(combined_data, pairwise=do_pairwise, summary_statistic=None)

        isc_results[comb_cond] = isc_utils.isc_1sample(combined_data, pairwise=do_pairwise,n_boot = n_boot, summary_statistic=None)

        # save results
        save_path = os.path.join(concat_cond_save, f"isc_results_{comb_cond}_{n_boot}boot_pairWise{do_pairwise}.pkl")
        isc_utils.save_data(save_path, isc_results[comb_cond])
        result_paths["isc_combined_results"][comb_cond] = save_path

        end_cond_time = time.time()
        print(f" {cond} done in {end_cond_time - start_cond_time:.2f} sec")
        print(f"ISC results saved for {comb_cond} at {save_path}")


# %%
#==============================
# ISFC per conditions
do_isfc = setup.do_isfc

if do_isfc:

    isfc_results_per_cond = {}
    isfc_save_folder = os.path.join(results_dir, 'isfc_1sample')
    os.makedirs(isfc_save_folder, exist_ok=True)
    result_paths['isfc_results'] = {}

    for cond in ['all_sugg', 'ana_run']: #conditions: 

        start_cond_time = time.time()
        print(f' Performing ISFC for condition: {cond}')

        data = transformed_data_per_cond[cond]
        print('Data for ISFC : ' , data.shape)

        isfc_results_per_cond[cond] = isc_utils.isfc_1sample(data, pairwise=do_pairwise,n_boot = n_boot, summary_statistic=None)
     
        save_path = os.path.join(isfc_save_folder, f"isfc_results_{cond}_{n_boot}boot_pairWise{do_pairwise}.pkl")
        isc_utils.save_data(save_path, isfc_results_per_cond[cond])
        result_paths["isfc_results"][cond] = save_path

        end_cond_time = time.time()
        print(f" {cond} done in {end_cond_time - start_cond_time:.2f} sec")
        print(f"ISC results saved for {cond} at {save_path}")

    # Group permutation
    isfc_permutation_group_results = {}
    isfc_cols = ['Chge_hypnotic_depth_median_grp',
    'SHSS_score_median_grp',
    'total_chge_pain_hypAna_median_grp']

    # 'Abs_diff_automaticity_median_grp']
    # 'raw_change_HYPER_median_grp', 'raw_change_ANA_median_grp'
    results_save_dir = os.path.join(results_dir, 'isfc_group_permutation')
    os.makedirs(results_save_dir, exist_ok=True)

    for cond, isfc_dict in isfc_results_per_cond.items():  # isfc_results[cond] = subjects x roi_pairs
        
        print(f"Performing permutation ISFC for condition: {cond}")
        result_paths["isfc_group_permutation_results"] = {}
        isfc_permutation_group_results[cond] = {}
        isfc = isfc_dict['isfc'] 
        
        for var in isfc_cols:
            print(f" --- ISFC for : {var}")
            group_assignment = group_labels_df[var].values  # binary labels (0 or 1), length = n_subjects
            
            # Run permutation test per ROI pair (e.g., vectorized ISFC values)
            isfc_permutation_group_results[cond][var] = isc_utils.group_permutation(
                isfc, group_assignment,
                n_perm=5,
                do_pairwise=do_pairwise, 
                side='two-sided',
                summary_statistic='median'
            )

        save_path = os.path.join(results_save_dir, f'{cond}_isfc_group_permutation_{n_perm}perm.pkl')
        isc_utils.save_data(save_path, isfc_permutation_group_results[cond])
        result_paths["isfc_group_permutation_results"][cond] = save_path
    
        print(f" ==Done permutation ISFC for all variables in: {cond}")

# %%


    # %%
    # Permutation based on group labels    #%%
    # import numpy as np

    # def prepare_contrast_data_with_trimmed_TRs(cond_trials1, cond_trials2, tr_trim1=None, tr_trim2=None, transformed_data=None):
    #     """
    #     Prepares and concatenates data across trials for two condition groups,
    #     trimming the number of TRs per trial as needed.

    #     Parameters
    #     ----------
    #     cond_trials1 : list of str
    #         Trial names for condition 1 (e.g., Ana).
    #     cond_trials2 : list of str
    #         Trial names for condition 2 (e.g., N_Ana).
    #     tr_trim1 : list of int or None
    #         List of number of TRs to keep per trial for cond1.
    #     tr_trim2 : list of int or None
    #         List of number of TRs to keep per trial for cond2.
    #     transformed_data : dict
    #         Condition name → list of arrays (TRs x ROIs x subjects).

    #     Returns
    #     -------
    #     task1_data : ndarray
    #         (TRs x ROIs x subjects), concatenated across trimmed trials of cond1.
    #     task2_data : ndarray
    #         Same for cond2.
    #     """

    #     def trim_trials(trial_list, tr_trim, label):
    #         trimmed = []
    #         for i, trial in enumerate(trial_list):
    #             data = transformed_data[trial]
    #             # Assume data is list of subjects: [sub1_data, sub2_data, ...]
    #             trimmed_trial = [sub[:tr_trim[i], :] for sub in data]
    #             trimmed_array = np.stack(trimmed_trial, axis=-1)  # (TRs x ROIs x subjects)
    #             trimmed.append(trimmed_array)
    #         return np.concatenate(trimmed, axis=0)  # concatenate TRs

    #     if tr_trim1 is None:
    #         tr_trim1 = [transformed_data[trial][0].shape[0] for trial in cond_trials1]
    #     if tr_trim2 is None:
    #         tr_trim2 = [transformed_data[trial][0].shape[0] for trial in cond_trials2]

    #     task1_data = trim_trials(cond_trials1, tr_trim1, "task1")
    #     task2_data = trim_trials(cond_trials2, tr_trim2, "task2")

    #     assert task1_data.shape == task2_data.shape, f"Trimmed task shapes don't match: {task1_data.shape} vs {task2_data.shape}"
    #     return task1_data, task2_data

#%% 
summary_stat = 'median'

do_contrast_permutation = True
if do_contrast_permutation:
    # Paired sample permutation test
    # contrast_conditions = ['Hyper-Ana', 'Ana-Hyper', 'NAna-NHyper', 'ana_run-hyper_run']
    # contrast_to_test = [conditions[0:2], conditions[0:2][::-1], conditions[2:4], combined_conditions[3:]]
    contrast_conditions = setup.contrast_conditions
    contrast_to_test = setup.contrast_to_test

    if 'single-trial' in setup.model_id:
        contrast_conditions = ['N_ANA1_instrbk_1-N_HYPER1_instrbk_1', 'Ana-N_Ana', 'Hyper-N_Hyper'] #, 'Ana-N_Ana', 'Hyper-N_HYPER']

        Ana_trials = ['ANA1_instrbk_1', 'ANA2_instrbk_1']
        N_Ana_trials = ['N_ANA1_instrbk_1', 'N_ANA2_instrbk_1', 'N_ANA3_instrbk_1']
        Hyper_trials = ['HYPER1_instrbk_1', 'HYPER2_instrbk_1']
        N_Hyper_trials = ['N_HYPER1_instrbk_1', 'N_HYPER2_instrbk_1', 'N_HYPER3_instrbk_1']
        
        contrast_to_test = [['N_ANA1_instrbk_1', 'N_HYPER1_instrbk_1'], [Ana_trials, N_Ana_trials], [Hyper_trials, N_Hyper_trials]]
                            
    isc_permutation_cond_contrast = {}
    contrast_perm_save = os.path.join(results_dir, 'cond_contrast_permutation')
    os.makedirs(contrast_perm_save, exist_ok=True)

    for i, contrast in enumerate(contrast_conditions):
        print(f'== Performing 2 group permutation ISC : {contrast}')

        start_cond_time = time.time()

        if 'instrbk' in contrast: #single trial model
            to_test = contrast_to_test[i]            
            task1 = transformed_data_per_cond[to_test[0]]
            task2 = transformed_data_per_cond[to_test[1]]
            assert task1.shape == task2.shape, f"Data shape {task1.shape} does not match {task2.shape}"
            combined_data = np.concatenate([task1, task2], axis=2) 

        elif 'single-trial' in setup.model_id:
            
            if contrast == 'Ana-N_Ana':  # cont idx 1
                tr_trim1 = [50, 50]
                tr_trim2 = [34, 33, 33]

                all_trials_trimed1 = []
                for i, trial in enumerate(Ana_trials):  # modulation trials
                    trial_data = transformed_data_per_cond[trial] # shape: TRs x ROI x subjects
                    all_trials_trimed1.append(trial_data[:tr_trim1[i], :, :])
                task1_data = np.concatenate(all_trials_trimed1, axis=0)  # shape: (total_TRs x ROI x subjects)

                all_trials_trimmed2 = []
                for i, trial in enumerate(N_Ana_trials):  # neutral trials
                    trial_data = transformed_data_per_cond[trial]
                    all_trials_trimmed2.append(trial_data[:tr_trim2[i], :, :])
                task2_data = np.concatenate(all_trials_trimmed2, axis=0)

                transformed_data_per_cond['Ana_trimmed100'] = task1_data
                transformed_data_per_cond['N_Ana_trimmed100'] = task2_data

            elif contrast == 'Hyper-N_Hyper':  # cont idx 2
                tr_trim1 = [51, 49]
                tr_trim2 = [34, 33, 33]

                all_trials_trimmed1 = []
                for i, trial in enumerate(Hyper_trials):  # modulation trials
                    trial_data = transformed_data_per_cond[trial]
                    all_trials_trimmed1.append(trial_data[:tr_trim1[i], :, :])
                task1_data = np.concatenate(all_trials_trimmed1, axis=0)

                all_trials_trimmed2 = []
                for i, trial in enumerate(N_Hyper_trials):  # neutral trials
                    trial_data = transformed_data_per_cond[trial]
                    all_trials_trimmed2.append(trial_data[:tr_trim2[i], :, :])
                task2_data = np.concatenate(all_trials_trimmed2, axis=0)
                
                transformed_data_per_cond['Hyper_trimmed100'] = task1_data
                transformed_data_per_cond['N_Hyper_trimmed100'] = task2_data
                
            combined_data = np.concatenate([task1_data, task2_data], axis=2)

        else: 
            # here we need to comcat the data along the subject axis, like repeated measures
            # and this subjects is taken into account in the permutation. See brainIAK.
            combined_data_ls = [transformed_data_per_cond[task] for task in contrast_to_test[i]]
            combined_data = np.concatenate(combined_data_ls, axis=2)

        isc_grouped = isc(combined_data, pairwise=do_pairwise, summary_statistic=None)
        n_scans = combined_data.shape[0]

        # if do_pairwise:
        group_ids = np.array([0] * n_sub + [1] * n_sub)

        isc_permutation_cond_contrast[contrast] = isc_utils.group_permutation(isc_grouped, group_ids, n_perm, do_pairwise, side = 'two-sided', summary_statistic=summary_stat)
        diff_vector = isc_permutation_cond_contrast[contrast]['observed_diff']
        print(f'Max & min ISC diff : {np.max(diff_vector)}, {np.min(diff_vector)}')
        save_path = os.path.join(contrast_perm_save, f"isc_results_{contrast}_{n_perm}perm_pairWise{do_pairwise}.pkl")
        isc_utils.save_data(save_path, isc_permutation_cond_contrast[contrast])
        result_paths["condition_contrast_results"][contrast] = save_path
        
        print(f'     ... done in {time.time() - start_cond_time:.2f} sec')

#%%
do_shss_split = setup.do_shss_split

if 'single-trial' in setup.model_id:
    contrast_conditions = ['N_ANA1_instrbk_1-N_HYPER1_instrbk_1', 'Ana-N_Ana', 'Hyper-N_Hyper'] #, 'Ana-N_Ana', 'Hyper-N_HYPER']
    contrast_to_test = [['N_ANA1_instrbk_1', 'N_HYPER1_instrbk_1'], ['Ana_trimmed100', 'N_Ana_trimmed100'], ['Hyper_trimmed100', 'N_Hyper_trimmed100']]
    do_shss_permutation = True

else: 
    contrast_conditions = ['Hyper-Ana', 'Ana-Hyper', 'NHyper-NAna']
    contrast_to_test = [conditions[0:2], conditions[0:2][::-1], conditions[2:4]]

if do_shss_split:

    shss_grps = ['low_shss', 'high_shss']
    isc_permutation_cond_contrast = {}

    for g, shss_grp in enumerate(shss_grps):
        start_grp_time = time.time()
        # print(f'==== Doing {shss_grp}, suppose to have 12 if low and 11 if high ====')
        contrast_perm_shss = os.path.join(results_dir, f'group_perm_{shss_grp}')
        os.makedirs(contrast_perm_shss, exist_ok=True)

        if shss_grp == 'low_shss':
            shss_idx = np.array(group_labels_df['SHSS_score_median_grp'] == 0)
            keep_n = len(shss_idx[shss_idx == True]) # inverse!
        elif shss_grp == 'high_shss':
            shss_idx = np.array(group_labels_df['SHSS_score_median_grp'] == 1)
            keep_n = len(shss_idx[shss_idx == True])

        shss_idx_concat = np.concatenate([shss_idx, shss_idx])
        group_ids_selected = np.array([0] * keep_n + [1] * keep_n)
        # else:
        #     group_ids_selected = np.array([1] * keep_n)

        print(f'-----{shss_grp} grp with {keep_n} subjects')
        for i, contrast in enumerate(contrast_conditions):

            combined_data_ls = [transformed_data_per_cond[task] for task in contrast_to_test[i]]
            combined_data = np.concatenate(combined_data_ls, axis=2)
            combined_data_selected = combined_data[:, :, shss_idx_concat]
            print(f'{contrast} : Repeated mesaure isc having shape : ', combined_data_selected.shape)
        # print('group id unique : ', np.unique(group_ids_selected, return_counts=True))  
        
            isc_grouped_selected = isc(combined_data_selected, pairwise=do_pairwise, summary_statistic=None)
            # group_ids = np.array([0] * n_sub + [1] * n_sub)
            isc_permutation_cond_contrast[contrast] = isc_utils.group_permutation(isc_grouped_selected, group_ids_selected, n_perm, do_pairwise, side = 'two-sided', summary_statistic='median')

            save_path = os.path.join(contrast_perm_shss, f"isc_results_{keep_n}sub_{contrast}_{n_perm}perm_pairWise{do_pairwise}.pkl")
            isc_utils.save_data(save_path, isc_permutation_cond_contrast[contrast])
            result_paths["condition_contrast_results"][contrast] = save_path
        print(f'Contrasts for {shss_grp} done in {time.time() - start_grp_time:.2f} sec')    


    # single_trial test for cunterbalcing order!!
    counter_balance_effect = True if 'single-trial' in setup.model_id else False

    if 'single-trial' in setup.model_id:
        contrast_conditions = ['N_ANA1_instrbk_1-N_HYPER1_instrbk_1', 'Ana-N_Ana', 'Hyper-N_Hyper'] #, 'Ana-N_Ana', 'Hyper-N_HYPER']
        contrast_to_test = [['N_ANA1_instrbk_1', 'N_HYPER1_instrbk_1'], ['Ana_trimmed100', 'N_Ana_trimmed100'], ['Hyper_trimmed100', 'N_Hyper_trimmed100']]
        # do_shss_permutation = True


    # if counter_balance_effect:
    
    #     shss_grps = ['H1', 'H2']
    #     subject_to_cb_group = {
    #     'sub-02': 2,
    #     'sub-03': 1,
    #     'sub-06': 2,
    #     'sub-07': 1,
    #     'sub-09': 1,
    #     'sub-12': 2,
    #     'sub-16': 1,
    #     'sub-17': 2,
    #     'sub-20': 2,
    #     'sub-22': 2,
    #     'sub-26': 1,
    #     'sub-27': 2,
    #     'sub-28': 1,
    #     'sub-29': 1,
    #     'sub-33': 1,
    #     'sub-36': 1,
    #     'sub-37': 2,
    #     'sub-38': 1,
    #     'sub-40': 1,
    #     'sub-41': 2,
    #     'sub-42': 2,
    #     'sub-43': 2,
    #     'sub-47': 2
    #     }
    #     # adjust for test n subjects
    #     subject_to_cb_group = {k: v for k, v in subject_to_cb_group.items() if k in subjects}

    #     cb_groups = ['H1', 'H2']
    #     isc_permutation_cond_contrast = {}

    #     for cb_grp in cb_groups:
    #         start_grp_time = time.time()
    #         print(f'==== Doing {cb_grp} counterbalancing group ====')

    #         contrast_perm_cb = os.path.join(results_dir, f'group_perm_counterbalance_{cb_grp}')
    #         os.makedirs(contrast_perm_cb, exist_ok=True)

    #         # Get subject indices for this CB group
    #         cb_value = 1 if cb_grp == 'H1' else 2
    #         subjects_in_cb = [s for s, v in subject_to_cb_group.items() if v == cb_value]
    #         keep_n = len(subjects_in_cb)
    #         print(f'----- {cb_grp} group with {keep_n} subjects')

    #         # Build mask for subject selection
    #         subject_indices = [i for i, s in enumerate(subjects) if s in subjects_in_cb]
    #         cb_idx = np.zeros(len(subjects), dtype=bool)
    #         cb_idx[subject_indices] = True
    #         cb_idx_concat = np.concatenate([cb_idx, cb_idx])  # for 2-condition ISC data
    #         group_ids_selected = np.array([0] * keep_n + [1] * keep_n)

    #         for i, contrast in enumerate(contrast_conditions):
    #             combined_data_ls = [transformed_data_per_cond[task] for task in contrast_to_test[i]]
    #             combined_data = np.concatenate(combined_data_ls, axis=2)
    #             combined_data_selected = combined_data[:, :, cb_idx_concat]

    #             print(f'{contrast} : Repeated measure ISC shape: {combined_data_selected.shape}')

    #             isc_grouped_selected = isc(combined_data_selected, pairwise=do_pairwise, summary_statistic=None)

    #             isc_permutation_cond_contrast[contrast] = isc_utils.group_permutation(
    #                 isc_grouped_selected,
    #                 group_ids_selected,
    #                 n_perm,
    #                 do_pairwise,
    #                 side='two-sided',
    #                 summary_statistic='median'
    #             )

    #             save_path = os.path.join(contrast_perm_cb, f"isc_results_{keep_n}sub_{contrast}_{n_perm}perm_pairWise{do_pairwise}.pkl")
    #             isc_utils.save_data(save_path, isc_permutation_cond_contrast[contrast])
    #             result_paths["condition_contrast_results"][contrast] = save_path

    #         print(f'Contrasts for {cb_grp} done in {time.time() - start_grp_time:.2f} sec')

    #         group_labels_df['subject_to_cb'] = group_labels_df.index.map(subject_to_cb_group)


# %%

# ------------
# Perform permutation based on group labels
do_group_permutation = True

if do_group_permutation:
    reload(isc_utils)

    isc_permutation_group_results = {}

    for cond, isc_dict in isc_results.items():  # `isc_results` should already contain ISC values
        print(f"Performing permutation ISC for condition: {cond}")
        var_isc_results = {}
        isc_values = isc_dict['isc']
        isc_permutation_group_results[cond] = {}

        for var in group_labels_df.columns:
            print(f"Performing permutation ISC for condition: {cond}, variable: {var}")
            group_assignment = group_labels_df[var].values  # Get group labels for this variable
            isc_permutation_group_results[cond][var] = isc_utils.group_permutation(isc_values, group_assignment, n_perm, do_pairwise, side = 'two-sided', summary_statistic=summary_stat)
        
        print(f"Completed permutation ISC for {group_labels_df.columns}")

    results_save_dir = os.path.join(results_dir, 'behavioral_group_permutation')
    os.makedirs(results_save_dir, exist_ok=True)

    for cond, cond_results in isc_permutation_group_results.items():
        save_path = os.path.join(results_save_dir, f'{cond}_group_permutation_results_{n_perm}perm.pkl')
        isc_utils.save_data(save_path, cond_results)
        result_paths["group_permutation_results"][cond] = save_path

        # print(f"Saved ISC permutation results for condition: {cond} at {save_path}")

# %%
# %%
if do_rsa :
    # Load existing ISC results
    # task_to_test = setup.combined_task_to_test # [conditions, conditions[0:2], conditions[2:4]]
    combined_conditions = setup.combined_conditions #['all_sugg', 'modulation', 'neutral']
    # all_conditions = setup.all_conditions # ['Hyper', 'Ana', 'NHyper', 'NAna', 'all_sugg', 'modulation', 'neutral']
    all_conditions = conditions + combined_conditions

    isc_results = {}
    
    for cond in all_conditions:
        if cond in ['HYPER', 'ANA', 'NHYPER', 'NANA']:
            f = os.path.join(results_dir, cond, f"isc_results_{cond}_{n_boot}boot_pairWise{do_pairwise}.pkl")
            isc_results[cond] = isc_utils.load_pickle(f)
        elif cond in combined_conditions:
            # i = combined_conditions.index(cond)
            # combined_data = np.concatenate([transformed_data_per_cond[task] for task in task_to_test[i]], axis=0)
            # n_scans = combined_data.shape[0]
            f = os.path.join(concat_cond_save, f"isc_results_{cond}_{n_boot}boot_pairWise{do_pairwise}.pkl")
            isc_results[cond] = isc_utils.load_pickle(f) 

    # check if all cond are in isc_results

    # ISC-RSA
    # reload(utils)
    print('============================')
    print('ISC-RSA for each condition')
    subjectwise_rsa_results = {} # to save a subject x ROI matrix of ISC-RSA values

    for sim_model in ['euclidean', 'annak']:
        result_paths['rsa_isc_results'][sim_model] = {}

        rsa_save_dir = os.path.join(results_dir, f'rsa_isc_results_{sim_model}')
        os.makedirs(rsa_save_dir, exist_ok=True)

        for cond in all_conditions:
            start_cond_time = time.time()
            result_paths['rsa_isc_results'][sim_model][cond] = {}
            subjectwise_rsa_results[cond] = {}

            # # check mkdirs ?
            # if cond == 'all_sugg':
            #     breakpoint()
            save_cond_rsa = os.path.join(rsa_save_dir, f'rsa-isc_{cond}') # make condition folder
            os.makedirs(save_cond_rsa, exist_ok=True)

            isc_pairwise = isc_results[cond]['isc']
            values_rsa_perm = {}
            distribution_rsa_perm = {}

            if atlas_labels is None:
                atlas_labels = list(range(isc_pairwise.shape[1]))

            for behav_y in y_interest: # repeated for each Yi
                start_var_time = time.time()
                y = np.array(X_pheno[behav_y])
                y = (y - np.mean(y)) / np.std(y)
                sim_behav = isc_utils.compute_behav_similarity(y, metric = sim_model)
                result_paths['rsa_isc_results'][sim_model][cond][behav_y] = {}
                df_subjectwise_rsa = pd.DataFrame(index=range(isc_pairwise.shape[0]), columns=atlas_labels)

                for col_j in range(isc_pairwise.shape[1]): # j ROIs
                    if atlas_name == 'voxelWise':
                        roi_name = f'voxel_{col_j}'
                    else:
                        roi_name = atlas_labels[col_j]

                    isc_roi_vec = isc_pairwise[:, col_j]
                    rsa_results = isc_utils.matrix_permutation(sim_behav, isc_roi_vec, n_permute=n_perm_rsa, metric="spearman", how="upper", tail=1, return_perms = True)
                    values_rsa_perm[roi_name] = {'correlation': rsa_results['correlation'], 'p_value': rsa_results['p']}
                    distribution_rsa_perm[roi_name] = rsa_results['perm_dist']

                print(f'{behav_y} done in {time.time() - start_var_time:.2f} sec')
                distribution_rsa_perm['similarity_matrix'] = sim_behav
                #save
                df_rsa = pd.DataFrame.from_dict(values_rsa_perm, orient='index')
                csv_path = os.path.join(save_cond_rsa, f'{behav_y}_rsa_isc_{sim_model}simil_{n_perm_rsa}perm_pvalues.csv')
                df_rsa.to_csv(csv_path)
                dist_path = os.path.join(save_cond_rsa, f'{behav_y}_rsa_isc_{n_perm_rsa}perm_distribution.pkl')
                isc_utils.save_data(dist_path, distribution_rsa_perm)
                result_paths['rsa_isc_results'][sim_model][cond][behav_y] = {'csv': csv_path, 'distribution': dist_path}
            print(f'Done RSA-ISC ({sim_model} simil. model) for condition:', cond)
            print(f'{cond} time : ', time.time() - start_cond_time)
    # # %%
    # sim_model= 'annak'
    # rsa_save_dir = os.path.join(results_dir, f'rsa_isc_results_{sim_model}simil')
    # os.makedirs(rsa_save_dir, exist_ok=True)

    # for cond in conditions:
    #     save_cond_rsa = os.path.join(rsa_save_dir, f'rsa-isc_{cond}') # make condition folder
    #     os.makedirs(save_cond_rsa, exist_ok=True)

    #     isc_pairwise = isc_results[cond]['isc']
    #     values_rsa_perm = {}
    #     distribution_rsa_perm = {}

    #     for behav_y in y_interest: # repeated for each Yi
    #         y = np.array(X_pheno[behav_y])
    #         sim_behav_annak = utils.compute_behav_similarity(y, metric = sim_model)

    #         for col_j in range(isc_pairwise.shape[0]):
    #             if atlas_name == 'voxelWise':
    #                 roi_name = f'voxel_{col_j}'
    #             else:
    #                 roi_name = atlas_labels[col_j]

    #             isc_roi_vec = isc_pairwise[:, col_j]
    #             rsa_results = utils.matrix_permutation(sim_behav_annak, isc_roi_vec, n_permute=n_perm_rsa, metric="spearman", how="upper", return_perms = True)
    #             values_rsa_perm[roi_name] = {'correlation': rsa_results['correlation'], 'p_value': rsa_results['p']}
    #             distribution_rsa_perm[roi_name] = rsa_results['perm_dist']

    #         #save
    #         df_rsa = pd.DataFrame.from_dict(values_rsa_perm, orient='index')
    #         df_rsa.to_csv(os.path.join(save_cond_rsa, f'{behav_y}_rsa_isc_{sim_model}simil_{n_perm}perm_pvalues.csv'))
    #         utils.save_data(os.path.join(save_cond_rsa, f'{behav_y}_rsa_isc_{n_perm}perm_distribution.pkl'), distribution_rsa_perm)



# %%

    # %%

    # result_paths = {
    #     "isc_results": {},
    #     "isc_combined_results": {},
    #     "condition_contrast_results": {},
    #     "group_permutation_results": {},
    #     "rsa_isc_results": {},
    #     "setup_parameters": save_setup
    # }

    # # Save ISC results per condition
    # for cond, isc_dict in isc_results.items():
    #     save_path = os.path.join(results_dir, cond, f"isc_results_{cond}_{n_boot}boot_pairWise{do_pairwise}.pkl")
    #     result_paths["isc_results"][cond] = save_path

    # # Save combined condition ISC results
    # combined_conditions = ['all_sugg', 'modulation', 'neutral']
    # task_to_test = [conditions, conditions[0:2], conditions[2:4]]
    # concat_cond_save = os.path.join(results_dir, 'concat_suggs_1samp_boot')

    # for i, comb_cond in enumerate(combined_conditions):
    #     combined_data = np.concatenate([transformed_data_per_cond[task] for task in task_to_test[i]], axis=0)
    #     n_scans = combined_data.shape[0]
    #     save_path = os.path.join(concat_cond_save, f"isc_results_{comb_cond}_{n_scans}TRs_{n_boot}boot_pairWise{do_pairwise}.pkl")
    #     result_paths["isc_combined_results"][comb_cond] = save_path

    # # Save condition contrast permutation results
    # # contrast_perm_save = os.path.join(results_dir, 'condition_contrast_results')
    # # for contrast in contrast_conditions:
    # #     save_path = os.path.join(contrast_perm_save, f"isc_results_{contrast}_{n_perm}perm_pairWise{do_pairwise}.pkl")
    # #     result_paths["condition_contrast_results"][contrast] = save_path
        
    # # save condition contrast permutation results
    # # contrast_conditions = ['Hyper-Ana', 'Ana-Hyper', 'NHyper-NAna']
    # # contrast_to_test = [conditions[0:2], conditions[0:2][::-1], conditions[2:4]]
    # # contrast_perm_save = os.path.join(results_dir, 'cond_contrast_permutation')

    # # for i, contrast in enumerate(contrast_conditions):

    # #     combined_data_ls = [transformed_data_per_cond[task] for task in contrast_to_test[i]]
    # #     if contrast == 'Hyper-Ana':
    # #         adjusted_Hyper, adjusted_Ana = utils.trim_TRs(combined_data_ls[0], combined_data_ls[1])
    # #     elif contrast == 'Ana-Hyper':
    # #         adjusted_Ana, adjusted_Hyper = utils.trim_TRs(combined_data_ls[1], combined_data_ls[0])
    # #     if i == 2: # for the neutral condtions, no need to trim
    # #         combined_data = np.concatenate(combined_data_ls, axis=2)
    # #     else:
    # #         combined_data = np.concatenate([adjusted_Hyper, adjusted_Ana], axis=2)
    # #     n_scans = combined_data.shape[0]
    # #     save_path = os.path.join(contrast_perm_save, f"isc_results_{contrast}_{n_scans}TRs_{n_perm}perm_pairWise{do_pairwise}.pkl")
    # #     result_paths["condition_contrast_results"][contrast] = save_path
        
    # # # Save behavioral group permutation results
    # # results_save_dir = os.path.join(results_dir, 'behavioral_group_permutation')
    # # for cond, cond_results in isc_permutation_group_results.items():
    # #     save_path = os.path.join(results_save_dir, f"{cond}_group_permutation_results_{n_perm}perm.pkl")
    # #     result_paths["group_permutation_results"][cond] = save_path

    # # # ISC-RSA results
    # # sim_model= 'euclidean'
    # # for sim_model in ['euclidean', 'annak']: 
    # #     result_paths['rsa_isc_results'][sim_model] = {}
    # #     rsa_save_dir = os.path.join(results_dir, f'rsa_isc_results_{sim_model}')
    # #     for cond in conditions:
    # #         result_paths['rsa_isc_results'][sim_model][cond] = {}
    # #         save_cond_rsa = os.path.join(rsa_save_dir, f'rsa-isc_{cond}') # make condition folder
    # #         for behav_y in y_interest:
    # #             csv_path = os.path.join(save_cond_rsa, f'{behav_y}_rsa_isc_{sim_model}simil_{n_perm}perm_pvalues.csv')
    # #             dist_path = os.path.join(save_cond_rsa, f'{behav_y}_rsa_isc_{n_perm}perm_distribution.pkl')
    # #             result_paths['rsa_isc_results'][sim_model][cond][behav_y] = {'csv': csv_path, 'distribution': dist_path}

# Save the result paths dictionary
result_paths_save_path = os.path.join(results_dir, "result_paths.json")
with open(result_paths_save_path, 'w') as f:
    json.dump(result_paths, f, indent=4)

print(f"Result paths saved at {result_paths_save_path}")
print('MODEL RAN : ', model_name)
print('Done with all!!')

    # %%

