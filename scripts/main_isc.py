

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
    print("Running in interactive mode. Using default setup: `atlas_setup`")
    setup_name = "setups.debug_setup"  # Default setup when running cells manually

print("Current Working Directory:", os.getcwd())

if "setups" not in sys.modules:
    sys.path.append(os.path.abspath("setups"))

setup_module = importlib.import_module(f"{setup_name}")
setup = setup_module.init()

print(f"Loaded setup: {setup_name}")
# print(f"Running model: {setup.model_name} with {setup.n_sub} subjects")
print(f"Results will be saved in: {setup.results_dir}")

# %%
project_dir = setup.project_dir
preproc_model_data = setup.preproc_model_data
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
do_rsa = setup.do_rsa
n_boot = setup.n_boot
do_pairwise = setup.do_pairwise
n_perm = setup.n_perm
n_perm_rsa = setup.n_perm_rsa
n_rois = setup.n_rois
atlas_name = setup.atlas_name
model_name = setup.model_name
results_dir = setup.results_dir

# behav data
isc_data_df = isc_utils.load_isc_data(base_path, folder_is='subject', model = model_is) # !! change params if change base_dir
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

# initialize results dir
setup.subjects = subjects
setup.check_and_create_results_dir()
setup.save_to_json()
print('Results directory at :' , setup.results_dir)

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
voxel_masker = Bunch(name="voxel_wise")
ref_img = index_img(nib.load(subject_file_dict[conditions[0]][subjects[0]]), 0)
data_shape = ref_img.shape[0:3] #3D
data_affine = ref_img.affine

if atlas_name=='voxelWise' or atlas_name == f'schafer{n_rois}_2mm':
    mask = nib.load('/data/rainville/Hypnosis_ISC/masks/brainmask_91-109-91.nii')
    # resamp_mask_cont = resample_to_img(mask, ref_img)
    resamp_mask = qc_utils.resamp_to_img_mask(mask, ref_img)
    mask_path = os.path.join(results_dir, 'resamp_mask_mni.nii.gz')
    mask.to_filename(mask_path)
    # resamp_mask = binarize_img(resamp_mask_cont,mask_img=mask, threshold=0)
    plot_roi(resamp_mask, title='resampled_mask', display_mode='ortho', draw_cross=False, output_file=os.path.join(results_dir, 'resampled_mask.png'))

elif atlas_name=='voxelWise_lanA800':
    full_mask = nib.load(os.path.join(project_dir, 'masks/lipkin2022_lanA800', 'LanA_n806.nii'))
    at_thresh = 0.30
    mask_native = binarize_img(full_mask, threshold=at_thresh)
    mask = qc_utils.resamp_to_img_mask(mask_native, ref_img)
    # mask = resample_to_img(mask_native, ref_img)
    mask_path = os.path.join(results_dir, f'bin_lanA800_{at_thresh}thresh.nii.gz')
    mask.to_filename(mask_path)
    plot_roi(mask, title='resampled_mask', display_mode='ortho', draw_cross=False, output_file=os.path.join(results_dir, 'resampled_mask.png'))

masker_params_dict = {
    "standardize": 'zscore_sample',
    "mask_img": mask_path,
    "target_shape": None,
    "target_affine": None,
    "detrend": False,
    "low_pass": None,
    "high_pass": 0.01,  # 1/428 sec.
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

elif atlas_name == f'schafer{n_rois}_2mm':
    atlas_data = fetch_atlas_schaefer_2018(n_rois = n_rois, resolution_mm=2)
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
y_interest = ['SHSS_score', 'total_chge_pain_hypAna', 'Abs_diff_automaticity']
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
    elif atlas_name == f'schafer{n_rois}_2mm':
        masker = NiftiLabelsMasker(labels_img=atlas, labels = atlas_labels, mask_img=resamp_mask,resampling_target='data', standardize=True,high_variance_confounds=False, memory='nilearn_cache', verbose=5, n_jobs= 1)

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
        if 'voxelWise' in atlas_name or atlas_name == f'schafer{n_rois}_2mm':
            # ls_voxel_wise = [masker.fit_transform(concatenated_subjects[sub],
            #                                        confounds= confounds_ls[i][cond])
            #                                          for i, sub in enumerate(subjects)
            # ]
            print(f'Fitting {atlas_name} masker for {cond}')
            ls_voxel_wise = []
            fitted_maskers[cond] = []
            
            for i, sub in enumerate(subjects):

                if reg_conf == True:
                    conf = confounds_ls[i][cond]
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

    # mask_img = nib.load(voxel_masker.obj.mask_img)
    # resmap_mask = resample_img(mask_img, target_affine=data_affine, target_shape=data_shape)
    # cond= 'Ana'
    
    qc_path = os.path.join(setup.results_dir, 'QC')
    os.makedirs(qc_path, exist_ok=True)
    print('==Carpet plots==')

    if atlas_name =='voxelWise':
        carpet_mask = resamp_mask    
        carpet_lab = None
    elif atlas_name == f'schafer{n_rois}_2mm':
        carpet_mask = atlas
        carpet_lab = None
    elif atlas_name =='voxelWise_lanA800':
        carpet_mask = resamp_mask
        carpet_lab = None

    for sub in range(n_sub):
        subject = subjects[sub]
        shapes = []
        for cond in conditions:
            file_path = os.path.join(qc_path, f'{subject}_{cond}_carpet_detrend.png')
            inv_img = fitted_maskers[cond][sub].inverse_transform(transformed_data_per_cond[cond][sub])
            shapes.append(inv_img.shape)
            display = plot_carpet(
                inv_img,
                mask_img=carpet_mask,
                mask_labels=carpet_lab,
                detrend=True,
                t_r=3,
                standardize=True,
                output_file=file_path,
                title=f"global patterns {subject} in cond {cond}",
            )

    # combined carpet plot
    for cond in conditions:
        carpet_files = glob.glob(os.path.join(results_dir, 'QC', f'sub-*_{cond}_carpet_*.png'))
        carpet_files = sorted(carpet_files)
        save_cond= os.path.join(results_dir,'QC', f'combined_sub_{cond}_carpet.png')
        title = f'{cond} for model : {model_name}'
        visu_utils.plot_images_grid(carpet_files, title, save_to=save_cond, show=True)

        print('Subject : ', subject, 'imgs shapes : ', shapes)

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
    
# sim_model= 'annak'
# rsa_save_dir = os.path.join(results_dir, f'rsa_isc_results_{sim_model}simil')
# os.makedirs(rsa_save_dir, exist_ok=True)

# for cond in conditions:
#     save_cond_rsa = os.path.join(rsa_save_dir, f'rsa-isc_{cond}') # make condition folder
#     os.makedirs(save_cond_rsa, exist_ok=True)

#     isc_pairwise = isc_results[cond]['isc']
#     values_rsa_perm = {}
#     distribution_rsa_perm = {}>

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


else:
# Loading data
    print(f'Loading existing data and fitted maskers')
    transformed_data_per_cond = {}
    fitted_maskers = {}
    transformed_sphere_per_roi = {}

    for cond in setup.conditions:
        load_path = os.path.join(setup.results_dir, cond)
        transformed_path =  os.path.join(load_path, f'transformed_data_{atlas_name}_{cond}_{n_sub}sub.npz')
        transformed_data_per_cond[cond] = np.load(transformed_path)['arr_0'] #utils.load_pickle(os.path.join(load_path, transformed_path))
        fitted_mask_file = f'maskers_{atlas_name}_{cond}_{n_sub}sub.pkl'
        fitted_maskers[cond] = isc_utils.load_pickle(os.path.join(load_path, fitted_mask_file))

        #load_roi = os.path.join(load_path, roi_folder)
        #transformed_sphere_per_roi[cond] =utils.load_pickle(os.path.join(load_roi, f"{len(roi_coords)}ROIs_timeseries.pkl")) 

    print(f'Loading existing data and fitted maskers from : {load_path}')


#%%
# Perform ISC

# # isc with spheres ROI
# isc_results_roi = {}
# for cond in conditions:

#     print(f'Performing sphere ISC for condition: {cond}')
#     roi_ls = [] #stack list of roi timepoints x subjects 
#     for roi in roi_coords.keys():
#         roi_dict = transformed_sphere_per_roi[cond][roi]
#         roi_ts = [] # stack timepoints x subjects
#         for sub in roi_dict.keys():
#             roi_ts.append(np.array(roi_dict[sub])) # list timepoints x 1 vector
#         roi_ls.append(np.array(roi_ts)) # list of roi timepoints x subjects
#     roi_data_3d = np.stack(np.squeeze(roi_ls), axis=1).T # timepoints x roi x subjects
    
#     isc_results_roi[cond] = utils.isc_1sample(roi_data_3d, pairwise=do_pairwise, n_boot=n_boot, summary_statistic=None)
    
#roi_isc['roi_names'] = list(roi_coords.keys())

# for cond, isc_dict in isc_results_roi.items():
#     save_path = os.path.join(results_dir, cond, roi_folder, f"isc_{len(roi_coords)}spheres_{cond}_{n_boot}boot_pairWise{do_pairwise}.pkl")
#     utils.save_data(save_path, isc_dict)
#     print(f"ISC results saved for {cond} at {save_path}")

if atlas_name != f'schafer{n_rois}_2mm':
    # Assess 0 variance subjects
    sub_to_remove = isc_utils.identify_zero_variance_subjects(transformed_data_per_cond, subjects)
    print(f"Subjects with 0 variance will be removed : {sub_to_remove}")
    transformed_data_per_cond, fitted_masker, subjects = isc_utils.remove_subjects(transformed_data_per_cond, fitted_maskers, subjects, sub_to_remove)
    setup.kept_subjects = subjects
    n_sub= len(subjects)

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

    for cond, data in transformed_data_per_cond.items():
        print(f'Performing ISC for condition: {cond}')
        start_cond_time = time.time()
        # Convert list of 2D arrays to 3D array (n_TRs, n_voxels, n_subjects)
        print(data_3d.shape)

        isc_results[cond] = isc_utils.isc_1sample(data_3d, pairwise=do_pairwise,n_boot = n_boot, summary_statistic=None)

        # save results
        save_path = os.path.join(results_dir, cond, f"isc_results_{cond}_{n_boot}boot_pairWise{do_pairwise}.pkl")
        isc_utils.save_data(save_path, isc_results[cond])
        result_paths["isc_results"][cond] = save_path

        end_cond_time = time.time()
        print(f" {cond} done in {end_cond_time - start_cond_time:.2f} sec")
        print(f"ISC results saved for {cond} at {save_path}")

# %%

    # %%

    # Combined condition bootstrap
    print('====================================')
    # code conditions for all_sugg, (ANA+Hyper), (Neutral_H + Neutral_A)
    combined_conditions = ['all_sugg', 'modulation', 'neutral']
    task_to_test = [conditions, conditions[0:2], conditions[2:4]]
    concat_cond_save = os.path.join(results_dir, 'concat_suggs_1samp_boot')
    os.makedirs(concat_cond_save, exist_ok=True)

    for i, comb_cond in enumerate(combined_conditions):
        start_cond_time = time.time()
        print(f'Performing bootstraped ISC for combined condition: {comb_cond}')
        
        combined_data = np.concatenate([transformed_data_per_cond[task]for task in task_to_test[i]], axis=0)
        n_scans = combined_data.shape[0] #  concat along TR axis for each 3d task array
        n_sub = combined_data.shape[-1]
        # isc_combined = isc(combined_data, pairwise=do_pairwise, summary_statistic=None)

        isc_results[comb_cond] = isc_utils.isc_1sample(combined_data, pairwise=do_pairwise,n_boot = n_boot, summary_statistic=None)

        # save results
        save_path = os.path.join(concat_cond_save, f"isc_results_{comb_cond}_{n_scans}TRs_{n_boot}boot_pairWise{do_pairwise}.pkl")
        isc_utils.save_data(save_path, isc_results[comb_cond])
        result_paths["isc_combined_results"][comb_cond] = save_path

        end_cond_time = time.time()
        print(f" {cond} done in {end_cond_time - start_cond_time:.2f} sec")
        print(f"ISC results saved for {comb_cond} at {save_path}")


    # %%
    #==============================
    # ISFC per conditions

    # from brainiak.isfc import isfc

    # conditions = ['Hyper', 'Ana', 'NHyper', 'NAna']
    # isfc_results_per_cond = {}
    # isc_results = {}
    # for cond, data in transformed_data_per_cond.items():
    #     print(f'Performing ISC for condition: {cond}')
    #     # Convert list of 2D arrays to 3D array (n_TRs, n_voxels, n_subjects)
    #     if isinstance(data, list):
    #         data_3d = np.stack(data, axis=-1)
    #     elif isinstance(data, np.ndarray):
    #         data_3d = data
    #     assert data_3d.shape[-1] == setup.n_sub, f"Data shape {data_3d.shape} does not match number of subjects {n_sub}"
        
    #     isfc_results_per_cond[cond] = isfc(data, pairwise=True, vectorize_isfcs=True)



    # %%    
    # Project isc values and p values to brain

    # # inverse trnasform isc with fitted maskers
    # for cond, isc_dict in isc_results.items():

    #     masker = fitted_maskers[cond]
    #     isc_img = masker.inverse_transform(isc_dict['observed'])
    #     p_img = masker.inverse_transform(isc_dict['p_values'])

    #     # save images
    #     save_boot = os.path.join(results_dir, cond)
    #     isc_img_name = f"isc_val_{cond}_boot{n_boot}_pariwise{do_pairwise}.nii.gz"
    #     isc_img.to_filename(os.path.join(save_boot, isc_img_name))
    #     print(f"ISC image saved to {save_boot}")

    #     isc_pval_img_name =  f"p_values_{cond}_boot{n_boot}_pairwise{do_pairwise}.nii.gz"
    #     p_img.to_filename(os.path.join(save_boot, isc_pval_img_name))
    #     print(f"p-values group image saved to {save_boot}")

    #     # plot images
    #     plot_path = os.path.join(results_dir, cond, f"isc_plot_{cond}_boot{n_boot}_pairwise{do_pairwise}.png")
    #     plotting.plot_stat_map(isc_img, title=f'ISC {cond}', colorbar=True, threshold=0.1, display_mode='z', cut_coords=10, output_file=plot_path)
    #     print(f"ISC plot saved to {plot_path}")

    #     plot_path = os.path.join(results_dir, cond, f"p_values_plot_{cond}_boot{n_boot}_pairwise{do_pairwise}.png")
    #     plotting.plot_stat_map(p_img, title=f'p-values {cond}', colorbar=True, threshold=0.05, display_mode='z', cut_coords=10, output_file=plot_path)
    #     print(f"p-values plot saved to {plot_path}")


    # %%
    # Permutation based on group labels

    # ----------------
    # sphere permutation
    # isc_results_roi = {}
    # for cond in conditions:

    #     print(f'Performing sphere ISC for condition: {cond}')
    #     roi_ls = [] #stack list of roi timepoints x subjects 
    #     for roi in roi_coords.keys():
    #         roi_dict = transformed_sphere_per_roi[cond][roi]
    #         roi_ts = [] # stack timepoints x subjects
    #         for sub in roi_dict.keys():
    #             roi_ts.append(np.array(roi_dict[sub])) # list timepoints x 1 vector
    #         roi_ls.append(np.array(roi_ts)) # list of roi timepoints x subjects
        
    #     roi_data_3d = np.stack(np.squeeze(roi_ls), axis=1).T # timepoints x roi x subjects
    #     isc_results_roi[cond] = utils.isc_1sample(roi_data_3d, pairwise=do_pairwise, n_boot=n_boot, summary_statistic=None)
        
    # #roi_isc['roi_names'] = list(roi_coords.keys())

    # for cond, isc_dict in isc_results_roi.items():
    #     save_path = os.path.join(results_dir, cond, roi_folder, f"isc_{len(roi_coords)}spheres_{cond}_{n_boot}boot_pairWise{do_pairwise}.pkl")
    #     utils.save_data(save_path, isc_dict)
    #     print(f"ISC results saved for {cond} at {save_path}")

    # isc_sphere_permutation_group_results = {}
    # for cond, isc_dict in isc_results_roi.items():  # `isc_results` should already contain ISC values
    #     print(f"Performing permutation ISC for condition: {cond}")

    #     isc_values = isc_dict['isc']

    #     for var in group_labels_df.columns:
    #         group_assignment = group_labels_df[var].values  # Get group labels for this variable

    #         # Perform ISC permutation test
    #         observed, p_value, distribution = permutation_isc(
    #             isc_values,  # This should be the ISC matrix from the main analysis
    #             group_assignment=group_assignment,
    #             pairwise=do_pairwise,
    #             summary_statistic="median",  # Median ISC
    #             n_permutations=n_perm,
    #             side="two-sided",
    #             random_state=42
    #         )

    #         if cond not in isc_sphere_permutation_group_results:
    #             isc_sphere_permutation_group_results[cond] = {}

    #         isc_sphere_permutation_group_results[cond][var] = {
    #             "observed": observed,
    #             "p_value": p_value,
    #             "distribution": distribution
    #         }

    #         print(f"Completed SPHERE permutation ISC for condition: {cond}, variable: {var}")

    # # Save the permutation results
    # for cond, cond_results in isc_sphere_permutation_group_results.items():
    #     save_path = os.path.join(results_dir, cond, roi_folder, f"isc_{n_perm}permutation_results_{cond}_pairwise{do_pairwise}.pkl")
    #     utils.save_data(save_path, cond_results)
    #     print(f"sphere Permutation ISC results saved for {cond} at {save_path}")

    #%% 
    # Paired sample permutation test
    contrast_conditions = ['Hyper-Ana', 'Ana-Hyper', 'NHyper-NAna']
    contrast_to_test = [conditions[0:2], conditions[0:2][::-1], conditions[2:4]]

    isc_permutation_cond_contrast = {}

    contrast_perm_save = os.path.join(results_dir, 'cond_contrast_permutation')
    os.makedirs(contrast_perm_save, exist_ok=True)

    reload(isc_utils)
    for i, contrast in enumerate(contrast_conditions):
        print(f'Performing 2 group permutation ISC : {contrast}')
        start_cond_time = time.time()
        combined_data_ls = [transformed_data_per_cond[task] for task in contrast_to_test[i]]

        combined_data = np.concatenate(combined_data_ls, axis=2)
        n_scans = combined_data.shape[0]

        isc_grouped = isc(combined_data, pairwise=do_pairwise, summary_statistic=None)
        group_ids = np.array([0] * n_sub + [1] * n_sub)
        isc_permutation_cond_contrast[contrast] = isc_utils.group_permutation(isc_grouped, group_ids, n_perm, do_pairwise, side = 'two-sided', summary_statistic='median')

        save_path = os.path.join(contrast_perm_save, f"isc_results_{contrast}_{n_scans}TRs_{n_perm}perm_pairWise{do_pairwise}.pkl")
        isc_utils.save_data(save_path, isc_permutation_cond_contrast[contrast])
        result_paths["condition_contrast_results"][contrast] = save_path
        
        print(f'contrast {contrast} done in {time.time() - start_cond_time:.2f} sec')

    #%%
    shss_grps = ['low_shss', 'high_shss']
    contrast_conditions = ['Hyper-Ana', 'Ana-Hyper', 'NHyper-NAna']
    contrast_to_test = [conditions[0:2], conditions[0:2][::-1], conditions[2:4]]

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

        print(f'-----{shss_grp} grp with {keep_n} subjects')
        for i, contrast in enumerate(contrast_conditions):

            combined_data_ls = [transformed_data_per_cond[task] for task in contrast_to_test[i]]
            combined_data = np.concatenate(combined_data_ls, axis=2)
            combined_data_selected = combined_data[:, :, shss_idx_concat]
            print(f'{contrast} : Repeated mesaure isc having shape : ', combined_data_selected.shape)
        # print('group id unique : ', np.unique(group_ids_selected, return_counts=True))  
        
            isc_grouped_selected = isc(combined_data_selected, pairwise=do_pairwise, summary_statistic=None)
            group_ids = np.array([0] * n_sub + [1] * n_sub)
            isc_permutation_cond_contrast[contrast] = isc_utils.group_permutation(isc_grouped_selected, group_ids_selected, n_perm, do_pairwise, side = 'two-sided', summary_statistic='median')

            save_path = os.path.join(contrast_perm_shss, f"isc_results_{keep_n}sub_{contrast}_{n_perm}perm_pairWise{do_pairwise}.pkl")
            isc_utils.save_data(save_path, isc_permutation_cond_contrast[contrast])
            result_paths["condition_contrast_results"][contrast] = save_path
        print(f'Contrasts for {shss_grp} done in {time.time() - start_grp_time:.2f} sec')    

    # %%
    # ------------
    # Perform permutation based on group labels
    reload(isc_utils)

    isc_permutation_group_results = {}
    for cond, isc_dict in isc_results.items():  # `isc_results` should already contain ISC values
        print(f"Performing permutation ISC for condition: {cond}")
        var_isc_results = {}
        isc_values = isc_dict['isc']
        isc_permutation_group_results[cond] = {}

        for var in group_labels_df.columns:
            group_assignment = group_labels_df[var].values  # Get group labels for this variable
            isc_permutation_group_results[cond][var] = isc_utils.group_permutation(isc_values, group_assignment, n_perm, do_pairwise, side = 'two-sided', summary_statistic='median')
        
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
    task_to_test = setup.combined_task_to_test # [conditions, conditions[0:2], conditions[2:4]]
    combined_conditions = setup.combined_conditions #['all_sugg', 'modulation', 'neutral']
    all_conditions = setup.all_conditions # ['Hyper', 'Ana', 'NHyper', 'NAna', 'all_sugg', 'modulation', 'neutral']

    isc_results = {}
    
    for cond in all_conditions:
        if cond in ['HYPER', 'ANA', 'NHYPER', 'NANA']:
            f = os.path.join(results_dir, cond, f"isc_results_{cond}_{n_boot}boot_pairWise{do_pairwise}.pkl")
            isc_results[cond] = isc_utils.load_pickle(f)
        elif cond in combined_conditions:
            i = combined_conditions.index(cond)
            combined_data = np.concatenate([transformed_data_per_cond[task] for task in task_to_test[i]], axis=0)
            n_scans = combined_data.shape[0]
            f = os.path.join(concat_cond_save, f"isc_results_{cond}_{n_scans}TRs_{n_boot}boot_pairWise{do_pairwise}.pkl")
            isc_results[cond] = isc_utils.load_pickle(f) 

    # check if all cond are in isc_results
    breakpoint()
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
print('Done with all!!')

    # %%

