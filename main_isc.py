

# %%
import utils
from importlib import reload
import nibabel as nib
import numpy as np
import os
import pandas as pd
from nilearn.image import concat_imgs
import brainiak
from brainiak.isc import isc, bootstrap_isc, permutation_isc, compute_summary_statistic
reload(utils)
# %% Load the data
# Example usage
base_path = "/data/rainville/dSutterlin/projects/ISC_hypnotic_suggestions/results/data/test_data_sugg_3sub"
base_path = '/home/dsutterlin/projects/test_data/suggestion_block_concat_4D_3subj'

project_dir = '/home/dsutterlin/projects/ISC_hypnotic_suggestions'
results_dir = os.path.join(project_dir, 'results/imaging/ISC')
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

isc_data_df = utils.load_isc_data(base_path)
subjects = isc_data_df['subject'].unique()
isc_data_df = isc_data_df.sort_values(by='subject')
# Display the DataFrame
print(isc_data_df.head())

# %%
# get dict will all cond files from all subjects

subjects = isc_data_df['subject'].unique()
n_sub = len(subjects)
n_boot = 1000

print("Subjects:", subjects)

tasks_ordered = sorted(isc_data_df['task'].unique())
# code conditions for all_sugg, (ANA+Hyper), (Neutral_H + Neutral_A)
conditions = ['all_sugg', 'modulation', 'neutral']
task_combinations = [tasks_ordered, tasks_ordered[0:2], tasks_ordered[2:4]]
subject_file_dict = {}
for i, cond in enumerate(conditions):
    subject_file_dict[cond] = utils.get_files_for_condition_combination(subjects, task_combinations[i], isc_data_df)

print("Conditions:", conditions)
print("Condition combinations:", task_combinations)
print(subject_file_dict)

# %%

# load difumo atlas
atlas_path = os.path.join(project_dir, 'masks/DiFuMo256/3mm/maps.nii.gz')
atlas_dict_path = os.path.join(project_dir, 'masks/DiFuMo256/labels_256_dictionary.csv')
atlas = nib.load(atlas_path)
atlas_df = pd.read_csv(atlas_dict_path)
atlas_name = 'Difumo64'
print('atlas loaded with N ROI : ', atlas.shape)

# %%
# extract timseries from atlas
from nilearn.maskers import MultiNiftiMapsMasker

masker = MultiNiftiMapsMasker(maps_img=atlas, standardize=False, memory='nilearn_cache', verbose=5)
masker.fit()

transformed_data_per_cond = {}
fitted_maskers = {}
# extract time series for each subject and condition
for cond in conditions:
    cond = conditions[1]

    condition_files = subject_file_dict[cond]
    concatenated_subjects = {sub : concat_imgs(sub_files) for sub, sub_files in condition_files.items()}
    # assert all images have the same shape
    for sub, img in concatenated_subjects.items():
        assert img.shape == concatenated_subjects[subjects[0]].shape

    print(f'fitting images for condition : {cond} with shape {concatenated_subjects[subjects[0]][0].shape}')
    transformed_data_per_cond[cond] = masker.transform(concatenated_subjects.values())
    fitted_maskers[cond] = masker

# save transformed data and masker
for cond, data in transformed_data_per_cond.items():
    cond_folder = os.path.join(results_dir, cond)
    if not os.path.exists(cond_folder):
        os.makedirs(cond_folder)

    save_path = os.path.join(cond_folder, f'transformed_{atlas_name}_{cond}.pkl')
    utils.save_data(save_path, data)

    masker_path = os.path.join(cond_folder, f'maskers_{atlas_name}_{cond}.pkl')
    utils.save_data(masker_path, fitted_maskers[cond])
    print(f'Transformed timseries and maskers saved to {cond_folder}')



# %%

# test moddule
test_results = '/home/dsutterlin/projects/ISC_hypnotic_suggestions/results/imaging/ISC/modulation'
ts_path = os.path.join(test_results, 'transformed_Difumo256_modulation.pkl')
masker_path = os.path.join(test_results, 'maskers_Difumo256_modulation.pkl')
test_data = utils.load_pickle(ts_path)
test_masker = utils.load_pickle(masker_path)
transformed_data_per_cond = {}
transformed_data_per_cond['modulation'] = test_data
fitted_maskers = {}
fitted_maskers['modulation'] = test_masker



print(test_data[0].shape)
print(test_masker)

# %%
# Perform ISC
n_boot = 100
isc_results = {}

for cond, data in transformed_data_per_cond.items():
    print(f'Performing ISC for condition: {cond}')
    # Convert list of 2D arrays to 3D array (n_TRs, n_voxels, n_subjects)
    data_3d = np.stack(data, axis=-1)  # Perform ISC
    isc_result = isc(data_3d, pairwise=True, summary_statistic=None)
    isc_results[cond] = isc_result

    observed, ci, p, distribution = bootstrap_isc(
    isc_result,
    pairwise=False,
    summary_statistic="median",
    n_bootstraps=n_boot,
    ci_percentile=95,
)
    
    median_isc = compute_summary_statistic(isc_result, 'median', axis=0) # per ROI : 1, n_voxels
    total_median_isc = compute_summary_statistic(isc_result, 'median', axis=None)
    print(f'Median ISC for {cond}: {total_median_isc}')
    

    isc_results[cond] = {
    "isc": isc_result,
    "observed": observed,
    "confidence_intervals": ci,
    "p_values": p,
    "distribution": distribution,
}

# save results
for cond, results in isc_results.items():
    save_path = os.path.join(results_dir, f"isc_results_{cond}.pkl")
    utils.save_data(save_path, results)
    print(f"ISC results saved for {cond} at {save_path}")

# summary stats

# %%
# Project isc values and p values to brain
from nilearn import plotting
from nilearn.plotting import plot_stat_map
from nilearn.image import new_img_like

# inverse trnasform isc with fitted maskers
for cond in conditions:
    if cond != 'modulation':
        continue
    masker = fitted_maskers[cond]
    isc_result = isc_results[cond]
    isc_img = masker.inverse_transform(isc_result['observed'])
    p_img = masker.inverse_transform(isc_result['p_values'])
   
    # save images
    # save_path = os.path.join(results_dir, f"isc_{cond}.nii.gz")
    # isc_img.to_filename(save_path)
    # print(f"ISC image saved to {save_path}")

    # save_path = os.path.join(results_dir, f"p_values_{cond}.nii.gz")
    # p_img.to_filename(save_path)
    # print(f"p-values image saved to {save_path}")

    # save_path = os.path.join(results_dir, f"confidence_intervals_{cond}.nii.gz")
    # ci_img.to_filename(save_path)
    # print(f"confidence intervals image saved to {save_path}")

    # plot images
    plot_stat_map(isc_img, title=f'ISC {cond}', colorbar=True, threshold=0.1, display_mode='z', cut_coords=10)
    plot_stat_map(p_img, title=f'p-values {cond}', colorbar=True, threshold=0.05, display_mode='z', cut_coords=10)


# %%
# Permutation based on group labels

# load behavioral data

behav_path = r"/data/rainville/dSutterlin/projects/resting_hypnosis/resting_state_hypnosis/atlases/Hypnosis_variables_20190114_pr_jc.xlsx"
APM_subjects = ['APM' + sub[4:] for sub in subjects] # make APMXX format instead of subXX

import func
import importlib
importlib.reload(func) 

phenotype= func.load_process_y(behav_path,APM_subjects)
phenotype.head()

# Create group labels based on sHSS and change in pain


# 