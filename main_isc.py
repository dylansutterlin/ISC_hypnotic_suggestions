

# %%
import utils
from importlib import reload
import nibabel as nib
import os
import pandas as pd
from nilearn.image import concat_imgs
#import brainiak
#from brainiak.isc import isc, bootstrap_isc, permutation_isc
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
transformed_data_per_cond = test_data
fitted_maskers = test_masker

print(test_data.shape)
print(test_masker)

# %%
# Perform ISC
isc_results = {}

for cond, data in transformed_data_per_cond.items():
    print(f'Performing ISC for condition: {cond}')
    # Convert list of 2D arrays to 3D array (subjects, timepoints, regions)
    data_3d = np.array(data)
    # Perform ISC
    isc_result = isc(data_3d, pairwise=False, summary_statistic=None)
    isc_results[cond] = isc_result

# Save ISC results
isc_results_path = os.path.join(results_dir, 'isc_results.npy')
np.save(isc_results_path, isc_results)
print(f'ISC results saved to {isc_results_path}')

