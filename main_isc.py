

# %%
import utils
from importlib import reload
import nibabel as nib
import numpy as np
import os
import pandas as pd
import json
from nilearn.image import concat_imgs
from brainiak.isc import isc, bootstrap_isc, permutation_isc, compute_summary_statistic, phaseshift_isc
from nilearn.maskers import MultiNiftiMapsMasker
from sklearn.utils import Bunch
#reload(utils)
# %% Load the data

project_dir = "/data/rainville/dSutterlin/projects/ISC_hypnotic_suggestions"
# base_path = "/data/rainville/dSutterlin/projects/ISC_hypnotic_suggestions/results/data/test_data_sugg_3sub"
preproc_model_data = '23subjects_zscore_sample_detrend_FWHM6_low-pass428_10-12-24/suggestion_blocks_concat_4D_23sub'
base_path = os.path.join(project_dir, 'results/imaging/preproc_data', preproc_model_data)
behav_path = os.path.join(project_dir, 'results/behavioral/behavioral_data_cleaned.csv')
exclude_sub = ['sub-02']
keep_n_subjects = None

import sys
sys.path.append(os.path.join(project_dir, 'QC_py310'))
import func

# base_path = '/home/dsutterlin/projects/test_data/suggestion_block_concat_4D_3subj'
# project_dir = '/home/dsutterlin/projects/ISC_hypnotic_suggestions'
# behav_path = os.path.join('/home/dsutterlin/projects/ISC_hypnotic_suggestions/results/behavioral', 'behavioral_data_cleaned.csv')

isc_data_df = utils.load_isc_data(base_path)
sub_check = {}
# Select a subset of subjects
if keep_n_subjects is not None:
    isc_data_df = isc_data_df[isc_data_df['subject'].isin(isc_data_df['subject'].unique()[:keep_n_subjects])]
    subjects = isc_data_df['subject'].unique()

# exclude subjects
isc_data_df = isc_data_df[~isc_data_df['subject'].isin(exclude_sub)]
isc_data_df = isc_data_df.sort_values(by='subject')
subjects = isc_data_df['subject'].unique()
n_sub = len(subjects)

# hyperparams
model_name = f'model2_zcore_sample-{str(n_sub)}sub'
conditions = ['all_sugg', 'modulation', 'neutral']
transform_imgs = False
nsub = len(subjects)
n_sub = len(subjects)
n_boot = 10000
do_pairwise = True
n_perm = 10000

results_dir = os.path.join(project_dir, f'results/imaging/ISC/{model_name}')
if not os.path.exists(results_dir):
    os.makedirs(results_dir)
elif transform_imgs == False:
    print(f"Results directory {results_dir} already exists and will be used to save (new) isc results!!")
else:
    print(f"Results directory {results_dir} already exists and will be overwritten!!")
    print("Press 'y' to overwrite or 'n' to exit")
    while True:
        user_input = input("Enter your choice: ").lower()
        if user_input == 'y':
            break
        elif user_input == 'n':
            print("Exiting...")
            exit()
        else:
            print("Invalid input. Please enter 'y' or 'n'.")


setup = Bunch()
setup.project_dir = project_dir
setup.preproc_model = preproc_model_data
setup.load_data_from = base_path
setup.behav_path = behav_path
setup.condition = conditions,
setup.exclude_sub = exclude_sub
setup.n_sub = n_sub
setup.model_name = model_name
setup.results_dir = results_dir
setup.subjects = list(subjects)
setup.n_boot = n_boot
setup.do_pairwise = do_pairwise
setup.n_perm = n_perm

save_setup = os.path.join(setup.results_dir, "setup_parameters.json")
with open(save_setup, 'w') as fp:
    json.dump(dict(setup), fp, indent=4)

# %%
# get dict will all cond files from all subjects
print("Subjects:", subjects)
sub_check['init'] = subjects

tasks_ordered = sorted(isc_data_df['task'].unique())
# code conditions for all_sugg, (ANA+Hyper), (Neutral_H + Neutral_A)
task_combinations = [tasks_ordered, tasks_ordered[0:2], tasks_ordered[2:4]]
subject_file_dict = {}
for i, cond in enumerate(conditions):
    subject_file_dict[cond] = utils.get_files_for_condition_combination(subjects, task_combinations[i], isc_data_df)

print("Conditions:", conditions)
print("Condition combinations:", task_combinations)
#print(subject_file_dict)

sub_check['cond_files_sub'] = list(subject_file_dict[conditions[0]].keys())

# save subjects paths
for cond in conditions:
    cond_folder = os.path.join(results_dir, cond)
    if not os.path.exists(cond_folder):
        os.makedirs(cond_folder)
    utils.save_data(os.path.join(cond_folder, f'setup_func_path_{cond}.pkl'), subject_file_dict[cond])

# %%

# load difumo atlas
atlas_path = os.path.join(project_dir, 'masks/DiFuMo256/3mm/maps.nii.gz')
atlas_dict_path = os.path.join(project_dir, 'masks/DiFuMo256/labels_256_dictionary.csv')
atlas = nib.load(atlas_path)
atlas_df = pd.read_csv(atlas_dict_path)
atlas_labels = atlas_df['Difumo_names']
atlas_name = 'Difumo256'
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

APM_subjects = ['APM' + sub[4:] for sub in subjects] # make APMXX format instead of subXX
print(APM_subjects)
sub_check['APM_behav'] = APM_subjects

# phenotype= func.load_process_y(behav_path,APM_subjects)
phenotype =pd.read_csv(behav_path, index_col=0)
phenotype.head()
y_interest = ['SHSS_score', 'total_chge_pain_hypAna', 'Abs_diff_automaticity']
# Create group labels based on sHSS and change in pain
group_data = phenotype[y_interest]

#rewrite subjects (APM) to match subjects in the ISC data
pheno_sub = phenotype.index
pheno_sub = ['sub-' + sub[3:] for sub in pheno_sub]
group_data.index = pheno_sub

sub_check['pheno_sub'] = pheno_sub

# ensure that the subjects are in the same order et same number
group_data = group_data.loc[subjects]

# create group labels
group_labels = {}

for var in y_interest:
    median_value = group_data[var].median()
    group_labels[var] = (group_data[var] > median_value).astype(int)  # 1 if above median, 0 otherwise
new_group_col = [col+'median_split' for col in group_labels.keys()]
group_labels_df = pd.DataFrame(group_labels)

group_labels_df.index = group_labels_df.index.astype(str).str.strip()
group_data.index = group_data.index.astype(str).str.strip()
# concatenate the group labels to the data
combined_df = pd.concat([group_data, group_labels_df], axis=1)

#group_data_with_labels.reset_index(inplace=True)
#group_data_with_labels.rename(columns={"index": "Subject"}, inplace=True)
output_csv_path = os.path.join(setup.results_dir, "behav_data_group_labels.csv")
combined_df.to_csv(output_csv_path, index=True)

print("Group Labels Based on Median Split:")
print(group_labels_df.head())
setup.group_labels = group_labels_df

# %%
# extract timseries from atlas

if transform_imgs == True:
    print(f'will fit and trasnform images for {n_sub} subjects')
    transformed_data_per_cond = {}
    fitted_maskers = {}
    transformed_sphere_per_roi = {}

    masker = MultiNiftiMapsMasker(maps_img=atlas, standardize=False, memory='nilearn_cache', verbose=5)
    masker.fit()

	# extract time series for each subject and condition
    for cond in conditions:

        condition_files = subject_file_dict[cond]
        concatenated_subjects = {sub : concat_imgs(sub_files) for sub, sub_files in condition_files.items()}
        print('Imgs shape : ', [img.shape for _, img in  concatenated_subjects.items()])

	    #print(f'fitting images for condition : {cond} with shape {concatenated_subjects[subjects[0]][0].shape}')
        transformed_data_per_cond[cond] = masker.transform(concatenated_subjects.values())
        fitted_maskers[cond] = masker

        # sphere masker
        transformed_sphere_per_roi[cond] = utils.extract_save_sphere(concatenated_subjects, cond, results_dir, roi_coords, sphere_radius=sphere_radius)

	# save transformed data and masker
    for cond, data in transformed_data_per_cond.items():
        cond_folder = os.path.join(results_dir, cond)
        if not os.path.exists(cond_folder):
            os.makedirs(cond_folder)

        save_path = os.path.join(cond_folder, f'transformed_data_{atlas_name}_{cond}_{nsub}sub.pkl')
        utils.save_data(save_path, data)

        masker_path = os.path.join(cond_folder, f'maskers_{atlas_name}_{cond}_{nsub}sub.pkl')
        utils.save_data(masker_path, fitted_maskers[cond])
        print(f'Transformed timseries and maskers saved to {cond_folder}')

    print('====Done with data extraction====')

# %%

# test moddule
# test_results = '/home/dsutterlin/projects/ISC_hypnotic_suggestions/results/imaging/ISC/modulation'
# ts_path = os.path.join(test_results, f'transformed_Difumo256_modulation_{nsub}sub.pkl')
# masker_path = os.path.join(test_results, f'maskers_Difumo256_modulation_{nsub}sub.pkl')
# test_data = utils.load_pickle(ts_path)
# test_masker = utils.load_pickle(masker_path)
# transformed_data_per_cond = {}
# transformed_data_per_cond['modulation'] = test_data
# fitted_maskers = {}
# fitted_maskers['modulation'] = test_masker

else:
# Loading data
    print(f'Loading existing data and fitted maskers')
    transformed_data_per_cond = {}
    fitted_maskers = {}
    transformed_sphere_per_roi = {}

    for cond in conditions:
        load_path = os.path.join(results_dir, cond)
        transformed_file =  f'transformed_data_{atlas_name}_{cond}_{nsub}sub.pkl'
        transformed_data_per_cond[cond] = utils.load_pickle(os.path.join(load_path, transformed_file))
        fitted_mask_file = f'maskers_{atlas_name}_{cond}_{nsub}sub.pkl'
        fitted_maskers[cond] = utils.load_pickle(os.path.join(load_path, fitted_mask_file))

        load_roi = os.path.join(load_path, roi_folder)
        transformed_sphere_per_roi[cond] =utils.load_pickle(os.path.join(load_roi, f"{len(roi_coords)}ROIs_timeseries.pkl")) 

    print(f'Loading existing data and fitted maskers from : {load_path}')

# %%
# Perform ISC

# isc with spheres ROI
isc_results_roi = {}
for cond in conditions:

    print(f'Performing sphere ISC for condition: {cond}')
    roi_ls = [] #stack list of roi timepoints x subjects 
    for roi in roi_coords.keys():
        roi_dict = transformed_sphere_per_roi[cond][roi]
        roi_ts = [] # stack timepoints x subjects
        for sub in roi_dict.keys():
            roi_ts.append(np.array(roi_dict[sub])) # list timepoints x 1 vector
        roi_ls.append(np.array(roi_ts)) # list of roi timepoints x subjects
    roi_data_3d = np.stack(np.squeeze(roi_ls), axis=1).T # timepoints x roi x subjects
    
    isc_results_roi[cond] = utils.isc_1sample(roi_data_3d, pairwise=do_pairwise, n_boot=n_boot, summary_statistic=None)
    
#roi_isc['roi_names'] = list(roi_coords.keys())

for cond, isc_dict in isc_results_roi.items():
    save_path = os.path.join(results_dir, cond, roi_folder, f"isc_{len(roi_coords)}spheres_{cond}_{n_boot}boot_pairWise{do_pairwise}.pkl")
    utils.save_data(save_path, isc_dict)
    print(f"ISC results saved for {cond} at {save_path}")

# ISC with ROI from atlas
isc_results = {}
for cond, data in transformed_data_per_cond.items():
    print(f'Performing ISC for condition: {cond}')
    # Convert list of 2D arrays to 3D array (n_TRs, n_voxels, n_subjects)
    data_3d = np.stack(data, axis=-1)  # Perform ISC
    isc_results[cond] = utils.isc_1sample(data_3d, pairwise=do_pairwise,n_boot = n_boot, summary_statistic=None)

# save results
for cond, isc_dict in isc_results.items():
    save_path = os.path.join(results_dir, cond, f"isc_results_{cond}_{n_boot}boot_pairWise{do_pairwise}.pkl")
    utils.save_data(save_path, isc_dict)
    print(f"ISC results saved for {cond} at {save_path}")



# %%
# Project isc values and p values to brain
from nilearn import plotting
from nilearn.plotting import plot_stat_map
from nilearn.image import new_img_like

# inverse trnasform isc with fitted maskers
for cond, isc_dict in isc_results.items():

    masker = fitted_maskers[cond]
    isc_img = masker.inverse_transform(isc_dict['observed'])
    p_img = masker.inverse_transform(isc_dict['p_values'])

    # save images
    save_boot = os.path.join(results_dir, cond)
    isc_img_name = f"isc_val_{cond}_boot{n_boot}_pariwise{do_pairwise}.nii.gz"
    isc_img.to_filename(os.path.join(save_boot, isc_img_name))
    print(f"ISC image saved to {save_boot}")

    isc_pval_img_name =  f"p_values_{cond}_boot{n_boot}_pairwise{do_pairwise}.nii.gz"
    p_img.to_filename(os.path.join(save_boot, isc_pval_img_name))
    print(f"p-values group image saved to {save_boot}")

    # plot images
    plot_path = os.path.join(results_dir, cond, f"isc_plot_{cond}_boot{n_boot}_pairwise{do_pairwise}.png")
    plotting.plot_stat_map(isc_img, title=f'ISC {cond}', colorbar=True, threshold=0.1, display_mode='z', cut_coords=10, output_file=plot_path)
    print(f"ISC plot saved to {plot_path}")

    plot_path = os.path.join(results_dir, cond, f"p_values_plot_{cond}_boot{n_boot}_pairwise{do_pairwise}.png")
    plotting.plot_stat_map(p_img, title=f'p-values {cond}', colorbar=True, threshold=0.05, display_mode='z', cut_coords=10, output_file=plot_path)
    print(f"p-values plot saved to {plot_path}")


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

isc_sphere_permutation_group_results = {}
for cond, isc_dict in isc_results_roi.items():  # `isc_results` should already contain ISC values
    print(f"Performing permutation ISC for condition: {cond}")

    isc_values = isc_dict['isc']

    for var in group_labels_df.columns:
        group_assignment = group_labels_df[var].values  # Get group labels for this variable

        # Perform ISC permutation test
        observed, p_value, distribution = permutation_isc(
            isc_values,  # This should be the ISC matrix from the main analysis
            group_assignment=group_assignment,
            pairwise=do_pairwise,
            summary_statistic="median",  # Median ISC
            n_permutations=n_perm,
            side="two-sided",
            random_state=42
        )

        if cond not in isc_sphere_permutation_group_results:
            isc_sphere_permutation_group_results[cond] = {}

        isc_sphere_permutation_group_results[cond][var] = {
            "observed": observed,
            "p_value": p_value,
            "distribution": distribution
        }

        print(f"Completed SPHERE permutation ISC for condition: {cond}, variable: {var}")

# Save the permutation results
for cond, cond_results in isc_sphere_permutation_group_results.items():
    save_path = os.path.join(results_dir, cond, roi_folder, f"isc_{n_perm}permutation_results_{cond}_pairwise{do_pairwise}.pkl")
    utils.save_data(save_path, cond_results)
    print(f"sphere Permutation ISC results saved for {cond} at {save_path}")



# ------------
# Perform permutation based on group labels

isc_permutation_group_results = {}
for cond, isc_dict in isc_results.items():  # `isc_results` should already contain ISC values
    print(f"Performing permutation ISC for condition: {cond}")

    isc_values = isc_dict['isc']

    for var in group_labels_df.columns:
        group_assignment = group_labels_df[var].values  # Get group labels for this variable

        # Perform ISC permutation test
        observed, p_value, distribution = permutation_isc(
            isc_values,  # This should be the ISC matrix from the main analysis
            group_assignment=group_assignment,
            pairwise=do_pairwise,
            summary_statistic="median",  # Median ISC
            n_permutations=n_perm,
            side="two-sided",
            random_state=42
        )

        if cond not in isc_permutation_group_results:
            isc_permutation_group_results[cond] = {}

        isc_permutation_group_results[cond][var] = {
            "observed": observed,
            "p_value": p_value,
            "distribution": distribution
        }

        print(f"Completed permutation ISC for condition: {cond}, variable: {var}")

# Save the permutation results
for cond, cond_results in isc_permutation_group_results.items():
    save_path = os.path.join(results_dir, cond, f"isc_{n_perm}permutation_results_{cond}_pairwise{do_pairwise}.pkl")
    utils.save_data(save_path, cond_results)
    print(f"Permutation ISC results saved for {cond} at {save_path}")

save_gen = os.path.join(results_dir, 'check_sub.pkl')

utils.save_data(save_gen, sub_check)


# %%
print('Done with all!!')


# import utils
# p = '/data/rainville/dSutterlin/projects/ISC_hypnotic_suggestions/results/imaging/ISC/model1-22sub/modulation/setup_func_path_modulation.pkl'
# data = utils.load_pickle(p)

# %%
t = r'/data/rainville/dSutterlin/projects/ISC_hypnotic_suggestions/results/imaging/ISC/model2_zcore_sample-22sub/all_sugg/sphere_10mm_3ROIS_isc/isc_3spheres_all_sugg_10000boot_pairWiseFalse.pkl'
t = r'/data/rainville/dSutterlin/projects/ISC_hypnotic_suggestions/results/imaging/ISC/model2_zcore_sample-22sub/all_sugg/isc_results_all_sugg_10000boot_pairWiseFalse.pkl'
isc = utils.load_pickle(t)
