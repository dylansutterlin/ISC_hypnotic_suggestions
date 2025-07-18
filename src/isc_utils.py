import os
import nibabel as nib
import pandas as pd
import numpy as np
import pickle as pkl
import json
from nilearn.maskers import NiftiSpheresMasker
from nilearn.image import concat_imgs
from brainiak.isc import phaseshift_isc, compute_summary_statistic, isc, bootstrap_isc, phaseshift_isc, permutation_isc
from scipy.stats import rankdata, spearmanr
from sklearn.metrics import pairwise_distances

import src.visu_utils


def initialize_setup():
    project_dir = "/data/rainville/dSutterlin/projects/ISC_hypnotic_suggestions"
    # base_path = "/data/rainville/dSutterlin/projects/ISC_hypnotic_suggestions/results/data/test_data_sugg_3sub"
    preproc_model_data = '23subjects_zscore_sample_detrend_FWHM6_low-pass428_10-12-24/suggestion_blocks_concat_4D_23sub'
    base_path = os.path.join(project_dir, 'results/imaging/preproc_data', preproc_model_data)

    #jeni prepoc
    base_path =  r'/data/rainville/Hypnosis_ISC/4D_data/segmented/concat_bks'
    behav_path = os.path.join(project_dir, 'results/behavioral/behavioral_data_cleaned.csv')
    exclude_sub = [] #['sub-02']
    keep_n_subjects = 3

    import sys
    sys.path.append(os.path.join(project_dir, 'QC_py310'))
    import func

    # base_path = '/home/dsutterlin/projects/test_data/suggestion_block_concat_4D_3subj'
    # project_dir = '/home/dsutterlin/projects/ISC_hypnotic_suggestions'
    # behav_path = os.path.join('/home/dsutterlin/projects/ISC_hypnotic_suggestions/results/behavioral', 'behavioral_data_cleaned.csv')

    isc_data_df = utils.load_isc_data(base_path, folder_is='subject') # !! change params if change base_dir
    isc_data_df = isc_data_df.sort_values(by='subject')

    ana_lvl = '/data/rainville/Hypnosis_ISC/4D_data/segmented_Ana_Instr_leveled/concat_Ana_Instr_leveled'
    ana_df = utils.load_isc_data(ana_lvl, folder_is='subject')
    ana_df['task'] = 'Ana'
    ana_df = ana_df.sort_values(by='subject')

    ana_file_map = ana_df.set_index('subject')['file_path']  # Map of subject to Ana file paths
    isc_data_df.loc[isc_data_df['task'] == 'Ana', 'file_path'] = isc_data_df.loc[
        isc_data_df['task'] == 'Ana', 'subject'
    ].map(ana_file_map)


    # exclude subjects
    isc_data_df = isc_data_df[~isc_data_df['subject'].isin(exclude_sub)]
    #isc_data_df = isc_data_df.sort_values(by='subject')
    subjects = isc_data_df['subject'].unique()
    n_sub = len(subjects)

    sub_check = {}
    # Select a subset of subjects
    if keep_n_subjects is not None:
        isc_data_df = isc_data_df[isc_data_df['subject'].isin(isc_data_df['subject'].unique()[:keep_n_subjects])]
        subjects = isc_data_df['subject'].unique()
        n_sub = len(subjects)



    #model_name = f'model3_jeni_preproc-23sub'
    conditions = ['Hyper', 'Ana', 'NHyper', 'NAna'] #['all_sugg', 'modulation', 'neutral']
    transform_imgs = True #False
    do_isc_analyses = False     
    nsub = len(subjects)
    n_sub = len(subjects)
    n_boot = 5000
    do_pairwise = False
    n_perm = 5000
    n_perm_rsa = 10000 #change to 10 000!
    # hyperparams
    atlas_name = 'voxelWise' #'schafer100_2mm' #'voxelWise' #Difumo256' # !!!!!!! 'Difumo256'
    atlas_bunch = Bunch(name=atlas_name)
    model_name = f'model7_noise_QC-{str(n_sub)}sub_{atlas_name}_pairwise{do_pairwise}'

    results_dir = os.path.join(project_dir, f'results/imaging/ISC/{model_name}')
    mkdirs_if_not_exist(results_dir, transform_imgs)

    setup = Bunch()
    setup.project_dir = project_dir
    setup.load_data_from = base_path
    setup.behav_path = behav_path
    setup.conditions = conditions
    setup.exclude_sub = exclude_sub
    setup.n_sub = n_sub
    setup.model_name = model_name
    setup.results_dir = results_dir
    setup.subjects = list(subjects)
    setup.n_boot = n_boot
    setup.do_pairwise = do_pairwise
    setup.n_perm = n_perm
    setup.atlas_name = atlas_name


    return setup


def mkdirs_if_not_exist(results_dir, bool_condition):
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    elif bool_condition == False:
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



def load_isc_data(base_path, folder_is='task', model = 'sugg'):
    """
    Load ISC data from the specified folder structure.

    Parameters
    ----------
    base_path : str
        The base directory containing the subfolders with ISC data.
    folder : str
        if folder is subjecct instead of task, the df is reorganized

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the loaded data with columns:
        'folder', 'subject', 'file_path', and 'data'.
    """

    data_list = []


    # Iterate through folders in the base path
    for folder in os.listdir(base_path):
        folder_path = os.path.join(base_path, folder)
        if os.path.isdir(folder_path):  
    
            for file in os.listdir(folder_path): #file is subject normally
                if model in file : 
                    # print(file)
                    if file.endswith('.nii.gz'):  # Check for .nii.gz files
                        file_path = os.path.join(folder_path, file)
                        file_id = file.split('_')[0]  # Extract subject from filename

                        if folder_is == 'task':
                            # Append metadata and data to the list
                            data_list.append({
                                'subject': file_id,
                                'task': folder,
                                'file_path': file_path,
            
                            })
                        # adjusted 15-01-25 for jeni preproc where subj/ contains all tasks files
                        elif folder_is =='subject':
                            if file_id == 'N': # adjust for neutral cond : N_Ana / N_Hyper
                                file_id = file.split('_')[0] + file.split('_')[1]
                            data_list.append({
                                'subject': folder,
                                'task': file_id,
                                'file_path': file_path,

                            })

    df = pd.DataFrame(data_list)

    return df

# modif 8 avril 25 for single trial model (compare 1st neutral blocks)
def load_single_trial_isc_data(base_path):
    """
    Load single-trial ISC data from folders named by subject (e.g., sub-01, sub-02),
    where each folder contains individual condition volumes like 'ANA1_instrbk_1_53-vol.nii.gz'.

    Parameters
    ----------
    base_path : str
        The base directory containing the subject folders.

    Returns
    -------
    pd.DataFrame
        A DataFrame with columns: 'subject', 'condition', 'file_path'.
    """
    data_list = []

    for subject_folder in os.listdir(base_path):
        subject_path = os.path.join(base_path, subject_folder)
        if not os.path.isdir(subject_path):
            continue

        for file in os.listdir(subject_path):   
            if file.endswith(".nii.gz"):
                file_path = os.path.join(subject_path, file)
                core_name = file.replace("-vol.nii.gz", "")
                parts = core_name.split('_')

                if core_name.startswith("N_") and len(parts) >= 5:
                    condition_name = '_'.join(parts[:4])  # e.g. 'N_HYPER3_instrbk_1'
                elif len(parts) >= 4:
                    condition_name = '_'.join(parts[:3])  # e.g. 'ANA1_instrbk_1'

                else:
                    continue

                data_list.append({
                    'subject': subject_folder,
                    'task': condition_name,
                    'file_path': file_path
                })

    df = pd.DataFrame(data_list)
    return df

#----
def load_confounds(base_path):
    sub_conf = {}
    for folder in os.listdir(base_path):
        folder_path = os.path.join(base_path, folder)
        if os.path.isdir(folder_path):  
    
            for file in [f for f in os.listdir(folder_path) if f.startswith('mvmnt_reg')]:
                #print(folder, file)
                p = os.path.join(base_path, folder, file)
                reg_dct = load_pickle(p)
                sub_conf[folder] = reg_dct

    return sub_conf


def filter_and_rename_confounds(dct_conf_unsorted, subjects, model_is):
    """
    Filters and renames confound keys for each subject based on `model_is`.
    If the key starts with 'N_', it concatenates the first two parts without an underscore.

    Parameters
    ----------
    dct_conf_unsorted : dict
        Dictionary of confounds per subject.
    subjects : list
        List of subject IDs.
    model_is : str
        Substring to filter relevant confound keys.

    Returns
    -------
    list of dict
        List of dictionaries with filtered and renamed confound keys per subject.
    """
    ls_dct_conf = []

    for sub in subjects:
        sub_dct = dct_conf_unsorted[sub].copy()
        filtered_dct = {}

        for cond_key in list(sub_dct.keys()):
            if model_is in cond_key:
                if 'instrbk' in cond_key:
                    novel_key = cond_key
                else:
                    parts = cond_key.split('_')
                    if parts[0] == 'N':
                        novel_key = parts[0] + parts[1]  # Merge "N_" into "NANA", "NHYPER", etc.
                    else:
                        novel_key = parts[0]
                filtered_dct[novel_key] = sub_dct[cond_key]

        ls_dct_conf.append(filtered_dct)

    return ls_dct_conf


def get_files_for_condition_combination(subjects, task_combinations, sub_task_files):
    func_file_dict = {}

    for sub in subjects:
        sub_data = sub_task_files[sub_task_files['subject'] == sub]
        task_data = sub_data.set_index('task')
        

        if isinstance(task_combinations, str):
            if task_combinations in task_data.index:
                func_file_dict[sub] = task_data.loc[task_combinations, 'file_path']
            else:
                print(f"{task_combinations} missing for {sub}")
                print(task_data)
        else:
            available_tasks = [t for t in task_combinations if t in task_data.index]
            if available_tasks:
                func_file_dict[sub] = task_data.loc[available_tasks, 'file_path'].tolist()
            else:
                print(f"[] No tasks from {task_combinations} found for {sub}")

    return func_file_dict
def save_data(save_path, data):
    with open(save_path, 'wb') as f:
        pkl.dump(data, f)

def load_pickle(file_path):
    with open(file_path, 'rb') as f:
        data = pkl.load(f)
    return data

def save_json(file_path, data):
    with open(file_path, "w") as f:
        json.dump(data, f, indent=4)

def load_json(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

def load_process_behav(phenotype, y_interest, setup, sub_check):
    '''
        Load and process behavioral data for ISC analysis.
        Makes groups based on median split of the y_interest variables.

        return : 
        - combined_df : DataFrame with behavioral data and group labels
        - sub_check * is just a test to keep track of subjects update*
        '''
    subjects = setup.subjects
    # Create group labels based on sHSS and change in pain
    proproc_data = phenotype[y_interest]

    #rewrite subjects (APM) to match subjects in the ISC data
    pheno_index = phenotype.index
    pheno_sub = ['sub-' + sub[3:] for sub in pheno_index]
    proproc_data.index = pheno_sub

    sub_check['pheno_sub'] = pheno_sub

    # ensure that the subjects are in the same order et same number
    proproc_data = proproc_data.loc[subjects]
    group_labels = {}

    for var in y_interest:
        median_value = proproc_data[var].median()
        group_labels[var] = (proproc_data[var] > median_value).astype(int)  # 1 if above median, 0 otherwise
    new_group_col = [col+'_median_grp' for col in group_labels.keys()]
    group_labels_df = pd.DataFrame(group_labels)
    group_labels_df.columns = new_group_col
    group_labels_df.index = group_labels_df.index.astype(str).str.strip()
    proproc_data.index = proproc_data.index.astype(str).str.strip()
    # concatenate the group labels to the data
    combined_df = pd.concat([proproc_data, group_labels_df], axis=1)

    #group_data_with_labels.reset_index(inplace=True)
    #group_data_with_labels.rename(columns={"index": "Subject"}, inplace=True)
    output_csv_path = os.path.join(setup.results_dir, "behav_data_group_labels.csv")
    combined_df.to_csv(output_csv_path, index=True)

    print("Group Labels Based on Median Split:")
    print(group_labels_df.head())

    return combined_df, group_labels_df, sub_check


# Function to extract timeseries with NiftiSpheresMasker
def extract_save_sphere(concatenated_imgs, condition, results_dir, roi_coords, sphere_radius):

    roi_timeseries = {}
    n_rois = len(list(roi_coords.keys()))
    print(f"---Processing spheres for condition: {condition}---")

    condition_path = os.path.join(results_dir, condition)
    sphere_results_dir = os.path.join(condition_path, f"sphere_{sphere_radius}mm_{n_rois}ROIS_isc")
    os.makedirs(sphere_results_dir, exist_ok=True)

    # func_path = os.path.join(condition_path, f"setup_func_path_{condition}.pkl")
    # subject_file_dict = load_pickle(func_path)

    for roi_name, roi_coord in roi_coords.items():
        print(f"Processing ROI: {roi_name} at {roi_coord}")
        
        masker = NiftiSpheresMasker(seeds=[roi_coord], radius=sphere_radius, standardize=False)
        roi_timeseries[roi_name] = {}

        # Process each subject
        for subject, concatenated_img in concatenated_imgs.items():
            timeseries = masker.fit_transform(concatenated_img)
            roi_timeseries[roi_name][subject] = timeseries
        
    # Save timeseries and masker
    timeseries_path = os.path.join(sphere_results_dir, f"{n_rois}ROIs_timeseries.pkl")
    # masker_path = os.path.join(sphere_results_dir, f"{roi_name}_masker.pkl")
    save_data(timeseries_path, roi_timeseries)
    # save_data(masker_path, masker)
    print(f"Saved sphere timeseries of {list(roi_coords.keys())} as {timeseries_path}")

    return roi_timeseries



def identify_zero_variance_subjects(data_per_cond, subjects):
    """
    Identifies subjects who have zero variance in any condition and prints which conditions are affected.

    Parameters
    ----------
    data_per_cond : dict
        Dictionary where keys are condition names and values are arrays of shape (TRs, voxels, subjects).
    subjects : list
        List of subject IDs.

    Returns
    -------
    subjects_to_remove : list
        List of subject IDs with zero variance in any condition.
    """
    n_subjects = len(subjects)
    zero_variance_subjects = np.zeros(n_subjects, dtype=bool)
    subject_condition_log = {sub: [] for sub in subjects}

    for cond, data in data_per_cond.items():
        print(f"Checking zero variance for condition: {cond}")

        if isinstance(data, list):
            data_3d = np.stack(data, axis=-1)
        elif isinstance(data, np.ndarray):
            data_3d = data
        else:
            raise ValueError("Unexpected data format.")

        assert data_3d.shape[-1] == n_subjects, f"Data shape {data_3d.shape} does not match subject count {n_subjects}"

        # Identify which subjects have zero variance across all voxels
        subject_has_zero_variance = np.any(np.var(data_3d, axis=0) == 0, axis=0)

        for idx, has_zero in enumerate(subject_has_zero_variance):
            if has_zero:
                subject_condition_log[subjects[idx]].append(cond)

        zero_variance_subjects |= subject_has_zero_variance  # Mark for removal

    # Print condition-specific warnings
    for sub in subjects:
        if subject_condition_log[sub]:
            print(f"{sub} has zero variance in: {subject_condition_log[sub]}")

    subjects_to_remove = list(np.array(subjects)[zero_variance_subjects])
    print(f"\nSubjects to remove due to zero-variance in any condition: {subjects_to_remove}")
    
    return subjects_to_remove



def remove_subjects(data_per_cond,fitted_maskers, subjects, subjects_to_remove):
    """
    Removes identified subjects across all conditions.

    Parameters:
    - transformed_data_per_cond (dict): Condition-wise fMRI data (TRs, voxels, subjects).
    - subjects (list): List of subjects.
    - subjects_to_remove (list): List of subjects to be removed.

    Returns:
    - updated_data_per_cond (dict): Cleaned fMRI data without removed subjects.
    - updated_subjects (list): Updated list of subjects.
    """
    n_sub = len(subjects)
    subjects_array = np.array(subjects)
    keep_indices = ~np.isin(subjects_array, subjects_to_remove)
    print(f"Keeping subjects: {subjects_array[keep_indices]}")
    print('keeping indices :', keep_indices)

    updated_data_per_cond = {}
    for cond, data in data_per_cond.items():

        if isinstance(data, list):
            data_3d = np.stack(data, axis=-1)  # Convert list of 2D arrays to 3D (TRs, voxels, subjects)
        elif isinstance(data, np.ndarray):
            data_3d = data
        else:
            raise TypeError(f"Unexpected data type for condition {cond}: {type(data)}")
        assert data_3d.shape[-1] == n_sub, f"Data shape {data_3d.shape} does not match expected number of subjects {n_sub}"
        
        updated_data_per_cond[cond] = data_3d[:, :, keep_indices]

        assert updated_data_per_cond[cond].shape[-1] == sum(keep_indices), \
            f"Post-removal shape {updated_data_per_cond[cond].shape} does not match expected {sum(keep_indices)} subjects"

    updated_maskers = {
        cond: [masker for i, masker in enumerate(maskers) if keep_indices[i]]
        for cond, maskers in fitted_maskers.items()
    }
    updated_subjects = subjects_array[keep_indices].tolist()

    print(f"Removed: {subjects_to_remove}")

    return updated_data_per_cond, updated_maskers, updated_subjects


from matplotlib import pyplot as plt
def plot_flat_rois(transformed_data_per_cond, subjects, roi_names, condition_names, save_dir):

    for cond in condition_names:
        cond_dir = os.path.join(save_dir, cond)
        os.makedirs(cond_dir, exist_ok=True)

        for i, sub in enumerate(subjects):
            ts = transformed_data_per_cond[cond][i]  # (TRs x ROIs)
            variances = np.var(ts, axis=0)
            flat_idx = np.where(variances == 0)[0]

            if len(flat_idx) == 0:
                continue  # nothing to plot

            print(f"{sub} – {cond}: {len(flat_idx)} flat ROI(s): {[roi_names[idx] for idx in flat_idx if idx < len(roi_names)]}")

            plt.figure(figsize=(10, 4))
            for idx in flat_idx:
                label = roi_names[idx] if idx < len(roi_names) else f"ROI-{idx}"
                plt.plot(ts[:, idx], label=label)

            plt.title(f"Zero-variance ROIs – {sub} | {cond}")
            plt.xlabel("Time (TRs)")
            plt.ylabel("Signal")
            plt.legend(fontsize=8, loc='upper right')
            plt.tight_layout()
            plt.savefig(os.path.join(cond_dir, f"{sub}_{cond}_flat_rois.png"))
            plt.close()


# def assert_zero_variance_sub(data_3d, subjects):
#     """
#     Checks for zero-variance voxels in fMRI time series data, reports if found, 
#     and replaces zero-value voxels with NaNs.

#     Parameters:
#     - data_3d: ndarray of shape (TRs, ROIs, subjects)
#     """
#     # data_copy = data_3d.copy()
#     remove_sub_zero_variance =[]
#     for i, sub in enumerate(subjects):
#         time_series = data_3d[:, :, i]
#         zero_variance_voxels = np.var(time_series, axis=0) == 0

#         if np.any(zero_variance_voxels):  # Only print if zero-variance voxels exist
#             zero_voxels = np.sum(time_series == 0, axis=0) == time_series.shape[0]
#             print(f"[Warning!!] Zero variance detected for {sub} in {zero_variance_voxels.sum()} voxels/parcel")
#             print(f"Number of oxels with all zero values: {zero_voxels.sum()}")
#             remove_sub_zero_variance.append(True)
#         else:
#             remove_sub_zero_variance.append(False)
#     return np.array(remove_sub_zero_variance)



def isc_1sample(data_3d, pairwise, n_boot=5000, side = 'two-sided', summary_statistic=None):
    
    isc_result = isc(data_3d, pairwise=pairwise, summary_statistic=None)
    print(f"ISC shape: {isc_result.shape}")
    print(f'--bootstrap {n_boot}--')
    observed, ci, p, distribution = bootstrap_isc(
    isc_result,
    pairwise=pairwise,
    summary_statistic="median",
    n_bootstraps=n_boot,
    side = side,
    ci_percentile=95,
    )
    #phase_obs, phase_p, phase_dist = phaseshift_isc(isc_result, pairwise=pairwise, summary_statistic="median",side= side, n_shifts=n_boot)
    print('done--')
    median_isc = compute_summary_statistic(isc_result, 'median', axis=0) # per ROI : 1, n_voxels
    total_median_isc = compute_summary_statistic(isc_result, 'median', axis=None)
    print(f'Median ISC : {total_median_isc}')

    isc_results = {
    "isc": isc_result,
    "observed": observed,
    "confidence_intervals": ci,
    "p_values": p,
    "distribution": distribution,
    }
    # isc_results = {
    # "isc": isc_result,
    # "observed": observed,
    # "confidence_intervals": ci,
    # "p_values": p,
    # "distribution": distribution,
    # "phase_obs": phase_obs,
    # "phase_p": phase_p,
    # "phase_dist": phase_dist,
    # "median_isc": median_isc
    # }

    return isc_results

from brainiak.isc import isfc

def isfc_1sample(data_3d, pairwise, n_boot=5000, side = 'two-sided', summary_statistic=None):
    
    # isc_result = isfc(data_3d, pairwise=pairwise, summary_statistic=None)
    isfc_result, isc_diag = isfc(data_3d, pairwise=True, vectorize_isfcs=True)
    print(f"ISFC shape: {isfc_result.shape}")
    print(f'--bootstrap {n_boot}--')

    observed, ci, p, distribution = bootstrap_isc(
    isfc_result,
    pairwise=pairwise,
    summary_statistic="median",
    n_bootstraps=n_boot,
    side = side,
    ci_percentile=95,
    )

    total_median_isc = compute_summary_statistic(isfc_result, 'median', axis=None)
    print(f'Median ISC : {total_median_isc}')

    isfc_dict = {
    "isfc": isfc_result,
    "observed": observed,
    "confidence_intervals": ci,
    "p_values": p,
    "distribution": distribution,
    }

    return isfc_dict


def trim_TRs(hyper_data_3d, ana_data_3d):
    """
    Adjust Ana and Hyper conditions to match TRs for direct contrast.

    Parameters
    ----------
    ana_data : np.ndarray
        Ana condition data (TRs x ROIs x subjects).
    hyper_data : np.ndarray
        Hyper condition data (TRs x ROIs x subjects).

    Returns
    -------
    adjusted_ana : np.ndarray
        Adjusted Ana data with the first and last planes removed.
    adjusted_hyper : np.ndarray
        Adjusted Hyper data with the last plane repeated.
    """
    # Remove the first and last planes from Ana
    adjusted_ana = ana_data_3d[1:-1, :, :]
    
    # Repeat the last plane in Hyper
    last_tr = hyper_data_3d[-1:, :, :]  # Select the last plane
    adjusted_hyper = np.concatenate([hyper_data_3d, last_tr], axis=0)  # Append repeated plane

    return adjusted_hyper, adjusted_ana

def group_permutation(isc_values, group_ids, n_perm, do_pairwise, side = 'two-sided', summary_statistic='median'):


    # Perform ISC permutation test
    observed, p_value, distribution = permutation_isc(
        isc_values,  # This should be the ISC matrix from the main analysis
        group_assignment=group_ids,
        pairwise=do_pairwise,
        summary_statistic=summary_statistic,  # Median ISC
        n_permutations=n_perm,
        side=side,
        random_state=42
    )

    perm_results = {
        "grouped_isc": isc_values,
        "observed_diff": observed,
        "p_value": p_value,
        "distribution": distribution
    }

    return perm_results


from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import pairwise_distances, cosine_similarity

def compute_behav_similarity(behavior, 
                             metric='euclidean', 
                             standardize=True, 
                             vectorize=False):
    """
    Compute a subject-by-subject similarity matrix from 1D or multi-D behavioral data.

    Parameters
    ----------
    behavior : array-like, shape (n_subjects,) or (n_subjects, n_features)
        Raw behavioral scores per subject (either a single score or multiple condition scores).
    metric : {'euclidean', 'annak', 'cosine', 'rbf'}, default='euclidean'
        Similarity metric to use:
          - 'euclidean':  1 - (pairwise Euclidean distance / max distance)
          - 'annak':      AnnaK mean-based (only for 1D input)
          - 'cosine':     cosine similarity of the vectors

    standardize : bool, default=True
        If True and input is 2D, z‑score each feature column before computing similarity.
    vectorize : bool, default=False
        If True, return only the upper‑triangle (k=1) of the similarity matrix as a flat array.

    Returns
    -------
    sim_matrix : ndarray, shape (n_subjects, n_subjects) or (n_pairs,)
        Full similarity matrix, or the vectorized upper triangle if `vectorize=True`.
    """
    
    X = np.asarray(behavior, dtype=float)
    # ensure shape (n_subjects, n_features)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    n_subj = X.shape[0]

    # standardize each column if multivariate
    if standardize and X.shape[1] > 1:
        X = StandardScaler().fit_transform(X)

    # compute similarity
    if metric == 'euclidean':
        D = pairwise_distances(X, metric='euclidean')
        sim = 1 - (D / D.max())

    elif metric == 'annak':
        if X.shape[1] > 1:
            raise ValueError("AnnaK only supported for univariate input.")
        values = X.ravel()
        max_val = values.max()
        sim = np.zeros((n_subj, n_subj))
        for i in range(n_subj):
            for j in range(n_subj):
                sim[i, j] = (values[i] + values[j]) / (2 * max_val)

    elif metric == 'cosine':
        sim = cosine_similarity(X)

    else:
        raise ValueError(f"Unsupported metric '{metric}'. "
                         "Choose from 'euclidean', 'annak', 'cosine'")

    if vectorize:
        iu = np.triu_indices(n_subj, k=1)
        return sim[iu]

    return sim


def compute_behav_similarity_LOO(behavior, metric='euclidean'):
    """
    Compute leave-one-out (LOO) behavioral similarity for each subject,
    comparing each subject to the rest of the group.

    Parameters
    ----------
    behavior : array-like, shape (n_subjects,)
        Behavioral scores for each subject.
    metric : str, optional
        Similarity metric ('euclidean' or 'annak'), default is 'euclidean'.

    Returns
    -------
    sim_loo : np.ndarray, shape (n_subjects,)
        LOO similarity value for each subject.
    """
    behavior = behavior.reshape(-1, 1)
    n_subs = len(behavior)
    sim_loo = np.zeros(n_subs)

    if metric == 'euclidean':
        for i in range(n_subs):
            other_subs = np.delete(behavior, i)
            mean_other = np.mean(other_subs)
            dist = np.abs(behavior[i] - mean_other)
            max_dist = np.max(np.abs(behavior - mean_other))
            sim_loo[i] = 1 - (dist / max_dist) if max_dist != 0 else 1

    elif metric == 'annak':
        max_behavior = np.max(behavior)
        for i in range(n_subs):
            other_subs = np.delete(behavior, i)
            mean_other = np.mean(other_subs)
            sim_loo[i] = np.mean([behavior[i], mean_other]) / max_behavior if max_behavior != 0 else 1

    return sim_loo


from scipy.stats import spearmanr
from sklearn.utils import check_random_state
from joblib import Parallel, delayed

#===========================================
# Permutation tests implemented frmo nlTools 
# https://nltools.org/api.html#nltools.stats.matrix_permutation
# Permutation from chen 2016

# Helper function for correlation computation
def _permute_func(data1, vec2, metric, how, include_diag=False, random_state=None):
    random_state = check_random_state(random_state)
    permuted_ix = random_state.permutation(len(data1))

    if data1.ndim == 1:
        permuted_vec = data1[permuted_ix]
    else:
        permuted_matrix = data1[np.ix_(permuted_ix, permuted_ix)]
        if how == "upper":
            permuted_vec = permuted_matrix[np.triu_indices(permuted_matrix.shape[0], k=1)]
        elif how == "lower":
            permuted_vec = permuted_matrix[np.tril_indices(permuted_matrix.shape[0], k=-1)]
        elif how == "full":
            if include_diag:
                permuted_vec = permuted_matrix.ravel()
            else:
                permuted_vec = np.concatenate([
                    permuted_matrix[np.triu_indices(permuted_matrix.shape[0], k=1)],
                    permuted_matrix[np.tril_indices(permuted_matrix.shape[0], k=-1)]
                ])

    if metric == "spearman":
        r, _ = spearmanr(permuted_vec, vec2)
    else:
        raise ValueError("Only 'spearman' is supported.")

    return r


# Main Mantel test function [! 15 jan. 25 DSG changed data2 to vec2
# for inputing vectorized output isc values]
def matrix_permutation(
    data1,
    vec2,
    n_permute=10000,
    metric="spearman",
    how="upper",
    include_diag=False,
    tail=2,
    n_jobs=-1,
    return_perms=False,
    random_state=None,
):
    """
    Perform a permutation-based Mantel test. Shuffles data1 !

    Parameters:
    - data1: np.array, square similarity matrix
    - vec2: np.array, square similarity matrix
    - n_permute: int, number of permutations
    - metric: str, correlation metric ('spearman')
    - how: str, which part of the matrix to use ('upper', 'lower', 'full')
    - include_diag: bool, include diagonal elements for 'full'
    - tail: int, 1 for one-tailed or 2 for two-tailed test
    - n_jobs: int, number of CPUs to use (-1 for all CPUs)
    - return_perms: bool, return permutation distribution
    - random_state: int or RandomState, random seed or state

    Returns:
    - stats: dict, contains observed correlation, p-value, and optionally permutation distribution
    """
    random_state = check_random_state(random_state)

    if data1.ndim == 1: # can already pass a vector
        vec1 = data1
    else:
        # Validate and extract parts of the matrices
        if how == "upper":
            vec1 = data1[np.triu_indices(data1.shape[0], k=1)]
            #vec2 = data2[np.triu_indices(data2.shape[0], k=1)]
        elif how == "lower":
            vec1 = data1[np.tril_indices(data1.shape[0], k=-1)]
            #vec2 = data2[np.tril_indices(data2.shape[0], k=-1)]
        elif how == "full":
            if include_diag:
                vec1 = data1.ravel()
                #vec2 = data2.ravel()
            else:
                vec1 = np.concatenate([
                    data1[np.triu_indices(data1.shape[0], k=1)],
                    data1[np.tril_indices(data1.shape[0], k=-1)],
                ])
                #vec2 = np.concatenate([
                    # data2[np.triu_indices(data2.shape[0], k=1)],
                    # data2[np.tril_indices(data2.shape[0], k=-1)],
                # ])
    if len(vec1) != len(vec2):
        raise ValueError("Similarity vectors must be the same length.")

    # Compute observed correlation
    observed_r, _ = spearmanr(vec1, vec2)

    # Run permutations in parallel
    seeds = random_state.randint(0, 2**32 - 1, size=n_permute)
    permuted_r = Parallel(n_jobs=n_jobs)(
        delayed(_permute_func)(
            vec1, vec2, metric=metric, how=how, include_diag=include_diag, random_state=seeds[i]
        ) for i in range(n_permute)
    )

    #  p-value
    permuted_r = np.array(permuted_r)
    if tail == 2:
        p_value = np.mean(np.abs(permuted_r) >= np.abs(observed_r))
    elif tail == 1:
        p_value = np.mean(permuted_r >= observed_r)
    else:
        raise ValueError("tail must be 1 or 2.")

    stats = {"correlation": observed_r, "p": p_value}
    if return_perms:
        stats["perm_dist"] = permuted_r

    return observed_r, p_value, permuted_r


def fdr(p, q=0.05):
    """Determine FDR threshold given a p value array and desired false
    discovery rate q. Written by Tal Yarkoni

    Args:
        p: (np.array) vector of p-values
        q: (float) false discovery rate level

    Returns:
        fdr_p: (float) p-value threshold based on independence or positive
                dependence

    """

    if not isinstance(p, np.ndarray):
        raise ValueError("Make sure vector of p-values is a numpy array")
    if any(p < 0) or any(p > 1):
        raise ValueError("array contains p-values that are outside the range 0-1")

    if np.any(p > 1) or np.any(p < 0):
        raise ValueError("Does not include valid p-values.")

    s = np.sort(p)
    nvox = p.shape[0]
    null = np.array(range(1, nvox + 1), dtype="float") * q / nvox
    below = np.where(s <= null)[0]
    return s[max(below)] if len(below) else -1

def bonferroni(p, alpha=0.05):
    """Determine Bonferonni corrected p-value threshold.

    Args:
        p: (np.array) vector of p-values
        alpha: (float) desired alpha level

    Returns:
        bonf_p: (float) p-value threshold based on Bonferonni correction

    """

    if not isinstance(p, np.ndarray):
        raise ValueError("Make sure vector of p-values is a numpy array")
    if any(p < 0) or any(p > 1):
        raise ValueError("array contains p-values that are outside the range 0-1")

    return alpha / p.shape[0]

# from nltools
def expand_mask(mask, custom_mask=None):
    """expand a mask with multiple integers into separate binary masks

    Args:
        mask: nibabel or Brain_Data instance
        custom_mask: nibabel instance or string to file path; optional

    Returns:
        out: Brain_Data instance of multiple binary masks

    """

    from nltools.data import Brain_Data

    if isinstance(mask, nib.Nifti1Image):
        mask = Brain_Data(mask, mask=custom_mask)
    if not isinstance(mask, Brain_Data):
        raise ValueError("Make sure mask is a nibabel or Brain_Data instance.")
    mask.data = np.round(mask.data).astype(int)
    tmp = []
    for i in np.nonzero(np.unique(mask.data))[0]:
        tmp.append((mask.data == i) * 1)
    out = mask.empty()
    out.data = np.array(tmp)
    return out

# from nltools
def roi_to_brain(data, mask_x):
    """This function will create convert an expanded binary mask of ROIs
    (see expand_mask) based on a vector of of values. The dataframe of values
    must correspond to ROI numbers.

    This is useful for populating a parcellation scheme by a vector of Values

    Args:
        data: Pandas series, dataframe, list, np.array of ROI by observation
        mask_x: an expanded binary mask

    Returns:
        out: (Brain_Data) Brain_Data instance where each ROI is now populated
             with a value
    """
    from nltools.data import Brain_Data

    if not isinstance(data, (pd.Series, pd.DataFrame)):
        if isinstance(data, list):
            data = pd.Series(data)
        elif isinstance(data, np.ndarray):
            if len(data.shape) == 1:
                data = pd.Series(data)
            elif len(data.shape) == 2:
                data = pd.DataFrame(data)
                if data.shape[0] != len(mask_x):
                    if data.shape[1] == len(mask_x):
                        data = data.T
                    else:
                        raise ValueError(
                            "Data must have the same number of rows as rois in mask"
                        )
            else:
                raise NotImplementedError

        else:
            raise ValueError("Data must be a pandas series or data frame.")

    if len(mask_x) != data.shape[0]:
        raise ValueError("Data must have the same number of rows as mask has ROIs.")

    if isinstance(data, pd.Series):
        out = mask_x[0].copy()
        out.data = np.zeros(out.data.shape)
        for roi in range(len(mask_x)):
            out.data[np.where(mask_x.data[roi, :])] = data[roi]
        return out
    else:
        out = mask_x.copy()
        out.data = np.ones((data.shape[1], out.data.shape[1]))
        for roi in range(len(mask_x)):
            roi_data = np.reshape(data.iloc[roi, :].values, (-1, 1))
            out.data[:, mask_x[roi].data == 1] = np.repeat(
                roi_data.T, np.sum(mask_x[roi].data == 1), axis=0
            ).T
        return out

def r_to_z(r_values):

    # Ensure values are within the valid range (-1, 1)
    r_values = np.clip(r_values, -0.9999, 0.9999)
    z_values = np.arctanh(r_values)
    return z_values


def check_confounfs_isc(confounds_ls, subjects, conditions, show=True):

    isc_conf = {}

    for cond in conditions:
        arr_ls = []
        for sub in range(len(subjects)):
            sub_arr = confounds_ls[sub][cond]
            arr_ls.append(sub_arr)
        arrays = np.stack(arr_ls, axis=-1)[:,0:6,:] 
        
        print(f'Confounds ISC in cond {cond} wih {arrays.shape}')

        isc_conf[cond] = isc_1sample(arrays, pairwise=True, n_boot=5000, summary_statistic=None)
        
        median = np.median(isc_conf[cond]['isc'], axis=0)
        p_val = isc_conf[cond]['p_values']
        conf_names = ['reg'+ str(i) for i in range(1,7)]
        visu_utils.plot_isc_median_with_significance(median,p_val,conf_names, show=show)

    return isc_conf


from scipy.spatial.distance import squareform


def vector_to_isc_matrix(vec, diag=1):
    mat = squareform(vec)           # reconstruct symmetric matrix, zeros on diag
    np.fill_diagonal(mat, diag)     # set diagonal to desired value
    return mat
