import os
import nibabel as nib
import pandas as pd
import numpy as np
import pickle as pkl
import json
from nilearn.maskers import NiftiSpheresMasker
from nilearn.image import concat_imgs
from brainiak.isc import phaseshift_isc, compute_summary_statistic, isc, bootstrap_isc, phaseshift_isc, permutation_isc


def load_isc_data(base_path, folder_is='task'):
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


def get_files_for_condition_combination(subjects, task_combinations, sub_task_files):
    '''
    Assumes that sub_task_files is a DataFrame with columns 'subject', 'task', and 'file_path'
    '''
    func_file_dict = {}
    for sub in subjects:
        # get file paths for each task
        sub_data = sub_task_files[sub_task_files['subject'] == sub]
        task_data = sub_data.set_index('task').loc[task_combinations]
        if isinstance(task_combinations, str):
            func_file_dict[sub] = task_data['file_path']
        else:
            func_file_dict[sub] = task_data['file_path'].tolist()

    return func_file_dict # sub : : [file1, file2, file3]

def save_data(save_path, data):
    with open(save_path, 'wb') as f:
        pkl.dump(data, f)

def load_pickle(file_path):
    with open(file_path, 'rb') as f:
        data = pkl.load(f)
    return data

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


def isc_1sample(data_3d, pairwise, n_boot=5000, side = 'two-sided', summary_statistic=None):
    
    isc_result = isc(data_3d, pairwise=pairwise, summary_statistic=None)
    print(f"ISC shape: {isc_result.shape}")
    
    observed, ci, p, distribution = bootstrap_isc(
    isc_result,
    pairwise=pairwise,
    summary_statistic="median",
    n_bootstraps=n_boot,
    side = side,
    ci_percentile=95,
    )
    #phase_obs, phase_p, phase_dist = phaseshift_isc(isc_result, pairwise=pairwise, summary_statistic="median",side= side, n_shifts=n_boot)

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