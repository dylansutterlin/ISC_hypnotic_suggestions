import os
import nibabel as nib
import pandas as pd
import pickle as pkl

def load_isc_data(base_path):
    """
    Load ISC data from the specified folder structure.

    Parameters
    ----------
    base_path : str
        The base directory containing the subfolders with ISC data.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the loaded data with columns:
        'folder', 'subject', 'file_path', and 'data'.
    """
    # Initialize a list to store metadata and data
    data_list = []


    # Iterate through folders in the base path
    for folder in os.listdir(base_path):
        folder_path = os.path.join(base_path, folder)
        if os.path.isdir(folder_path):  # Ensure it's a folder
            # Iterate through files in each folder
            for file in os.listdir(folder_path):
                if file.endswith('.nii.gz'):  # Check for .nii.gz files
                    file_path = os.path.join(folder_path, file)
                    subject = file.split('_')[0]  # Extract subject from filename
                    
                    # Append metadata and data to the list
                    data_list.append({
                        'subject': subject,
                        'task': folder,
                        'file_path': file_path,
    
                    })
    
    # Convert the list to a pandas DataFrame
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
        func_file_dict[sub] = task_data['file_path'].tolist()

    return func_file_dict # sub : : [file1, file2, file3]

def save_data(save_path, data):
    with open(save_path, 'wb') as f:
        pkl.dump(data, f)

def load_pickle(file_path):
    with open(file_path, 'rb') as f:
        data = pkl.load(f)
    return data