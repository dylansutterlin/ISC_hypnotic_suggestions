import os
import glob as glob
import pandas as pd
import numpy as np
import json
import nibabel as nib
import matplotlib.cm as cm
from scipy import stats
from sklearn.utils import Bunch
from nilearn import plotting
from nilearn.image import new_img_like, load_img
from nilearn import datasets, image
from matplotlib import cm
from nilearn.plotting import plot_glass_brain
from nilearn.plotting import find_probabilistic_atlas_cut_coords
from sklearn.covariance import GraphicalLassoCV
from sklearn.preprocessing import MinMaxScaler
from nilearn.connectome import GroupSparseCovarianceCV
import scipy 
import os
import matplotlib.pyplot as plt
import seaborn as sns
from nilearn.maskers import NiftiLabelsMasker
import pickle as pkl

def save_pickle(save_path, data):
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

def load_process_y(xlsx_path, subjects):
    '''Load behavioral variables from xlsx file and process them for further analysis
    '''
    # dependant variables
    rawY = pd.read_excel(xlsx_path, sheet_name=0, index_col=1, header=2).iloc[
        2:, [4,5,6,7,8,9,10,11,12,17, 18, 19, 38, 48, 65, 67]
    ]
    columns_of_interest = [
        "SHSS_score",
        "mean_VAS_Nana_int", 
        "mean_VAS_ana_int", 
        "mean_VAS_Nhyper_int", 
        "mean_VAS_hyper_int", 
        "mean_VAS_Nana_UnP", 
        "mean_VAS_ana_UnP", 
        "mean_VAS_Nhyper_UnP", 
        "mean_VAS_hyper_UnP",
        "raw_change_ANA",
        "raw_change_HYPER",
        "total_chge_pain_hypAna",
        "Chge_hypnotic_depth",
        "Mental_relax_absChange",
        "Automaticity_post_ind",
        "Abs_diff_automaticity",
    ]
    rawY.columns = columns_of_interest
    cleanY = rawY.iloc[:-6, :]  # remove sub04, sub34 and last 6 rows
    cutY = cleanY.drop(["APM04*", "APM34*"])

    filledY = cutY.fillna(cutY.astype(float).mean()).astype(float)
    filledY["SHSS_groups"] = pd.cut(
        filledY["SHSS_score"], bins=[0, 4, 8, 12], labels=["0", "1", "2"]
    )  # encode 3 groups for SHSS scores

    # bin_edges = np.linspace(min(data_column), max(data_column), 4) # 4 bins
    filledY["auto_groups"] = pd.cut(
        filledY["Abs_diff_automaticity"],
        bins=np.linspace(
            min(filledY["Abs_diff_automaticity"]) - 1e-10,
            max(filledY["Abs_diff_automaticity"]) + 1e-10,
            4,
        ),
        labels=["0", "1", "2"],
    )

    # rename 'APM_XX_HH' to 'APMXX' format, for compatibility with Y.rows
    #subjects_rewritten = ["APM" + s.split("_")[1] for s in subjects]
    subjects_rewritten = subjects
    # reorder to match subjects order
    Y = pd.DataFrame(columns=filledY.columns)
    for namei in subjects_rewritten: 
        row = filledY.loc[namei]
        Y.loc[namei] = row

    return Y


def extract_timeseries_and_generate_individual_reports(subjects, func_list, atlas_masker_to_fit, masker_name, save_path, confounds = None,confounf_files= None, condition_name="Analgesia", do_heatmap = True):
    """
    Fit each subject 4D image and saves individual reports and ROI heatmaps for each subject.
    Saves masker params in the 

    Parameters
    ----------
    subjects : list of str
        List of subject identifiers. This should correspond to the order of functional files in `func_list`.
    func_list : list of str
        List of paths to 4D functional files for each subject.
    atlas_masker_to_fit : NiftiMasker, NiftiLabelsMasker or NiftiMapsMasker
        Initialized Nifti masker object to fit. This can be a NiftiLabelsMasker or NiftiMapsMasker 
        depending on the type of atlas used.
        If niftiMasker, heatmap is not very informative, suggest do_heatmap=False.
    masker_name : str
        Name of the masker or atlas. This name will be included in the report and heatmap filenames.
    project_dir : str
        Path to the project directory where the results will be saved.
    condition_name : str, optional
        Name of the experimental condition (default is "Analgesia"). This will be used in naming the
        report and heatmap files.
    do_heatmap : bool, optional
        Whether to generate and save heatmaps for each subject (default is True).

    Returns
    -------
    list of numpy.ndarray
        List of masked timeseries for all subjects.
    """
    # Directory to save individual reports
    report_dir = os.path.join(save_path, f'reports_{condition_name}_{len(subjects)}-subjects')
    os.makedirs(report_dir, exist_ok=True)

    print(f"------{condition_name}-----")
    print(f"Masker initialized with {masker_name} atlas")

    # Storage for masked timeseries
    fitted_maskers = []
    masked_timeseries = []

    for i, file in enumerate(func_list):
        sub_id = subjects[i]
        print(f"Processing subject {sub_id}...")
        
        # Extract timeseries for the subject
        fit_masker = atlas_masker_to_fit.fit(file)
        fitted_maskers.append(fit_masker)

        if confounds != None: #atlas_masker_to_fit.get_params()['high_variance_confounds'] == True:
            print("Using confound file : ", confounf_files[i])
            ts = fit_masker.transform(file, confounds = confounds[i])
            masked_timeseries.append(ts)
        else:
            ts = fit_masker.transform(file)
            masked_timeseries.append(ts)
        #fit_mask_path = os.path.join(report_dir, f'fitted_masker_{sub_id}_{condition_name}.pkl')
        
        # Generate and save the report for this subject
        report = fit_masker.generate_report()
        report_path = os.path.join(report_dir, f'report_{sub_id}_{condition_name}.html')
        report.save_as_html(report_path)

        print(f"Report saved for {sub_id} at {report_path}")
        
        if do_heatmap:
            
            # Plot heatmap of ROI × TRs
            fig, ax = plt.subplots(figsize=(10, 6))

            sns.heatmap(ts.T, cmap='coolwarm', cbar=True, ax=ax)

            ax.set_title(f"Subject {i + 1} - Time Series of All ROIs ({condition_name})", fontsize=16)
            ax.set_xlabel('Timepoints', fontsize=12)
            ax.set_ylabel('ROI Index', fontsize=12)
            plt.tight_layout()

            # Save heatmap
            heatmap_path = os.path.join(report_dir, f'{sub_id}_ROI-heatmap_{condition_name}-run.png')
            plt.savefig(heatmap_path, dpi=300)
            plt.close()

            print(f"Heatmap saved for {sub_id} at {heatmap_path}")
       

    # save masker params
    masker_params_path = os.path.join(report_dir, f'{masker_name}_masker_prams.txt')

    with open(masker_params_path, 'w') as f:
        f.write("Masker Parameters:\n")
        f.write("="*20 + "\n")
        params = atlas_masker_to_fit.get_params()
        for key, value in params.items():
            f.write(f"{key}: {value}\n")

    print(f"Masker parameters saved to {masker_params_path}")

        #print(f'saved fitted masker for {sub_id} as {fit_mask_path}')

    return masked_timeseries, fitted_maskers



def get_timestamps(
    data_path, subj_name, timestamps_path_root, condition_file, return_path=None
):
    """
    Parameters
    ----------
    data_path : path to fMRI data in order to know in which conditions the subject is
    subj_name : subject's name, e.g., APM_02_H2 to identify the particular subjects
    timestamps_path_root : root path to the timestamps
    condition_file : file name indicating the experimental condition
    return_path : if True, the function returns a path to timestamps. If False, it returns a pandas DataFrame.

    Returns
    -------
    A pandas DataFrame with timestamps or a path to the timestamps file.
    """

    # Normalize the subject name to remove `_H2`
    subj_name = subj_name.replace("_H2", "")

    # ======================================
    if "Hyperalgesia" in condition_file:  # Need to return the right timestamps
        # TIMESTAMPS
        if subj_name == "APM02":
            timestamps_file = "ASTREFF_Model6_TxT_model3_multicon_APM02_HYPER"
        elif subj_name == "APM05":
            timestamps_file = "ASTREFF_Model6_TxT_model3_multicon_APM05_HYPER"
        elif subj_name == "APM17":
            timestamps_file = "ASTREFF_Model6_TxT_model3_multicon_APM17_HYPER"
        elif subj_name == "APM20":
            timestamps_file = "ASTREFF_Model6_TxT_model3_multicon_APM20_HYPER"
        else:  # For all other subjects
            timestamps_file = "ASTREFF_Model6_TxT_model3_multicon_HYPER"

    elif "Analgesia" in condition_file:  # For Analgesia/Hypoalgesia
        if subj_name == "APM02":
            timestamps_file = "ASTREFF_Model6_TxT_model3_multicon_APM02_ANA"
        elif subj_name == "APM05":
            timestamps_file = "ASTREFF_Model6_TxT_model3_multicon_APM05_ANA"
        elif subj_name == "APM17":
            timestamps_file = "ASTREFF_Model6_TxT_model3_multicon_APM17_ANA"
        elif subj_name == "APM20":
            timestamps_file = "ASTREFF_Model6_TxT_model3_multicon_APM20_ANA"
        else:  # For all other subjects
            timestamps_file = "ASTREFF_Model6_TxT_model3_multicon_ANA"

    # Return the appropriate file
    if return_path is False:
        timestamps = scipy.io.loadmat(
            os.path.join(timestamps_path_root, timestamps_file + ".mat"),
            simplify_cells=True,
        )  # .mat option
        df_timestamps = pd.concat(
            [
                pd.DataFrame(timestamps["onsets"]),
                pd.DataFrame(timestamps["durations"]),
                pd.DataFrame(timestamps["names"]),
            ],
            axis=1,
        )
        df_timestamps.columns = ["onset", "duration", "trial_type"]
        return df_timestamps
    else:
        timestamps_path = os.path.join(
            timestamps_path_root, timestamps_file + ".xlsx"
        )  # .xlsx option
        return timestamps_path

#----------------------------
# Quality check function

import os
import matplotlib.pyplot as plt
import seaborn as sns
from nilearn.maskers import NiftiLabelsMasker

def generate_individual_labelMasker_reports_heatmap(subjects, func_list, mask_nifti, project_dir, condition_name="Analgesia"):
    """
    Generate and save individual ROI reports and heatmaps for each subject.

    Parameters
    ----------
    func_list : list of str
        List of paths to the functional files for each subject.
    mask_nifti : str
        Path to the 4D mask NIfTI file.
    project_dir : str
        Path to the project directory where results will be saved.
    condition_name : str, optional
        Name of the condition (default is "Analgesia").
        This name will be used in folder and report/heatmap names.

    Returns
    -------
    list of numpy.ndarray
        List of masked timeseries for all subjects.
    """
    # Directory to save individual reports
    report_dir = os.path.join(project_dir, 'results/imaging', f'ROI_reports_{condition_name}')
    os.makedirs(report_dir, exist_ok=True)

    # Initialize NiftiLabelsMasker
    label_masker = NiftiLabelsMasker(mask_nifti, standardize=True, detrend=True)

    # Storage for masked timeseries
    masked_timeseries = []

    for i, file in enumerate(func_list):
        sub_id = subjects[i]
        print(f"Processing subject {sub_id}...")
        
        # Extract timeseries for the subject
        ts = label_masker.fit_transform(file)
        masked_timeseries.append(ts)

        # Generate and save the report for this subject
        report = label_masker.generate_report()
        report_path = os.path.join(report_dir, f'ROI_report_{sub_id}_{condition_name}.html')
        report.save_as_html(report_path)

        # Plot heatmap of ROI × TRs
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(ts.T, cmap='coolwarm', cbar=True, ax=ax)
        ax.set_title(f"{sub_id} - Time Series of All ROIs ({condition_name})", fontsize=16)
        ax.set_xlabel('Timepoints', fontsize=12)
        ax.set_ylabel('ROI Index', fontsize=12)
        plt.tight_layout()

        heatmap_path = os.path.join(report_dir, f'ROI_heatmap_{sub_id}_{condition_name}.png')
        plt.savefig(heatmap_path, dpi=300)
        plt.close()

        print(f"Heatmap saved for {sub_id} at {heatmap_path}")
        print(f"Report saved for {sub_id} at {report_path}")

    return masked_timeseries



import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

def count_plot_TRs_2conditions(
    subjects,
    event_files, 
    TR=3, 
    neutral_pattern="N_ANA.*instrbk", 
    modulation_pattern=r"ANA.*instrbk", 
    title='Neutral and Modulation Suggestion TRs per Subjects',
    neutral_color="skyblue",
    modulation_color="salmon",
    save_path=False
):
    """
    Counts and compares the total number of TRs for neutral and modulation suggestion blocks across subjects,
    with shading based on individual block durations.

    Parameters
    ----------
    event_files : list of pandas.DataFrame
        A list where each element is a DataFrame containing the event file data for one subject.
        Each DataFrame must have columns: 'onset', 'duration', 'trial_type'.

    TR : int, optional
        The TR (Repetition Time) duration in seconds. Default is 3.

    neutral_pattern : str, optional
        Regex pattern to identify neutral suggestion blocks. Default is "N_ANA.*instrbk".

    modulation_pattern : str, optional
        Regex pattern to identify modulation suggestion blocks. Default is "ANA.*instrbk".

    title : str, optional
        The title for the plot. Default is 'Neutral and Modulation Suggestion TRs per Subjects'.

    Returns
    -------
    pd.DataFrame
        A DataFrame summarizing the total TRs for neutral and modulation suggestion blocks for all subjects.

    Notes
    -----
    Neutral and modulation suggestion blocks are identified using provided regex patterns,
    and each bar is shaded based on the individual block durations.
    """

    neutral_TRs_list = []
    modulation_TRs_list = []

    neutral_block_details = []
    modulation_block_details = []

    for i, events in enumerate(event_files):
        sub_id = subjects[i]
        # Filter for neutral suggestion blocks
        neutral_blocks = events[events['trial_type'].str.contains(neutral_pattern, na=False, regex=True)]
        neutral_durations = neutral_blocks['duration'].tolist()
        neutral_total_duration = sum(neutral_durations)
        print('--count TRs--')
        print('sub :', sub_id, 'neutral_total_duration :', neutral_total_duration)

        neutral_total_TRs = int(neutral_total_duration / TR)
        neutral_TRs_list.append(neutral_total_TRs)
        neutral_block_details.append([(d / TR) for d in neutral_durations])

        # Filter for modulation suggestion blocks, excluding the neutral ones
        modulation_blocks = events[
            events['trial_type'].str.contains(modulation_pattern, na=False, regex=True) &
            ~events['trial_type'].str.contains(neutral_pattern, na=False, regex=True)
        ]
        modulation_durations = modulation_blocks['duration'].tolist()
        modulation_total_duration = sum(modulation_durations)
        modulation_total_TRs = int(modulation_total_duration / TR)
        modulation_TRs_list.append(modulation_total_TRs)
        modulation_block_details.append([(d / TR) for d in modulation_durations])

    TRs_df = pd.DataFrame({
        "Subject": [f"{subjects[i]}" for i in range(len(event_files))],
        "Neutral_TRs": neutral_TRs_list,
        "Modulation_TRs": modulation_TRs_list
    })

    print(TRs_df)

    # Check for discrepancies in TR counts
    if TRs_df["Neutral_TRs"].nunique() == 1 and TRs_df["Modulation_TRs"].nunique() == 1:
        print("All subjects have the same number of TRs for both conditions.")
    else:
        print("Discrepancies found in the number of TRs across subjects.")

    # Plot the results with shading
    fig, ax = plt.subplots(figsize=(12, 6))
    bar_width = 0.4
    x_positions = np.arange(len(TRs_df))

    # Plot neutral and modulation TRs with shading
    for i, x in enumerate(x_positions):
        # Neutral blocks
        bottom = 0
        for block_TRs in neutral_block_details[i]:
            ax.bar(x - bar_width / 2, block_TRs, width=bar_width, color=neutral_color, bottom=bottom, edgecolor="black", alpha=0.8)
            bottom += block_TRs

        # Modulation blocks
        bottom = 0
        for block_TRs in modulation_block_details[i]:
            ax.bar(x + bar_width / 2, block_TRs, width=bar_width, color=modulation_color, bottom=bottom, edgecolor="black", alpha=0.8)
            bottom += block_TRs

    ax.set_xlabel("Subjects", fontsize=14)
    ax.set_ylabel("Total TRs", fontsize=14)
    ax.set_title(title, fontsize=16)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(TRs_df["Subject"], rotation=45, fontsize=10)

    # Create a manual legend with distinct colors
    neutral_patch = mpatches.Patch(color=neutral_color, label="Neutral TRs")
    modulation_patch = mpatches.Patch(color=modulation_color, label="Modulation TRs")
    ax.legend(handles=[neutral_patch, modulation_patch], fontsize=12)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)

    plt.show()
    plt.close()

    return TRs_df, fig

import os
from nilearn.maskers import MultiNiftiLabelsMasker


def generate_multinifti_report(func_list, mask_nifti, atlas_name, save_dir, condition_name="Analgesia"):
    """
    Generate and save a MultiNiftiLabelsMasker report for a given condition.

    Parameters
    ----------
    func_list : list of str
        List of paths to the functional files for the condition.
    mask_nifti : str
        Path to the 4D mask NIfTI file.
    project_dir : str
        Path to the project directory where results will be saved.
    condition_name : str, optional
        Name of the condition (default is "Analgesia").
        This name will be used in folder and report names.

    Returns
    -------
    None
    """
    # Ensure mask_nifti is a valid 4D mask
    assert len(mask_nifti.shape) == 4, "Mask image must be 3D."

    nsub = len(func_list)
    
    # Initialize MultiNiftiLabelsMasker
    masker_type = 'MultiNiftiLabelsMasker'
    multi_masker = MultiNiftiLabelsMasker(mask_nifti, standardize=True, detrend=True)

    # Fit-transform the functional files
    fitted_mask = multi_masker.fit_transform(func_list)


    # Generate and save the report
    report = multi_masker.generate_report()
    report_name = f'{atlas_name}_Multireport_{condition_name}_{nsub}-subjects.html'
    report_path = os.path.join(save_dir, report_name)
    report.save_as_html(report_path)

    print(f"Report saved at: {report_path}")

    return fitted_mask


#======First LEvel GLM function==================


def create_tr_masks_for_suggestion(combined_dm, regressor,all_conds = ['ANA', 'N_ANA', 'HYPER', 'N_HYPER'], verbose=True):
    """
    Create boolean masks for each condition from the combined design matrix columns,
    ensuring no overlap with other conditions.

    Parameters:
        combined_dm : pandas.DataFrame
            The combined design matrix containing conditions as columns.
        regressor_names : list
            List of condition names or patterns to extract from the design matrix.

    Returns:
        dict: A dictionary with condition names as keys and masks (list of indices) as values.
    """
 
    condition = regressor[:-5] # extract the cond by removing '_sugg'
    print(f'extracting {condition} from {regressor}')

    # Include only columns that start with the condition and contain '*instrbk*'
    include_columns = [
        col for col in combined_dm.columns 
        if col.startswith(condition) and 'instrbk' in col
    ]
    # print('include_columns', include_columns)
    # Exclude overlapping columns from other conditions
    exclude_columns = [
        col for other_condition in all_conds if other_condition != condition
        for col in combined_dm.columns 
        if col.startswith(other_condition) and 'instrbk' in col
    ]
    # print('exclude_columns', exclude_columns)
    target_columns = include_columns
    regressor_signal = combined_dm[target_columns].sum(axis=1).to_numpy()
    active_mask = regressor_signal != 0  # Selects positive and negative values
    labeled_array, num_features = label(active_mask)
    clean_indices = np.where((labeled_array > 0) & (regressor_signal != 0))[0] # account for 0 in HRF block
    # indices = np.where(labeled_array > 0)[0]

    print('n regressor kept : ', len(target_columns))

    return clean_indices, target_columns

from scipy.ndimage import label
def create_tr_masks_for_shock(combined_dm, regressor, all_conds = ['ANA', 'N_ANA', 'HYPER', 'N_HYPER'], verbose=True):
    """
    Create boolean masks for each condition from the combined design matrix columns,
    ensuring no overlap with other conditions.

    Parameters:
        combined_dm : pandas.DataFrame
            The combined design matrix containing conditions as columns.
        regressor_names : list
            List of condition names or patterns to extract from the design matrix.

    Returns:
        dict: A dictionary with condition names as keys and masks (list of indices) as values.
    """
    condition = regressor[:-6] # e.g. ANA, extract the cond by removing '_shock       

    # Include only columns that start with the condition and contain '*instrbk*'
    include_columns = [
        col for col in combined_dm.columns 
        if col.startswith(condition) and 'shock' in col
    ]
  
    # Exclude overlapping columns from other conditions
    # exclude_columns = [
    #     col for other_condition in all_conds if other_condition != condition
    #     for col in combined_dm.columns 
    #     if col.startswith(other_condition) and 'shock' in col
    # ]
    # print('exclude_columns', exclude_columns)
    # target_columns = set(include_columns) - set(exclude_columns)
    # target_columns = include_columns
    # mask = combined_dm[list(target_columns)].sum(axis=1).to_numpy() > 0
    # # Get indices where the condition is true
    # indices = np.where(mask)[0]
    target_columns = include_columns
    regressor_signal = combined_dm[target_columns].sum(axis=1).to_numpy()
    active_mask = regressor_signal != 0  # Selects positive and negative values
    labeled_array, num_features = label(active_mask)
    clean_indices = np.where((labeled_array > 0) & (regressor_signal != 0))[0]
    # indices = np.where(labeled_array > 0)[0]

    print('n regressor kept : ', len(target_columns))

    return clean_indices, target_columns

def extract_tr_indices_from_events(events_df, regressor, TR, total_scans, need_str = 'instrbk'):
    """
    Extract TR indices for each condition based on event file timings.

    Parameters:
    - events_df (pd.DataFrame): Event file containing 'onset', 'duration', and 'trial_type'.
    - regressors_names (list): List of target regressors (e.g., 'ANA_sugg', 'HYPER_sugg', etc.).
    - TR (float): Repetition time (time per fMRI volume acquisition).
    - total_scans (int): The total number of scans in the fMRI run.

    Returns:
    - dict: A dictionary where keys are regressors and values are lists of TR indices.
    """
    

    # Extract base condition (removing '_sugg' or '_shock')
    if need_str == 'instrbk':
        condition = regressor[:-5]  
    if need_str == 'shock':
        condition = regressor[:-6]
        
    # Filter only events matching the condition and containing 'instrbk'
    relevant_events = events_df[
        (events_df['trial_type'].str.startswith(condition)) & 
        (events_df['trial_type'].str.contains(need_str))
    ]

    # Initialize empty list to store TR indices
    condition_trs = []

    for _, row in relevant_events.iterrows():
        onset_tr = int(round(row['onset'] / TR))  # Convert onset to TR index
        duration_tr = int(round(row['duration'] / TR))  # Convert duration to TR count

        # Generate all TRs spanning the event
        tr_range = np.arange(onset_tr, onset_tr + duration_tr)

        # Clip TRs to ensure they do not exceed total scan count
        tr_range = tr_range[tr_range < total_scans]

        # Append to the list
        condition_trs.extend(tr_range)

    # Store indices for this regressor
    tr_indices = np.array(condition_trs)  # Convert to NumPy array for easy handling

    print(f"Extracted {len(condition_trs)} TRs for {regressor}: {condition_trs}")

    return tr_indices, list(relevant_events['trial_type'])


def extract_regressor_timeseries(masked_2d_timeseries,indices_dct, conditions):
    """
    Extracts the timeseries for each condition based on specific regressors from design matrix indices.
    takes a dict with all the indices/rows to extract in the timeseries.

    Parameters:

        masked_2d_timeseries : np.ndarray
            2d Timeseries array. transformed from e.g. niftiMasker
        indices_dct : dict
            Condition indices. 1 key per regressor, value contains indices to extract (from design matrix).


    Returns:
        dict : Extracted timeseries for each condition.
    """
    
    # Extract timeseries for each condition
    condition_timeseries = {}
    for cond in conditions:
        condition_timeseries[cond] = masked_2d_timeseries[indices_dct[cond], :]

    return condition_timeseries


def plot_timecourses_from_ls(df_list, labels, save_to = False, n_rows=4, n_cols=4, condition_name="_", show=False, measure_lengend = 'Mean'):
     
    """
    Plots the timecourse of all ROIs from a list of DataFrames. For each ROI (column), 
    it plots the timecourse of all subjects in pale gray and the mean signal for that ROI in orange.

    Parameters
    ----------
    df_list : list of pd.DataFrame
        List of DataFrames, where each DataFrame corresponds to one subject.
        Each DataFrame should have shape (timepoints, ROIs).
    labels : list of str
        ROI names used for subplot titles.
    save_to : str or bool, optional
        Directory path to save the plot. If False, the plot is not saved. Default is False.
    n_rows : int, optional
        Number of rows in the subplot layout. Default is 4.
    n_cols : int, optional
        Number of columns in the subplot layout. Default is 4.
    condition_name : str, optional
        A label for the condition (e.g., "Pre" or "Post") to include in the title. Default is "_".
    show : bool, optional
        If True, displays the plot. Default is False.
    measure_lengend : str, optional
        Legend label for the mean signal. Default is 'Mean'.

    Returns
    -------
    None
    """

    n_subjects = len(df_list)  # Number of subjects
    n_rois = df_list[0].shape[1]  # Number of ROIs (columns in the DataFrame)

    # Create a figure with specified rows and columns
    plt.figure(figsize=(15, n_rows * 3))  # Adjust figure size based on the number of rows

    for roi in range(n_rois):
        if roi >= (n_rows*n_cols):
            break 
        label = labels[roi]
        plt.subplot(n_rows, n_cols, roi + 1)  # Create a subplot for each ROI

        # Plot each subject's timecourse for this ROI (pale gray lines)
        for subj in range(n_subjects):
            plt.plot(df_list[subj][:, roi], color='lightgray', alpha=0.5)
        
        # Calculate and plot the mean timecourse for this ROI (orange line)
        mean_roi_signal = np.mean([df[:, roi] for df in df_list], axis=0)
        plt.plot(mean_roi_signal, color='orange', linewidth=2, label=f'{measure_lengend} ROI {roi + 1}')
        
        # Customize the plot
        plt.title(f"{label}")
        #plt.xlabel("Timepoints")
        plt.ylabel("Signal")
        plt.legend()

    # Display the figure with a tight layout
    plt.tight_layout()
    fig_name = 'ROI_timecourse_cond-{}.png'.format(condition_name)
    if type(save_to) is str:
        plt.savefig(os.path.join(save_to, fig_name))
        print('[plot_timecourse] Saving a timecourse figure in {}'.format(os.path.join(save_to, fig_name)))
    if show == True:
        plt.show()


def write_masker_params(masker_get_param, masker_params_path):
    """
    Save the parameters of the masker to a text file.

    Parameters
    ----------
    masker_get_param : dict
        dict from masker.get_params() method.
    masker_params_path : str
        The path to save the masker parameters to.

    Returns
    -------
    None
    """

    with open(masker_params_path, 'w') as f:
            f.write("Masker Parameters:\n")
            f.write("="*20 + "\n")

            for key, value in masker_get_param.items():
                if key != 'labels_img':
                    f.write(f"{key}: {value}\n")
