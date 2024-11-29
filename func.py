import os
import glob as glob
import pandas as pd
import numpy as np
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
        ax.set_title(f"Subject {i + 1} - Time Series of All ROIs ({condition_name})", fontsize=16)
        ax.set_xlabel('Timepoints', fontsize=12)
        ax.set_ylabel('ROI Index', fontsize=12)
        plt.tight_layout()

        # Save the heatmap
        heatmap_path = os.path.join(report_dir, f'ROI_heatmap_{sub_id}_{condition_name}.png')
        plt.savefig(heatmap_path, dpi=300)
        plt.close()

        print(f"Heatmap saved for {sub_id} at {heatmap_path}")
        print(f"Report saved for {sub_id} at {report_path}")

    return masked_timeseries



import pandas as pd
import matplotlib.pyplot as plt

def count_TRs_2conds_general(
    event_files, 
    TR=3, 
    neutral_pattern="N_ANA.*instrbk", 
    modulation_pattern=r"ANA.*instrbk", 
    title='Neutral and Modulation Suggestion TRs per Subjects',
    neutral_color="skyblue",
    modulation_color="salmon",
    save_path = False
):
    """
    Counts and compares the total number of TRs for neutral and modulation suggestion blocks across subjects.

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
    Neutral and modulation suggestion blocks are identified using provided regex patterns.

    Example
    -------
    TRs_df = count_TRs_2conds_general(
        events_hyper, 
        TR=3, 
        neutral_pattern="N_HYPER.*instrbk", 
        modulation_pattern="HYPER.*instrbk",
        title='Neutral and Hyperalgesia Suggestion TRs per Subjects'
    )
    """

    # Initialize lists to store TR counts for each condition
    neutral_TRs_list = []
    modulation_TRs_list = []

    # Loop through all subjects' event data
    for i, events in enumerate(event_files):
        # Filter for neutral suggestion blocks
        neutral_blocks = events[events['trial_type'].str.contains(neutral_pattern, na=False, regex=True)]
        neutral_durations = neutral_blocks['duration'].tolist()
        neutral_total_duration = sum(neutral_durations)
        neutral_total_TRs = int(neutral_total_duration / TR)
        neutral_TRs_list.append(neutral_total_TRs)

        # Filter for modulation suggestion blocks, excluding the neutral ones
        modulation_blocks = events[
            events['trial_type'].str.contains(modulation_pattern, na=False, regex=True) &
            ~events['trial_type'].str.contains(neutral_pattern, na=False, regex=True)
        ]
        modulation_durations = modulation_blocks['duration'].tolist()
        modulation_total_duration = sum(modulation_durations)
        modulation_total_TRs = int(modulation_total_duration / TR)
        modulation_TRs_list.append(modulation_total_TRs)

    # Create a DataFrame for easier comparison
    TRs_df = pd.DataFrame({
        "Subject": [f"Subject_{i+1}" for i in range(len(event_files))],
        "Neutral_TRs": neutral_TRs_list,
        "Modulation_TRs": modulation_TRs_list
    })

    print(TRs_df)

    # Check for discrepancies in TR counts
    if TRs_df["Neutral_TRs"].nunique() == 1 and TRs_df["Modulation_TRs"].nunique() == 1:
        print("All subjects have the same number of TRs for both conditions.")
    else:
        print("Discrepancies found in the number of TRs across subjects.")

    # Plot the results
    plt.figure(figsize=(12, 6))
    bar_width = 0.4
    x_positions = range(len(TRs_df))

    # Plot neutral and modulation TRs
    plt.bar([x - bar_width/2 for x in x_positions], TRs_df["Neutral_TRs"], width=bar_width, label="Neutral TRs", color=neutral_color)
    plt.bar([x + bar_width/2 for x in x_positions], TRs_df["Modulation_TRs"], width=bar_width, label="Modulation TRs", color=modulation_color)
    
    # Add labels, title, and legend
    plt.xlabel("Subjects", fontsize=14)
    plt.ylabel("Total TRs", fontsize=14)
    plt.title(title, fontsize=16)
    plt.xticks(x_positions, TRs_df["Subject"], rotation=45, fontsize=10)
    plt.legend()
    plt.tight_layout()

 
    if save_path:
        plt.savefig(save_path)

    plt.show()
    plt.close() 
 
    return TRs_df, fig


#======First LEvel GLM function==================

def create_tr_masks_for_regressors(combined_dm, regressor_names=["ANA", "N_ANA", "HYPER", "N_HYPER"], verbose=True):
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
    condition_indices = {}

    for condition in regressor_names:
        # Include only columns that start with the condition and contain '*instrbk*'
        include_columns = [
            col for col in combined_dm.columns 
            if col.startswith(condition) and 'instrbk' in col
        ]
    
        # Exclude overlapping columns from other conditions
        exclude_columns = [
            col for other_condition in regressor_names if other_condition != condition
            for col in combined_dm.columns 
            if col.startswith(other_condition) and 'instrbk' in col
        ]
        #print(f"Excluding columns: {exclude_columns}")

       
        target_columns = set(include_columns) - set(exclude_columns)

        mask = combined_dm[list(target_columns)].sum(axis=1).to_numpy() > 0

        # Get indices where the condition is true
        indices = np.where(mask)[0]
        condition_indices[condition] = indices
        
        if verbose:
            print(f'Including regressors : {set(include_columns)}')
    if verbose:
        print(f"Conditions '{condition_indices.keys()}' have  {[len(condition_indices[key]) for key in condition_indices.keys()]} TRs.")
        
    return condition_indices



def extract_regressor_timeseries(masked_2d_timeseries,indices_dct):
    """
    Extracts the timeseries for each condition based on specific regressors from design matrix indices.
    takes a dict with all the indices/rows to extract in the timeseries.

    Parameters:

        masked_2d_timeseries : np.ndarray
            Timeseries from the 'Analgesia' run.
        indices_dct : dict
            Condition indices. 1 key per regressor, value contains indices to extract (from design matrix).


    Returns:
        dict : Extracted timeseries for each condition.
    """
    
    # Extract timeseries for each condition
    condition_timeseries = {
        "ANA": masked_2d_timeseries[indices_dct["ANA"], :],
        "N_ANA": masked_2d_timeseries[indices_dct["N_ANA"], :],
        "HYPER": masked_2d_timeseries[indices_dct["HYPER"], :],
        "N_HYPER": masked_2d_timeseries[indices_dct["N_HYPER"], :]
    }

    return condition_timeseries
