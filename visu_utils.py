import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import rankdata
from nilearn.maskers  import NiftiLabelsMasker, NiftiMapsMasker
import os 
import nibabel as nib
import pandas as pd
from nilearn.plotting import find_parcellation_cut_coords


def load_isc_results(isc_results):
    isc = isc_results['isc']
    observed_isc = isc_results['observed']
    p_values = isc_results['p_values']
    ci = isc_results['confidence_intervals']
    distributions = isc_results['distribution']
    median_isc = isc_results['median_isc'] 

    return isc, observed_isc, p_values, ci, median_isc, distributions

def load_isc_results_pairwise(isc_results):
    isc = isc_results['isc']
    observed_isc = isc_results['observed']
    p_values = isc_results['p_values']
    ci = isc_results['confidence_intervals']
    distributions = isc_results['distribution']
    
    return isc, observed_isc, p_values, ci, distributions

def load_isc_results_permutation(isc_results):
    
    observed_isc = isc_results['observed']
    p_values = isc_results['p_value']
    distributions = isc_results['distribution']
    return observed_isc, p_values, distributions


def plot_isc_distributions(observed_isc, p_values, median_isc, bootstrap_distributions,save_to=None, title="ISC Distributions"):
    """
    Plots the distributions of observed ISC values per column with the median ISC as a line,
    p-value annotation, and bootstrap distributions.

    Parameters
    ----------
    observed_isc : pd.DataFrame
        The observed ISC values (timepoints x ROIs or timepoints x subjects) as a DataFrame.
    p_values : np.ndarray
        P-values corresponding to the observed ISC values.
    median_isc : np.ndarray
        Median ISC values to be displayed as a line on the plots.
    bootstrap_distributions : np.ndarray
        Bootstrap distributions for ISC values (n_boot x ROIs).
    title : str, optional
        Title for the entire figure. Default is "ISC Distributions".
    """
    col_names = observed_isc.columns
    observed_isc = observed_isc.to_numpy()
    n_rois = 5
    n_cols = observed_isc.shape[1]  # Number of columns
    fig, axes = plt.subplots(1, n_rois)
    fig.suptitle(title, fontsize=24)

    for i in range(n_cols):
        ax = axes[i]
        data = observed_isc[:, i]
        bootstrap_data = bootstrap_distributions[:, i]

        # Plot the histogram
        ax.hist(data, bins=20, alpha=0.7, color='blue', label='Observed ISC')

        # Plot the bootstrap distribution
        ax.hist(bootstrap_data, bins=30, alpha=0.4, color='green', label='Bootstrap Distribution', density=True)

        # Add the median line
        ax.axvline(median_isc[i], color='red', linestyle='--', label=f'Median ISC = {median_isc[i]:.4f}')

        # Annotate the p-value
        ax.text(
            0.05, 0.95, f'p = {p_values[i]:.4f}',
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
        )
        fontsize = 20
        # Labels and formatting
        ax.set_xlabel("ISC Values", fontsize=fontsize)
        ax.set_ylabel("Frequency", fontsize=fontsize)
        ax.set_title(f"ROI : {col_names[i]}", fontsize=fontsize)
        ax.legend()

        if save_to is not None:
            plt.savefig(save_to, bbox_inches='tight', dpi = 500)
    print(f"Plot saved to {save_to}")


    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust layout to fit title
    plt.show()  


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import squareform

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import squareform
import pandas as pd

def heatmap_pairwise_isc_combined(isc_df, subjects, behavioral_scores=None,roi_to_plot = 'all', save_to=None, show=True, title="Pairwise ISC Matrices"):
    """
    Visualizes subject-by-subject similarity matrices for all ROIs in a single row of plots.

    Parameters
    ----------
    isc_df : pd.DataFrame
        DataFrame containing ISC vectorized results, with columns corresponding to ROIs.
    subjects : list
        List of subject IDs corresponding to the ISC data.
    behavioral_scores : pd.Series, optional
        A pandas Series containing behavioral scores for each subject. If provided, subjects are reordered
        based on the rank of their behavioral scores.
    output_dir : str, optional
        Directory to save the plots. If None, plots are not saved. Default is None.
    show : bool, optional
        Whether to display the plots. Default is False.
    title : str, optional
        Title for the combined figure. Default is "Pairwise ISC Matrices".
    """

    n_rois = 5

    if behavioral_scores is not None:
        ranked_scores = rankdata(behavioral_scores)
        sorted_indices = np.argsort(ranked_scores)  # Sort indices based on rank
        sorted_subjects = [subjects[i] for i in sorted_indices]
    else:
        sorted_indices = range(len(subjects))
        sorted_subjects = subjects
        
    columns = isc_df.columns
    observed_isc = isc_df.to_numpy()

    fig, axes = plt.subplots(1, n_rois-1, figsize=(5 * n_rois, 5))
    fig.suptitle(title, fontsize=16)

    # Handle single ROI case where axes is not iterable
    if n_rois == 1:
        axes = [axes]

    for i, roi_name in enumerate(isc_df.columns):

        isc_matrix = squareform(observed_isc[:, i])

        if behavioral_scores is not None:
            isc_matrix = isc_matrix[np.ix_(sorted_indices, sorted_indices)]

        ax = axes[i]
        sns.heatmap(
            isc_matrix, 
            annot=False, 
            cmap='coolwarm', 
            square=True, 
            cbar_kws={'label': 'ISC'}, 
            ax=ax, 
            linewidths=0.5
        )
        ax.set_title(f"{roi_name}")
        ax.set_xticks(np.arange(len(subjects)) + 0.5)
        ax.set_yticks(np.arange(len(subjects)) + 0.5)
        ax.set_xticklabels(sorted_subjects, rotation=90, fontsize=8)
        ax.set_yticklabels(sorted_subjects, fontsize=8)

    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust layout to fit the title

    # Save the plot
    if save_to is not None:
        plt.savefig(save_to, bbox_inches="tight", dpi=300)
        print(f"Combined ISC matrix plot saved to {save_to}")

    # Show the plot
    if show:
        plt.show()
    else:
        plt.close()


def load_boot_images(results_dir, condition):
    isc_img_path = os.path.join(results_dir, condition, f"isc_val_{condition}_boot5000_pariwiseFalse.nii.gz")
    pval_img_path = os.path.join(results_dir, condition, f"p_values_{condition}_boot5000_pairwiseFalse.nii.gz")
    
    isc_img = nib.load(isc_img_path)
    pval_img = nib.load(pval_img_path)
    
    return isc_img, pval_img

def load_perm_images(results_dir, condition):
    isc_img_path = os.path.join(results_dir, condition, f"isc_val_{condition}_boot5000_pariwiseFalse.nii.gz")
    pval_img_path = os.path.join(results_dir, condition, f"p_values_{condition}_boot5000_pairwiseFalse.nii.gz")
    
    isc_img = nib.load(isc_img_path)
    pval_img = nib.load(pval_img_path)
    
    return isc_img, pval_img

def load_difumo():
    project_dir = project_dir = "/data/rainville/dSutterlin/projects/ISC_hypnotic_suggestions"
    atlas_path = os.path.join(project_dir, 'masks/DiFuMo256/3mm/maps.nii.gz')
    atlas_dict_path = os.path.join(project_dir, 'masks/DiFuMo256/labels_256_dictionary.csv')
    atlas = nib.load(atlas_path)
    atlas_df = pd.read_csv(atlas_dict_path)
    atlas_labels = atlas_df['Difumo_names']
    atlas_name = 'Difumo256' # !!!!!!! 'Difumo256'

    masker = NiftiMapsMasker(maps_img=atlas, standardize=True, memory='nilearn_cache', verbose=5)
    return atlas, masker.fit(), atlas_df, atlas_labels


import os
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.stats.multitest import multipletests

def plot_isc_median_with_significance(isc_median, p_values, atlas_labels, p_threshold=0.01, 
                                      significant_color='red', nonsignificant_color='gray', 
                                      save_path=None, show=False, fdr_correction=False):
    """
    Plots ISC median values as a bar plot with significant regions highlighted.

    Parameters
    ----------
    isc_median : np.ndarray
        Array of median ISC values for each ROI.
    p_values : np.ndarray
        Array of p-values corresponding to the ISC values.
    atlas_labels : list
        List of ROI labels corresponding to the atlas.
    p_threshold : float, optional
        Threshold for significance of p-values. Default is 0.01.
    significant_color : str, optional
        Color for bars representing significant ROIs. Default is 'red'.
    nonsignificant_color : str, optional
        Color for bars representing non-significant ROIs. Default is 'gray'.
    save_path : str, optional
        Path to save the plot. Default is None.
    show : bool, optional
        Whether to display the plot. Default is False.
    fdr_correction : bool, optional
        Whether to apply FDR correction to p-values. Default is False.
    """
    # Apply FDR correction 
    if fdr_correction:
        _, corrected_p_values, _, _ = multipletests(p_values, alpha=p_threshold, method='fdr_bh')
        sig_mask = corrected_p_values < p_threshold
    else:
        sig_mask = p_values < p_threshold

    # Highlight significant ROIs
    significant_labels = [label if sig else " " for label, sig in zip(atlas_labels, sig_mask)]
    bar_colors = [significant_color if sig else nonsignificant_color for sig in sig_mask]

    # Create the bar plot
    plt.figure(figsize=(14, 7))
    plt.bar(range(len(isc_median)), isc_median, color=bar_colors, alpha=0.8)
    plt.axhline(0, color='black', linestyle='--', linewidth=0.8)
    plt.xticks(range(len(isc_median)), significant_labels, rotation=90, fontsize=8)
    plt.xlabel("ROIs", fontsize=12)
    plt.ylabel("Median ISC", fontsize=12)
    plt.title(f"Median ISC Values (Significant Regions Highlighted, p < {p_threshold})", fontsize=14)
    plt.tight_layout()

    # Add histogram of p-values in the corner
    inset_ax = plt.gcf().add_axes([0.75, 0.75, 0.2, 0.2])  # x, y, width, height in figure coordinates
    inset_ax.hist(p_values, bins=20, color='blue', alpha=0.7)
    inset_ax.axvline(p_threshold, color='red', linestyle='--', linewidth=0.8)
    inset_ax.set_title("P-value Distribution", fontsize=10)
    inset_ax.set_xlabel("P-value", fontsize=8)
    inset_ax.set_ylabel("Count", fontsize=8)
    inset_ax.tick_params(axis='both', which='major', labelsize=8)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
        print(f"Bar plot saved to {save_path}")
    if show:
        plt.show()
    else:
        plt.close()

    return sig_mask

import nibabel as nib
import numpy as np
import os
from nilearn.plotting import plot_stat_map

def project_isc_to_brain(atlas_path, isc_median, atlas_labels, p_values=None, p_threshold=0.01,title = '"ISC Median Values (Thresholded)', save_path=None, show=True):
    """
    Projects ISC values to brain space and optionally thresholds by significance.

    Parameters
    ----------
    atlas_path : str
        Path to the atlas file used for analysis.
    isc_median : np.ndarray
        Array of median ISC values for each ROI.
    atlas_labels : list
        List of ROI labels corresponding to the atlas.
    p_values : np.ndarray, optional
        P-values corresponding to the ISC values. Default is None.
    p_threshold : float, optional
        Threshold for significance of p-values. Default is 0.01.
    save_path : str, optional
        Path to save the projected ISC map. Default is None.
    show : bool, optional
        Whether to display the plot. Default is True.
    """
    
    atlas_img = nib.load(atlas_path)
    atlas_data = atlas_img.get_fdata()

    # Create an empty volume to store ISC values
    isc_vol = np.zeros_like(atlas_data)
    non_sig_max_isc = []
    sig_min_isc = []
    # Assign ISC values to corresponding atlas regions
    for roi_idx, label in enumerate(atlas_labels):
        roi_mask = atlas_data == roi_idx + 1
        if p_values is not None and p_values[roi_idx] < p_threshold:
            # Assign significant ISC value
            isc_vol[roi_mask] = isc_median[roi_idx]
            sig_min_isc.append(isc_median[roi_idx])
        else:
            # Track the highest ISC value for non-significant ROIs
            non_sig_max_isc.append(isc_median[roi_idx])

    isc_img = nib.Nifti1Image(isc_vol, atlas_img.affine, atlas_img.header)
    
    # max_isc_thresh = np.max(np.array(non_sig_max_isc))
    if len(sig_min_isc) > 0:
        min_sig = np.min(np.array(sig_min_isc))
    else : min_sig = 0.0001
    # Save the ISC map if save_path is provided
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        nib.save(isc_img, save_path)
        print(f"ISC projection saved to {save_path}")
    max_isc = np.max(isc_vol)
    print(f"Max ISC value: {max_isc}")

    if show:
        plot_stat_map(isc_img, title=title, threshold=min_sig,vmax=max_isc, colorbar=True, display_mode='z', cut_coords=7, draw_cross=False)

    return isc_img, min_sig 


def project_isc_to_brain_perm(atlas_path, isc_median, atlas_labels, p_values=None, p_threshold=0.01,title = '"ISC Median Values (Thresholded)', save_path=None, show=True):
    """
    Projects ISC values to brain space and optionally thresholds by significance.

    Parameters
    ----------
    atlas_path : str
        Path to the atlas file used for analysis.
    isc_median : np.ndarray
        Array of median ISC values for each ROI.
    atlas_labels : list
        List of ROI labels corresponding to the atlas.
    p_values : np.ndarray, optional
        P-values corresponding to the ISC values. Default is None.
    p_threshold : float, optional
        Threshold for significance of p-values. Default is 0.01.
    save_path : str, optional
        Path to save the projected ISC map. Default is None.
    show : bool, optional
        Whether to display the plot. Default is True.
    """
    # Load the atlas
    atlas_img = nib.load(atlas_path)
    atlas_data = atlas_img.get_fdata()
    roi_coords = find_parcellation_cut_coords(labels_img=atlas_img)

    # Create an empty volume to store ISC values
    isc_vol = np.zeros_like(atlas_data)
    non_sig_max_isc = []
    sig_labels_data = []

    # Assign ISC values to corresponding atlas regions
    for roi_idx, label in enumerate(atlas_labels):
        roi_mask = atlas_data == roi_idx + 1
        if p_values is not None and p_values[roi_idx] < p_threshold:
            # Assign significant ISC value
            isc_vol[roi_mask] = isc_median[roi_idx]
            sig_labels_data.append({
                "ROI": label,
                "Difference": round(isc_median[roi_idx], 2),
                "p-value": round(p_values[roi_idx], 2),
                "Coordinates": tuple(np.round(roi_coords[roi_idx], 2))
            })
        else:
            # Track the highest ISC value for non-significant ROIs
            non_sig_max_isc.append(isc_median[roi_idx])

    # Create a Nifti image for the ISC projection
    isc_img = nib.Nifti1Image(isc_vol, atlas_img.affine, atlas_img.header)

    max_isc_thresh = np.max(np.array(non_sig_max_isc))
    # Save the ISC map if save_path is provided
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        nib.save(isc_img, save_path)
        print(f"ISC projection saved to {save_path}")

    max_diff = np.max(np.abs(isc_vol))
    sign_max = np.max(isc_vol) if np.max(isc_vol) > 0 else np.min(isc_vol)
    print(f"Max ISC abs diff: {max_diff} and sign : {sign_max}")

    if show:
        plot_stat_map(isc_img, title=title, threshold=max_isc_thresh,vmax=max_diff, colorbar=True, display_mode='x', cut_coords=6, draw_cross=False)

    return isc_img, max_isc_thresh, pd.DataFrame(sig_labels_data)


import seaborn as sns
import matplotlib.pyplot as plt

def plot_similarity_and_histogram(similarity_matrix, correlations, p_values, atlas_labels, behav_name, save_path=None):
    """
    Plots the similarity matrix as a heatmap and the RSA correlation histogram.

    Parameters
    ----------
    similarity_matrix : np.ndarray
        Similarity matrix for the behavioral variable.
    correlations : np.ndarray
        RSA correlations for each ROI.
    p_values : np.ndarray
        P-values corresponding to the RSA correlations.
    atlas_labels : list
        List of ROI labels.
    behav_name : str
        Name of the behavioral variable.
    save_path : str, optional
        Path to save the plots. Default is None.

    Returns
    -------
    None
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # Plot the similarity matrix heatmap
    sns.heatmap(similarity_matrix, annot=False, cmap='coolwarm', ax=axes[0])
    axes[0].set_title(f"Behavioral Similarity Matrix ({behav_name})", fontsize=14)
    axes[0].set_xlabel("Subjects", fontsize=12)
    axes[0].set_ylabel("Subjects", fontsize=12)

    # Plot the correlation histogram
    axes[1].hist(correlations, bins=20, color='blue', alpha=0.7, edgecolor='black')
    axes[1].set_title(f"RSA Correlations Distribution ({behav_name})", fontsize=14)
    axes[1].set_xlabel("Correlation Values", fontsize=12)
    axes[1].set_ylabel("Frequency", fontsize=12)

    # Add a line for zero correlation
    axes[1].axvline(0, color='red', linestyle='--', linewidth=1, label='Zero Correlation')
    axes[1].legend(fontsize=10)

    plt.tight_layout()

    # Save the plot if save_path is provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plots saved to {save_path}")

    plt.show()


def vector_to_symmetric_matrix(vec, size):
   
    mat = np.zeros((size, size))
    triu_indices = np.triu_indices(size, k=1)
    mat[triu_indices] = vec
    mat = mat + mat.T
    
    return mat

#=====================
# Atlas related

def yeo_networks_from_schaeffer(label_list):
    """
    Classifies the Yeo 7 networks based on the label list.

    Parameters
    ----------
    label_list : list
        List of labels from the atlas, where each label contains the network name.

    Returns
    -------
    network_mapping : dict
        A dictionary where the keys are network names (e.g., 'Vis', 'SomMot') 
        and the values are lists of indices corresponding to each network.
    labels_by_index : list
        A list of network names (e.g., 'Vis') corresponding to each ROI.
    """
    network_mapping = {}
    labels_by_index = []

    for idx, label in enumerate(label_list):
        # Extract the network name from the label (e.g., 'Vis' from '7Networks_LH_Vis_9')
        network = label.decode().split('_')[2]
        labels_by_index.append(network)

        if network not in network_mapping:
            network_mapping[network] = []
        network_mapping[network].append(idx)

    return network_mapping, labels_by_index


import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.cm as cm
import matplotlib.colors as mcolors


def plot_scatter_legend(correl1, correl2, var_name = ['var1', 'var2'], grp_id=None, legend=True, title=None, save_path=None):
    """
    Plots a scatter plot of two ISC-RSA correlation models, optionally grouped by categories.

    Parameters
    ----------
    correl1 : np.ndarray or list
        Correlation values for the first model (e.g., Euclidean).
    correl2 : np.ndarray or list
        Correlation values for the second model (e.g., AnnaK).
    grp_id : list, optional
        Group IDs for each point (e.g., Yeo network names). Default is None.
    legend : bool, optional
        Whether to include a legend in the plot. Default is True.
    title : str, optional
        Title for the plot. Default is None.
    save_path : str, optional
        File path to save the plot. Default is None.

    Returns
    -------
    None
    """
    correl1 = np.array(correl1)
    correl2 = np.array(correl2)
    grp_id = np.array(grp_id) if grp_id is not None else None

   
    plt.figure(figsize=(6, 4))

    if grp_id is not None:
  
        unique_groups = np.unique(grp_id)
        num_grps = len(unique_groups)
        colors = cm.get_cmap('tab20', num_grps).colors
        color_map = {group: colors[i] for i, group in enumerate(unique_groups)}

        # Plot each group with a unique color
        for group in unique_groups:
            mask = grp_id == group
            plt.scatter(correl1[mask], correl2[mask], label=group, color=color_map[group], alpha=0.7, edgecolor='k')
    else:
        plt.scatter(correl1, correl2, color='blue', alpha=0.7, edgecolor='k')

    # Add diagonal line
    max_val = max(np.max(correl1), np.max(correl2))
    min_val = min(np.min(correl1), np.min(correl2))
    plt.plot([min_val, max_val], [min_val, max_val], linestyle='--', color='black', linewidth=1)

    # Add labels and title
    plt.xlabel(f"{var_name[0]}", fontsize=14)
    plt.ylabel(f'{var_name[1]}', fontsize=14)
    plt.title(title if title else "scatter plot", fontsize=16)

    if legend and grp_id is not None:
        plt.legend(title="Yeo Networks", loc='best', fontsize=10, title_fontsize=12)

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"Plot saved to {save_path}")
    else:
        plt.show()


# Function to plot images in a grid
from PIL import Image

def plot_images_grid(image_paths, title,save_to=False, show=True):
    """Plots images in a flexible grid layout."""
    num_images = len(image_paths)
    cols = min(5, num_images)  # Define max columns to keep layout balanced
    rows = int(np.ceil(num_images / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))

    # Flatten axes if needed
    axes = np.array(axes).reshape(-1)

    for ax, img_path in zip(axes, image_paths):
        img = Image.open(img_path)
        ax.imshow(img)
        ax.axis("off")
        ax.set_title(os.path.basename(img_path).split("_")[0])  

    # Hide empty subplots
    for ax in axes[len(image_paths):]:
        ax.axis("off")

    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    if show:
        plt.show()

    if save_to:
        plt.savefig(save_to, dpi=300, bbox_inches='tight')
        print(f"Image grid saved to {save_to}")


