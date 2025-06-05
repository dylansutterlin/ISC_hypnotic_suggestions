import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import rankdata
from nilearn import datasets
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

def plot_isc_median_with_significance(isc_median, p_values,atlas, atlas_labels, p_threshold=0.01, 
                                      significant_color='red', nonsignificant_color='gray',coords_bool_mask=None, 
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

    roi_coords = find_parcellation_cut_coorcoordsds(labels_img=atlas)
    if coords_bool_mask is not None:
        roi_coords = roi_coords[coords_bool_mask]
    
    # Highlight significant ROIs
    significant_labels = [label if sig else " " for label, sig in zip(atlas_labels, sig_mask)]
    bar_colors = [significant_color if sig else nonsignificant_color for sig in sig_mask]

    sig_coords = np.array(roi_coords)[sig_mask]

    sig_df = pd.DataFrame({
        'ROI': [label for label, sig in zip(atlas_labels, sig_mask) if sig],
        'ISC': isc_median[sig_mask],
        'p_value': p_values[sig_mask],
        'Coordinates': [tuple(np.round(coord, 2)) for coord in sig_coords]
    })

    # Create the bar plot
    plt.figure(figsize=(14, 7))
    plt.bar(range(len(isc_median)), isc_median, color=bar_colors, alpha=0.8)
    plt.axhline(0, color='black', linestyle='--', linewidth=0.8)
    plt.xticks(range(len(isc_median)), significant_labels, rotation=90, fontsize=8)
    plt.xlabel("ROIs", fontsize=12)
    plt.ylabel("Median ISC", fontsize=12)
    plt.title(f"Median ISC Values (Significant regions in red, p < {np.round(p_threshold,2)})", fontsize=14)
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

    return sig_mask, sig_df 

import nibabel as nib
import numpy as np
import os
from nilearn.plotting import plot_stat_map

def project_isc_to_brain(atlas_img, isc_median, atlas_labels,roi_coords, p_values=None, p_threshold=0.01,title = '"ISC Median Values (Thresholded)', coords_bool_mask = None, color = 'Reds',save_path=None, show=True, cut_coords_plot = None, display_mode = 'z'):
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
        **assumes start at 1, 0 is background
    p_values : np.ndarray, optional
        P-values corresponding to the ISC values. Default is None.
    p_threshold : float, optional
        Threshold for significance of p-values. Default is 0.01.
    save_path : str, optional
        Path to save the projected ISC map. Default is None.
    show : bool, optional
        Whether to display the plot. Default is True.
    """
    
    bg_mni =  datasets.load_mni152_template(resolution=1)

    # atlas_img = nib.load(atlas_path)
    # atlas_data = atlas_img.get_fdata()

    labels = list(atlas_labels.values())

    # atlas_data = atlas_img.get_fdata()
    atlas_data = np.rint(atlas_img.get_fdata()).astype(int)

    if len(roi_coords) != len(labels):
        roi_coords = roi_coords[coords_bool_mask]
        raise ValueError("Mismatch between number of ROI coordinates and labels.")
        
    # Create an empty volume to store ISC values
    isc_vol = np.zeros_like(atlas_data, dtype=float)
    non_sig_max_isc = []
    sig_isc_values = []
    sig_labels_data = []

    # Assign ISC values to corresponding atlas regions
    for (roi_idx, label) in atlas_labels.items():

        i = int(roi_idx) - 1  # Adjust for zero-based indexing

        roi_mask = atlas_data ==int(roi_idx) #+1 !!! ok pre Sensaas
        if p_values is not None and p_values[i] < p_threshold:
            # print('Sig ROI', label, p_values[i], isc_median[i])
            # Assign significant ISC value
            isc_vol[roi_mask] = isc_median[i]
            sig_isc_values.append(isc_median[i])

            sig_labels_data.append({
                "ROI": label,
                "ISC": round(isc_median[i], 2),
                "p-value": round(p_values[i], 2),
                "Coordinates": tuple(np.round(roi_coords[i], 2))
            })
            
        else:
            # Track the highest ISC value for non-significant ROIs
            non_sig_max_isc.append(isc_median[i])

    isc_img = nib.Nifti1Image(isc_vol, atlas_img.affine, atlas_img.header)
    
    # max_isc_thresh = np.max(np.array(non_sig_max_isc))
    if len(sig_isc_values) > 0:
        min_sig = np.min(np.array(sig_isc_values)) -0.01
    else : min_sig = 0.0001

    # Save the ISC map if save_path is provided
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        nib.save(isc_img, save_path)
        print(f"ISC projection saved to {save_path}")

    max_isc = np.max(isc_vol)
    print(f"Max ISC value: {max_isc}")

    if max_isc == 0 :
        print(f'Max non significant ISC value: {np.max(isc_median)}')

    if show:
        plot_stat_map(
            isc_img,
            bg_img=bg_mni,
            title=title,
            threshold=min_sig,
            vmax=max_isc,
            black_bg=False,
            colorbar=True,
            display_mode=display_mode,
            cut_coords=cut_coords_plot,
            cmap=color,
            draw_cross=False
        )

    return isc_img, min_sig, pd.DataFrame(sig_labels_data)


def project_isc_to_brain_perm(atlas_img, isc_median, atlas_labels,roi_coords, p_values=None, p_threshold=0.01,title = '"ISC Median Values (Thresholded)',color='seismic', save_path=None, show=True, display_mode = 'z',cut_coords_plot = None):
    """
    Projects ISC values to brain space and optionally thresholds by significance.

    Parameters
    ----------
    atlas_path : str
        Path to the atlas file used for analysis.
    isc_median : np.ndarray
        Array of median ISC values for each ROI.
    atlas_labels : dict 
        ROI label (integer) : label (string) mapping.
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
    bg_mni =  datasets.load_mni152_template(resolution=1)
    # atlas_img = nib.load(atlas_path)
    atlas_data = atlas_img.get_fdata()
    # if coords is not None:
    #     roi_coords = list(zip(
    #     coords['Xmm'].astype(float),
    #     coords['Ymm'].astype(float),
    #     coords['Zmm'].astype(float)
    #     ))
    # else:
    #     roi_coords = find_parcellation_cut_coords(labels_img=atlas_img)

    # Create an empty volume to store ISC values
    isc_vol = np.zeros_like(atlas_data, dtype=float)
    img_unthresholded = np.zeros_like(atlas_data, dtype=float)

    sig_labels_data = []
    sig_min_isc = []

    # Assign ISC values to corresponding atlas regions
    for (roi_idx, label) in atlas_labels.items():

        i = int(roi_idx) - 1  # Adjust for zero-based indexing
        roi_mask = atlas_data ==int(roi_idx)

        if isc_median[i] == 0:
            isc_median[i] = 0.0000

        if p_values is not None and p_values[i] <= p_threshold:
            # Assign significant ISC value
            isc_vol[roi_mask] = isc_median[i]
            sig_min_isc.append(isc_median[i])
            sig_labels_data.append({
                "ROI": roi_idx,
                "Atlas Label": label,
                "r Difference": round(isc_median[i], 4),
                "p-value": f"{p_values[i]:.5f}",
                "Coordinates": tuple(np.round(roi_coords[i], 0).astype(int))
            })

        img_unthresholded[roi_mask] = np.arctanh(isc_median[i])

    # Create a Nifti image for the ISC projection
    isc_img = nib.Nifti1Image(isc_vol, atlas_img.affine, atlas_img.header)
    z_img_unthresholded = nib.Nifti1Image(img_unthresholded, atlas_img.affine, atlas_img.header)

    if len(sig_min_isc) > 0:
        min_sig_thresh = np.min(np.abs(sig_min_isc)) -0.01
    else:
        min_sig_thresh = 0.0001

    # Save the ISC map if save_path is provided
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        nib.save(isc_img, save_path)
        print(f"ISC projection saved to {save_path}")

    max_diff = np.max(np.abs(isc_vol))
    sign_max = np.max(isc_vol) if np.max(isc_vol) > 0 else np.min(isc_vol)
    print(f"Max ISC abs diff: {max_diff} and sign : {sign_max}")

    if max_diff == 0 :
        print(f'Max non significant ISC value: {np.max(isc_median)}')

    
    if show:
        # plot_stat_map(isc_img, title=title, threshold=min_sig_thresh, vmax=max_diff, 
        #               colorbar=True, display_mode='x', cmap=color, cut_coords=6, draw_cross=False)
        plot_stat_map(
            isc_img,
            bg_img=bg_mni,
            title=title,
            threshold=min_sig_thresh,
            vmax=max_diff,
            black_bg=False,
            colorbar=True,
            display_mode=display_mode,
            cut_coords=cut_coords_plot,
            cmap=color,
            draw_cross=False
        )
    return isc_img, min_sig_thresh, pd.DataFrame(sig_labels_data), z_img_unthresholded


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
        network =  label.split('_')[2]
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
    
    plt.axhline(0, linestyle='--', color='gray', lw=1)
    plt.axvline(0, linestyle='--', color='gray', lw=1)

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
    if len(image_paths) < 9:
        max_col = 4
    else : max_col = 5

    cols = min(max_col, num_images)  # Define max columns to keep layout balanced
    rows = int(np.ceil(num_images / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 6, rows * 6))

    # Flatten axes if needed
    axes = np.array(axes).reshape(-1)

    for ax, img_path in zip(axes, image_paths):
        img = Image.open(img_path)
        ax.imshow(img)
        ax.axis("off")
        ax.set_title("_".join(os.path.basename(img_path).split("_")[:-1]))  

    # Hide empty subplots
    for ax in axes[len(image_paths):]:
        ax.axis("off")

    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    
    if save_to:
        plt.savefig(save_to, dpi=300, bbox_inches='tight')
        print(f"Image grid saved to {save_to}")

    if show:
        plt.show()

    
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import linregress
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import linregress

def jointplot(x, y, x_label="X", y_label="Y", 
              title=None, default_color=[0.2, 0.5, 1], 
              density_color_x='#4c80ff', density_color_y='#fd6262'):
    """
    Create a jointplot with regression line for X and Y with KDE marginals.

    Parameters:
    -----------
    x : array-like
        Predictor variable.
    y : array-like
        Outcome variable.
    x_label : str
        Label for the x-axis.
    y_label : str
        Label for the y-axis.
    title : str or None
        Title of the plot.
    default_color : list
        RGB color used for plotting points.
    density_color_x : str
        Color code for the X-axis marginal KDE distribution.
    density_color_y : str
        Color code for the Y-axis marginal KDE distribution.

    Returns:
    --------
    g : seaborn.axisgrid.JointGrid
        The seaborn jointplot object.
    """
    #ensure np arrays
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    # Handle NaNs
    valid_mask = ~np.isnan(x) & ~np.isnan(y)

    x_valid = x[valid_mask]
    y_valid = y[valid_mask]

    # Compute correlation
    slope, intercept, r_value, p_value, _ = linregress(x_valid, y_valid)
    corr_label = f'r = {r_value:.2f}, p = {p_value:.3f}'

    # Determine consistent limits
    x_margin = (x_valid.max() - x_valid.min()) * 0.05
    y_margin = (y_valid.max() - y_valid.min()) * 0.05
    xlim = (x_valid.min() - x_margin, x_valid.max() + x_margin)
    ylim = (y_valid.min() - y_margin, y_valid.max() + y_margin)

    # Create JointGrid with fixed axis limits
    g = sns.JointGrid(x=x_valid, y=y_valid, height=8, xlim=xlim, ylim=ylim)

    # Scatter plot
    g.ax_joint.scatter(x_valid, y_valid, alpha=0.7, s=50, edgecolor='black', color=default_color)
    g.ax_joint.tick_params(axis='both', which='major', labelsize=22)


    # Add regression line
    sns.regplot(x=x_valid, y=y_valid, scatter=False, ax=g.ax_joint,
                line_kws={'color': 'black', 'linewidth': 5})

    # KDE marginal distributions with separate colors
    sns.kdeplot(x=x_valid, ax=g.ax_marg_x, fill=True, color=density_color_x, alpha=0.5)
    sns.kdeplot(y=y_valid, ax=g.ax_marg_y, fill=True, color=density_color_y, alpha=0.5)

    # Add correlation text with larger box
    g.ax_joint.text(0.05, 0.95, corr_label, transform=g.ax_joint.transAxes,
                    fontsize=25, verticalalignment='top', 
                    bbox=dict(boxstyle="round,pad=0.48", alpha=0.5, facecolor = 'lightgray'))

    # Add labels and title
    g.set_axis_labels(x_label, y_label, fontsize=30)
    if title:
        g.fig.suptitle(title, fontsize=30, y=0.98)

    plt.tight_layout()
    plt.show()

    return g


# =============
# RADAR CHART
#==============
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter

def plot_network_radar(network_list, title="Network Distribution of Significant ROIs"):
    """
    Plot a radar chart with discrete scale and fewer radial labels for clarity.

    Parameters
    ----------
    network_list : list of str
        List of network names (e.g., ['Default', 'Limbic', ...]).
    title : str
        Title for the radar chart.
    """
    counts = Counter(network_list)

    network_order = ['Visual', 'SomMot', 'DorsAttn', 'SalVentAttn', 'Limbic',
                     'Cont', 'Default', 'TempPar', 'Subcortical', 'Cerebellum']
    
    values = [counts.get(n, 0) for n in network_order]
    max_val = max(values) if max(values) > 0 else 1

    labels = network_order
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    values += values[:1]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    ax.plot(angles, values, color='tab:blue', linewidth=2)
    ax.fill(angles, values, color='tab:blue', alpha=0.25)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=10)

    yticks = list(range(1, max_val + 1))
    ax.set_yticks(yticks)

    # Label every 2nd radial line (e.g., 2, 4, 6)
    ytick_labels = [str(y) if y % 2 == 0 else "" for y in yticks]
    ax.set_yticklabels(ytick_labels, fontsize=9)
    ax.set_ylim(0, max_val)

    ax.yaxis.grid(True, linestyle='dotted')
    ax.xaxis.grid(True, linestyle='dotted')

    plt.title(title, fontsize=13)
    plt.tight_layout()
    plt.show()


#for multiple networks/conditions

def plot_overlay_network_radar(sig_dfs_dict, full_cond_names, title="Overlay Network Distribution"):
    network_order = ['Visual', 'SomMot', 'DorsAttn', 'SalVentAttn', 'Limbic', 'Cont', 'Default', 'TempPar', 'Subcortical', 'Cerebellum']
    all_used_networks = set()
    for df in sig_dfs_dict.values():
        nets = [roi.split('_')[2] for roi in df['ROI']]
        all_used_networks.update(nets)
    network_order = [n for n in network_order if n in all_used_networks]

    angles = np.linspace(0, 2 * np.pi, len(network_order), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    colors = plt.cm.tab10.colors

    for i, (cond, full_cond) in enumerate(full_cond_names.items()):
        df = sig_dfs_dict[cond]
        nets = [roi.split('_')[2] for roi in df['ROI']]
        counts = Counter(nets)
        values = [counts.get(n, 0) for n in network_order]
        values += values[:1]

        ax.plot(angles, values, label=full_cond, linewidth=2, color=colors[i % len(colors)])
        ax.fill(angles, values, alpha=0.25, color=colors[i % len(colors)])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(network_order, fontsize=12)
    max_val = max([max(Counter([roi.split('_')[2] for roi in df['ROI']]).values()) for df in sig_dfs_dict.values()])
    yticks = list(range(1, max_val + 1))
    ax.set_yticks(yticks)
    ax.set_yticklabels([str(y) if y % 2 == 0 else '' for y in yticks], fontsize=10)
    ax.set_ylim(0, max_val)

    ax.yaxis.grid(True, linestyle='dotted')
    ax.xaxis.grid(True, linestyle='dotted')
    legend = plt.legend(loc='upper right', bbox_to_anchor=(1.1, 1.1), fontsize=12)
    plt.setp(legend.get_texts(), fontsize=12)
    plt.title(title, fontsize=14)
    plt.tight_layout()
    plt.show()

def plot_median_isc_dots(sig_dfs_one_sample, 
                          title="Intersubject Correlation (ISC) per Condition", 
                          conditions_full_names=None,
                          jitter=True,
                          jitter_scale=0.08,
                          y_spacing=1.3):
    """
    Plot ISC values as dots for each significant ROI per condition,
    with median and MAD error bars, condition color coding, and ROI counts.

    Parameters
    ----------
    sig_dfs_one_sample : dict
        Dictionary of condition -> DataFrame with 'ISC' column and 'ROI'.
    title : str
        Title for the plot.
    conditions_full_names : dict or None
        Mapping from short to full condition labels.
    jitter : bool
        Whether to jitter y-axis to separate overlapping points.
    jitter_scale : float
        Amount of vertical jitter.
    y_spacing : float
        Vertical spacing between condition rows.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import matplotlib.patches as patches

    # Color per condition
    color_map = {
        'ANA': '#1f77b4',
        'HYPER': '#d62728',
        'neutral': '#2ca02c',
        'modulation': '#9467bd',
        'all_sugg': '#ff7f0e',
        'NANA': '#17becf',
        'NHYPER': '#e377c2',
    }

    fig, ax = plt.subplots(figsize=(11, 7))
    yticks = []
    ylabels = []

    for i, (cond, df) in enumerate(sig_dfs_one_sample.items()):
        if df.empty:
            continue

        y_base = i * y_spacing
        yticks.append(y_base)
        full_label = conditions_full_names.get(cond, cond) if conditions_full_names else cond
        ylabels.append(full_label)

        isc_vals = df['ISC'].values
        n = len(isc_vals)

        # Jitter for dot spread
        y_vals = y_base + np.random.uniform(-jitter_scale, jitter_scale, size=n) if jitter else np.full(n, y_base)

        color = color_map.get(cond, f"C{i}")

        # Plot dots
        ax.scatter(isc_vals, y_vals, s=85, alpha=0.9, color=color, edgecolor='black', linewidth=0.4)

        # Compute median and MAD
        med = np.median(isc_vals)
        mad = np.median(np.abs(isc_vals - med))

        # Add box
        box_height = 0.35
        box = patches.FancyBboxPatch(
            (med - mad, y_base - box_height / 2),
            width=2 * mad,
            height=box_height,
            boxstyle="round,pad=0.02",
            linewidth=2.8,
            edgecolor='black',
            facecolor=color,
            alpha=0.25
        )
        ax.add_patch(box)

        # Add median line inside box
        ax.plot([med, med], [y_base - box_height / 2, y_base + box_height / 2],
                color='black', linewidth=2.5)

    # Final layout
    ax.set_yticks(yticks)
    ax.set_yticklabels(ylabels, fontsize=18)
    ax.set_xlabel('Median ISC per region', fontsize=24)
    ax.set_title(title, fontsize=26)
    ax.tick_params(axis='x', labelsize=18)
    ax.tick_params(axis='y', labelsize=18)
    ax.set_ylim(-y_spacing, yticks[-1] + y_spacing)
    ax.grid(axis='x', linestyle='dotted', alpha=0.5)
    plt.tight_layout()
    plt.show()
