import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import rankdata
from nilearn import datasets
from nilearn.maskers import NiftiLabelsMasker, NiftiMapsMasker
import os
import nibabel as nib
import pandas as pd
from nilearn.plotting import find_parcellation_cut_coords


def load_isc_results(isc_results):
    isc = isc_results["isc"]
    observed_isc = isc_results["observed"]
    p_values = isc_results["p_values"]
    ci = isc_results["confidence_intervals"]
    distributions = isc_results["distribution"]
    median_isc = isc_results["median_isc"]

    return isc, observed_isc, p_values, ci, median_isc, distributions


def load_isc_results_pairwise(isc_results):
    isc = isc_results["isc"]
    observed_isc = isc_results["observed"]
    p_values = isc_results["p_values"]
    ci = isc_results["confidence_intervals"]
    distributions = isc_results["distribution"]

    return isc, observed_isc, p_values, ci, distributions


def load_isc_results_permutation(isc_results):

    observed_isc = isc_results["observed"]
    p_values = isc_results["p_value"]
    distributions = isc_results["distribution"]
    return observed_isc, p_values, distributions


def plot_isc_distributions(
    observed_isc,
    p_values,
    median_isc,
    bootstrap_distributions,
    save_to=None,
    title="ISC Distributions",
):
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
        ax.hist(data, bins=20, alpha=0.7, color="blue", label="Observed ISC")

        # Plot the bootstrap distribution
        ax.hist(
            bootstrap_data,
            bins=30,
            alpha=0.4,
            color="green",
            label="Bootstrap Distribution",
            density=True,
        )

        # Add the median line
        ax.axvline(
            median_isc[i],
            color="red",
            linestyle="--",
            label=f"Median ISC = {median_isc[i]:.4f}",
        )

        # Annotate the p-value
        ax.text(
            0.05,
            0.95,
            f"p = {p_values[i]:.4f}",
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )
        fontsize = 20
        # Labels and formatting
        ax.set_xlabel("ISC Values", fontsize=fontsize)
        ax.set_ylabel("Frequency", fontsize=fontsize)
        ax.set_title(f"ROI : {col_names[i]}", fontsize=fontsize)
        ax.legend()

        if save_to is not None:
            plt.savefig(save_to, bbox_inches="tight", dpi=500)
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


def heatmap_pairwise_isc_combined(
    isc_df,
    subjects,
    behavioral_scores=None,
    roi_to_plot="all",
    save_to=None,
    show=True,
    title="Pairwise ISC Matrices",
):
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

    fig, axes = plt.subplots(1, n_rois - 1, figsize=(5 * n_rois, 5))
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
            cmap="coolwarm",
            square=True,
            cbar_kws={"label": "ISC"},
            ax=ax,
            linewidths=0.5,
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
    isc_img_path = os.path.join(
        results_dir, condition, f"isc_val_{condition}_boot5000_pariwiseFalse.nii.gz"
    )
    pval_img_path = os.path.join(
        results_dir, condition, f"p_values_{condition}_boot5000_pairwiseFalse.nii.gz"
    )

    isc_img = nib.load(isc_img_path)
    pval_img = nib.load(pval_img_path)

    return isc_img, pval_img


def load_perm_images(results_dir, condition):
    isc_img_path = os.path.join(
        results_dir, condition, f"isc_val_{condition}_boot5000_pariwiseFalse.nii.gz"
    )
    pval_img_path = os.path.join(
        results_dir, condition, f"p_values_{condition}_boot5000_pairwiseFalse.nii.gz"
    )

    isc_img = nib.load(isc_img_path)
    pval_img = nib.load(pval_img_path)

    return isc_img, pval_img


def load_difumo():
    project_dir = project_dir = (
        "/data/rainville/dSutterlin/projects/ISC_hypnotic_suggestions"
    )
    atlas_path = os.path.join(project_dir, "masks/DiFuMo256/3mm/maps.nii.gz")
    atlas_dict_path = os.path.join(
        project_dir, "masks/DiFuMo256/labels_256_dictionary.csv"
    )
    atlas = nib.load(atlas_path)
    atlas_df = pd.read_csv(atlas_dict_path)
    atlas_labels = atlas_df["Difumo_names"]
    atlas_name = "Difumo256"  # !!!!!!! 'Difumo256'

    masker = NiftiMapsMasker(
        maps_img=atlas, standardize=True, memory="nilearn_cache", verbose=5
    )
    return atlas, masker.fit(), atlas_df, atlas_labels


import os
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.stats.multitest import multipletests


def plot_isc_median_with_significance(
    isc_median,
    p_values,
    atlas,
    atlas_labels,
    p_threshold=0.01,
    significant_color="red",
    nonsignificant_color="gray",
    coords_bool_mask=None,
    save_path=None,
    show=False,
    fdr_correction=False,
):
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
        _, corrected_p_values, _, _ = multipletests(
            p_values, alpha=p_threshold, method="fdr_bh"
        )
        sig_mask = corrected_p_values < p_threshold
    else:
        sig_mask = p_values < p_threshold

    roi_coords = find_parcellation_cut_coorcoordsds(labels_img=atlas)
    if coords_bool_mask is not None:
        roi_coords = roi_coords[coords_bool_mask]

    # Highlight significant ROIs
    significant_labels = [
        label if sig else " " for label, sig in zip(atlas_labels, sig_mask)
    ]
    bar_colors = [
        significant_color if sig else nonsignificant_color for sig in sig_mask
    ]

    sig_coords = np.array(roi_coords)[sig_mask]

    sig_df = pd.DataFrame(
        {
            "ROI": [label for label, sig in zip(atlas_labels, sig_mask) if sig],
            "ISC": isc_median[sig_mask],
            "p_value": p_values[sig_mask],
            "Coordinates": [tuple(np.round(coord, 2)) for coord in sig_coords],
        }
    )

    # Create the bar plot
    plt.figure(figsize=(14, 7))
    plt.bar(range(len(isc_median)), isc_median, color=bar_colors, alpha=0.8)
    plt.axhline(0, color="black", linestyle="--", linewidth=0.8)
    plt.xticks(range(len(isc_median)), significant_labels, rotation=90, fontsize=8)
    plt.xlabel("ROIs", fontsize=12)
    plt.ylabel("Median ISC", fontsize=12)
    plt.title(
        f"Median ISC Values (Significant regions in red, p < {np.round(p_threshold,2)})",
        fontsize=14,
    )
    plt.tight_layout()

    # Add histogram of p-values in the corner
    inset_ax = plt.gcf().add_axes(
        [0.75, 0.75, 0.2, 0.2]
    )  # x, y, width, height in figure coordinates
    inset_ax.hist(p_values, bins=20, color="blue", alpha=0.7)
    inset_ax.axvline(p_threshold, color="red", linestyle="--", linewidth=0.8)
    inset_ax.set_title("P-value Distribution", fontsize=10)
    inset_ax.set_xlabel("P-value", fontsize=8)
    inset_ax.set_ylabel("Count", fontsize=8)
    inset_ax.tick_params(axis="both", which="major", labelsize=8)

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


def project_isc_to_brain(
    atlas_img,
    isc_median,
    atlas_labels,
    roi_coords,
    p_values=None,
    p_threshold=0.01,
    title='"ISC Median Values (Thresholded)',
    coords_bool_mask=None,
    color="Reds",
    save_path=None,
    show=True,
    save_fig_as=None,
    cut_coords_plot=None,
    display_mode="z",
    CI = None
):
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

    bg_mni = datasets.load_mni152_template(resolution=1)

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
    img_unthresholded = np.zeros_like(atlas_data, dtype=float)
    non_sig_max_isc = []
    sig_isc_values = []
    sig_labels_data = []
    
    # Assign ISC values to corresponding atlas regions
    for roi_idx, label in atlas_labels.items():

        i = int(roi_idx) - 1  # Adjust for zero-based indexing

        roi_mask = atlas_data == int(roi_idx)  # +1 !!! ok pre Sensaas
        if p_values is not None and p_values[i] < p_threshold:
            # print('Sig ROI', label, p_values[i], isc_median[i])
            # Assign significant ISC value
            isc_vol[roi_mask] = isc_median[i]
            sig_isc_values.append(isc_median[i])

            sig_labels_data.append(
                {
                    "ROI": roi_idx,
                    "Label": label,
                    "ISC": round(isc_median[i], 2),
                    "p-value": round(p_values[i], 4),
                    "Coordinates": tuple(np.round(roi_coords[i], 0).astype(int)),
                }
            )
    
        else:
            # Track the highest ISC value for non-significant ROIs
            non_sig_max_isc.append(isc_median[i])

        # img_unthresholded[roi_mask] = np.arctanh(isc_median[i]) #r to z transfrom
        img_unthresholded[roi_mask] = isc_median[i] #r to z transfrom

    isc_img = nib.Nifti1Image(isc_vol, atlas_img.affine, atlas_img.header)
    z_img_unthresholded = nib.Nifti1Image(
        img_unthresholded, atlas_img.affine, atlas_img.header
    )
    # max_isc_thresh = np.max(np.array(non_sig_max_isc))
    if len(sig_isc_values) > 0:
        min_sig = np.min(np.array(sig_isc_values)) - 0.01
    else:
        min_sig = 0.0001

    # Save the ISC map if save_path is provided
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        nib.save(isc_img, save_path)
        print(f"ISC projection saved to {save_path}")

    max_isc = np.max(isc_vol)
    print(f"Max ISC value: {max_isc}")

    if max_isc == 0:
        print(f"Max non significant ISC value: {np.max(isc_median)}")

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
            draw_cross=False,
        )
        if save_fig_as is not None:
            plt.savefig(save_fig_as, dpi=1000, bbox_inches="tight")
            print(f"ISC projection plot saved to {save_fig_as}")
        plt.show()

    return isc_img, min_sig, pd.DataFrame(sig_labels_data), z_img_unthresholded


def project_isc_to_brain_perm(
    atlas_img,
    isc_median,
    atlas_labels,
    roi_coords,
    p_values=None,
    p_threshold=0.01,
    title='"ISC Median Values (Thresholded)',
    color="seismic",
    save_path=None,
    save_fig_as=None,
    show=True,
    display_mode="z",
    cut_coords_plot=None,
):
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
    bg_mni = datasets.load_mni152_template(resolution=1)
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
    for i, (roi_id, label) in enumerate(atlas_labels.items()):

        # i = int(roi_idx) - 1  # Adjust for zero-based indexing
        roi_mask = atlas_data == int(roi_id)

        if isc_median[i] == 0:
            isc_median[i] = 0.0000

        if p_values is not None and p_values[i] <= p_threshold:
            # Assign significant ISC value
            isc_vol[roi_mask] = isc_median[i]
            sig_min_isc.append(isc_median[i])
            sig_labels_data.append(
                {
                    "ROI": roi_id,
                    "Label": label,
                    "r Difference": round(isc_median[i], 2),
                    "p-value": f"{p_values[i]:.4f}",
                    "Coordinates": tuple(np.round(roi_coords[i], 0).astype(int)),
                }
            )

        img_unthresholded[roi_mask] = np.arctanh(isc_median[i])

    # Create a Nifti image for the ISC projection
    isc_img = nib.Nifti1Image(isc_vol, atlas_img.affine, atlas_img.header)
    z_img_unthresholded = nib.Nifti1Image(
        img_unthresholded, atlas_img.affine, atlas_img.header
    )

    if len(sig_min_isc) > 0:
        min_sig_thresh = np.min(np.abs(sig_min_isc)) - 0.01
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

    if max_diff == 0:
        print(f"Max non significant ISC value: {np.max(isc_median)}")

    if show:
        # plot_stat_map(isc_img, title=title, threshold=min_sig_thresh, vmax=max_diff,
        #               colorbar=True, display_mode='x', cmap=color, cut_coords=6, draw_cross=False)
        im = plot_stat_map(
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
            draw_cross=False,
        )
        if save_fig_as is not None:
            plt.savefig(save_fig_as, dpi=1000, bbox_inches="tight")
            print(f"ISC projection plot saved to {save_fig_as}")
        plt.show()

    return isc_img, min_sig_thresh, pd.DataFrame(sig_labels_data), z_img_unthresholded


import seaborn as sns
import matplotlib.pyplot as plt


def plot_similarity_and_histogram(
    similarity_matrix, correlations, p_values, atlas_labels, behav_name, save_path=None
):
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
    sns.heatmap(similarity_matrix, annot=False, cmap="coolwarm", ax=axes[0])
    axes[0].set_title(f"Behavioral Similarity Matrix ({behav_name})", fontsize=14)
    axes[0].set_xlabel("Subjects", fontsize=12)
    axes[0].set_ylabel("Subjects", fontsize=12)

    # Plot the correlation histogram
    axes[1].hist(correlations, bins=20, color="blue", alpha=0.7, edgecolor="black")
    axes[1].set_title(f"RSA Correlations Distribution ({behav_name})", fontsize=14)
    axes[1].set_xlabel("Correlation Values", fontsize=12)
    axes[1].set_ylabel("Frequency", fontsize=12)

    # Add a line for zero correlation
    axes[1].axvline(
        0, color="red", linestyle="--", linewidth=1, label="Zero Correlation"
    )
    axes[1].legend(fontsize=10)

    plt.tight_layout()

    # Save the plot if save_path is provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Plots saved to {save_path}")

    plt.show()


def vector_to_symmetric_matrix(vec, size):

    mat = np.zeros((size, size))
    triu_indices = np.triu_indices(size, k=1)
    mat[triu_indices] = vec
    mat = mat + mat.T

    return mat


# =====================
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
        network = label.split("_")[2]
        labels_by_index.append(network)

        if network not in network_mapping:
            network_mapping[network] = []
        network_mapping[network].append(idx)

    return network_mapping, labels_by_index


import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.cm as cm
import matplotlib.colors as mcolors


def plot_scatter_legend(
    correl1,
    correl2,
    var_name=["var1", "var2"],
    grp_id=None,
    legend=True,
    title=None,
    save_path=None,
):
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
        colors = cm.get_cmap("tab20", num_grps).colors
        color_map = {group: colors[i] for i, group in enumerate(unique_groups)}

        # Plot each group with a unique color
        for group in unique_groups:
            mask = grp_id == group
            plt.scatter(
                correl1[mask],
                correl2[mask],
                label=group,
                color=color_map[group],
                alpha=0.7,
                edgecolor="k",
            )
    else:
        plt.scatter(correl1, correl2, color="blue", alpha=0.7, edgecolor="k")

    # Add diagonal line
    max_val = max(np.max(correl1), np.max(correl2))
    min_val = min(np.min(correl1), np.min(correl2))

    plt.plot(
        [min_val, max_val],
        [min_val, max_val],
        linestyle="--",
        color="black",
        linewidth=1,
    )

    plt.axhline(0, linestyle="--", color="gray", lw=1)
    plt.axvline(0, linestyle="--", color="gray", lw=1)

    # Add labels and title
    plt.xlabel(f"{var_name[0]}", fontsize=14)
    plt.ylabel(f"{var_name[1]}", fontsize=14)
    plt.title(title if title else "scatter plot", fontsize=16)

    if legend and grp_id is not None:
        plt.legend(title="Yeo Networks", loc="best", fontsize=10, title_fontsize=12)

    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
        print(f"Plot saved to {save_path}")
    else:
        plt.show()


# Function to plot images in a grid
from PIL import Image


def plot_images_grid(image_paths, title, save_to=False, show=True):
    """Plots images in a flexible grid layout."""
    num_images = len(image_paths)
    if len(image_paths) < 9:
        max_col = 4
    else:
        max_col = 5

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
    for ax in axes[len(image_paths) :]:
        ax.axis("off")

    plt.suptitle(title, fontsize=16)
    plt.tight_layout()

    if save_to:
        plt.savefig(save_to, dpi=300, bbox_inches="tight")
        print(f"Image grid saved to {save_to}")

    if show:
        plt.show()


def plot_roi_timeseries(
    timeseries,
    region_label="ROI",
    plot_mean=True,
    save_to=None,
    show=True,
    mean_label="Mean",
    subject_scores=None,
    score_label="Score",
    colormap="viridis",
):
    """
    Plots each subject's timeseries and the mean timeseries for a given region.
    Optionally colors each subject by an external score and adds a colorbar.

    Parameters
    ----------
    timeseries : np.ndarray
        Array of shape (datapoints, subjects) containing the timeseries data for one region.
    region_label : str, optional
        Label for the region (used in the plot title). Default is "ROI".
    plot_mean : bool, optional
        Whether to plot the mean timeseries on top. Default is True.
    save_to : str or None, optional
        If provided, path to save the figure as PNG. Default is None.
    show : bool, optional
        Whether to display the plot. Default is True.
    mean_label : str, optional
        Legend label for the mean signal. Default is "Mean".
    subject_scores : array-like or None, optional
        If provided, array of shape (subjects,) with external scores to color lines.
    score_label : str, optional
        Label for the colorbar if subject_scores is provided.
    colormap : str, optional
        Name of the matplotlib colormap to use for subject_scores. Default is "viridis".
    """
    import matplotlib.pyplot as plt

    n_timepoints, n_subjects = timeseries.shape

    plt.figure(figsize=(15, 5))
    ax = plt.gca()

    if subject_scores is not None:
        # Normalize scores for colormap
        subject_scores = np.asarray(subject_scores)
        assert (
            subject_scores.shape[0] == n_subjects
        ), "subject_scores must match number of subjects"
        norm = plt.Normalize(np.min(subject_scores), np.max(subject_scores))
        cmap = plt.get_cmap(colormap)
        colors = cmap(norm(subject_scores))
        for subj in range(n_subjects):
            ax.plot(timeseries[:, subj], color=colors[subj], alpha=0.85, linewidth=1.5)
        # Add colorbar
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, pad=0.02)
        cbar.set_label(score_label, fontsize=12)
    else:
        # Use a different color for each subject
        cmap = plt.get_cmap("tab10") if n_subjects <= 10 else plt.get_cmap("tab20")
        for subj in range(n_subjects):
            color = cmap(subj % cmap.N)
            ax.plot(timeseries[:, subj], color=color, alpha=0.85, linewidth=1.5)

    # Plot mean timeseries if requested
    if plot_mean:
        mean_signal = np.mean(timeseries, axis=1)
        ax.plot(mean_signal, color="red", linewidth=3.5, label=mean_label, zorder=10)

    ax.set_title(f"Timecourse for {region_label}")
    ax.set_xlabel("Timepoints")
    ax.set_ylabel("Signal")
    if plot_mean:
        ax.legend()
    plt.tight_layout()

    if save_to is not None:
        plt.savefig(save_to, dpi=300)
        print(f"[plot_roi_timeseries] Saved plot to {save_to}")

    if show:
        plt.show()
    else:
        plt.close()


# BEHAVIORAL
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import linregress
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import linregress

def jointplot(
    x,
    y,
    x_label="X",
    y_label="Y",
    title=None,
    default_color=[0.2, 0.5, 1],
    density_color_x="#4c80ff",
    density_color_y="#fd6262",
    alpha=1,
    plot_w=10,
    plot_h=11,
    dash_horz=None,
    dash_vert=None,
):
    import numpy as np
    import seaborn as sns
    import matplotlib.pyplot as plt
    from scipy.stats import linregress

    # Clean inputs
    valid_mask = ~np.isnan(x) & ~np.isnan(y)
    x_valid = x[valid_mask]
    y_valid = y[valid_mask]

    # Compute correlation
    slope, intercept, r_value, p_value, _ = linregress(x_valid, y_valid)
    if p_value < 0.001:
        corr_label = f"r = {r_value:.2f}***"
    elif p_value < 0.01:
        corr_label = f"r = {r_value:.2f}**"
    elif p_value < 0.05:
        corr_label = f"r = {r_value:.2f}*"
    else:
        corr_label = f"r = {r_value:.2f}, p = {p_value:.3f}"

    # Axis limits with margin
    x_margin = (x_valid.max() - x_valid.min()) * 0.05
    y_margin = (y_valid.max() - y_valid.min()) * 0.05
    xlim = (x_valid.min() - x_margin, x_valid.max() + x_margin)
    ylim = (y_valid.min() - y_valid.min()) * 0.05
    ylim = (y_valid.min() - y_margin, y_valid.max() + y_margin)

    # Create plot
    g = sns.JointGrid(x=x_valid, y=y_valid, height=10, xlim=xlim, ylim=ylim)
    g.fig.set_size_inches(plot_w, plot_h)

    # Main scatter plot
    g.ax_joint.scatter(
        x_valid,
        y_valid,
        alpha=0.7,
        s=200,
        edgecolor="black",
        linewidth=2,
        color=default_color,
    )

    # Regression line
    sns.regplot(
        x=x_valid,
        y=y_valid,
        scatter=False,
        ax=g.ax_joint,
        line_kws={"color": "black", "linewidth": 5},
    )

    # KDE marginals
    sns.kdeplot(x=x_valid, ax=g.ax_marg_x, fill=True, color=density_color_x, alpha=alpha)
    sns.kdeplot(y=y_valid, ax=g.ax_marg_y, fill=True, color=density_color_y, alpha=alpha)

    # Optional reference lines
    if dash_vert is not None:
        g.ax_joint.axvline(dash_vert, linestyle="--", color="black", linewidth=3)
        g.ax_joint.set_xlim(min(xlim[0], dash_vert), max(xlim[1], dash_vert))
    if dash_horz is not None:
        g.ax_joint.axhline(dash_horz, linestyle="--", color="black", linewidth=3)
        g.ax_joint.set_ylim(min(ylim[0], dash_horz), max(ylim[1], dash_horz))

    # Ticks on both sides
    g.ax_joint.tick_params(
        axis='both',
        which='major',
        labelsize=40,
        width=2.5,
        length=10,             # Length of the ticks
        direction='inout',     # Ticks inside and outside
    )

    # Thicker spines
    for spine in g.ax_joint.spines.values():
        spine.set_linewidth(3.5)

    # Correlation text position based on sign
    if r_value >= 0:
        text_x = 0.05
        text_y = 0.95
        v_align = "top"
    else:
        text_x = 0.05
        text_y = 0.05
        v_align = "bottom"

    g.ax_joint.text(
        text_x,
        text_y,
        corr_label,
        transform=g.ax_joint.transAxes,
        fontsize=45,
        verticalalignment=v_align,
        bbox=dict(
            boxstyle="round,pad=0.48",
            alpha=0.5,
            facecolor="lightgray"
        ),
    )

    # Labels and title
    g.ax_joint.set_xlabel(x_label, fontsize=45, labelpad=24)
    g.ax_joint.set_ylabel(y_label, fontsize=45)
    if title:
        g.fig.suptitle(title, fontsize=36, y=0.97)

    plt.tight_layout()
    plt.show()

    return g

def jointplot_brain_correl(
    x,
    y,
    x_label="X",
    y_label="Y",
    title=None,
    default_color=[0.2, 0.5, 1],
    density_color_x="#4c80ff",
    density_color_y="#fd6262",
    alpha=1,
    plot_w=10,
    plot_h=11,
    dash_horz=None,
    dash_vert=None,
    input_correl_text=None,
    text_legend=None  # NEW PARAMETER
):
    """
    Create a jointplot with regression line for X and Y with KDE marginals.

    Parameters
    ----------
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
    dash_horz : number or None
        If provided, draw a horizontal dashed line at this y value.
    dash_vert : number or None
        If provided, draw a vertical dashed line at this x value.
    input_correl_text : str or None
        Custom correlation text override.
    text_legend : str or None
        If provided, adds a color legend dot with this text.

    Returns
    -------
    g : seaborn.axisgrid.JointGrid
        The seaborn jointplot object.
    """
    valid_mask = ~np.isnan(x) & ~np.isnan(y)
    x_valid = x[valid_mask]
    y_valid = y[valid_mask]

    # Compute correlation
    slope, intercept, r_value, p_value, _ = linregress(x_valid, y_valid)
    if isinstance(input_correl_text, str):
        corr_label = input_correl_text
    else:
        corr_label = f"r = {r_value:.2f}"
        if p_value < 0.001:
            corr_label += "***"
        elif p_value < 0.01:
            corr_label += "**"
        elif p_value < 0.05:
            corr_label += "*"
        else:
            corr_label = f"r = {r_value:.2f}, p = {p_value:.3f}"

    # Axis limits
    x_margin = (x_valid.max() - x_valid.min()) * 0.05
    y_margin = (y_valid.max() - y_valid.min()) * 0.05
    xlim = (x_valid.min() - x_margin, x_valid.max() + x_margin)
    ylim = (y_valid.min() - y_margin, y_valid.max() + y_margin)

    # Create jointplot
    g = sns.JointGrid(x=x_valid, y=y_valid, height=10, xlim=xlim, ylim=ylim)
    g.fig.set_size_inches(plot_w, plot_h)

    # Scatter points
    g.ax_joint.scatter(
        x_valid,
        y_valid,
        alpha=0.7,
        s=150,
        edgecolor="black",
        linewidth=1.5,
        color=default_color,
    )

    for spine in g.ax_joint.spines.values():
        spine.set_linewidth(3.5)

    # Regression line
    sns.regplot(
        x=x_valid,
        y=y_valid,
        scatter=False,
        ax=g.ax_joint,
        line_kws={"color": "black", "linewidth": 5},
    )

    # KDE marginal distributions
    sns.kdeplot(x=x_valid, ax=g.ax_marg_x, fill=True, color=density_color_x, alpha=alpha)
    sns.kdeplot(y=y_valid, ax=g.ax_marg_y, fill=True, color=density_color_y, alpha=alpha)

    # Optional dashed lines
    if dash_vert is not None:
        g.ax_joint.axvline(dash_vert, linestyle="--", color="black", linewidth=3)
        xmin, xmax = g.ax_joint.get_xlim()
        g.ax_joint.set_xlim(min(xmin, dash_vert), max(xmax, dash_vert))

    if dash_horz is not None:
        g.ax_joint.axhline(dash_horz, linestyle="--", color="black", linewidth=3)
        ymin, ymax = g.ax_joint.get_ylim()
        g.ax_joint.set_ylim(min(ymin, dash_horz), max(ymax, dash_horz))

    # Correlation text
    g.ax_joint.text(
        0.05,
        0.95,
        corr_label,
        transform=g.ax_joint.transAxes,
        fontsize=35,
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.48", alpha=0.5, facecolor="lightgray"),
    )

    # Axis labels and title
    g.ax_joint.set_xlabel(x_label, fontsize=45, labelpad=24)
    g.ax_joint.set_ylabel(y_label, fontsize=45)
    if title:
        g.fig.suptitle(title, fontsize=36, y=0.97)

    # Tick styling
    for axis in ['x', 'y']:
        g.ax_joint.tick_params(
            axis=axis,
            which='both',
            direction='inout',
            length=8,
            width=2.5,
            labelsize=40,
            bottom=True, top=False,
            left=True, right=False,
        )

    if text_legend is not None:
        legend_fontsize = 32
        dot_size = 260

        legend_y_pos = 0.075  # ⬆️ Shifted slightly up

        # Render text and calculate bounding box
        text_obj = g.ax_joint.text(
            0.98, legend_y_pos,
            f"  {text_legend}",
            transform=g.ax_joint.transAxes,
            fontsize=legend_fontsize,
            verticalalignment="center",
            horizontalalignment="right",
            bbox=dict(
                boxstyle="round,pad=0.55",
                facecolor="white",
                edgecolor="black",
                linewidth=1.8,
                alpha=0.85
            ),
            zorder=9
        )

        # Dynamically position dot to left of text
        renderer = g.fig.canvas.get_renderer()
        bbox = text_obj.get_window_extent(renderer=renderer).transformed(g.ax_joint.transAxes.inverted())
        dot_x_pos = bbox.x0 - 0.015
        dot_y_pos = legend_y_pos  # Keep aligned with shifted text

        g.ax_joint.scatter(
            [dot_x_pos], [dot_y_pos],
            transform=g.ax_joint.transAxes,
            s=dot_size,
            color=default_color,
            edgecolor="black",
            linewidth=2,
            zorder=10
        )

    plt.tight_layout()
    plt.show()
    return g

# Re-import necessary modules after kernel reset
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


def plot_four_condition_violin(
    data_dict,
    palette=None,
    title="Pain Ratings by Condition",
    x_name="Hypnotic suggestion conditions",
    save_as=None,
    min_y=0,
    max_y=110,
):
    """
    Plot asymmetric violin plots for four conditions with:
    - Mean (dashed line)
    - Standard deviation (rectangle)
    - Connecting lines between paired conditions

    Parameters
    ----------
    data_dict : dict
        Dictionary with condition names as keys and lists/arrays of VAS values as values.
        Expected order: Analgesia, Neutral (Ana.), Hyperalgesia, Neutral (Hyper.)
    palette : list of str
        List of 4 hex color codes.
    title : str
        Title for the plot.
    save_as : str or None
        If set, path to save the figure.
    """

    assert len(data_dict) == 4, "Must provide exactly four conditions."

    condition_names = list(data_dict.keys())

    # Prepare DataFrame
    all_data = []
    for cond, values in data_dict.items():
        for i, v in enumerate(values):
            all_data.append({"Subject": i, "Condition": cond, "Pain Rating": v})

    df_plot = pd.DataFrame(all_data)

    if palette is None:
        palette = ["#4c80ff", "#a8c7ff", "#fd6262", "#ffaaaa"]

    fig, ax = plt.subplots(figsize=(14, 10))
    sns.set_style("white")

    # Violin plot
    sns.violinplot(
        x="Condition",
        y="Pain Rating",
        data=df_plot,
        palette=palette,
        dodge=False,
        scale="width",
        inner=None,
        linewidth=2,
        cut=0,
        ax=ax,
    )

    # Clip violins in half
    for i, violin in enumerate(ax.collections[:4]):
        path = violin.get_paths()[0]
        bbox = path.get_extents()
        x0, y0, width, height = bbox.bounds
        if i in [0, 2]:
            clip_rect = plt.Rectangle(
                (x0, y0), width / 2, height, transform=ax.transData
            )
        else:
            clip_rect = plt.Rectangle(
                (x0 + width / 2, y0), width / 2, height, transform=ax.transData
            )
        violin.set_clip_path(clip_rect)

    for violin in ax.collections[:4]:  # adjust if more violins exist
        violin.set_alpha(0.9)

    # Stripplot (with offset)
    strip = sns.stripplot(
        x="Condition",
        y="Pain Rating",
        data=df_plot,
        palette=palette,
        dodge=False,
        size=10,
        alpha=0.7,
        ax=ax,
    )
    for i, dots in enumerate(strip.collections):
        offsets = dots.get_offsets()
        shift = 0.12 if i % 2 == 0 else -0.12
        dots.set_offsets(offsets + np.array([shift, 0]))

    # Draw connecting lines between paired conditions
    n_subjects = len(data_dict[condition_names[0]])
    for i in range(n_subjects):
        # Left pair: Analgesia <-> Neutral (Ana.)
        ax.plot(
            [0, 1],
            [data_dict[condition_names[0]][i], data_dict[condition_names[1]][i]],
            color="gray",
            alpha=0.4,
            linewidth=1.5,
            zorder=0,
        )
        # Right pair: Hyperalgesia <-> Neutral (Hyper.)
        ax.plot(
            [2, 3],
            [data_dict[condition_names[2]][i], data_dict[condition_names[3]][i]],
            color="gray",
            alpha=0.4,
            linewidth=1.5,
            zorder=0,
        )

    # Add mean and standard deviation box for each condition
    for i, (cond, values) in enumerate(data_dict.items()):
        mean = np.mean(values)
        std = np.std(values)
        ax.hlines(mean, i - 0.2, i + 0.2, color="black", linestyle="--", linewidth=2.5)
        ax.add_patch(
            plt.Rectangle(
                (i - 0.15, mean - std),
                0.3,
                2 * std,
                edgecolor="black",
                facecolor="none",
                lw=2.5,
                zorder=10,
            )
        )

    # Formatting
    ax.set_title(title, fontsize=28, y=1.02)
    ax.set_xlabel(x_name, fontsize=40, labelpad=20)
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=37)
    ax.set_ylabel("Pain ratings", fontsize=40)
    ax.tick_params(axis="both", which="major", labelsize=37, width=4, length=12)

    # more space on top
    ax.set_ylim(min_y, max_y)
    ticks = ax.get_yticks()
    ticks = [tick for tick in ticks if tick < 110]
    ax.set_yticks(ticks)

    ax.yaxis.set_major_locator(MaxNLocator(nbins="auto", integer=True))
    for spine in ax.spines.values():
        spine.set_linewidth(2)

    ax.tick_params(axis="x", bottom=True)  # show ticks on bottom
    ax.tick_params(axis="y", left=True)  # show ticks on left

    plt.tight_layout()
    if save_as:
        plt.savefig(save_as, dpi=1000)
    plt.show()

    return fig


def plot_delta_condition_violin(
    data_dict,
    palette=None,
    title="Pain Ratings by Condition",
    save_as=None,
    min_y=0,
    max_y=110,
    xlabel="Hypnotic suggestions",
    ylabel="Pain ratings",
    median_line=True,
    pair_lines=True,
    std_box=True,
    dot_size=10,
    condition_labels=None,
    lab_rotation=0,
):
    n_conditions = len(data_dict)
    assert n_conditions % 2 == 0, "Number of conditions must be even (e.g., 2, 4, 6)."
    condition_names = list(data_dict.keys())
    n_subjects = len(next(iter(data_dict.values())))

    all_data = [
        {"Subject": i, "Condition": cond, "Pain Rating": v}
        for cond, values in data_dict.items()
        for i, v in enumerate(values)
    ]
    df_plot = pd.DataFrame(all_data)

    if palette is None:
        default_palette = [
            "#4c80ff",
            "#a8c7ff",
            "#fd6262",
            "#ffaaaa",
            "#53b37f",
            "#b6eadf",
        ]
        palette = default_palette[:n_conditions]

    fig, ax = plt.subplots(figsize=(14.5, 10))
    sns.set_style("white")

    sns.violinplot(
        x="Condition",
        y="Pain Rating",
        data=df_plot,
        palette=palette,
        dodge=False,
        scale="width",
        inner=None,
        linewidth=2,
        cut=0,
        ax=ax,
    )

    for i, violin in enumerate(ax.collections[:n_conditions]):
        path = violin.get_paths()[0]
        bbox = path.get_extents()
        x0, y0, width, height = bbox.bounds
        if i % 2 == 0:
            clip_rect = plt.Rectangle(
                (x0, y0), width / 2, height, transform=ax.transData
            )
        else:
            clip_rect = plt.Rectangle(
                (x0 + width / 2, y0), width / 2, height, transform=ax.transData
            )
        violin.set_clip_path(clip_rect)
        violin.set_alpha(0.9)

    strip = sns.stripplot(
        x="Condition",
        y="Pain Rating",
        data=df_plot,
        palette=palette,
        dodge=False,
        size=dot_size,
        alpha=0.7,
        ax=ax,
    )
    for i, dots in enumerate(strip.collections):
        shift = 0.12 if i % 2 == 0 else -0.12
        dots.set_offsets(dots.get_offsets() + np.array([shift, 0]))

    if pair_lines:
        for i in range(n_subjects):
            for pair in range(n_conditions // 2):
                x1, x2 = 2 * pair, 2 * pair + 1
                y1, y2 = (
                    data_dict[condition_names[x1]][i],
                    data_dict[condition_names[x2]][i],
                )
                ax.plot(
                    [x1, x2], [y1, y2], color="gray", alpha=0.4, linewidth=1.5, zorder=0
                )

    for i, (cond, values) in enumerate(data_dict.items()):
        if std_box:
            mean = np.mean(values)
            std = np.std(values)
            ax.hlines(
                mean, i - 0.2, i + 0.2, color="black", linestyle="--", linewidth=2.5
            )
            ax.add_patch(
                plt.Rectangle(
                    (i - 0.15, mean - std),
                    0.3,
                    2 * std,
                    edgecolor="black",
                    facecolor="none",
                    lw=2.5,
                    zorder=10,
                )
            )
        elif median_line:
            median = np.median(values)
            ax.hlines(
                median, i - 0.2, i + 0.2, color="black", linestyle="-", linewidth=2.5
            )

    ax.set_title(title, fontsize=40, y=1.02)
    ax.set_xlabel(xlabel, fontsize=40, labelpad=20)
    ax.set_ylabel(ylabel, fontsize=40)
    if condition_labels:
        xticklabels = [condition_labels.get(name, name) for name in condition_names]
    else:
        xticklabels = condition_names

    ax.set_xticklabels(xticklabels, fontsize=37, rotation=lab_rotation)
    for tick in ax.get_xticklabels():
        tick.set_ha("right")  # Aligns the label's right edge to the tick mark

    ax.tick_params(axis="both", which="major", labelsize=37, width=4, length=12)
    ax.set_ylim(min_y, max_y)
    ax.set_yticks([tick for tick in ax.get_yticks() if tick < max_y])
    ax.yaxis.set_major_locator(MaxNLocator(nbins="auto", integer=True))
    for spine in ax.spines.values():
        spine.set_linewidth(2)
    ax.tick_params(axis="x", bottom=True)
    ax.tick_params(axis="y", left=True)

    plt.tight_layout()
    if save_as:
        plt.savefig(save_as, dpi=1000)
    plt.show()

    return fig


# =============
# RADAR CHART
# ==============
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter


import matplotlib.pyplot as plt
import numpy as np
from collections import Counter


def plot_network_polarbar(
    network_list,
    all_networks=None,
    max_radius=5,
    title=None,
    color_map=None,
    font_size=10,
    label_distance=1.05,
    show_radial_ticks=True,
    tick_fontsize=12,
    divider_color=None,
    divider_linewidth=1.2,
    show_only_max_tick=True,
    save_as=None,
    return_fig=False,
):
    """
    Create a polar bar chart showing the distribution of significant ROIs across networks.

    Parameters
    ----------
    network_list : list of str
        List of network names (one per ROI).
    all_networks : list of str, optional
        Full set of possible network names, defines ordering.
    max_radius : int
        Fixed radial scale maximum.
    title : str or None
        Title for the plot. If None, no title is displayed.
    color_map : dict, optional
        Mapping from network name to color.
    font_size : int
        Font size for network labels.
    label_distance : float
        Radius multiplier to control label distance from origin.
    show_radial_ticks : bool
        Whether to display inner circle tick labels.
    tick_fontsize : int
        Font size for the radial tick labels.
    divider_color : str, optional
        Color of lines separating network segments.
    divider_linewidth : float
        Width of divider lines.
    show_only_max_tick : bool
        If True, display only the outermost radial tick label.
    save_as : str or None
        If not None, path to save the figure (e.g., 'fig.png').
    return_fig : bool
        If True, return the matplotlib figure object.
    """
    counts = Counter(network_list)

    if all_networks is None:
        network_order = [
            "Visual",
            "SomMot",
            "DorsAttn",
            "SalVentAttn",
            "Limbic",
            "Cont",
            "Default",
            "TempPar",
            "Subcortical",
            "Cerebellum",
        ]
    else:
        network_order = all_networks

    values = np.array([counts.get(net, 0) for net in network_order])
    N = len(network_order)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)

    # Assign colors
    if color_map:
        colors = [color_map.get(net, "lightgray") for net in network_order]
    else:
        base_cmap = plt.get_cmap("tab10")
        colors = base_cmap.colors[:N]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={"polar": True})
    width = 2 * np.pi / N
    bars = ax.bar(
        angles,
        values,
        width=width,
        bottom=0.0,
        color=colors,
        alpha=0.9,
        edgecolor="white",
        linewidth=1.2,
    )

    # Divider color defaults to polar axis spine color
    if divider_color is None:
        divider_color = ax.spines["polar"].get_edgecolor()

    # Draw dividers between network slices
    edge_angles = [(angle - width / 2) % (2 * np.pi) for angle in angles]
    for angle in edge_angles:
        ax.plot(
            [angle, angle],
            [0, max_radius],
            color=divider_color,
            linestyle="dotted",
            linewidth=divider_linewidth,
        )

    # Add network labels
    for angle, label in zip(angles, network_order):
        ax.text(
            angle,
            max_radius * label_distance,
            label,
            fontsize=font_size,
            ha="center",
            va="center",
            rotation=0,
            rotation_mode="anchor",
            transform_rotates_text=False,
        )

    ax.set_ylim(0, max_radius)
    ax.set_xticks([])

    # Show only the max radial tick
    if show_radial_ticks:
        if show_only_max_tick:
            ax.set_yticks([max_radius])
            ax.set_yticklabels([str(max_radius)], fontsize=tick_fontsize)
        else:
            ax.set_yticks(range(1, max_radius + 1))
            ax.set_yticklabels(
                [str(y) for y in range(1, max_radius + 1)], fontsize=tick_fontsize
            )
    else:
        ax.set_yticks([])
        ax.set_yticklabels([])

    if title:
        ax.set_title(title, fontsize=14, pad=20)

    plt.tight_layout()

    if save_as:
        fig.savefig(save_as, dpi=300)

    if return_fig:
        return fig
    else:
        plt.show()


def plot_median_isc_dots(
    sig_dfs_one_sample,
    title="Intersubject Correlation (ISC) per Condition",
    conditions_full_names=None,
    jitter=True,
    jitter_scale=0.08,
    y_spacing=1.3,
):
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
        "ANA": "#1f77b4",
        "HYPER": "#d62728",
        "neutral": "#2ca02c",
        "modulation": "#9467bd",
        "all_sugg": "#ff7f0e",
        "NANA": "#17becf",
        "NHYPER": "#e377c2",
    }

    fig, ax = plt.subplots(figsize=(11, 7))
    yticks = []
    ylabels = []

    for i, (cond, df) in enumerate(sig_dfs_one_sample.items()):
        if df.empty:
            continue

        y_base = i * y_spacing
        yticks.append(y_base)
        full_label = (
            conditions_full_names.get(cond, cond) if conditions_full_names else cond
        )
        ylabels.append(full_label)

        isc_vals = df["ISC"].values
        n = len(isc_vals)

        # Jitter for dot spread
        y_vals = (
            y_base + np.random.uniform(-jitter_scale, jitter_scale, size=n)
            if jitter
            else np.full(n, y_base)
        )

        color = color_map.get(cond, f"C{i}")

        # Plot dots
        ax.scatter(
            isc_vals,
            y_vals,
            s=85,
            alpha=0.9,
            color=color,
            edgecolor="black",
            linewidth=0.4,
        )

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
            edgecolor="black",
            facecolor=color,
            alpha=0.25,
        )
        ax.add_patch(box)

        # Add median line inside box
        ax.plot(
            [med, med],
            [y_base - box_height / 2, y_base + box_height / 2],
            color="black",
            linewidth=2.5,
        )

    # Final layout
    ax.set_yticks(yticks)
    ax.set_yticklabels(ylabels, fontsize=18)
    ax.set_xlabel("Median ISC per region", fontsize=24)
    ax.set_title(title, fontsize=26)
    ax.tick_params(axis="x", labelsize=18)
    ax.tick_params(axis="y", labelsize=18)
    ax.set_ylim(-y_spacing, yticks[-1] + y_spacing)
    ax.grid(axis="x", linestyle="dotted", alpha=0.5)
    plt.tight_layout()
    plt.show()


from mpl_toolkits.axes_grid1 import make_axes_locatable


def plot_simmat_behav(
    simmat,
    title,
    title_id="SHSS",
    y_label=None,
    x_label=None,
    cmap="RdBu_r",
    colorbar_name="Similarity",
    save_path=None,
    dpi=1000,
):
    """
    Plot a similarity matrix sorted by behavioral scores, styled for publication.

    Parameters
    ----------
    simmat : np.ndarray
        Square similarity matrix.
    title : str
        Title of the plot.
    title_id : str, optional
        Identifier for the title or output filename (used if save_path is provided).
    y_name : str, optional
        Name of the behavioral score used for sorting.
    cmap : str, optional
        Colormap for the matrix.
    save_path : str, optional
        Path to save the figure. If None, the figure is not saved.
    dpi : int, optional
        Resolution of the saved figure in dots per inch.
    """
    fig, ax = plt.subplots(figsize=(10, 10))

    vmax = np.max(np.abs(simmat))
    im = ax.imshow(simmat, vmin=-vmax, vmax=vmax, cmap=cmap, aspect="equal")

    # Custom colorbar: thick and aligned with heatmap
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.2)  # 5% width, 10% pad
    cbar = plt.colorbar(im, cax=cax, ticks=[1, 0, -1])
    cbar.ax.tick_params(labelsize=30, width=3, length=6)
    cbar.set_label(colorbar_name, fontsize=35)

    # Titles and labels
    ax.set_title(title, fontsize=35, pad=15)
    ax.set_xlabel(x_label, fontsize=35, labelpad=14)
    ax.set_ylabel(y_label, fontsize=35, labelpad=14)

    # Tick labels only at 0, 5, 10, 15, 20 (within matrix size)
    max_idx = simmat.shape[0] - 1
    ticks = [i for i in [0, 5, 10, 15, 20] if i <= max_idx]
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(ticks, fontsize=30)
    ax.set_yticklabels(ticks, fontsize=30)

    ax.tick_params(length=8, width=3)

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
        print(f"Figure saved to: {save_path}")

    plt.show()


from mpl_toolkits.axes_grid1.inset_locator import inset_axes

def plot_simmat_isc(
    simmat,
    title=None,
    title_id="SHSS",
    y_label=None,
    x_label="Subjects",
    cmap="RdBu_r",
    colorbar_name="Similarity",
    show_colorbar=False,
    vmin=None,
    vmax=None,
    save_path=None,
    dpi=300,
    figsize=(10, 10),
    tick_fontsize=40,
    cbar_ticks=None,
    ticks=None,
    label_fontsize=60,
):
    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes

    fig, ax = plt.subplots(figsize=figsize)

    # Define color range if not specified
    if vmin is None or vmax is None:
        abs_max = np.max(np.abs(simmat))
        vmin = -abs_max
        vmax = abs_max

    im = ax.imshow(simmat, vmin=vmin, vmax=vmax, cmap=cmap, aspect="equal")

    if show_colorbar:
        divider = make_axes_locatable(ax)
        cax = inset_axes(
            ax,
            width="5%",
            height="96%",
            loc="right",
            bbox_to_anchor=(0.1, 0.0, 1, 1),
            bbox_transform=ax.transAxes,
            borderpad=0,
        )
        cbar = plt.colorbar(im, cax=cax, ticks=cbar_ticks)
        cbar.ax.tick_params(labelsize=tick_fontsize, width=3, length=6)

        if colorbar_name:
            cbar.set_label(
                colorbar_name, fontsize=label_fontsize, rotation=270, labelpad=0
            )
            cbar.ax.yaxis.label.set_position((0.98, 0.5))
        else:
            cbar.ax.yaxis.label.set_visible(False)

    if ticks is None:
        ax.set_xticks([])
        ax.set_yticks([])
    else:
        # Custom tick selection
        max_idx = simmat.shape[0] - 1
        ticks = [i for i in [0, 20] if i <= max_idx]
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        ax.set_xticklabels(ticks, fontsize=tick_fontsize)
        ax.set_yticklabels(ticks, fontsize=tick_fontsize)
        ax.tick_params(length=8, width=3)

    if title is not None:
        ax.set_title(title, fontsize=label_fontsize, pad=20)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
        print(f"Figure saved to: {save_path}")

    plt.show()


import matplotlib.pyplot as plt
import matplotlib as mpl


def plot_isc_colorbar(
    vmin,
    vmax,
    cmap="RdBu_r",
    label=None,
    figsize=None,
    label_fontsize=60,
    tick_fontsize=50,
    tick_labels=None,
    save_path=None,
    dpi=300,
    horizontal=False,
):
    """
    Plot and optionally save a standalone colorbar image with specified limits and styling.

    Parameters
    ----------
    vmin : float
        Minimum value of the color scale.
    vmax : float
        Maximum value of the color scale.
    cmap : str
        Colormap name.
    label : str or None
        Label for the colorbar.
    figsize : tuple or None
        Figure size in inches (width, height). If None, set automatically.
    label_fontsize : int
        Font size for the label.
    tick_fontsize : int
        Font size for tick labels.
    save_path : str or None
        Path to save the figure. If None, shows the plot interactively.
    dpi : int
        DPI for saved figure.
    horizontal : bool
        If True, plot horizontal colorbar. Default is vertical.
    """
    if figsize is None:
        figsize = (8, 2) if horizontal else (2, 8)

    fig, ax = plt.subplots(figsize=figsize)

    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    orientation = "horizontal" if horizontal else "vertical"
    ticks = [vmin, 0, vmax]

    cbar = mpl.colorbar.ColorbarBase(
        ax, cmap=cmap, norm=norm, orientation=orientation, ticks=ticks
    )

    if label:
        if horizontal:
            cbar.set_label(label, fontsize=label_fontsize, labelpad=10)
        else:
            cbar.set_label(label, fontsize=label_fontsize, rotation=270, labelpad=15)

    cbar.ax.tick_params(labelsize=tick_fontsize, width=3, length=6)

    if tick_labels is None:
        tick_labels = [str(t) if t != 0 else "0" for t in ticks]

    if horizontal:
        cbar.ax.set_xticklabels(tick_labels)
    else:
        cbar.ax.set_yticklabels(tick_labels)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
        print(f"Colorbar saved to {save_path}")
    else:
        plt.show()


# ==================================
# BRAIN PLOTS FINAL //Like pain gen project
# from M-E Picard FEPS + Coll et al.
from nilearn.image import resample_to_img


def save_slices(img, coords_to_plot, save_to, img_id, cmap="hot", **kwargs):
    ticksfontsize = 14

    bg_img = datasets.load_mni152_template(resolution=1)
    new_img = resample_to_img(img, bg_img, interpolation="nearest")

    for axis, coord_list in coords_to_plot.items():
        for c in coord_list:
            fig, ax = plt.subplots(figsize=(2, 2))  # matched style
            disp = plot_stat_map(
                new_img,
                bg_img=bg_img,
                cmap=cmap,
                colorbar=False,
                dim=0,
                black_bg=False,
                display_mode=axis,
                axes=ax,
                vmax=None,
                cut_coords=(c - 2,),
                alpha=1,
                annotate=False,
                interpolation="nearest",
            )
            print(
                " WARNING: Shifting Cut coord by -2, but the coord label is as inputed in COORDS"
            )
            # Temporarily bug fix

            # disp.annotate(size=ticksfontsize, left_right=False, xy=(0.5, -0.08))

            # disp.annotate(size=ticksfontsize, left_right=False, xy=(0.5, -0.08))
            if axis == "z":  # bigger brain on z cuts
                x_pos = 0.5
                y_pos = -0.029
            elif axis == "y":
                x_pos = 0.5
                y_pos = 0.1
            elif axis == "x":
                x_pos = 0.65
                y_pos = 0.145

            ax.text(
                x_pos,
                y_pos,
                f"{axis} = {c}",
                transform=ax.transAxes,
                fontsize=ticksfontsize,
                ha="center",
                va="top",
                bbox=dict(
                    boxstyle="round,pad=0.3",
                    facecolor="white",
                    alpha=0.7,
                    edgecolor="none",
                ),
            )

            fig.savefig(
                os.path.join(save_to, f"{img_id}_sig_{axis}{c}.png"),
                transparent=True,
                bbox_inches="tight",
                dpi=600,
            )
            plt.close(fig)


# ===================================
# color bars
# ===================================

# Adapted from Nilearn (https://github.com/nilearn/nilearn),
# Copyright (c) 2007-2024 The Nilearn developers,
# BSD-3-Clause license

from numbers import Number
import numpy as np


def get_cbar_ticks(vmin, vmax, offset, n_ticks=5):
    """Extracted from Nilearn, adapted for standalone use."""
    if vmin == vmax:
        return np.linspace(vmin, vmax, 1)

    if vmax == 0:
        vmax += np.finfo(np.float32).eps

    ticks = np.linspace(vmin, vmax, n_ticks)
    if offset is not None and offset / vmax > 0.12:
        diff = [abs(abs(tick) - offset) for tick in ticks]
        if diff.count(min(diff)) == 4:
            idx_closest = np.sort(np.argpartition(diff, 4)[:4])
            idx_closest = np.isin(ticks, np.sort(ticks[idx_closest])[1:3])
        else:
            idx_closest = np.sort(np.argpartition(diff, 2)[:2])
            if 0 in ticks[idx_closest]:
                idx_closest = np.sort(np.argpartition(diff, 3)[:3])
                idx_closest = idx_closest[[0, 2]]
        ticks[idx_closest] = [-offset, offset]
    if len(ticks) > 0 and ticks[0] < vmin:
        ticks[0] = vmin
    return ticks


def get_colorbar_and_data_ranges(
    stat_map_data,
    vmin=None,
    vmax=None,
    symmetric_cbar=True,
    force_min_stat_map_value=None,
):
    """Extracted from Nilearn, adapted for standalone use."""
    if (not isinstance(vmin, Number)) or (not np.isfinite(vmin)):
        vmin = None
    if (not isinstance(vmax, Number)) or (not np.isfinite(vmax)):
        vmax = None

    if hasattr(stat_map_data, "_mask"):
        stat_map_data = np.asarray(stat_map_data[np.logical_not(stat_map_data._mask)])

    if force_min_stat_map_value is None:
        stat_map_min = np.nanmin(stat_map_data)
    else:
        stat_map_min = force_min_stat_map_value
    stat_map_max = np.nanmax(stat_map_data)

    if symmetric_cbar == "auto":
        if vmin is None or vmax is None:
            min_value = stat_map_min if vmin is None else max(vmin, stat_map_min)
            max_value = stat_map_max if vmax is None else min(stat_map_max, vmax)
            symmetric_cbar = min_value < 0 < max_value
        else:
            symmetric_cbar = np.isclose(vmin, -vmax)

    if symmetric_cbar:
        if vmin is None and vmax is None:
            vmax = max(-stat_map_min, stat_map_max)
            vmin = -vmax
        elif vmin is None:
            vmin = -vmax
        elif vmax is None:
            vmax = -vmin
        elif not np.isclose(vmin, -vmax):
            raise ValueError(
                "vmin must be equal to -vmax unless symmetric_cbar is False."
            )
        cbar_vmin = vmin
        cbar_vmax = vmax
    else:
        negative_range = stat_map_max <= 0
        positive_range = stat_map_min >= 0
        if positive_range:
            cbar_vmin = 0 if vmin is None else vmin
            cbar_vmax = vmax
        elif negative_range:
            cbar_vmax = 0 if vmax is None else vmax
            cbar_vmin = vmin
        else:
            cbar_vmin = vmin
            cbar_vmax = vmax

    if vmin is None:
        vmin = stat_map_min
    if vmax is None:
        vmax = stat_map_max

    return cbar_vmin, cbar_vmax, float(vmin), float(vmax)


import matplotlib.pyplot as plt
from matplotlib.colorbar import ColorbarBase
from matplotlib.colors import Normalize


def save_colorbar(
    img, cmap, fontsize = 16, outpath=None, symmetric_cbar=False, offset=None, n_ticks=5, transparent=True
):
    img_data = img.get_fdata()

    import numpy as np  # bug fix, somehow numpy is not imported in the scritp

    # Automatically set vmax and vmin based on actual data range
    data_max = np.nanmax(img_data)
    data_min = np.nanmin(img_data)

    # Important adjustment here:
    if symmetric_cbar:
        vmax = max(abs(data_min), abs(data_max))
        vmin = -vmax
    else:
        vmax = data_max
        vmin = data_min  # <-- set to actual data min, not 0 explicitly

    cbar_vmin, cbar_vmax, _, _ = get_colorbar_and_data_ranges(
        img_data, symmetric_cbar=symmetric_cbar, vmax=vmax, vmin=vmin
    )

    # Guard against None values
    if cbar_vmin is None:
        cbar_vmin = data_min
    if cbar_vmax is None:
        cbar_vmax = data_max

    # Generate ticks safely
    ticks = np.linspace(cbar_vmin, cbar_vmax, n_ticks)

    # Plotting the standalone colorbar
    fig, ax = plt.subplots(figsize=(0.4, 3.5))
    norm = Normalize(vmin=cbar_vmin, vmax=cbar_vmax)
    cb = ColorbarBase(ax, cmap=cmap, norm=norm, ticks=ticks, orientation="vertical")
    cb.ax.set_yticklabels([f"{t:.2f}" for t in ticks], fontsize=fontsize)

    cb.ax.yaxis.set_ticks_position("left")
    cb.ax.yaxis.set_label_position("left")
    cb.outline.set_visible(False)

    if outpath is None:
        plt.show()
    else:
        fig.savefig(outpath, transparent=transparent, bbox_inches="tight", dpi=300)
        plt.close(fig)

    return fig

# VIAULIZATION ROI IN ATLAS
#----------------------------
from nilearn.image import math_img, new_img_like
from nilearn import plotting

def filter_atlas(atlas_img, atlas_labels_dict, keep_ids):
    atlas_data = atlas_img.get_fdata()
    mask_data = np.isin(atlas_data, keep_ids) * atlas_data
    filtered_img = new_img_like(atlas_img, mask_data)

    # Build new labels dict for the kept IDs
    new_labels = {
        roi: label for roi, label in atlas_labels_dict.items() if roi in keep_ids
    }
    return filtered_img, new_labels


# Plot the filtered atlas with clear discrete colorbar
def plot_filtered_atlas(filtered_img, filtered_labels, cmap="tab20"):
    label_ids = list(filtered_labels.keys())
    label_names = list(filtered_labels.values())

    return plotting.view_img(
        filtered_img,
        cmap=cmap,
        colorbar=True,
        title="Filtered Atlas",
        symmetric_cmap=False,
    )

def plot_single_roi(atlas_img, roi_id, atlas_labels=None, title=None, show = True):

    mask_img = math_img(f"img == {roi_id}", img=atlas_img)

    title = f"ROI {roi_id}"
    if atlas_labels and roi_id in atlas_labels:
        title += f" {atlas_labels[roi_id]}"

    print(f"Displaying region {roi_id}: {title}")

    if show is False:
        return mask_img
    else:
        return mask_img, plotting.view_img(
            mask_img, cmap="Reds", title=title, colorbar=False
        )

def plot_many_roi(atlas_img, roi_ids, atlas_labels=None, title=None):

    combined_data = np.isin(atlas_img.get_fdata(), roi_ids) * atlas_img.get_fdata()
    combined_img = new_img_like(atlas_img, combined_data)

    title = "Selected ROIs"
    if title:
        title += f": {title}"

    return combined_img, plotting.view_img(
        combined_img, cmap="tab20", title=title, colorbar=True, symmetric_cmap=False
    )

####################
# PLOTTING 
####################
from nilearn.plotting import plot_surf_stat_map
from nilearn import datasets, surface

def plot_surface_views(z_map, output_dir, threshold=3.1, fine_resolution=False, vmax = None):
    """
    Project a volumetric map to fsaverage surface and save:
    - left/right hemispheres
    - lateral + medial views
    """

    print("Loading fsaverage surface...")
    if fine_resolution:
        fsaverage = datasets.fetch_surf_fsaverage(mesh="fsaverage7")
    else:
        fsaverage = datasets.fetch_surf_fsaverage()

    print("Projecting volume to surface...")
    texture_left = surface.vol_to_surf(z_map, fsaverage.pial_left)
    texture_right = surface.vol_to_surf(z_map, fsaverage.pial_right)

    # Define views to iterate over
    hemis = [
        ("left", fsaverage.infl_left, texture_left),
        ("right", fsaverage.infl_right, texture_right),
    ]

    views = ["lateral"] # "medial"]

    for hemi_name, surf_mesh, texture in hemis:
        for view in views:

            fig = plt.figure(figsize=(4, 4))

            plot_surf_stat_map(
                surf_mesh,
                texture,
                hemi=hemi_name,
                view=view,
                threshold=threshold,
                cmap="Reds",
                colorbar=True,
                bg_map=fsaverage.sulc_left if hemi_name == "left" else fsaverage.sulc_right,
                figure=fig,
                vmax=vmax,
            )

            fname = f"surface_{hemi_name}_{view}.png"
            path = os.path.join(output_dir, fname)

            plt.savefig(path, dpi=1000, bbox_inches="tight")
            plt.close()

            print(f"Saved: {path}")

#===========================

schaeffer_region_mapping = {
    "7Networks_LH_SomMot_1": {
        "anatomical_label": "L STG",
        "region_index": 15,
    },  # validation 4 sept.
    "7Networks_LH_SomMot_2": {
        "anatomical_label": "L planum temporale",
        "region_index": 16,
    },  # validation 4 sept.
    "7Networks_LH_SomMot_6": {
        "anatomical_label": "Left Motor cortex",
        "region_index": 20,
    },
    "7Networks_LH_SomMot_7": {
        "anatomical_label": "Left precentral gyrus",
        "region_index": 21,
    },  # validation 4 sept.
    "7Networks_LH_SomMot_14": {
        "anatomical_label": "Left SMA",
        "region_index": 28,
    },  # validation 4 sept.
    "7Networks_LH_SomMot_15": {
        "anatomical_label": "Left paracentral lobule",
        "region_index": 29,
    },  # validation 4 sept.
    "7Networks_LH_SomMot_16": {
        "anatomical_label": "Left Somatosensory",
        "region_index": 30,
    },
    "7Networks_LH_SalVentAttn_ParOper_1": {
        "anatomical_label": "Parietal operculum",
        "region_index": 44,
    },
    "7Networks_LH_SalVentAttn_FrOperIns_1": {
        "anatomical_label": "L insula",
        "region_index": 47,
    },  # validation 30 sept.
    "7Networks_LH_SalVentAttn_FrOperIns_4": {
        "anatomical_label": "Frontal operculum",
        "region_index": 50,
    }, # validation 14 oct.
    "7Networks_LH_SalVentAttn_Med_1": {"anatomical_label": "aMCC", "region_index": 52},
    "7Networks_LH_SalVentAttn_Med_3": {"anatomical_label": "SMA", "region_index": 54},
    "7Networks_LH_Limbic_OFC_1" : {
        "anatomical_label": "Left OFC",
        "region_index": 55,
    },  # validation 13 sept.
    "7Networks_LH_Default_Temp_3": {
        "anatomical_label": "Left anterior STG",
        "region_index": 76,
    },  # validation 4 sept.
    "7Networks_LH_Default_Temp_4": {
        "anatomical_label": "Left mid. STS",
        "region_index": 77,
    },  # validation 4 sept.
    "7Networks_LH_Default_Temp_5": {
        "anatomical_label": "Left pSTG",
        "region_index": 78,
    },  # validation 4 sept.
    "7Networks_LH_Default_PFC_3": {
        "anatomical_label": "L IFG",
        "region_index": 85,
    }, # validation 13 sept.
    "7Networks_LH_Default_PFC_12": {
        "anatomical_label": "Left MFG",
        "region_index": 94,
    },  # validation 4 sept.
    "7Networks_LH_Default_PHC_1" : {
        "anatomical_label": "Left PHG",
        "region_index": 100,
    },  # validation 13 sept.
    "7Networks_RH_SomMot_1": {
        "anatomical_label": "R planum temporale",
        "region_index": 116,
    },  # validation 4 sept.
    "7Networks_RH_SomMot_2": {
        "anatomical_label": "R post. STG",
        "region_index": 117,
    },  # validation 4 sept.
    "7Networks_RH_SomMot_4": {
        "anatomical_label": "Right parietal operculum",
        "region_index": 119,
    },  # validation 4 sept.
    "7Networks_RH_SomMot_7": {
        "anatomical_label": "Right inf. Lateral Motor Cortex",
        "region_index": 122,
    },
    "7Networks_RH_SomMot_8": {
        "anatomical_label": "MCC",
        "region_index": 123,
    },  # valid. 14 oct.
    "7Networks_RH_SomMot_16": {
        "anatomical_label": "Right Motor Cortex",
        "region_index": 131,
    },
    "7Networks_RH_SomMot_17": {
        "anatomical_label": "Right Somatosensory",
        "region_index": 132,
    },
    "7Networks_RH_DorsAttn_Post_4": {
        "anatomical_label": "R supramarginal gyrus",
        "region_index": 138,
    }, # validation 13 sept.
    "7Networks_RH_SalVentAttn_TempOccPar_2": {
        "anatomical_label": "STG/SMG",
        "region_index": 149,
    },
    "7Networks_RH_SalVentAttn_FrOperIns_4": {
        "anatomical_label": "Frontal operculum",
        "region_index": 155,
    },  # valid. 14 oct.
    "7Networks_RH_SalVentAttn_Med_1": {
        "anatomical_label": " aMCC",
        "region_index": 156,
    }, # validation 30 sept.
    "7Networks_RH_SalVentAttn_Med_2": {
        "anatomical_label": "anterior PCC",
        "region_index": 157,
    },
    "7Networks_RH_SalVentAttn_Med_3": {
        "anatomical_label": "Right pre-SMA",
        "region_index": 158,
    },
    "7Networks_RH_Cont_PFCv_1": {
        "anatomical_label": "Right ant. insula",
        "region_index": 169,
    },  # validation 4 sept.
    "7Networks_RH_Cont_PFCl_4": {"anatomical_label": "Right MFG", "region_index": 173},
    "7Networks_RH_Cont_PFCmp_1": {
        "anatomical_label": "aMCC",
        "region_index": 180,
    },  # validation 30 sept.
    "7Networks_RH_Default_Par_3": {
        "anatomical_label": "Right angular gyrus",
        "region_index": 184,
    },  # validation 30 sept.
    "7Networks_RH_Default_Temp_3": {
        "anatomical_label": "Right anterior STG",
        "region_index": 187,
    },  # validation 4 sept.
    "7Networks_RH_Default_Temp_4": {
        "anatomical_label": "Right pMTG",
        "region_index": 188,
    },  # validation 4 sept.
    "7Networks_RH_Default_Temp_5": {
        "anatomical_label": "Right pSTS",
        "region_index": 189,
    },  # validation 4 sept.
    "7Networks_RH_Default_PFCdPFCm_6": {
        "anatomical_label": "Right dlPFC",
        "region_index": 196,
    },
    "PUT-rh": {"anatomical_label": "Right Putamen", "region_index": 207
} # 30 sept.
}


