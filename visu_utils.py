import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import rankdata


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

def heatmap_pairwise_isc_combined(isc_df, subjects, behavioral_scores=None, save_to=None, show=True, title="Pairwise ISC Matrices"):
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
    n_rois = isc_df.shape[1]

    if behavioral_scores is not None:
        ranked_scores = rankdata(behavioral_scores)
        sorted_indices = np.argsort(ranked_scores)  # Sort indices based on rank
        sorted_subjects = [subjects[i] for i in sorted_indices]
    else:
        sorted_indices = range(len(subjects))
        sorted_subjects = subjects
        
    columns = isc_df.columns
    observed_isc = isc_df.to_numpy()

    fig, axes = plt.subplots(1, n_rois, figsize=(5 * n_rois, 5))
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
