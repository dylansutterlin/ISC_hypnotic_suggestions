# %%
import os
import nibabel as nib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from nilearn import plotting
from nilearn.image import math_img, mean_img
from nilearn.maskers import MultiNiftiMapsMasker, NiftiMapsMasker
import utils
import os
import matplotlib.pyplot as plt
from statsmodels.stats.multitest import multipletests
import visu_utils as vu
# Threshold values
p_threshold_uncorrected = 0.1  # Uncorrected
p_threshold_001 = 0.001
fdr_threshold = 0.05

roi_coords = {
"amcc": (-2, 20, 32),
"rPO": (54, -28, 26),
"lPHG": (-20, -26, -14),
}
# %%
# Function to ensure visualization directories exist

def create_visu_folders(results_dir, conditions):
    for condition in conditions:
        visu_dir = os.path.join(results_dir, condition, "visu")
        try:
            os.makedirs(visu_dir, exist_ok=True)
        except PermissionError:
            print(f"Permission denied: Unable to create directory {visu_dir}. Check permissions.")
    return

# Function to load transformed data
def load_transformed_data(results_dir, condition):
    transformed_path = os.path.join(results_dir, condition, f"transformed_data_Difumo64_{condition}_22sub.pkl")
    return utils.load_pickle(transformed_path)

# Function to load ISC and p-value images
def load_images(results_dir, condition):
    isc_img_path = os.path.join(results_dir, condition, f"isc_val_{condition}_boot5000_pariwiseFalse.nii.gz")
    pval_img_path = os.path.join(results_dir, condition, f"p_values_{condition}_boot5000_pairwiseFalse.nii.gz")
    
    isc_img = nib.load(isc_img_path)
    pval_img = nib.load(pval_img_path)
    
    return isc_img, pval_img

# Function to threshold and mask ISC images
def threshold_and_mask(isc_img, pval_img, threshold):
    thresholded_pval = math_img(f"img < {threshold}", img=pval_img)
    masked_isc = math_img("img1 * img2", img1=isc_img, img2=thresholded_pval)
    return masked_isc

def visualize_mean_activation(results_dir, atlas_path, conditions):
    masker = NiftiMapsMasker(maps_img=atlas_path, standardize=False)
    masker.fit()
    
    for condition in conditions:
        visu_dir = os.path.join(results_dir, condition, "visu")
        os.makedirs(visu_dir, exist_ok=True)

        mean_img_path = os.path.join(visu_dir, f"{condition}_mean_activation.nii.gz")
        mean_plot_path = os.path.join(visu_dir, f"{condition}_mean_activation.png")
        if os.path.exists(mean_img_path) and os.path.exists(mean_plot_path):
            print(f"Mean activation and plot already exist for {condition}. Skipping...")
            continue
            
        data = load_transformed_data(results_dir, condition)
        mean_activation = np.mean(data, axis=0)
        activation_img = masker.inverse_transform(mean_activation)
        mean_activation_img = mean_img(activation_img)
            
        mean_activation_img.to_filename(mean_img_path)
        plotting.plot_stat_map(
            mean_activation_img,
            title=f"Mean Activation - {condition}",
            colorbar=True,
            display_mode='x',
            output_file=mean_plot_path
        )
        print(f"Saved: {mean_img_path} and {mean_plot_path}")


# 2. Visualize ISC Maps for All Thresholds
def visualize_isc_maps(results_dir, conditions):
    for condition in conditions:
        isc_img, pval_img = load_images(results_dir, condition)
        
        visu_dir = os.path.join(results_dir, condition, "visu")
        
        # Plot uncorrected ISC map
        isc_uncorrected_path = os.path.join(visu_dir, f"{condition}_isc_uncorrected.png")
        # display = plotting.plot_stat_map(isc_img, title=f"ISC Map (Uncorrected) - {condition}", colorbar=True)
        # plt.savefig(isc_uncorrected_path)
        # plt.show()
        # plt.close()
        
        # Mask ISC map at p < 0.001
        # isc_masked_001 = threshold_and_mask(isc_img, pval_img, p_threshold_001)
        
        isc_p001_path = os.path.join(visu_dir, f"{condition}_isc_p001.png")
        plotting.plot_stat_map(
            isc_img,
            title=f"unc. ISC maps - {condition}",
            colorbar=True,
            display_mode='x',
            output_file=isc_uncorrected_path
        )
        #display = plotting.plot_stat_map(isc_masked_001, title=f"ISC Map (p<0.001) - {condition}", colorbar=True)
        #plt.savefig(isc_p001_path)
        plt.show()
        plt.close()
        
        # # Mask ISC map with FDR correction
        # isc_masked_fdr = threshold_and_mask(isc_img, pval_img, fdr_threshold)
        # isc_fdr_path = os.path.join(visu_dir, f"{condition}_isc_fdr.png")
        # display = plotting.plot_stat_map(isc_masked_fdr, title=f"ISC Map (FDR Corrected) - {condition}", colorbar=True)
        # plt.savefig(isc_fdr_path)
        # plt.close()
        # print(f"ISC maps saved for {condition} in {visu_dir}")

import os
import matplotlib.pyplot as plt

def visu_isc(setup, condition, results_dir, atlas_labels, p_threshold=0.01, significant_color='red', nonsignificant_color='gray'):
    """
    Visualizes ISC values for a specific condition, highlighting significant ROIs.

    Parameters
    ----------
    condition : str
        The condition to visualize (e.g., "all_sugg").
    results_dir : str
        Path to the directory containing ISC results.
    atlas_labels : list
        List of ROI labels corresponding to the atlas.
    p_threshold : float, optional
        Threshold for significance of p-values. Default is 0.01.
    significant_color : str, optional
        Color for bars representing significant ROIs. Default is 'red'.
    nonsignificant_color : str, optional
        Color for bars representing non-significant ROIs. Default is 'gray'.
    """
    n_boot = setup['n_boot']
    # Load ISC results
    isc_file = os.path.join(results_dir, condition, f"isc_results_{condition}_{n_boot}boot_pairWiseFalse.pkl")
    isc_results = utils.load_pickle(isc_file)
    
    observed_isc = isc_results['observed'] # median
    p_values = isc_results['p_values']
    distributions = isc_results['distribution']

    plt.hist(observed_isc, bins=50, alpha=0.7, color='blue')
    plt.xlabel("ISC Values")
    plt.ylabel("Frequency")
    plt.title("Histogram of Observed ISC Values")
    plt.show()

    plt.hist(p_values, bins=50, alpha=0.7, color='blue')
    plt.xlabel("P-values")
    plt.ylabel("Frequency")
    plt.title("Histogram of Raw P-Values")
    plt.show()

    mean_null = np.mean(distributions, axis=0)
    plt.hist(mean_null, bins=50, alpha=0.7, color='orange', label="Null Distribution")
    plt.axvline(x=np.mean(observed_isc), color='blue', linestyle='--', label="Observed Mean ISC")
    plt.xlabel("ISC Values")
    plt.ylabel("Frequency")
    plt.legend()
    plt.title("Null Distribution vs. Observed ISC Values")
    plt.show()

    fdr_correction = False
    if p_threshold == 0.01:
        p_threshold_str = '01unc'
    elif p_threshold == 0.001:
        p_threshold_str = '001unc'
    elif p_threshold == 0.0001:
        p_threshold_str = '0001unc'
    elif p_threshold == 0.05:
        p_threshold_str = 'FDR05'
        fdr_correction = True
    
    if fdr_correction:
        fdr_threshold = p_threshold
        _, fdr_p_values, _, _ = multipletests(p_values, alpha=fdr_threshold, method='fdr_bh')
        sig_mask = fdr_p_values < fdr_threshold
  
    else:
        sig_mask = p_values < p_threshold
   
    sig_mask = p_values < p_threshold
    significant_labels = [label if sig else " " for label, sig in zip(atlas_labels, sig_mask)]
    #p_threshold_str = str(int(p_threshold * 100)).zfill(2)

    # Plot all ROIs with significant labels
    plt.figure(figsize=(12, 6))
    bar_colors = [significant_color if sig else nonsignificant_color for sig in sig_mask]
    plt.bar(range(len(observed_isc)), observed_isc, color=bar_colors, alpha=0.8)
    plt.axhline(0, color='black', linestyle='--', linewidth=0.8)
    plt.xticks(range(len(observed_isc)), significant_labels, rotation=90, fontsize=8)
    plt.xlabel("ROIs")
    plt.ylabel("ISC Values")
    plt.title(f"ISC Values for {condition} (Significant Regions Highlighted, p < {p_threshold})")
    plt.tight_layout()

    # Save the plot
    output_dir = os.path.join(results_dir, condition, "visu")
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, f"barplot_isc_{n_boot}boot_sig-{p_threshold_str}.png")
    plt.savefig(save_path)
    plt.close()
    print(f"Bar plot saved to {save_path}")



import matplotlib.pyplot as plt
import numpy as np
import os
from statsmodels.stats.multitest import multipletests

def visu_permutation(setup, condition, results_dir, atlas_labels,sphere_folder=None, p_threshold=0.01, 
                      significant_color='red', nonsignificant_color='gray', show=False):
    """
    Visualizes ISC permutation results for a specific condition and behavioral variable,
    including a small histogram of p-values in the corner of the bar plot.

    Parameters
    ----------
    setup : dict
        Setup information including number of permutations.
    condition : str
        The condition to visualize (e.g., "modulation").
    results_dir : str
        Path to the directory containing ISC results.
    atlas_labels : list
        List of ROI labels corresponding to the atlas.
    p_threshold : float, optional
        Threshold for significance of p-values. Default is 0.01.
    significant_color : str, optional
        Color for bars representing significant ROIs. Default is 'red'.
    nonsignificant_color : str, optional
        Color for bars representing non-significant ROIs. Default is 'gray'.
    show : bool, optional
        Whether to display the plot. Default is False.
    """
    fdr_correction = False
    if p_threshold == 0.01:
        p_threshold_str = '01unc'
    elif p_threshold == 0.001:
        p_threshold_str = '001unc'
    elif p_threshold == 0.0001:
        p_threshold_str = '0001unc'
    elif p_threshold == 0.05:
        p_threshold_str = 'FDR05'
        fdr_correction = True

    n_perm = setup['n_perm']

    # Load ISC results
    if sphere_folder != None:
        isc_file = os.path.join(results_dir, condition,sphere_folder, f"isc_{n_perm}permutation_results_{condition}_pairwiseFalse.pkl")
    else:
        isc_file = os.path.join(results_dir, condition, f"isc_permutation_results_{condition}_pairwiseFalse.pkl")
    isc_results = utils.load_pickle(isc_file)
    y_names = list(isc_results.keys())

    for y_name in y_names:
        observed_isc = isc_results[y_name]['observed']
        p_values = isc_results[y_name]['p_value']

        if fdr_correction:
            fdr_threshold = p_threshold
            _, fdr_p_values, _, _ = multipletests(p_values, alpha=fdr_threshold, method='fdr_bh')
            sig_mask = fdr_p_values < fdr_threshold
        else:
            sig_mask = p_values < p_threshold

        significant_labels = [label if sig else " " for label, sig in zip(atlas_labels, sig_mask)]

        # Plot ISC values with significant labels
        plt.figure(figsize=(12, 6))
        bar_colors = [significant_color if sig else nonsignificant_color for sig in sig_mask]
        plt.bar(range(len(observed_isc)), observed_isc, color=bar_colors, alpha=0.8)
        plt.axhline(0, color='black', linestyle='--', linewidth=0.8)
        plt.xticks(range(len(observed_isc)), significant_labels, rotation=90, fontsize=8)
        plt.xlabel("ROIs")
        plt.ylabel("ISC Values")
        plt.title(f"ISC Values for {y_name} (Significant Regions Highlighted, p < {p_threshold})")
        plt.tight_layout()

        # Add histogram of p-values in the corner
        inset_ax = plt.gcf().add_axes([0.88, 0.8, 0.1, 0.1])  # x, y, width, height in figure coordinates
        inset_ax.hist(p_values, bins=20, color='blue', alpha=0.7)
        #inset_ax.axvline(p_threshold, color='red', linestyle='--', linewidth=0.8)
        inset_ax.set_title("P-value Distribution", fontsize=8)
        inset_ax.set_xlabel("P-value", fontsize=6)
        inset_ax.set_ylabel("Count", fontsize=6)
        inset_ax.tick_params(axis='both', which='major', labelsize=6)

        # Save the plot
        if sphere_folder != None:
            output_dir = os.path.join(results_dir, condition,sphere_folder, "visu")
        else:
            output_dir = os.path.join(results_dir, condition, "visu")
        os.makedirs(output_dir, exist_ok=True)
        save_path = os.path.join(output_dir, f"barplot_group_{n_perm}permutation_isc_{y_name}_sig-{p_threshold_str}.png")
        plt.savefig(save_path, dpi=1000)

        if show:
            plt.show()
        else:
            plt.close()

        print(f"Bar plot saved to {save_path}")



# Main function to execute all visualizations
def main(results_dir, atlas_path, conditions):
    create_visu_folders(results_dir, conditions)
    print("Visualizing mean activations...")
    visualize_mean_activation(results_dir, atlas_path, conditions)
    
    print("Visualizing ISC maps...")
    visualize_isc_maps(results_dir, conditions)
    
    print("Visualizations complete. Results saved in 'visu' folders.")

# %% 
import importlib
importlib.reload(utils)

#model_name = "model1-22sub"
model_name = 'model2_zcore_sample-22sub'
project_dir = "/data/rainville/dSutterlin/projects/ISC_hypnotic_suggestions"
results_dir = os.path.join(project_dir, f'results/imaging/ISC/{model_name}')
atlas_path = os.path.join(project_dir, 'masks/DiFuMo256/3mm/maps.nii.gz')
conditions = ["all_sugg", "modulation", "neutral"]
setup = utils.load_json(os.path.join(results_dir, 'setup_parameters.json'))

atlas_path = os.path.join(project_dir, 'masks/DiFuMo256/3mm/maps.nii.gz')
atlas_dict_path = os.path.join(project_dir, 'masks/DiFuMo256/labels_256_dictionary.csv')
atlas = nib.load(atlas_path)
atlas_df = pd.read_csv(atlas_dict_path)
atlas_labels = atlas_df['Difumo_names']

setup = utils.load_json(os.path.join(results_dir, 'setup_parameters.json'))
n_boot = setup['n_boot']
n_perm = setup['n_perm']
do_pairwise = setup['do_pairwise']

behavioral_scores = pd.read_csv(r'/data/rainville/dSutterlin/projects/ISC_hypnotic_suggestions/results/imaging/ISC/model2_zcore_sample-22sub/behav_data_group_labels.csv', index_col=0)
# %%
#for cond in conditions:
visu_isc(setup, cond, results_dir, atlas_labels, p_threshold=0.001, significant_color='red', nonsignificant_color='gray')
    #visu_isc(cond, results_dir, atlas_labels, p_threshold=0.05, significant_color='red', nonsignificant_color='gray')


# %%

create_visu_folders(results_dir, conditions)
visualize_mean_activation(results_dir, atlas_path, conditions)
for cond in conditions:
    print(f'Permutation plots for {cond}')
    #visu_permutation(setup, cond, results_dir, atlas_labels, p_threshold=0.01, significant_color='red', nonsignificant_color='gray')
    visu_permutation(setup, cond, results_dir, atlas_labels, p_threshold=0.01,show=True, significant_color='red', nonsignificant_color='gray')

# %%
import importlib
importlib.reload(utils)
importlib.reload(vu)
cond = 'modulation'
masker = utils.load_pickle(os.path.join(results_dir, cond, 'maskers_Difumo256_modulation_22sub.pkl'))

# %%
# load isc results ROI LOO approach
cond = 'all_sugg'
#for cond in conditions:
# cond = 'modulation'
do_pairwise = False
cond_path = os.path.join(results_dir, cond)
isc_sphere = f'isc_3spheres_{cond}_{n_boot}boot_pairWiseFalse.pkl'
isc_sphere = os.path.join(cond_path, f'sphere_10mm_3ROIS_isc/isc_3spheres_{cond}_10000boot_pairWiseFalse.pkl')
isc_nopairwise = utils.load_pickle(isc_sphere)

isc, observed_isc, p_values, ci, median_isc, distributions = vu.load_isc_results(isc_nopairwise)
isc_df = pd.DataFrame(isc, columns=roi_coords.keys())

save_to = os.path.join(cond_path, f'sphere_10mm_3ROIS_isc', f'hist_isc_{n_boot}boot_pairWise{do_pairwise}.png')
#vu.plot_isc_distributions(isc_df, p_values, median_isc, distributions, save_to=save_to, title=f"{cond} ISC Distributions (LOO)")

# %%
#------------------
# Pairewise ISC
import importlib
importlib.reload(vu)

for cond in conditions:
    # cond = 'modulation'
    do_pairwise = True
    cond_path = os.path.join(results_dir, cond)
    isc_sphere = f'isc_3spheres_{cond}_{n_boot}boot_pairWiseTrue.pkl'
    isc_sphere = os.path.join(cond_path, f'sphere_10mm_3ROIS_isc/isc_3spheres_{cond}_10000boot_pairWiseTrue.pkl')
    isc_pairwise = utils.load_pickle(isc_sphere)

    isc, observed_isc, p_values, ci, distributions = vu.load_isc_results_pairwise(isc_pairwise)
    isc_df = pd.DataFrame(isc, columns=roi_coords.keys())
    behav = 'SHSS'
    pain_score = behavioral_scores['total_chge_pain_hypAna']
    pain_score = behavioral_scores['SHSS_score']
    save_to = os.path.join(cond_path, f'sphere_10mm_3ROIS_isc', f'heatmap_isc_{n_boot}boot_pairWise{do_pairwise}_sort{behav}.png')
    #vu.heatmap_pairwise_isc_combined(isc_df,setup['subjects'], save_to = save_to, behavioral_scores=pain_score, title=f"{cond} Pairwise ISC spheres (10mm)")



    # # isc_df = pd.DataFrame(isc, columns=roi_coords.keys())

    # save_to = os.path.join(cond_path, f'sphere_10mm_3ROIS_isc', f'hist_isc_{n_boot}boot_pairWise{do_pairwise}.png')
    # vu.plot_isc_distributions(isc_df, p_values, median_isc, distributions, save_to=save_to, title=f"{cond} ISC Distributions (LOO)")

#============
# %%
# ROI permutaion LOO
sphere_folder = 'sphere_10mm_3ROIS_isc'
visu_permutation(setup, cond,  results_dir, atlas_labels,sphere_folder =sphere_folder, p_threshold=0.01,show=True, significant_color='red', nonsignificant_color='gray')


# %%
# =============== atlas brain maps boot

visualize_isc_maps(results_dir, conditions)

# isc per ROI with bootstrap
# %%
condition='neutral'

isc_img, pval_img = load_images(results_dir, condition)
visu_dir = os.path.join(results_dir, condition, "visu")
isc_uncorrected_path = os.path.join(visu_dir, f"{condition}_isc_uncorrected.png")
# display = plotting.plot_stat_map(isc_img, title=f"ISC Map (Uncorrected) - {condition}", colorbar=True)
# plt.savefig(isc_uncorrected_path)
# plt.show()
# plt.close()

# Mask ISC map at p < 0.001
# isc_masked_001 = threshold_and_mask(isc_img, pval_img, p_threshold_001)

isc_p001_path = os.path.join(visu_dir, f"{condition}_isc_p001.png")
plotting.plot_stat_map(
    isc_img,

    title=f"unc. ISC maps - {condition}",
    colorbar=True,
    display_mode='z',
    output_file=isc_uncorrected_path
)

plt.show()

# +++++++++++++++++++++++=======
# %% VISUALIZE ATLAS
# load difumo atlas

import numpy as np
import matplotlib.pyplot as plt
from nilearn import plotting, datasets
from nilearn.maskers import NiftiSpheresMasker
import nibabel as nib
import os

atlas_path = os.path.join(project_dir, 'masks/DiFuMo256/3mm/maps.nii.gz')
atlas_dict_path = os.path.join(project_dir, 'masks/DiFuMo256/labels_256_dictionary.csv')
atlas = nib.load(atlas_path)
atlas_df = pd.read_csv(atlas_dict_path)
atlas_labels = atlas_df['Difumo_names']
atlas_name = 'Difumo256'
print('atlas loaded with N ROI : ', atlas.shape)

 # extract sphere signal

sphere_radius = 10
roi_folder = f"sphere_{sphere_radius}mm_{len(roi_coords)}ROIS_isc"
sphere_coords = [(-2, 20, 32), (54, -28, 26), (-20, -26, -14)]
# the atlas map
atlas_img = nib.load(atlas_path)

sphere_imgs = []

sphere_masker = NiftiSpheresMasker(sphere_coords, radius=sphere_radius, allow_overlap=True)
sphere_masker.fit()
sphere_masker.generate_report()

# %%
sphere_imgs.append(sphere_img)

# Combine sphere masks into one
combined_sphere_data = np.zeros(sphere_imgs[0].shape)
for img in sphere_imgs:
    combined_sphere_data += img.get_fdata()

combined_sphere_img = nib.Nifti1Image(combined_sphere_data, sphere_imgs[0].affine)

# Plot the atlas
fig, axes = plt.subplots(1, 2, figsize=(12, 6), subplot_kw={"projection": "3d"})

# Plot atlas
plotting.plot_roi(
    atlas_img,
    title="Atlas Maps",
    axes=axes[0],
    cut_coords=[0, 0, 0],
    draw_cross=False,
    cmap="tab10"
)

# Plot spheres
plotting.plot_roi(
    combined_sphere_img,
    title="Spheres",
    axes=axes[1],
    cut_coords=[0, 0, 0],
    draw_cross=False,
    cmap="autumn"
)

plt.tight_layout()
plt.show()



# %%
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import pairwise_distances
import seaborn as sns
import matplotlib.pyplot as plt

def distance_to_similarity(distance_matrix):
    """
    Converts a distance matrix to a similarity matrix.
    Similarity = 1 / (1 + distance).
    """
    similarity_matrix = 1 / (1 + distance_matrix)
    return similarity_matrix

def sort_square_matrix(matrix, behavioral_vector, subjects):
    """
    Sorts a square matrix based on the order of the behavioral vector.
    """
    sort_indices = np.argsort(behavioral_vector)
    sorted_matrix = matrix[np.ix_(sort_indices, sort_indices)]
    sorted_subjects = [subjects[i] for i in sort_indices]
    return sorted_matrix, sorted_subjects

behav = 'SHSS'
pain_score = behavioral_scores['SHSS_score']

subjects = setup['subjects']
behav_rank = rankdata(pain_score)
distance_matrix = pairwise_distances(behav_rank.reshape(-1, 1), metric="euclidean")

# Convert distances to similarities
behav_sim_nn = distance_to_similarity(distance_matrix)
sorted_matrix, sorted_subjects = sort_square_matrix(behav_sim_nn, behav_rank, subjects)

# Plot the similarity matrix
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

# Original similarity matrix
sns.heatmap(behav_sim_nn, ax=ax1, square=True, cmap="coolwarm", cbar_kws={"label": "Similarity"},
            xticklabels=subjects, yticklabels=subjects)
ax1.set_title(f"{behav} Similarity Matrix (Unsorted)", fontsize=14)
ax1.tick_params(axis='x', rotation=90)

# Sorted similarity matrix
sns.heatmap(sorted_matrix, ax=ax2, square=True, cmap="coolwarm", cbar_kws={"label": "Similarity"},
            xticklabels=sorted_subjects, yticklabels=sorted_subjects)
ax2.set_title(f"{behav} Similarity Matrix (Sorted)", fontsize=14)
ax2.tick_params(axis='x', rotation=90)

plt.tight_layout()
plt.show()

# %%
visualize_isc_maps(results_dir, conditions)

def threshold_and_mask(isc_img, pval_img, threshold):
    thresholded_pval = math_img(f"img < {threshold}", img=pval_img)
    masked_isc = math_img("img1 * img2", img1=isc_img, img2=thresholded_pval)
    return masked_isc


condition = "all_sugg"
isc_img, pval_img = load_images(results_dir, condition)
        
visu_dir = os.path.join(results_dir, condition, "visu")

# Plot uncorrected ISC map
isc_uncorrected_path = os.path.join(visu_dir, f"{condition}_isc_uncorrected.png")
display = plotting.plot_stat_map(isc_img, title=f"ISC Map (Uncorrected) - {condition}", colorbar=True)
plt.savefig(isc_uncorrected_path)
plt.close()

# Mask ISC map at p < 0.001
isc_masked_001 = threshold_and_mask(isc_img, pval_img, p_threshold_001)
isc_p001_path = os.path.join(visu_dir, f"{condition}_isc_p001.png")
display = plotting.plot_stat_map(isc_masked_001, title=f"ISC Map (p<0.001) - {condition}", colorbar=True)
plt.savefig(isc_p001_path)
plt.close()

# Mask ISC map with FDR correction
isc_masked_fdr = threshold_and_mask(isc_img, pval_img, fdr_threshold)
isc_fdr_path = os.path.join(visu_dir, f"{condition}_isc_fdr.png")
display = plotting.plot_stat_map(isc_masked_fdr, title=f"ISC Map (FDR Corrected) - {condition}", colorbar=True)
plt.savefig(isc_fdr_path)
plt.close()
print(f"ISC maps saved for {condition} in {visu_dir}")


# %%
# save mean images per sugg in preproc model, create
# visualization mean imgs per ana, hyper, neutral, neutral
preproc = r'/data/rainville/dSutterlin/projects/ISC_hypnotic_suggestions/results/imaging/preproc_data/23subjects_zscore_sample_detrend_FWHM6_low-pass428_10-12-24/'
setup_preproc = utils.load_pickle(os.path.join(preproc, 'setup_and_files.pkl'))
sugg_type = list(setup_preproc.suggestion_imgs_files.keys())
save_mean = os.path.join(preproc, 'mean_activation')
os.makedirs(save_mean, exist_ok=True)

for sugg in sugg_type:
    imgs = setup_preproc.suggestion_imgs_files[sugg]
    mean_imgs = mean_img(imgs)
    name = os.path.join(save_mean, f'{sugg}_mean_img.png')
    plotting.plot_stat_map(
            mean_imgs,
            title=f"Mean Activation - {sugg}",
            colorbar=True,
            display_mode='x',
            output_file=name
        )
    # save html plot
    html = plotting.view_img(mean_imgs)
    html.save_as_html(os.path.join(save_mean, f'{sugg}_mean_img.html'))


    # plot mean img

# %%
