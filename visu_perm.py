# %%
import os
import nibabel as nib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from nilearn import plotting
from nilearn.image import math_img, mean_img
from nilearn.maskers import MultiNiftiMapsMasker, NiftiMapsMasker
import scripts.isc_utils as isc_utils
import os
import matplotlib.pyplot as plt
from statsmodels.stats.multitest import multipletests
import scripts.visu_utils as vu
# Threshold values
p_threshold_uncorrected = 0.1  # Uncorrected
p_threshold_001 = 0.001
fdr_threshold = 0.05

roi_coords = {
"amcc": (-2, 20, 32),
"rPO": (54, -28, 26),
"lPHG": (-20, -26, -14),
}

import importlib
importlib.reload(isc_utils)

#model_name = "model1-22sub"
model_name = 'model2_zcore_sample-22sub'
project_dir = "/data/rainville/dSutterlin/projects/ISC_hypnotic_suggestions"
results_dir = os.path.join(project_dir, f'results/imaging/ISC/{model_name}')
atlas_path = os.path.join(project_dir, 'masks/DiFuMo256/3mm/maps.nii.gz')
conditions = ["all_sugg", "modulation", "neutral"]
setup = isc_utils.load_json(os.path.join(results_dir, 'setup_parameters.json'))

atlas_path = os.path.join(project_dir, 'masks/DiFuMo256/3mm/maps.nii.gz')
atlas_dict_path = os.path.join(project_dir, 'masks/DiFuMo256/labels_256_dictionary.csv')
atlas = nib.load(atlas_path)
atlas_df = pd.read_csv(atlas_dict_path)
atlas_labels = atlas_df['Difumo_names']

setup = isc_utils.load_json(os.path.join(results_dir, 'setup_parameters.json'))
n_boot = setup['n_boot']
n_perm = setup['n_perm']
do_pairwise = setup['do_pairwise']

behavioral_scores = pd.read_csv(r'/data/rainville/dSutterlin/projects/ISC_hypnotic_suggestions/results/imaging/ISC/model2_zcore_sample-22sub/behav_data_group_labels.csv', index_col=0)
# %%
cond = 'modulation'
masker = isc_utils.load_pickle(os.path.join(results_dir, cond, 'maskers_Difumo256_modulation_22sub.pkl'))
# %%
import importlib
importlib.reload(vu)


# Load the ISC results like isc_10000permutation_results_modulation_pairwiseTrue.pkl
cond = 'all_sugg'
perm_results_name = f'isc_{n_perm}permutation_results_{cond}_pairwise{do_pairwise}.pkl'
perm_results_path = os.path.join(results_dir,cond, perm_results_name)
perm_dict = isc_utils.load_pickle(perm_results_path)

y_var = 'total_chge_pain_hypAna'
observed_isc, p_values, distributions = vu.load_isc_results_permutation(perm_dict[y_var])

_, fdr_corrected_pvals, _, _ = multipletests(p_values, alpha=0.05, method='fdr_bh')
significant_mask = p_values < 0.05

# Inverse transform ISC values and corrected p-values to brain space
isc_img = masker.inverse_transform(observed_isc)
pval_img = masker.inverse_transform(-np.log10(fdr_corrected_pvals))  # Use -log10(p) for visualization
significant_img = masker.inverse_transform(significant_mask.astype(int))  # Binary significance map

# %%
# Save and plot ISC map

from nilearn.glm import threshold_stats_img

thresholded_map1, threshold1 = threshold_stats_img(
    isc_img,
    alpha=0.05,
    height_control="fdr",
    two_sided=True,
)
print(f"Threshold: {threshold1}")
plotting.view_img(thresholded_map1, title=f"ISC Interactive Map ({cond})")
# %%

isc_save_path = os.path.join(results_dir, cond, "isc_map_fdr_corrected.nii.gz")
isc_img.to_filename(isc_save_path)
print(f"ISC map saved to {isc_save_path}")

plotting.plot_stat_map(
    isc_img,
    title=f"ISC Map ({cond}, FDR Corrected)",
    display_mode="z",
    cut_coords=10,
    colorbar=True,
    output_file=os.path.join(results_dir, cond, "isc_map_fdr_corrected.png")
)

# Save and plot -log10(p) map
pval_save_path = os.path.join(results_dir, cond, "pval_map_fdr_corrected.nii.gz")
pval_img.to_filename(pval_save_path)
print(f"p-value map saved to {pval_save_path}")

plotting.plot_stat_map(
    pval_img,
    title=f"-log10(p) Map ({cond}, FDR Corrected)",
    display_mode="z",
    cut_coords=10,
    colorbar=True,
    output_file=os.path.join(results_dir, cond, "pval_map_fdr_corrected.png")
)

# Interactive plot (HTML)
interactive_plot_path = os.path.join(results_dir, cond, "interactive_isc_map.html")
view = plotting.view_img(isc_img, title=f"ISC Interactive Map ({cond})")
view.save_as_html(interactive_plot_path)
print(f"Interactive ISC map saved to {interactive_plot_path}")

# Optional: Visualize the binary significant mask
plotting.plot_roi(
    significant_img,
    title=f"Significant Regions (FDR Corrected, {cond})",
    display_mode="z",
    cut_coords=10,
    colorbar=False,
    output_file=os.path.join(results_dir, cond, "significant_regions_fdr.png")
)

