from nilearn.image import math_img
from nilearn.plotting import plot_stat_map
import os

def threshold_and_save_isc_maps(isc_img, p_img, masker, results_dir, cond, atlas_name, n_sub, thresholds=("unc", 0.001), fdr_correct=False):
    """
    Threshold and save ISC brain maps based on p-values.
    
    Parameters:
    - isc_img: Nifti image of ISC values.
    - p_img: Nifti image of p-values.
    - masker: The masker object for inverse transform.
    - results_dir: Directory to save results.
    - cond: The condition name.
    - atlas_name: Name of the atlas used.
    - n_sub: Number of subjects.
    - thresholds: Tuple defining ('unc', value) or None for unthresholded maps.
    - fdr_correct: Boolean to indicate if FDR correction is applied.
    """
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    # Prepare file names
    base_filename = f"isc_thresholded_{cond}_{atlas_name}_{n_sub}sub"
    fdr_suffix = "_fdr" if fdr_correct else ""
    
    for threshold_type, value in thresholds:
        # Apply uncorrected or FDR thresholds
        if threshold_type == "unc":
            thresholded_p = math_img(f"img < {value}", img=p_img)
        else:
            raise ValueError("Unsupported threshold type. Use 'unc'.")
        
        # Apply threshold to ISC image
        isc_thresholded = math_img("isc * (p < thresh)", isc=isc_img, p=p_img, thresh=value)
        
        # Save thresholded ISC map
        isc_thresholded_filename = os.path.join(
            results_dir, f"{base_filename}_thresh_{threshold_type}{value}{fdr_suffix}.nii.gz"
        )
        isc_thresholded.to_filename(isc_thresholded_filename)
        print(f"Saved thresholded ISC map at: {isc_thresholded_filename}")
        
        # Plot and save visualization
        plot_filename = os.path.join(
            results_dir, f"{base_filename}_thresh_{threshold_type}{value}{fdr_suffix}.png"
        )
        plot_stat_map(
            isc_thresholded,
            title=f"ISC Map ({cond}) Threshold {threshold_type}: {value}",
            display_mode="z",
            cut_coords=10,
            colorbar=True,
            output_file=plot_filename,
        )
        print(f"Saved thresholded ISC plot at: {plot_filename}")

# Example Usage
# Load ISC and p-value maps (assuming you have already calculated and saved them)
for cond, isc_dict in isc_results.items():
    if cond != "modulation":  # Modify as per your requirement
        continue

    # Paths
    isc_path = os.path.join(results_dir, cond, f"isc_val_{cond}_boot{n_boot}_pariwise{do_pairwise}.nii.gz")
    pval_path = os.path.join(results_dir, cond, f"p_values_{cond}_boot{n_boot}_pairwise{do_pairwise}.nii.gz")
    isc_img = nib.load(isc_path)
    p_img = nib.load(pval_path)
    
    # Call the function for thresholds
    thresholds = [("unc", 0.001)]  # Uncorrected threshold
    threshold_and_save_isc_maps(isc_img, p_img, masker, results_dir, cond, atlas_name, n_sub, thresholds, fdr_correct=False)
