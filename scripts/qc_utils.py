import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from nilearn.maskers import NiftiLabelsMasker, MultiNiftiLabelsMasker
from nilearn.image import index_img, resample_to_img
# quality check for nifti files 


def assert_same_affine(func_ls, subjects, check_other_img=None):
    '''
    Check if all subjects have the same affine matrix and shape
    Dict. of subject with 4D timseries

    Parameters
    ----------
    concatenated_subjects : dict
    check_other_img : nibabel image to compare with the concatenated subjects

    '''
    looks_good = True
    
    for i, sub in enumerate(subjects):
        if len(func_ls) == 1:
            vols = func_ls
        else:
            vols = index_img(func_ls[i], 0)

        if i == 0:
            ref_aff = vols.affine
            ref_shape = vols.shape
        else:
            curr_aff = vols.affine
            curr_shape = vols.shape

            if not np.allclose(curr_aff, ref_aff):
                print(f"Warning: Subject {sub} has a different affine matrix.")
                looks_good = False
            if curr_shape != ref_shape:
                print(f"Warning: Subject {sub} has a different shape {curr_shape} compared to reference {ref_shape}.")
                looks_good = False
    if check_other_img is not None:
        if not np.allclose(ref_aff, check_other_img.affine):
            print("Warning: The affine matrix of the other image is different from the concatenated subjects.")
            print(f"Reference affine: {ref_aff}")
            print(f"Other affine: {check_other_img.affine}")
            looks_good = False
        if ref_shape != check_other_img.shape:
            print(f"Warning: The shape of the other image {check_other_img.shape} is different from the concatenated subjects.")
            print(f"Reference shape: {ref_shape}")
        
            looks_good = False

    if looks_good:
        print("Affine matrix and shape are consistent across all subjects.")

    return looks_good

def resamp_to_img_mask(img, ref_img):
    return resample_to_img(img, ref_img,
        interpolation='nearest',
        force_resample=True,
        copy_header=True )

# def downsample_for_tutorial(nii_file, output_dir):
#     """Downsample atlas to match developmental dataset.
#     Modified from https://osf.io/wjtyq
#     """
#     fname = os.path.basename(nii_file)
#     aff_orig = np.array([ -96., -132.,  -78.,    1.])  # from developmental dataset
#     target_affine = np.column_stack([np.eye(4, 3) * 4, aff_orig])
#     downsample_data = image.resample_img(
#         nii_file,
#         target_affine=target_affine,
#         target_shape=(50, 59, 50),
#         interpolation='nearest',
#         force_resample=True,
#         copy_header=True
#     )
#     downsample_data.to_filename(Path(output_dir) / f'downsample_{fname}')
#     return Path(output_dir) / f'downsample_{fname}'


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
