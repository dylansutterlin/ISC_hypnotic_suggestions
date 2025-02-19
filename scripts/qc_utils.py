# quality check for nifti files 



def assert_same_affine(concatenated_subjects, check_other_img=None):
    '''
    Check if all subjects have the same affine matrix and shape
    Dict. of subject with 4D timseries

    Parameters
    ----------
    concatenated_subjects : dict
    check_other_img : nibabel image to compare with the concatenated subjects

    '''
    subjects = concatenated_subjects.keys()
    for i, sub in enumerate(subjects):
        vols = concatenated_subjects[sub]

        if i == 0:
            ref_aff = vols.affine
            ref_shape = vols.shape
        else:
            curr_aff = vols.affine
            curr_shape = vols.shape

            if not np.allclose(curr_aff, ref_aff):
                print(f"Warning: Subject {sub} has a different affine matrix.")

            if curr_shape != ref_shape:
                print(f"Warning: Subject {sub} has a different shape {curr_shape} compared to reference {ref_shape}.")

    if check_other_img is not None:
        if not np.allclose(ref_aff, check_other_img.affine):
            print("Warning: The affine matrix of the other image is different from the concatenated subjects.")
        if ref_shape != check_other_img.shape:
            print(f"Warning: The shape of the other image {check_other_img.shape} is different from the concatenated subjects.")



def downsample_for_tutorial(nii_file, output_dir):
    """Downsample atlas to match developmental dataset.
    Modified from https://osf.io/wjtyq
    """
    fname = os.path.basename(nii_file)
    aff_orig = np.array([ -96., -132.,  -78.,    1.])  # from developmental dataset
    target_affine = np.column_stack([np.eye(4, 3) * 4, aff_orig])
    downsample_data = image.resample_img(
        nii_file,
        target_affine=target_affine,
        target_shape=(50, 59, 50),
        interpolation='nearest',
        force_resample=True,
        copy_header=True
    )
    downsample_data.to_filename(Path(output_dir) / f'downsample_{fname}')
    return Path(output_dir) / f'downsample_{fname}'