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
            vols = func_ls[0] # only 1 img in list
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

        if ref_shape[:3] != check_other_img.shape: # :3 in case its 4D.
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



def dot(path_to_img, path_output, to_dot_with = 'nps', dot_id = None, resample_to_mask=True, img_format = '*.nii',participant_folder = True, filter_data = False):
    """
    Parameters
    ----------
    path_to_img : string
        path to the files (nii, hdr or img) on which the signature will be applied
    path_output : string 
        path to save the ouput of the function (i.e., dot product and related file)
    to_dot_with : string, default = 'nps'
         signature or fmri images to apply on the data, the maks,e.g. NPS. The paths to the signature files are defined inside the function.
         Can also be a *list* of paths to images.
    dot_id : string, default = None
        name of the experimental condition or id to include in output filename
    img_format : string, default = '.nii'
        change according to the format of fmri images, e.g. '*.hdr' or '*.img'
    participant_folder : string, default = True
        If True, it's assumed that the path_to_img contains a folder for each participant that contain the fMRI image(s) that will be used to compute smt product. If False,
        it is assumed that all the images that will be used to compute dot product are directly in path_to_img
    filter_data : Bool. or list; False by default, if not False, it is exêcted that a list od string is given as arg. This will filter the data used for
                    dot product based on the string present in the filename. E.g. 'filename1','filename2' filter_data = ['2'], will noly keep filename2 for dot
    Returns
    -------
    dot_array : numpy array
    subj_array : list
    """
    pwd = os.getcwd()
    print(pwd)
    #Define signature path

    if to_dot_with == "nps":
        mask_path = os.path.join(pwd,r"signatures/weights_NSF_grouppred_cvpcr.hdr")
    if to_dot_with == "siips":
        mask_path = "nonnoc_v11_4_137subjmap_weighted_mean.nii"
    if to_dot_with == "vps":
        mask_path = "bmrk4_VPS_unthresholded.nii"

    if resample_to_mask:
        resamp = "img_to_mask"
    if resample_to_mask == False:
        resamp = "mask_to_img"

    #----------Formatting files--------

    #returns a list with all the paths of images in the provided argument 'path_to_img'
    if participant_folder:
        fmri_imgs = glob.glob(os.path.join(path_to_img,'*',img_format)) #e.g : path/*all_subjects'_folders/*.nii
    else:
        fmri_imgs = glob.glob(os.path.join(path_to_img,img_format)) #e.g : path/*.nii
        #fmri_imgs = glob.glob(os.path.join(path,'*','*.nii'))

    if filter_data:
        fmri_imgs = list(keep_sub_data(fmri_imgs, filter_data))


    #fmri_imgs = list(filter(lambda x: "hdr" in x, data_path))
    fmri_imgs.sort(reverse=True)
    fmri_imgs.sort()

    if type(to_dot_with) is list:
        to_dot_with.sort(reverse=True)
        to_dot_with.sort()
        print('NOTICE : \'Mask\' is a list of fmri images')
    else :
        mask = nib.load(mask_path) #Load the mask as a nii file
        print('mask shape :  {} '.format(mask.shape))
    #------------------------

    #Initializing empty arrays for results)
    dot_array = np.array([])
    subj_array = []
    masks_names =[] #to save either the mask name or the list of images'names

    #----------Main loop--------------

    #For each map/nii file in the provided path
    indx = 0
    for maps in fmri_imgs:
        if type(to_dot_with) is list:#i.e. if a list of images was given instead of a unique mask. Masks_names was only defined in this specific case
            mask = nib.load(to_dot_with[indx])
            mask_name = os.path.basename(os.path.normpath(to_dot_with[indx])) #mask_name

        else:
            mask_name = to_dot_with #by default nps

        #load ongoing image
        img = nib.load(maps)

        #saving the file/subject's name in a list
        subj = os.path.basename(os.path.normpath(maps))
        subj_array.append(subj)

        #if image is not the same shape as the mask, the image will be resample
        #mask is the original mask to dot product with the provided images
        if img.shape != mask.shape:
        #---------Resampling--------
            print('Resampling : ' + os.path.basename(os.path.normpath(maps)))
            #resampling img to mask's shape
            if resample_to_mask:
                resampled = image.resample_to_img(maps,mask)
                print('image has been resample to mask\'s shape : {} '.format(resampled.shape))
            else:
                resampled = image.resample_to_img(mask,maps)
                print('Mask has been resample to image\'s shape : {} '.format(resampled.shape))

        else:#if input and image are the same size
            if resample_to_mask:
                resampled = img
            else:
                resampled = mask

        #---------fitting images to 1 dimension------------
        #making mask and image a vector in order to dot product

        #Fitting the masker of the 'mask' for which we want to dot product the beta maps
        masker_all = NiftiMasker(mask_strategy="whole-brain-template")

        #masker of the initial mask provided, in our case the mask is called NPS
        if resample_to_mask:
            masker_NPS = masker_all.fit_transform(mask)
        else:
            masker_NPS = masker_all.fit_transform(img)

        #fitting temporary masker for ongoing beta map :'maps'
        masker_tmp = masker_all.fit_transform(resampled)

        #---------Dot product---------
        print(subj,' dot with : ', mask_name)
        print(f'Computing dot product : {indx + 1}/{len(fmri_imgs)}')
        print('---------------------------')
        #dot product of the image's masker with the mask(NPS)'s masker
        dot_res = np.dot(masker_tmp,masker_NPS.T)
        print(dot_res)
        #storing the result in array
        dot_array = np.append(dot_array,dot_res)
        masks_names.append(mask_name)

        indx += 1

    if type(to_dot_with) is list:
        to_dot_with = 'aslist'

    if dot_id == None:
        df_res = pd.concat([pd.DataFrame(dot_array.T, columns = [f'dot_results_{resamp}_{to_dot_with}']),
            pd.DataFrame(subj_array, columns = ['files']),
            pd.DataFrame(masks_names, columns = ['masks'])],axis=1)
        if path_output is False: # if no output path have been entered, e.i. = False, results will be saved as pickle in pwd
            with open(f'dot_{resamp}_{to_dot_with}.pickle', 'wb') as handle:
                pickle.dump(df_res, handle, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            df_res.to_csv(os.path.join(path_output,f'results_{resamp}_{to_dot_with}.csv'))

    else:
        df_res = pd.concat([pd.DataFrame(dot_array.T, columns = [f'dot_results_{resamp}_{to_dot_with}_{dot_id}']),
            pd.DataFrame(subj_array, columns = ['files']),
            pd.DataFrame(masks_names, columns = ['masks'])], axis=1)
        if path_output is False:
            with open(f'results_{resamp}_{to_dot_with}_{dot_id}.pickle', 'wb') as handle:
                pickle.dump(df_res, handle, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            df_res.to_csv(os.path.join(path_output,f'results_{resamp}_{to_dot_with}_{dot_id}.csv'))

    return dot_array, subj_array


import numpy as np
import nibabel as nib
import pandas as pd
from nilearn import image

def compute_similarity(data_files, signature_folder, pattern='NPS', metric='dot_product', resample_to_mask=True):
    """
    Compute similarity between a single signature image and multiple subject images.
    
    Parameters
    ----------
    data_files : dict
        Dictionary mapping subject names to the file path of their data image.
    signature_file : str
        File path of the signature image.
    metric : str, default='dot_product'
        Similarity metric to use. Supported options:
         - 'dot_product': Raw dot product of the flattened images.
         - 'cosine': Cosine similarity between the flattened images.
    resample_to_mask : bool, default=True
        If True, resample the data image to the signature's space.
        Otherwise, resample the signature image to the data image's space.
        
    Returns
    -------
    results_df : pandas.DataFrame
        DataFrame with subject names as the index and a column 'similarity'
        containing the computed similarity values.
    """
    # Load the signature image
    if pattern == 'NPS':
        signature_file = os.path.join(signature_folder, 'weights_NSF_grouppred_cvpcr.img')
    elif pattern == 'SIIPS':
        signature_file = os.path.join(signature_folder, 'nonnoc_v11_4_137subjmap_weighted_mean.nii')
    signature_img = nib.load(signature_file)
    
    # Get signature data and affine
    sig_data = signature_img.get_fdata()
    sig_affine = signature_img.affine
    
    results = []
    
    for subj, data_path in data_files.items():
 
        data_img = nib.load(data_path)
        data_data = data_img.get_fdata()
        data_affine = data_img.affine
        
        # Check if shape and affine match
        same_shape = data_data.shape == sig_data.shape
        same_affine = np.allclose(data_affine, sig_affine)
        
        if not (same_shape and same_affine):
            if resample_to_mask:
                # Resample subject image to signature image space
                data_img = image.resample_to_img(data_img, signature_img, interpolation='continuous', force_resample=True, copy_header=True)
                data_data = data_img.get_fdata()
            else:
                # Resample signature image to subject image space
                signature_img_res = image.resample_to_img(signature_img, data_img, interpolation='continuous', force_resample=True, copy_header=True)
                sig_data = signature_img_res.get_fdata()
                sig_affine = signature_img_res.affine
        
        # Flatten the images to vectors
        data_vector = data_data.ravel()
        sig_vector = sig_data.ravel()
        
        # Compute similarity
        if metric == 'dot_product':
            sim_value = np.dot(data_vector, sig_vector)
        elif metric == 'cosine':
            # Compute cosine similarity (avoid division by zero)
            norm_data = np.linalg.norm(data_vector)
            norm_sig = np.linalg.norm(sig_vector)
            if norm_data == 0 or norm_sig == 0:
                sim_value = np.nan
            else:
                sim_value = np.dot(data_vector, sig_vector) / (norm_data * norm_sig)
        else:
            raise ValueError("Unsupported metric. Use 'dot_product' or 'cosine'.")
        
        results.append({'subject': subj, 'similarity': sim_value})
    
    # Return as a DataFrame with subject names as index
    results_df = pd.DataFrame(results).set_index('subject')


    return results_df
