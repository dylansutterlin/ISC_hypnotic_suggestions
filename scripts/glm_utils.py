import numpy as np
import pandas as pd
import json
import pickle as pkl
import matplotlib.pyplot as plt
import seaborn as sns



def save_pickle(save_path, data):
    with open(save_path, 'wb') as f:
        pkl.dump(data, f)

def load_pickle(file_path):
    with open(file_path, 'rb') as f:
        data = pkl.load(f)
    return data

def save_json(file_path, data):
    with open(file_path, "w") as f:
        json.dump(data, f, indent=4)
    
def load_json(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

def load_data_mask(ref_img):
    mask = '/data/rainville/Hypnosis_ISC/masks/brainmask_91-109-91.nii'
    
    from qc_utils import resamp_to_img_mask, assert_same_affine
    from nilearn.image import concat_imgs
    mask = resamp_to_img_mask(mask, ref_img)
    assert_same_affine([ref_img], subjects=['mask'], check_other_img=mask)
    print(mask.shape)

    return mask

def make_contrasts(design_matrix, regressors_dict, plot=True):
    """
    Generate contrast vectors based on the design matrix and defined conditions.
    Also visualizes them to ensure correctness.

    Parameters:
    - design_matrix (pd.DataFrame): Subject's design matrix.
    - regressors_dict (dict): Mapping of conditions to lists of corresponding regressors.
    - plot (bool): Whether to visualize the contrast vectors.

    Returns:
    - dict: Contrast dictionary where each key is a condition, and the value is a contrast vector.
    """
    n_columns = design_matrix.shape[1]
    contrasts = {}

    for condition, regressors in regressors_dict.items():
        contrast_vector = np.zeros(n_columns)
        for regressor in regressors:
            if regressor in design_matrix.columns:
                contrast_vector[design_matrix.columns.get_loc(regressor)] = 1  # Activate regressor

        contrasts[condition] = contrast_vector

    #  Plot the contrast vectors as a heatmap to verify
    if plot:
        plt.figure(figsize=(18, len(contrasts) * 0.6))
        contrast_matrix = np.array(list(contrasts.values()))
        sns.heatmap(contrast_matrix, cmap="coolwarm", xticklabels=design_matrix.columns, 
                    yticklabels=contrasts.keys(), cbar=False, linewidths=0.5)

        # Make X-axis labels smaller and rotated
        plt.xticks(rotation=60, ha='right', fontsize=8)  # Rotate labels and make them smaller
        plt.yticks(fontsize=9)  # Adjust Y labels size
        plt.xlabel("Regressors in Design Matrix")
        plt.ylabel("Contrasts")
        plt.title("Contrast Vector Heatmap (Sanity Check)")
        plt.show()

    return contrasts
