from sklearn.metrics.pairwise import cosine_similarity
from nilearn.image import load_img
from sklearn.metrics import pairwise_distances
import numpy as np
import pandas as pd
import os
from sklearn.covariance import LedoitWolf
from scipy.spatial import distance


def compute_inter_subject_mvpa_similarity(condition_dict, atlas_img, labels_roi_dct, sim_method = 'cosine'):
    """
    Computes inter-subject cosine similarity matrices (subject x subject) for each ROI
    based on multivoxel patterns from a single condition.

    Parameters
    ----------
    condition_dict : dict
        Dictionary mapping subject IDs to NIfTI image file paths for a single condition.
    atlas_img : Nifti1Image
        NIfTI image of the brain atlas (e.g., Schaefer parcellation).
    roi_indices : list of int
        List of ROI indices to include.

    Returns
    -------
    similarity_matrices : dict
        Keys are ROI indices, values are subject x subject cosine similarity matrices.
    vectorized_df : pd.DataFrame
        DataFrame where rows are pairwise subject combinations and columns are ROIs.
        Each cell is the cosine similarity between two subjects for that ROI.
    """

    atlas_data = np.round(atlas_img.get_fdata()).astype(int)
    subjects = sorted(condition_dict.keys())
    n_subjects = len(subjects)
    roi_indices = list(labels_roi_dct.keys())
    roi_labels = list(labels_roi_dct.values())

    # Load all data once
    condition_data_dict = {
        subj: load_img(condition_dict[subj]).get_fdata() for subj in subjects
    }

    similarity_matrices = {}
    similarity_vectors = []

    for roi_idx in roi_indices:
        roi_mask = atlas_data == roi_idx
        if not roi_mask.any():
            print("Region id not found in atlas, skipping..")
            continue

        subj_vectors = []

        for subj in subjects:
            vec = condition_data_dict[subj][roi_mask].flatten()

            if np.linalg.norm(vec) == 0:
                subj_vectors.append(np.full(np.sum(roi_mask), np.nan))
                print("region mask is all 0 ")
            else:
                subj_vectors.append(vec)

        subj_vectors = np.array(subj_vectors)

        valid_mask = ~np.isnan(subj_vectors).any(axis=1)
        valid_vectors = subj_vectors[valid_mask]
        valid_subjects = np.array(subjects)[valid_mask]

        if valid_vectors.shape[0] < 2:
            print("Too few subjects for a ROI, skipping..")
            continue
        
        if sim_method == 'cosine':
            sim_matrix = cosine_similarity(valid_vectors)
        elif sim_method == 'dot':
            sim_matrix = np.dot(valid_vectors, valid_vectors.T)
        elif sim_method == 'euclidean':
            dist_matrix = pairwise_distances(valid_vectors, metric='euclidean')
            sim_matrix = 1 - (dist_matrix / np.max(dist_matrix)) # diag =1 and between 0-1 simil
        elif sim_method == 'pearson':
            sim_matrix = np.corrcoef(valid_vectors)  # correlation between rows
        elif sim_method == 'mahalanobis':
            cov = LedoitWolf().fit(valid_vectors).covariance_
            inv_cov = np.linalg.inv(cov)
            dist_matrix = pairwise_distances(valid_vectors, metric='mahalanobis', VI=inv_cov)
            sim_matrix = 1 - (dist_matrix / np.max(dist_matrix))  # normalize like Euclidean

        else:
            raise ValueError(f"Unsupported similarity method '{sim_method}'. Choose 'cosine' or 'dot'.")
        similarity_matrices[roi_idx] = pd.DataFrame(
            sim_matrix, index=valid_subjects, columns=valid_subjects
        )

        # Extract upper triangle for this ROI
        upper_tri_indices = np.triu_indices(sim_matrix.shape[0], k=1)
        upper_tri_values = sim_matrix[upper_tri_indices]
        similarity_vectors.append(pd.Series(upper_tri_values, name=roi_idx))

    vectorized_df = pd.concat(similarity_vectors, axis=1)
    vectorized_df.columns.name = "ROIs"
    vectorized_df.columns = roi_labels

    return similarity_matrices, vectorized_df


def build_subject_dict(file_list):
    """Build a dict: {subject_id: filepath} from a list of NIfTI file paths."""
    subject_dict = {}
    for path in file_list:
        fname = os.path.basename(path)
        subj_id = fname.split("_")[-1].replace(
            ".nii.gz", ""
        )  # expects '..._sub-01.nii.gz'
        subject_dict[subj_id] = path
    return subject_dict

