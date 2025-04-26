import numpy as np
import pandas as pd
import json
import pickle as pkl
import matplotlib.pyplot as plt
import seaborn as sns
from importlib import reload

import src.qc_utils as utils
import src.preproc_utils as preproc

import os
import glob     
import nibabel as nib
from nilearn.plotting import plot_design_matrix
from nilearn.image import concat_imgs
from nilearn.glm.first_level import make_first_level_design_matrix
from sklearn.utils import Bunch
from datetime import datetime


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

from nilearn import datasets
def load_data_mask(ref_img):
    mask = '/data/rainville/Hypnosis_ISC/masks/brainmask_91-109-91.nii'
    mask_native = datasets.load_mni152_brain_mask()
    # from qc_utils import resamp_to_img_mask, assert_same_affine

    # reload(utils)
    mask = utils.resamp_to_img_mask(mask_native, ref_img)
    utils.assert_same_affine([ref_img], subjects=['mask'], check_other_img=mask)
    print(mask.shape)

    return mask

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def make_localizer_vec_from_reg(design_matrix, regressors_dict, extra_indices_dct=None, direction=1, plot=True, save_to = None):

    """
    Generate contrast vectors for a localizer using two sources:
      1. The regular method: mapping condition names to lists of regressor names.
      2. Optionally, extra conditions specified by lists of indices. 
         These indices refer to positions in the ordered list of keys from regressors_dict.
         For each specified index, the corresponding condition’s regressors will be added (i.e., set to 1, summed if repeated)
         in the new contrast vector.
    
    Parameters
    ----------
    design_matrix : pd.DataFrame
        The subject's design matrix.
    regressors_dict : dict
        Mapping of condition names to lists of regressor names.
        For example: {'ANA_sugg': ['reg1', 'reg2'], 'HYPER_sugg': ['reg3']}
    extra_indices_dct : dict, optional
        Dictionary mapping extra condition names to lists of indices. 
        For example: {'all_sugg': [0, 1], 'all_shock': [2, 3]} 
        This means that for 'all_sugg', the function will add the regressors from the conditions at 
        positions 0 and 1 in the ordered keys of regressors_dict.
    plot : bool, default True
        Whether to plot a heatmap of the contrast vectors for a sanity check.
    
    Returns
    -------
    contrasts : dict
        A dictionary where each key is a condition and the value is the corresponding contrast vector.
    """
    n_columns = design_matrix.shape[1]
    contrasts = {}

    # Build contrasts using regressor names from regressors_dict.
    for condition, regressors in regressors_dict.items():
        contrast_vector = np.zeros(n_columns)
        for regressor in regressors:
            if regressor in design_matrix.columns:
                contrast_vector[design_matrix.columns.get_loc(regressor)] = direction
            else:
                raise ValueError(f"Regressor '{regressor}' not found in demake_contrast_vec_from_regsign matrix columns.")
        contrasts[condition] = contrast_vector

    # Add extra conditions based on provided indices from extra_indices_dct.
    # The indices refer to positions in the ordered list of keys of regressors_dict.
    if extra_indices_dct is not None:

        reg_keys = list(regressors_dict.keys())

        for new_condition, indices in extra_indices_dct.items():

            contrast_vector = np.zeros(n_columns)

            for idx in indices:

                if idx < 0 or idx >= len(reg_keys):
                    raise ValueError(f"Index {idx} out of bounds for regressors_dict with {len(reg_keys)} keys.")
                # Get the condition key corresponding to this index.
                cond_key = reg_keys[idx]

                # For each regressor in that condition, add 1.
                for reg in regressors_dict[cond_key]:
                    if reg in design_matrix.columns:
                        contrast_vector[design_matrix.columns.get_loc(reg)] += direction
                    else:
                        raise ValueError(f"Regressor '{reg}' from condition '{cond_key}' not found in design matrix columns.")
            # Add or override the contrast for this new condition.
            contrasts[new_condition] = contrast_vector

    # Optionally, plot a heatmap of the contrast vectors.
    if plot:
        plt.figure(figsize=(18, len(contrasts) * 0.6))
        contrast_matrix = np.array(list(contrasts.values()))
        sns.heatmap(contrast_matrix, cmap="coolwarm", 
                    xticklabels=design_matrix.columns, 
                    yticklabels=list(contrasts.keys()), 
                    cbar=False, linewidths=0.5)
        plt.xticks(rotation=60, ha='right', fontsize=8)
        plt.yticks(fontsize=9)
        plt.xlabel("Regressors in Design Matrix")
        plt.ylabel("Contrasts")
        plt.title("Contrast Vector Heatmap (Sanity Check)")

        if save_to != None:
            plt.savefig(os.path.join(save_to, 'localizer_weights.png'))

        plt.show()

    return contrasts



def make_contrast_vec_from_reg(design_matrix, regressors_dict, contrast_spec, plot=True, save_to=None):
    """
    Generate contrast vectors based on a contrast specification.
    
    The contrast_spec should be a dictionary where keys are of the form:
      "cond1_minus_cond2" or "cond1+cond2_minus_cond3+cond4"
    and values are tuples (w1, w2). For each key, all regressors in 
    regressors_dict[cond1] (and cond2 if present) will be given weight w1, 
    and those in regressors_dict[cond3] (and cond4 if present) will be given w2.
    
    Parameters:
      design_matrix (pd.DataFrame): Subject's design matrix.
      regressors_dict (dict): Mapping of condition names (e.g., "ANA_sugg") 
                              to lists of regressor names.
      contrast_spec (dict): For example:
          {
              "ANA_sugg_minus_N_ANA_sugg": (1.5, -1),
              "HYPER_sugg_minus_N_HYPER_sugg": (1.5, -1),
              "ANA_shock+HYPER_shock_minus_N_ANA_shock+N_HYPER_shock": (1, -1)
          }
      plot (bool): Whether to plot the contrast vectors.
      
    Returns:
      dict: A dictionary where each key is the contrast specification string and 
            the value is the corresponding contrast vector.
    """
    n_columns = design_matrix.shape[1]
    contrasts = {}
    
   
    for key, (w1, w2) in contrast_spec.items():
        parts = key.split('_minus_')
        if len(parts) != 2:
            raise ValueError("Contrast spec key must be of the form 'cond1_minus_cond2'")
        
        pos_str, neg_str = parts
        pos_conditions = pos_str.split('+')
        neg_conditions = neg_str.split('+')
        
        # Check that all extracted conditions match exactly keys in regressors_dict
        for cond in pos_conditions:
            if cond not in regressors_dict:
                raise ValueError(f"Positive condition '{cond}' not found in regressors_dict keys.")
        for cond in neg_conditions:
            if cond not in regressors_dict:
                raise ValueError(f"Negative condition '{cond}' not found in regressors_dict keys.")
        

        contrast_vector = np.zeros(n_columns)
        
        # Assign positive weights.
        for cond in pos_conditions:
            if cond in regressors_dict:
                for reg in regressors_dict[cond]:
                    if reg in design_matrix.columns:
                        contrast_vector[design_matrix.columns.get_loc(reg)] = w1
        
        # Assign negative weights.
        for cond in neg_conditions:
            if cond in regressors_dict:
                for reg in regressors_dict[cond]:
                    if reg in design_matrix.columns:
                        contrast_vector[design_matrix.columns.get_loc(reg)] = w2
        
        contrasts[key] = contrast_vector

    if plot:
        contrast_matrix = np.array(list(contrasts.values()))
        plt.figure(figsize=(15, len(contrasts) * 0.5))
        sns.heatmap(contrast_matrix, cmap="coolwarm", 
                    xticklabels=design_matrix.columns, 
                    yticklabels=list(contrasts.keys()), 
                    cbar=True, 
                    cbar_kws={"label": "Contrast Weight"},
                    linewidths=0.5)
        plt.xticks(rotation=90, ha="right", fontsize=8)
        plt.title("Condition Contrast Vector Heatmap")
        plt.tight_layout()

        if save_to != None:
            plt.savefig(os.path.join(save_to, 'contrasts_weights.png'))
    
        plt.show()
    
    return contrasts


def prep_glm(nsub=3):
    data_dir = '/data/rainville/Hypnosis_ISC/4D_data/full_run'
    ana_run = glob.glob(os.path.join(data_dir, 'sub*', '*analgesia*.nii.gz'))
    subjects = [os.path.basename(os.path.dirname(path)) for path in ana_run]
    nsub = nsub #len(subjects)


    setup = Bunch(
        data_dir="/data/rainville/Hypnosis_ISC/4D_data/full_run",
        run_names = ["Analgesia", "Hyperalgesia"],
        ana_run=glob.glob(
            os.path.join(
                "/data/rainville/Hypnosis_ISC/4D_data/full_run",
                "sub*",
                "*analgesia*.nii.gz",
            )
        )[0:nsub],
        hyper_run=glob.glob(
            os.path.join(
                "/data/rainville/Hypnosis_ISC/4D_data/full_run",
                "sub*",
                "*hyperalgesia*.nii.gz",
            )
        )[0:nsub],
        behav_path="/data/rainville/dSutterlin/projects/resting_hypnosis/resting_state_hypnosis/atlases/Hypnosis_variables_20190114_pr_jc.xlsx",
        events_dir = "/data/rainville/HYPNOSIS_IMAGING_DATA/timestamps",
        confound_dir = "/data/rainville/HYPNOSIS_IMAGING_DATA/Nii",
        project_dir="/data/rainville/dSutterlin/projects/ISC_hypnotic_suggestions",
        results_dir=os.path.join(
            "/data/rainville/dSutterlin/projects/ISC_hypnotic_suggestions",
            "results/imaging/GLM",
        ),  
        subjects=subjects[0:nsub]

    )
    setup.APM_subjects = ['APM' + sub[4:] for sub in setup.subjects][0:nsub] # make APMXX format instead of subXX
    setup.tr = 3

    voxel_masker = Bunch(name="voxel_wise")


    mask = '/data/rainville/Hypnosis_ISC/masks/brainmask_91-109-91.nii'
    voxel_masker.mask = mask
    masker_params_dict = {
        "standardize": 'zscore_sample',
        "mask_img": mask,
        "detrend": False,
        "low_pass": None,
        "high_pass": None,  # 1/428 sec.
        "t_r": setup.tr,
        "smoothing_fwhm": None,
        "standardize_confounds": True,
        "verbose": 3,
        "high_variance_confounds": False,
        "mask_strategy": "whole-brain-template", 
        "memory" : "nilearn_cache"
    }

    # initialize preproc model and save dirs
    preproc_model_name = "model1_{}subjects_{}_detrend_{}".format(
        len(setup.subjects),masker_params_dict['standardize'], datetime.today().strftime("%d-%m-%y")
    )
    save_dir = os.path.join(setup.results_dir, preproc_model_name)
    if os.path.exists(save_dir):
        print(f"The folder '{save_dir}' already exists. Results will be overwritten!!.")
        error = input("Do you want to continue? (y/n)")
    else:
        os.makedirs(save_dir, exist_ok=True)
    setup.model_name = preproc_model_name
    setup.save_dir = save_dir

    # save meta data
    # masker_params_path = os.path.join(save_dir, "preproc_params.json")
    # utils.save_json(masker_params_path, masker_params_dict)

    # save_setup_p = os.path.join(setup.save_dir, "setup_parameters.json")
    # utils.save_json(save_setup_p, setup)

    # %%
    # extract event files
    events_dir = setup.events_dir
    confound_dir = setup.confound_dir
    run_names = setup.run_names
    APM_subjects = setup.APM_subjects

    events_ana = [
        preproc.get_timestamps(
            data_dir, sub, events_dir, run_names[0], return_path=False
        ).sort_values(by="onset")
        for sub in APM_subjects
    ]
    events_hyper = [
        preproc.get_timestamps(
            data_dir, sub, events_dir, run_names[1], return_path=False
        ).sort_values(by="onset")
        for sub in APM_subjects
    ]

    # ANA run first!
    confound_files = []
    for sub in APM_subjects:

        formatted_sub = f"{sub[:3]}_{sub[3:]}*"
        # print(formatted_sub)
        # Extend the confound_files list with the matching files
        confound_files.extend(
            glob.glob(os.path.join(confound_dir, "H*", "*", formatted_sub, "*_8nuisreg_*"))
        )
        # print(glob.glob(os.path.join(confound_dir, 'H*', '*', formatted_sub, '*_8nuisreg_*')))
    print("len confound files : ", len(confound_files))

    cf4 = []  # extra 4 sub with diff TRs in H2/APM*/*nuis.txt
    for sub in APM_subjects:
        formatted_sub = f"{sub[:3]}_{sub[3:]}*"
        cf4.extend(
            glob.glob(os.path.join(confound_dir, "H*", formatted_sub, "*_8nuisreg_*"))
        )

    print(f"found {len(cf4)} extra irreglar-sub condound files")
    confound_files.extend(cf4)

    # Reorder matching APM_subjects
    ordered_confound_files = []
    for sub in APM_subjects:
        formatted_sub = f"{sub[:3]}_{sub[3:]}"
        match = [f for f in confound_files if formatted_sub in f]
        if match:
            ordered_confound_files.append(match[0])  # Take the first match

    confound_files = ordered_confound_files
    confound_files_dct = {sub: file for sub, file in zip(subjects, confound_files)}
    print("Total confound files : ", len(confound_files))

    data = Bunch()
    data.events_dfs = {}
    data.events_dfs["Ana"] = events_ana
    data.events_dfs["Hyper"] = events_hyper
    data.confound_files = confound_files

    # %%
    # Get the number of scans for each subject
    data.nscans = {}
    data.nscans["Ana"] = {
        sub: nib.load(img).shape[-1] for sub, img in zip(subjects, setup.ana_run)
    }
    data.nscans["Hyper"] = {
        sub: nib.load(img).shape[-1] for sub, img in zip(subjects, setup.hyper_run)
    }

    # Split movement regressor file in two according to scan number of each
    # output loaded np arrays fir each run
    ana_confounds = []
    hyper_confounds = []
    for i, sub in enumerate(setup.subjects):

        conf_path = confound_files[i]
        confounds = np.array(pd.read_csv(conf_path, sep="\s+", header=None))

        print(confounds.shape, type(confounds), "len % 2 :", len(confounds) / 2)

        nscan_ana = data.nscans["Ana"][sub]
        up_ana = confounds[0:nscan_ana, :]

        nscan_hyper = data.nscans["Hyper"][sub]
        low_hyper = confounds[nscan_ana : nscan_ana + nscan_hyper, :]

        print("low_ana shape:", up_ana.shape)
        print("low_hyper shape:", low_hyper.shape)

        ana_confounds.append(up_ana)
        hyper_confounds.append(low_hyper)

        # up, low = split_conf_matrix(confound_files[i], nscan, run_names[i]
        #    )

    data.confounds_Ana = ana_confounds
    data.confounds_Hyper = hyper_confounds

    # %%
    run_names_short = ['Ana','Hyper'] # [setup.conditions[0][0:3], setup.conditions[1][0:5]] # Ana and Hyper
    design_matrices_2runs = {}
    design_matrices_2runs_files = []
    dm_ana_df = []
    dm_hyper_df = []
    dm_ana_files = []
    dm_hyper_files = []

    save_path = os.path.join(setup.save_dir, "design_matrices_TxT")
    os.makedirs(save_path, exist_ok=True)

    for i, sub in enumerate(setup.subjects):

        tr = masker_params_dict["t_r"]
        # names = setup.run_names
        # ts_ana = voxel_masker.preproc_2d_Ana[i]
        # ts_hyper = voxel_masker.preproc_2d_Hyper[i]
        nscans_ana = nib.load(setup.ana_run[i]).shape[-1] #ts_ana.shape[0]
        frame_time_ana = np.arange(nscans_ana) * tr
        nscans_hyper = nib.load(setup.hyper_run[i]).shape[-1] #ts_hyper.shape[0]
        frame_time_hyper = np.arange(nscans_hyper) * tr

        # Assert that nscans match for both runs
        # warnings == ts_hyper.shape[0], f"Mismatch in nscans for subject {sub}"

        if masker_params_dict["high_pass"] is None:
            masker_params_dict["high_pass"] = 0.01 #default nilearn

        dm_ana = make_first_level_design_matrix(
            frame_time_ana,
            data.events_dfs["Ana"][i],
            hrf_model="spm",
            drift_model="cosine",
            high_pass=masker_params_dict["high_pass"],
            add_regs=data.confounds_Ana[i],
            min_onset=0,
            oversampling=3
        )

        dm_hyper = make_first_level_design_matrix(
            frame_time_hyper,
            data.events_dfs["Hyper"][i],
            hrf_model="spm",
            drift_model="cosine",
            high_pass=masker_params_dict["high_pass"],
            add_regs=data.confounds_Hyper[i],
            min_onset=0,
            oversampling=3
        )
        
        #rename movement reg 
        dm_ana.columns = [f"Ana_{col}" if col.startswith("nuis") else col for col in dm_ana.columns]
        dm_hyper.columns = [f"Hyper_{col}" if col.startswith("nuis") else col for col in dm_hyper.columns]


        # Combine design matrices
        zero_padding_ana = pd.DataFrame(0, index=dm_ana.index, columns=dm_hyper.columns)
        dm_ana_padded = pd.concat([dm_ana, zero_padding_ana], axis=1)

        zero_padding_hyper = pd.DataFrame(0, index=dm_hyper.index, columns=dm_ana.columns)
        dm_hyper_padded = pd.concat([zero_padding_hyper, dm_hyper], axis=1)

        dm_combined = pd.concat([dm_ana_padded, dm_hyper_padded], axis=0)
        design_matrices_2runs[sub] = dm_combined

        # save individual design matrices
        dm_ana_df.append(dm_ana)
        dm_hyper_df.append(dm_hyper)
        dm_ana_files.append(os.path.join(save_path, f"{sub}-Ana-design_matrix.csv"))
        dm_hyper_files.append(os.path.join(save_path, f"{sub}-Hyper-design_matrix.csv"))
        dm_ana_df[i].to_csv(dm_ana_files[-1])
        dm_hyper_df[i].to_csv(dm_hyper_files[-1])

        # Save combined design matrix as CSV
        csv_name = f'{sub}-{"-".join(run_names_short)}-design_matrix.csv'
        dm_combined_csv_path = os.path.join(save_path, csv_name)
        dm_combined.to_csv(dm_combined_csv_path)
        design_matrices_2runs_files.append(dm_combined_csv_path)

        # Save combined design matrix as PNG
        png_name = f'{sub}-{"-".join(run_names_short)}-design_matrix.png'
        dm_combined_png_path = os.path.join(save_path, png_name)

        plot_design_matrix(dm_combined, rescale=True)
        plt.title(f"Combined Design Matrix for {sub}")
        plt.tight_layout()
        plt.savefig(dm_combined_png_path)
        plt.close()

        print(
            f"Saved design matrix for subject {sub}: CSV ({csv_name}) and PNG ({png_name})."
        )

    # %%
        # Plot for the last subject
        if i == len(setup.subjects) - 1:

            plot_design_matrix(dm_ana, rescale=True)
            plt.title(f"Design Matrix for {sub} - Ana", fontsize=16)
            plt.tight_layout()
            plt.show()

            plot_design_matrix(dm_hyper, rescale=True)
            plt.title(f"Design Matrix for {sub} - Hyper", fontsize=16)
            plt.tight_layout()
            plt.show()

            plot_design_matrix(dm_combined, rescale=True)
            plt.title(f"Combined Design Matrix for {sub}", fontsize=16)
            plt.tight_layout()
            plt.show()

    data.design_mat_2runs_files = design_matrices_2runs_files
    data.ana_design_mat_files = dm_ana_files
    data.hyper_design_mat_files = dm_hyper_files

    return data
