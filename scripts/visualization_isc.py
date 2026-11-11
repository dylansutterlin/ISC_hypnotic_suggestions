# %%
from nilearn.image import math_img
from nilearn.plotting import plot_stat_map
import os
import seaborn as sns
from nilearn import plotting
from nilearn.plotting import view_img

import time
import matplotlib.pyplot as plt
from importlib import reload
import nibabel as nib
import numpy as np
import pandas as pd
from nilearn.image import concat_imgs
from brainiak.isc import (
    isc,
    bootstrap_isc,
    permutation_isc,
    compute_summary_statistic,
    phaseshift_isc,
)
from nilearn.maskers import MultiNiftiMapsMasker, MultiNiftiMasker
from nilearn.datasets import fetch_atlas_schaefer_2018
from sklearn.utils import Bunch
from nilearn.plotting import view_img_on_surf
from nilearn.maskers import NiftiLabelsMasker
from nilearn.plotting import plot_glass_brain

from nilearn.plotting import find_parcellation_cut_coords

import src.isc_utils as isc_utils
import src.visu_utils as visu_utils

reload(visu_utils)
reload(isc_utils)


# %% Load the data
model_names = {
    "model1_single-trial": "model1_single-trial-wb_sugg_23-sub_schafer_tian-200-2mm_mask-whole-brain_pairWise-True_preproc_reg-mvmnt-True-8",
    "model2_sugg": "model2_sugg_23-sub_schafer_tian-200-2mm_mask-whole-brain_pairWise-True_preproc_reg-mvmnt-True-8",
    "model3_shock": "model3_shock_23-sub_schafer_tian-200-2mm_mask-whole-brain_pairWise-True_preproc_reg-mvmnt-True-8",
}

model_is = "model2_sugg"

PROJECT_DIR = "/data/rainville/dSutterlin/projects/ISC_hypnotic_suggestions"

color_isc = "Reds"
# base_path = "/data/rainville/dSutterlin/projects/ISC_hypnotic_suggestions/results/data/test_data_sugg_3sub"
preproc_model_data = "23subjects_zscore_sample_detrend_FWHM6_low-pass428_10-12-24/suggestion_blocks_concat_4D_23sub"
base_path = os.path.join(
    PROJECT_DIR, "results/imaging/preproc_data", preproc_model_data
)
model_name = model_names[model_is]
# preproc_model_name =  model_names['model1_sugg'] #'model3_shock_23-sub_schafer-200-2mm_mask-whole-brain_pairWise-True_preproc_reg-mvmnt-True-8'

results_dir = os.path.join(PROJECT_DIR, f"results/imaging/ISC/{model_name}")
setup = isc_utils.load_json(os.path.join(results_dir, "setup_parameters.json"))

# create visualization dir.
save_visu = os.path.join(results_dir, "post-hoc_VISU-tables")
os.makedirs(save_visu, exist_ok=True)

parcel_name = setup["atlas_name"]
do_pairWise = setup["do_pairwise"]
n_boot = setup["n_boot"]

post_hoc_dir = os.path.join(results_dir, "post_hoc_results")
os.makedirs(post_hoc_dir, exist_ok=True)

# %%
# all_results_paths = utils.load_json(os.path.join(results_dir, "result_paths.json"))
# atlas_name = 'Difumo256' # change to setup['atlas_name']
# n_sub = setup['n_sub']
shaeffer_only = False
if shaeffer_only:

    atlas_data = fetch_atlas_schaefer_2018(n_rois=200, resolution_mm=2)
    atlas = nib.load(atlas_data["maps"])
    atlas_path = atlas_data[
        "maps"
    ]  # os.path.join(PROJECT_DIR,os.path.join(PROJECT_DIR, 'masks', 'k50_2mm', '*.nii*'))
    # labels_bytes = list(atlas_data['labels'])
    full_labels = [
        str(label, "utf-8") if isinstance(label, bytes) else str(label)
        for label in atlas_data["labels"]
    ]
    roi_index = [full_labels.index(lbl) + 1 for lbl in full_labels]
    id_labels_dct = dict(zip(roi_index, full_labels))

# ---------------
# up to date scheaffer + subcortical regions
else:
    atlas = nib.load(
        "/data/rainville/dSutterlin/projects/ISC_hypnotic_suggestions/masks/Tian2020_schaeffer200_subcortical16/combined_schaefer200_tian16_DSG.nii.gz"
    )
    id_labels_dct = isc_utils.load_json(
        "/data/rainville/dSutterlin/projects/ISC_hypnotic_suggestions/masks/Tian2020_schaeffer200_subcortical16/roi_labels_dict_DSG.json"
    )
    # remove background
    id_labels_dct.pop("0")
    full_labels = list(id_labels_dct.values())
    roi_index = list(id_labels_dct.keys())

atlas_masker = NiftiLabelsMasker(
    labels_img=atlas, labels=full_labels, standardize=False
)
atlas_masker.fit()

# # Sensaas
# # condition_target = ['all_sugg', 'A'
# atlas = nib.load('/data/rainville/dSutterlin/projects/ISC_hypnotic_suggestions/masks/sensaas/SENSAAS_MNI_ICBM_152_2mm.nii')
# atlas_data = pd.read_csv('/data/rainville/dSutterlin/projects/ISC_hypnotic_suggestions/masks/sensaas/SENSAAS_description.csv')
# # atlas = qc_utils.resamp_to_img_mask(atlas_native, ref_img)
# # view_img(atlas, threshold=0.5, title='SENSAAS atlas')
# #sensaas ids and labels
# roi_index = atlas_data['Index'].values
# atlas_data['annot_abbreviation'] = atlas_data['Abbreviation'] + '_' + atlas_data['Hemisphere']
# full_labels = atlas_data['annot_abbreviation'].values
# id_labels_dct = dict(zip(atlas_data['Index'], atlas_data['annot_abbreviation']))

if "SENSAAS" in setup["atlas_name"]:
    roi_coords = list(
        zip(
            atlas_data["Xmm"].astype(float),
            atlas_data["Ymm"].astype(float),
            atlas_data["Zmm"].astype(float),
        )
    )
else:
    roi_coords = find_parcellation_cut_coords(labels_img=atlas)

from nilearn import datasets, plotting

bg_mni = datasets.load_mni152_template(resolution=1)

# %%
# PUBLICATION surface plot for method section
from nilearn.plotting import plot_roi
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from nibabel.freesurfer import read_annot
from nilearn import datasets, plotting, surface

base_cmap = plt.get_cmap("tab20")
colors = base_cmap.colors * 11  # Repeat to cover 220+ parcels
colors = np.array(colors)

# plot_roi(atlas, bg_img =bg_mni, title='SENSAAS Atlas', display_mode = 'x', cut_coords = [0, -40, 45,48, 50,55, 60], view_type = 'continuous', colorbar=True, cmap=custom_cmap, linewidths=3)
shuffle_idx = np.argsort(
    np.sin(np.linspace(0, np.pi, len(colors)))
)  # or use np.random.permutation for random
shuffled_colors = colors[shuffle_idx]
shuffled_cmap = ListedColormap(shuffled_colors)

# Use fsaverage surface
fsaverage = datasets.fetch_surf_fsaverage(mesh="fsaverage6")
# Downloaded from shaeffer github repo
annot_path = (
    "/home/dsutterlin/Downloads/rh.Schaefer2018_200Parcels_7Networks_order.annot"
)

# Load as GIFTI label texture
labels, ctab, names = read_annot(annot_path)

fig = plt.figure(figsize=(8, 6))
ax = plt.gca()

im = plotting.plot_surf_stat_map(
    fsaverage.infl_right,
    stat_map=labels,
    hemi="right",
    bg_on_data=True,
    alpha=0.9,
    colorbar=False,
    title=None,
    cmap=shuffled_cmap,
    output_file=os.path.join(post_hoc_dir, "schaefer200_surface.png"),
)


# %%
# Load behavioral data
from src import preproc_utils

isc_results_roi = {}
behav_df = pd.read_csv(
    os.path.join(setup["project_dir"], f"results/behavioral_data_cleaned.csv"),
    index_col=0,
)
behav_df.index.name = "subjects"
behav_df = behav_df.sort_index()

xlsx_path = os.path.join(PROJECT_DIR, "masks/Hypnosis_variables_20190114_pr_jc.xlsx")
subjects = list(setup["subjects"])
apm_subjects = ["APM" + subj[4:] for subj in subjects]
Y, rawY = preproc_utils.load_process_y(xlsx_path, subjects)

# %%
# Bootstrap per condition visualization
# =====================================
reload(visu_utils)
reload(isc_utils)
result_key = "isc_results"
conditions = (
    ["NANA", "ANA", "NHYPER", "HYPER"]
    if "single-trial" not in model_name
    else ["N_ANA1_instrbk_1", "N_HYPER1_instrbk_1"]
)
all_conditions = (
    setup["conditions"] + setup["combined_conditions"]
)  # ['HYPER', 'ANA', 'NANA', 'NHYPER']
conditions = setup["conditions"]


# %%
# Visualizatino ISC 1-sample using global FDR threshold (based on x 4 conditions)
# Note : Excludes subsequent context grouping (ana_run + hyper_run)
# From threshold computing because it dilutes pvals to do FDR: .006 (6 conds) vs .003 (4 conds)
# Context grouping is redundant with the condition ISCs
# ======================
reload(visu_utils)

isc_matrices = {}
iscs_cond = {}
isc_p_cond = {}
isc_ci_cond = {}
save_plots_1sample = os.path.join(save_visu, "plots_1sample_isc")
os.makedirs(save_plots_1sample, exist_ok=True)
save_tables_1sample = os.path.join(save_visu, "tables_1sample_isc")
os.makedirs(save_tables_1sample, exist_ok=True)
save_imgs_1sample = os.path.join(save_visu, "imgs_1sample_isc")
os.makedirs(save_imgs_1sample, exist_ok=True)

# Load ISC results
for cond in setup["conditions"]:  #'NHYPER']: #conditions:
    print("----------------------------------------------------")
    print(cond)

    maskers = isc_utils.load_pickle(
        os.path.join(results_dir, f"{cond}/maskers_{parcel_name}_{cond}_23sub.pkl")
    )
    atlas = maskers[0].labels_img  # adjust number of ROI based on mask
    labels = list(maskers[0].region_names_.values())
    if labels == None:
        labels = [f"ROI_{i}" for i in range(isc_rois.shape[1])]

    # accounts for a case where we masked the brain, e.g. with LAN800
    # labels are ok, but need index to access proper coords
    if len(labels) != len(full_labels):
        labels_indices = [full_labels.index(lbl) for lbl in labels]
        print(labels_indices)
    else:
        labels_indices = list(range(len(labels)))
        # used for coords
    # load result variables
    isc_bootstrap = isc_utils.load_pickle(
        os.path.join(
            results_dir,
            f"{cond}/isc_results_{cond}_{n_boot}boot_pairWise{do_pairWise}.pkl",
        )
    )
    isc_rois = pd.DataFrame(isc_bootstrap["isc"], columns=labels)
    isc_results_roi[cond] = isc_rois
    isc_median = isc_bootstrap["observed"]
    ci = isc_bootstrap["confidence_intervals"]
    p_values = isc_bootstrap["p_values"]
    dist = isc_bootstrap["distribution"]
    n_boot = setup["n_boot"]

    isc_matrices[cond] = {
        region: isc_utils.vector_to_isc_matrix(isc_rois.iloc[:, col_j], diag=0)
        for col_j, region in enumerate(isc_rois.columns)
    }

    # stack for brain plotting/cond
    iscs_cond[cond] = isc_median
    isc_p_cond[cond] = p_values
    isc_ci_cond[cond] = ci

    print(
        f"min p value: {np.min(p_values):.6f}, and max ISC : {np.max(isc_median):.6f}"
    )
    print(
        f"Within condition FDR thresh : {isc_utils.fdr(p_values, q=0.05):.6f}, and Bonferroni : {isc_utils.bonferroni(p_values, alpha=0.05):.6f}"
    )

# FDR thresh across all conditions since
all_ps = []
all_iscs = []
for cond in setup["conditions"]:
    p = isc_p_cond[cond]
    all_ps.extend(p)
    all_iscs.extend(iscs_cond[cond])  # used for global scale/colorbar in plots
fdr_all_1samp = isc_utils.fdr(np.array(all_ps), q=0.05)
print("\nFDR across all conditions : ", fdr_all_1samp)

# -------------
# Project ISc results to brain maps

sig_isc_tables = {}
sig_isc_matrices = {}
fdr_isc_imgs = {}  # used further to produce brain plots of each conditions
unc_isc_imgs = {}
liberal_isc_imgs = {}

# Generate stats tables for each condition and save
for correction in ['unc_01', 'FDR']:  # between condition FDR / within bonferroni
    for cond in setup["conditions"]:
        print("-----------------------------------------------")
        print(cond)

        maskers = isc_utils.load_pickle(
            os.path.join(results_dir, f"{cond}/maskers_{parcel_name}_{cond}_23sub.pkl")
        )
        atlas = maskers[0].labels_img  # adjust number of ROI based on mask

        isc_median = iscs_cond[cond]
        p_values = isc_p_cond[cond]
        show = False

        if correction == "unc_01":
            p_thresh = 0.01
            p_thresh_str = "0.01"
        elif correction == "FDR":
            p_thresh = fdr_all_1samp  # all conditions!
            p_thresh_str = "FDR" + str(round(p_thresh, 4))
            show = True
        elif correction == "bonf":
            p_thresh = isc_utils.bonferroni(p_values, alpha=0.05)
            p_thresh_str = "Bonf" + str(round(p_thresh, 4))

        title_view = f"ISC_{cond}_{correction}"
        file_id = f"ISC_{cond}_FDR05"

        cut_coords = 5
        isc_img, isc_thresh, sig_df, unc_img = visu_utils.project_isc_to_brain(
            atlas_img=atlas,
            isc_median=isc_median,
            atlas_labels=id_labels_dct,
            roi_coords=roi_coords,
            p_values=p_values,
            p_threshold=p_thresh,
            title=title_view,
            coords_bool_mask=labels_indices,
            save_path=None,
            show=show,
            display_mode="x",
            cut_coords_plot=cut_coords,
        )

        # Always initialize the dict for the condition
        if cond not in sig_isc_matrices:
            sig_isc_matrices[cond] = {}

        unc_isc_imgs[cond] = unc_img

        if correction == "unc_01":
            liberal_isc_imgs[cond] = isc_img

        if correction == "FDR":
            fdr_isc_imgs[cond] = (isc_img, isc_thresh)
            p_mask = p_values < p_thresh
            sig_labels = [labels[i] for i in range(len(labels)) if p_mask[i]]

            # add CI to df
            low_ci_sig = isc_ci_cond[cond][0][p_mask] 
            up_ci_sig = isc_ci_cond[cond][1][p_mask]
            sig_df['CI'] = tuple((np.round(low, 2), np.round(up, 2)) for low, up in zip(low_ci_sig, up_ci_sig))

            sig_isc_tables[cond] = sig_df
            print("Sig ROIs :", sig_labels)

            nib.save(
                isc_img,
                os.path.join(save_imgs_1sample, f"ISC_{cond}_uncorrected.nii.gz"),
            )

            # Only add matrices for significant regions
            if sig_df.shape[0] > 0:
                for sig_region_name in sig_df["Label"]:
                    # Only add if region exists in isc_matrices[cond]
                    if sig_region_name in isc_matrices[cond]:
                        sig_isc_matrices[cond][sig_region_name] = isc_matrices[cond][sig_region_name]
                # save sig table
                sig_df.columns = [
                    "Region id",
                    "Label",
                    "ISC",
                    "P-value",
                    "Coordinates (X,Y,Z)",
                    "CI 95%",
                ]
                sig_df.to_csv(
                    os.path.join(
                        save_tables_1sample,
                        f"ISC_{cond}_table_{correction}_thresh{round(p_thresh,4)}.csv",
                    ),
                    index=False,
                )

# %%
# ----------
# PUBLICATION GLASS BRAINS per conditions
# Save glass brain to post-hoc plot directory )

# compute color bar across all values/conditions (all_iscs)
from matplotlib import cm
from nilearn.image import new_img_like
reload(visu_utils)
vmin, vmax = np.min(all_iscs), np.max(all_iscs)
fake_data = np.zeros_like(atlas.get_fdata())
fake_data.flat[: len(all_iscs)] = all_iscs
synthetic_img = new_img_like(atlas, fake_data)
cmap = cm.get_cmap("Reds")

cbar = visu_utils.save_colorbar(
    img=synthetic_img,
    cmap=cmap,
    fontsize = 30,
    outpath=os.path.join(save_plots_1sample, "shared_isc_colorbar.png"),
    symmetric_cbar=False,
    offset=None,
    n_ticks=3,
    transparent=True,
)

condition_names = {
    "HYPER": "Hyperalgesia",
    "ANA": "Hypoalgesia",
    "NHYPER": "Neutral (Hyper.)",
    "NANA": "Neutral (Hypo.)",
}

for i, (cond, (isc_img, isc_thresh)) in enumerate(fdr_isc_imgs.items()):
    # if i >= 1:
    #     break  # Stop after the first iteration
    print(cond)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.set_axis_off()

    title = f"{condition_names[cond]}"
    display = plot_glass_brain(
        isc_img,
        threshold=isc_thresh,
        colorbar=False,
        display_mode="lr",
        plot_abs=False,
        cmap="Reds",
        title=None,
        axes=ax,
        vmax=np.max(all_iscs),
    )
    y = 0.20
    ax.text(0.1, y, "L", transform=ax.transAxes, fontsize=20, ha="center")
    ax.text(0.9, y, "R", transform=ax.transAxes, fontsize=20, ha="center")

    # Add title
    fig.suptitle(title, fontsize=30, y=0.85)
    fig.savefig(
        os.path.join(save_plots_1sample, f"glass_brain_{cond}_FDR05.png"), dpi=1000
    )
cbar

#%%
# PUBLICATION Surface plot per ISC conditions
from src import visu_utils
import importlib
importlib.reload(visu_utils)

for i, (cond, (isc_img, isc_thresh)) in enumerate(fdr_isc_imgs.items()):
    
    surface_plot_dir = os.path.join(save_plots_1sample, f"surface_{cond}_FDR05.png")
    os.makedirs(surface_plot_dir, exist_ok=True)
    visu_utils.plot_surface_views(isc_img,
                                 output_dir=surface_plot_dir,
                                 threshold = isc_thresh,
                                 fine_resolution = True,
                                 vmax = np.max(all_iscs))

                       
    
# %%
# PUBLICATION violin plots of median ISC across all contrasts conditions
reload(visu_utils)
# violin plots displaying median SIc diff per region
color_conds = {"Ana": "#004197", "Hyper": "#df265e", "Neutral": "#a1a0a0"}

violin_palette = [
    color_conds["Hyper"],
    color_conds["Ana"],
    color_conds["Neutral"],
    color_conds["Neutral"],
    color_conds["Ana"],
    color_conds["Neutral"],
    color_conds["Hyper"],
    color_conds["Neutral"],
]

delta_dict = {cond: isc for cond, isc in iscs_cond.items()}
delta_dict.update({"ANA1": iscs_cond["ANA"], "NANA1": iscs_cond["NANA"]})
delta_dict.update({"HYPER1": iscs_cond["HYPER"], "NHYPER1": iscs_cond["NHYPER"]})

condition_labels = {
    "HYPER1": "Hyper",
    "ANA1": "Hypo",
    "NHYPER1": "Neutral (Hyper)",
    "NANA1": "Neutral (Hypo)",
    "ANA": "Hypo",
    "NANA": "Neutral (Hypo)",
    "HYPER": "Hyper",
    "NHYPER": "Neutral (Hyper)",
}
fig = visu_utils.plot_delta_condition_violin(
    delta_dict,
    min_y=np.min(all_iscs),
    max_y=np.max(all_iscs) + 0.05,
    condition_labels=condition_labels,
    dot_size=13,
    lab_rotation=45,
    title="Region-wise ISC change between conditions",
    xlabel=None,
    ylabel="Median ISC",
    palette=violin_palette,
    save_as=os.path.join(save_plots_1sample, "violin_medianISC_conditions.png"),
)

# %%
# ========================
# PLOT ISC matrices per conditions (SUPP MATERIAL)
from sklearn.preprocessing import StandardScaler

reload(visu_utils)
from src.visu_utils import schaeffer_region_mapping

region_mapping = {}
consistent_isc_roi = ["7Networks_LH_Default_Temp_3", "7Networks_RH_Default_Temp_3"]
# Concatenate matrices across all conditions and selected ROIs to find global color scale
concat_all = []

for region in consistent_isc_roi:
    for cond in conditions:
        mat = sig_isc_matrices[cond][region]
        concat_all.append(mat)

concat_mat = np.concatenate(concat_all, axis=0)
vmin, vmax = np.min(concat_mat), np.max(concat_mat)
print(f"Global ISC min: {vmin:.2f}, max: {vmax:.2f}")

# SHSS similarity matrices and rank
# ----------
Y_VAR = 'SHSS_score' #'total_chge_pain_hypAna'
shss = StandardScaler().fit_transform(Y[Y_VAR].values.reshape(-1, 1)).flatten()
sim_mat_behav_matrix = isc_utils.compute_behav_similarity(
    shss, metric="euclidean", vectorize=False
)
sim_mat_behav_annak = isc_utils.compute_behav_similarity(
    shss, metric="annak", vectorize=False
)

sorted_indices = np.argsort(shss)

def reorder_matrix(mat, indices):
    return mat[np.ix_(indices, indices)]


# Plot RANKED IS CMATRICES : 1SAMPLE ISC
# -----------------------------------------
for region in consistent_isc_roi:
    anat_name = schaeffer_region_mapping[region]["anatomical_label"]
    for cond in conditions:
        isc_mat = sig_isc_matrices[cond][region]
        reordered_mat = reorder_matrix(isc_mat, sorted_indices)

        title = f"{cond} – {anat_name}"
        visu_utils.plot_simmat_isc(
            simmat=reordered_mat,
            title=title,
            x_label="ranked subjects (SHSS)",
            vmin=-vmax,
            vmax=vmax,
            show_colorbar=False,
            save_path=None,  # or a path if saving,
            tick_fontsize=50,
            label_fontsize=50,
        )

visu_utils.plot_isc_colorbar(
    vmin=-vmax,
    vmax=vmax,
    cmap="RdBu_r",
    label="Pairwise r",
    save_path=None,  # or PNG, PDF
    label_fontsize=40,
    tick_fontsize=30,
    dpi=600,
)

# %%
# get isc matrix for each condition
region_name = consistent_isc_roi[0]
ls_mat = []
for cond in conditions:
    mat = sig_isc_matrices[cond][region_name]
    ls_mat.append(mat)

    concat_mat = np.concatenate(ls_mat, axis=0)
    print(concat_mat.shape)
    min, max = np.min(concat_mat), np.max(concat_mat)

# %%
# for voxel wise case, ignored otherwise
# masker = isc_utils.load_pickle(
#     "/data/rainville/dSutterlin/projects/ISC_hypnotic_suggestions/results/imaging/ISC/{model_name}/HYPER/maskers_voxelWise_lanA800_HYPER_23sub.pkl"
# )

# %%
# BLOCK MODELS : (ANA > HYPER, etc. run1>run2)
# ==========================
# Difference (Permutation) per condition visualization
model_name = model_names["model2_sugg"]
results_dir = os.path.join(PROJECT_DIR, f"results/imaging/ISC/{model_name}")
setup = isc_utils.load_json(os.path.join(results_dir, "setup_parameters.json"))

# create contrast output dir.
save_plots_contrasts = os.path.join(save_visu, "plots_contrasts_isc")
os.makedirs(save_plots_contrasts, exist_ok=True)
save_tables_contrasts = os.path.join(save_visu, "tables_contrasts_isc")
os.makedirs(save_tables_contrasts, exist_ok=True)

parcel_name = setup["atlas_name"]
do_pairWise = setup["do_pairwise"]
n_boot = setup["n_boot"]

result_key = "cond_contrast_permutation"
contrasts = (
    ["ana_run-hyper_run", "Ana-Hyper", "NAna-NHyper"]
    if "single-trial" not in model_name
    else ["Ana-N_Ana", "Hyper-N_Hyper"]
)
n_perm = setup["n_perm"]

sig_rois = {}
interactive_views = []

delta_iscs = {}
delta_p_values = {}
delta_ci = {} # not used

for i, cont in enumerate(contrasts):
    print(f"-------\n{cont}")
    file = os.path.join(
        results_dir,
        f"cond_contrast_permutation/isc_results_{cont}_{n_perm}perm_pairWise{do_pairWise}.pkl",
    )
    isc_contrast = isc_utils.load_pickle(file)

    grouped_isc = isc_contrast["grouped_isc"]  # Grouped ISC values
    distribution = isc_contrast["distribution"]
    delta_iscs[cont] = isc_contrast["observed_diff"]  # Observed differences in ISC
    delta_p_values[cont] = isc_contrast["p_value"]  # P-values for the ISC contrasts
    # delta_ci[cont] = isc_contrast[]
    print(
        f"min p value: {np.min(delta_p_values[cont]):.6f}, and max ISC diff. (delta) : {np.max(delta_iscs[cont]):.6f}"
    )
    print(
        f"Within condition FDR thresh : {isc_utils.fdr(delta_p_values[cont], q=0.05):.6f}, and Bonferroni : {isc_utils.bonferroni(delta_p_values[cont], alpha=0.05):.6f}"
    )

# FDR thresh across all contrasts
all_delta_p = []
all_delta_iscs = []
for cont in contrasts:
    all_delta_p.extend(delta_p_values[cont])
    all_delta_iscs.extend(delta_iscs[cont])  # used for global scale/colorbar in plots
fdr_all_block = isc_utils.fdr(np.array(all_delta_p), q=0.05)
print("\nFDR across all block (3) contrasts : ", fdr_all_block)

# %%
reload(visu_utils)
# ===========================
# SINGLE TRIAL MODEL : (Modulation - Neutre)
# Difference (Permutation) per condition visualization
model_name_single_trial = model_names["model1_single-trial"]
results_dir = os.path.join(
    PROJECT_DIR, f"results/imaging/ISC/{model_name_single_trial}"
)
setup = isc_utils.load_json(os.path.join(results_dir, "setup_parameters.json"))

# create visualization dir.
save_plots_contrasts = os.path.join(save_visu, "plots_contrasts_isc")
os.makedirs(save_plots_contrasts, exist_ok=True)
save_tables_contrasts = os.path.join(save_visu, "tables_contrasts_isc")
os.makedirs(save_tables_contrasts, exist_ok=True)

parcel_name = setup["atlas_name"]
do_pairWise = setup["do_pairwise"]
n_boot = setup["n_boot"]

result_key = "cond_contrast_permutation"
contrasts_trials = (
    ["Ana-Hyper", "NAna-NHyper"]
    if "single-trial" not in model_name_single_trial
    else ["Ana-N_Ana", "Hyper-N_Hyper"]
)
n_perm = setup["n_perm"]

# contrast_imgs_txt = {}

for i, cont in enumerate(contrasts_trials):
    print(f"----------\n {cont}")
    file = os.path.join(
        results_dir,
        f"cond_contrast_permutation/isc_results_{cont}_{n_perm}perm_pairWise{do_pairWise}.pkl",
    )
    isc_contrast = isc_utils.load_pickle(file)

    grouped_isc = isc_contrast["grouped_isc"]  # Grouped ISC values
    distribution = isc_contrast["distribution"]
    delta_iscs[cont] = isc_contrast["observed_diff"]  # Observed differences in ISC
    delta_p_values[cont] = isc_contrast["p_value"]  # P-values for the ISC contrasts

    print(
        f"min p value: {np.min(delta_p_values[cont]):.6f}, and max ISC diff. (delta) : {np.max(delta_iscs[cont]):.6f}"
    )
    print(
        f"Within condition FDR thresh : {isc_utils.fdr(delta_p_values[cont], q=0.05):.6f}, and Bonferroni : {isc_utils.bonferroni(delta_p_values[cont], alpha=0.05):.6f}"
    )

# %%
# compute gloabl FDR thresh across all (5) contrasts conditions
# p < 0.001* with 5 conditinns/ vs p < 0.003 with the 3 conditions from block model
# > more conservative
for cont in contrasts_trials:
    p = delta_p_values[cont]
    all_delta_p.extend(p)
fdr_all = isc_utils.fdr(np.array(all_delta_p), q=0.05)
print(f"fdr_all: {fdr_all:.6f}")

# %%
# -------
# Create CONTRAST stats images + tables for BLOCK MODEL & TRIAL model
# Note : within FDR thresh is always around 0.001, similar to the global FDR (5 cond.)
contrast_imgs = {}
contrast_fdr_df = {}

for correction in [
    "unc_01",
    "FDR",
]:  # between condition FDR / within bonferroni
    print(f"-----{correction}")
    for i, cont in enumerate(contrasts + contrasts_trials):

        observed_diff = delta_iscs[cont]
        p_values = delta_p_values[cont]

        show = False
        save_fig = None
        if correction == "unc_01":
            p_thresh = 0.01
            p_thresh_str = "0.01"
        elif correction == "within_FDR":
            p_thresh = isc_utils.fdr(p_values, q=0.05)
            p_thresh_str = "within_FDR" + str(round(p_thresh, 4))
        elif correction == "FDR":
            # p_thresh = isc_utils.fdr(p_values, q=0.05)
            p_thresh = fdr_all  # all contrasts!
            p_thresh_str = "FDR" + str(round(p_thresh, 4))
            show = True
            save_fig = save_plots_contrasts + f"/ISC_contrast_{cont}_{p_thresh_str}.png"
        elif correction == "bonf":
            p_thresh = isc_utils.bonferroni(p_values, alpha=0.05)
            p_thresh_str = "Bonf" + str(round(p_thresh, 4))
            show = True

        print(f"Correction thresh : {p_thresh}")

        if cont == "NAna-NHyper":
            observed_diff = -observed_diff
            # !! Reverse contrast since the name was opposite of difference
            # NAna has stronger ISC than Hyper (~ .20 max)

        diff_img, diff_thresh, sig_df, _ = visu_utils.project_isc_to_brain_perm(
            atlas_img=atlas,
            isc_median=observed_diff,
            atlas_labels=id_labels_dct,
            roi_coords=roi_coords,
            p_values=p_values,
            p_threshold=p_thresh,
            title=f"Difference in ISC between {cont} ({correction})",
            save_path=False,
            show=show,
            save_fig_as=save_fig,
            display_mode="x",
            color="Reds",
        )
        contrast_imgs[cont] = (diff_img, diff_thresh)

        if correction == "FDR":  # save within contrast threshold.
            contrast_fdr_df[cont] = sig_df

            if sig_df.shape[0] > 0:
                # sig_labels = pd.DataFrame(sig_labels, columns=['ROI', 'Label', 'spearman_r', 'p_values', 'Coordinates (X,Y,Z)'])
                sig_df.columns = [
                    "Region id",
                    "Label",
                    "Delta_r",
                    "p_values",
                    "Coordinates (X,Y,Z)",
                ]
                sig_df.to_csv(
                    os.path.join(
                        save_tables_contrasts, f"ISC_{cont}_table_{p_thresh_str}.csv"
                    ),
                    index=False,
                )

                print("Sig ROIs : ", sig_df["Label"].to_list())

# %%
# Load 1 sample ISC for runs/contexts : Hyper & Analgesia-context
# Will be used for plotting region ISC matrices
# stack contrast imgs in `contrast_imgs[cont]`

# Reload BLOCK model
model_name = model_names["model2_sugg"]
results_dir = os.path.join(PROJECT_DIR, f"results/imaging/ISC/{model_name}")
setup = isc_utils.load_json(os.path.join(results_dir, "setup_parameters.json"))

# Have been overwrite by trial model code !
save_plots_contrasts = os.path.join(save_visu, "plots_contrasts_isc")
save_tables_contrasts = os.path.join(save_visu, "tables_contrasts_isc")

result_key = "isc_results"
interactive_views = {}
combined_conditions = [
    "ana_run",
    "hyper_run",
]  # 'neutral']#setup['combined_conditions'] #['all_sugg', 'modulation', 'neutral']

for i, cond in enumerate(combined_conditions):
    print(
        f"Fetching 1 sample ISC values from BLOCK model, for contextual effect.\n {cond}"
    )

    isc_bootstrap = isc_utils.load_pickle(
        os.path.join(
            results_dir,
            f"concat_suggs_1samp_boot/isc_results_{cond}_{n_boot}boot_pairWise{do_pairWise}.pkl",
        )
    )
    isc_rois = pd.DataFrame(isc_bootstrap["isc"], columns=full_labels)
    isc_matrices[cond] = {
        region: isc_utils.vector_to_isc_matrix(isc_rois.iloc[:, col_j], diag=0)
        for col_j, region in enumerate(isc_rois.columns)
    }

    isc_median = isc_bootstrap["observed"]
    ci = isc_bootstrap["confidence_intervals"]
    p_values = isc_bootstrap["p_values"]
    dist = isc_bootstrap["distribution"]
    n_boot = setup["n_boot"]

    iscs_cond[cond] = isc_median  # one sample variables (vs. delta_iscs for contrasts)
    isc_p_cond[cond] = p_values
    unc_isc_maps = {}
# build brain images and save tables
for correction in ["FDR"]:  # ['unc_01', 'within_FDR', 'FDR', 'bonf']:

    for cond in combined_conditions:
        print(cond)

        isc_median = iscs_cond[cond]
        p_values = isc_p_cond[cond]
        show = False
        save_fig_as = None

        if correction == "unc_01":
            p_thresh = 0.01
            p_thresh_str = "0.01"
        elif correction == "FDR":
            p_thresh = fdr_all_1samp  # based on 4 conditions, .003. See note above
            p_thresh_str = "FDR" + str(round(p_thresh, 4))
            show = True
            save_fig_as = os.path.join(
                save_plots_1sample, f"ISC_stats_map_{cond}_FDR05.png"
            )

        elif correction == "bonf":
            p_thresh = isc_utils.bonferroni(p_values, alpha=0.05)
            p_thresh_str = "bonf" + str(round(p_thresh, 4))
        elif correction == "within_FDR":
            p_thresh = isc_utils.fdr(p_values, q=0.05)
            p_thresh_str = "within_FDR" + str(round(p_thresh, 4))
            print("Within FDR p-threshold : ", p_thresh_str)

        cut_coords = 5  # (-48, -2, 6, 50)
        isc_img, isc_thresh, sig_df, unc_img = visu_utils.project_isc_to_brain(
            atlas_img=atlas,
            isc_median=isc_median,
            atlas_labels=id_labels_dct,
            roi_coords=roi_coords,
            p_values=p_values,
            p_threshold=p_thresh,
            title=f"ISC {cond} - {p_thresh_str}",
            coords_bool_mask=labels_indices,
            save_path=None,
            show=show,
            save_fig_as=save_fig_as,
            display_mode="x",
            cut_coords_plot=cut_coords,
        )

        fdr_isc_imgs[cond] = (isc_img, isc_thresh)
        unc_isc_maps[cond] = unc_img

        if correction == "FDR":  # stack isc matrices for sig. regions
            sig_isc_matrices[cond] = {}
            sig_isc_tables[cond] = sig_df

            for sig_region_name in sig_df["Label"]:
                sig_isc_matrices[cond][sig_region_name] = isc_matrices[cond][
                    sig_region_name
                ]

        if sig_df.shape[0] > 0:
            # sig_labels = pd.DataFrame(sig_labels, columns=['ROI', 'Label', 'spearman_r', 'p_values', 'Coordinates (X,Y,Z)'])
            sig_df.columns = [
                "Region id",
                "Label",
                "Pearson_r",
                "p_values",
                "Coordinates (X,Y,Z)",
            ]
            sig_df.to_csv(
                os.path.join(
                    save_tables_1sample, f"ISC_{cond}_table_{p_thresh_str}.csv"
                ),
                index=False,
            )

            sig_df.sort_values(by="Pearson_r", ascending=False)
            print("Sig ROIs : ", sig_df["Label"].to_list(), "\n")

# %%
# Plot brain images for contextual effect (Ana vs Hyper)
# unthresholded maps GLASS brain and save unc_ISC maps
combined_cond_names = {
    "ana_run": "Ana-context",
    "hyper_run": "Hyper-context",
}
view_cut = ["lr", "z"]
for view in view_cut:
    for cond in combined_conditions:
        unc_isc_img = unc_isc_maps[cond]

        fig_path = os.path.join(
            save_plots_1sample, f"unc_ISC_glass-brain_{view}_{cond}.png"
        )

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.set_axis_off()
        title = f"{combined_cond_names[cond]} (unc.)"
        plotting.plot_glass_brain(
            unc_isc_img,
            threshold=0,
            colorbar=False,
            display_mode=view,
            plot_abs=False,
            cmap='RdBu_r',
            title=None,
            vmin=-np.max(all_iscs),
            vmax=np.max(all_iscs),
            axes = ax     
        )

        if view == "lr":
            y = 0.20
            ax.text(0.1, y, "L", transform=ax.transAxes, fontsize=20, ha="center")
            ax.text(0.9, y, "R", transform=ax.transAxes, fontsize=20, ha="center")

            # Add title
            fig.suptitle(title, fontsize=25, y=0.85)
        fig.savefig(
            fig_path, dpi=1000,
            transparent=True
        )

    fig_path_cb = os.path.join(
            save_plots_1sample, f"colorbar_unc_ISC_glass-brain_{cond}.png",
            
        )

    tick_fontsize=75
    tick_labels = [f"{-np.max(all_iscs):.2f}", " ", f"{np.max(all_iscs):.2f}"]

    visu_utils.plot_isc_colorbar(   
        vmin=-np.max(all_iscs),
        vmax=np.max(all_iscs),
        cmap='RdBu_r',
        figsize=(8,3),
        label=None,
        tick_fontsize=tick_fontsize,
        tick_labels=tick_labels,
        horizontal=True,
        save_path=fig_path_cb,
        dpi=1000)

# %%
# ================
# PUBLICATION ANATOMICAL brain plots for CONTRAST contextual effect (FDR global)

contrast_ana_hyper_run, delta_thresh = contrast_imgs["ana_run-hyper_run"]
max_diff = np.max(np.abs(contrast_ana_hyper_run.get_fdata()))

view_img(
    contrast_ana_hyper_run,
    threshold=delta_thresh,
    title="Ana-Hyper (FDR q < .05)",
    cmap="coolwarm",
    vmax=max_diff,
)
plotting.plot_stat_map(
    contrast_ana_hyper_run,
    threshold=delta_thresh,
    title="Ana-Hyper (FDR q < .05)",
    cmap="Reds",
    vmax=max_diff,
    display_mode="mosaic",
)
# %%
# PUBLICATION SLICES
COORDS = {
    "x": [54, 42, -48, -60],
    "y": [-22, 6],
    "z": [42, 58],
}

SAVE_TO = os.path.join(save_plots_contrasts, "ana_run-hyper_run")
os.makedirs(SAVE_TO, exist_ok=True)

cm = "Reds"

visu_utils.save_slices(
    contrast_ana_hyper_run, COORDS, SAVE_TO, img_id="run_contrast", cmap="Reds"
)
# save colorbar
fig_path = os.path.join(SAVE_TO, "colorbar.png")
cb_ab = visu_utils.save_colorbar(
    contrast_ana_hyper_run,
    cmap="Reds",
    fontsize=25,
    outpath=fig_path,
    symmetric_cbar=False,  # false because we dont keep negative coeff
    offset=0,
    n_ticks=3,
    transparent=True,
)
cb_ab
# %%
# Plot ISC MATRICES for PUBLICATION
# ------------------------------
# Ana-context &  Hyper-context only
min_pairwise_isc = 0
max_pairwise_isc = 0
for region, isc_mat in sig_isc_matrices[cond].items():
    min = np.min(isc_mat)
    max = np.max(isc_mat)
    if max > max_pairwise_isc:
        max_pairwise_isc = max
    if min < min_pairwise_isc:
        min_pairwise_isc = min

print(min_pairwise_isc, max_pairwise_isc)

# %%
reload(visu_utils)
for cond in combined_conditions:
    for region, isc_mat in sig_isc_matrices[cond].items():

        visu_utils.plot_simmat_isc(
            simmat=isc_mat,
            title=f"{region}",
            vmin=-max_pairwise_isc,
            vmax=max_pairwise_isc,
            show_colorbar=True,
            save_path=None,
        )

# %%
# Anatomical labeling significant parcels
# Find Atlas region of interest
def filter_atlas(atlas_img, atlas_labels_dict, keep_ids):
    atlas_data = atlas_img.get_fdata()
    mask_data = np.isin(atlas_data, keep_ids) * atlas_data
    filtered_img = new_img_like(atlas_img, mask_data)

    # Build new labels dict for the kept IDs
    new_labels = {
        roi: label for roi, label in atlas_labels_dict.items() if roi in keep_ids
    }
    return filtered_img, new_labels


# Plot the filtered atlas with clear discrete colorbar
def plot_filtered_atlas(filtered_img, filtered_labels, cmap="tab20"):
    label_ids = list(filtered_labels.keys())
    label_names = list(filtered_labels.values())

    return plotting.view_img(
        filtered_img,
        cmap=cmap,
        colorbar=True,
        title="Filtered Atlas",
        symmetric_cmap=False,
    )


def plot_single_roi(atlas_img, roi_id, atlas_labels=None, title=None):

    mask_img = math_img(f"img == {roi_id}", img=atlas_img)

    title = f"ROI {roi_id}"
    if atlas_labels and roi_id in atlas_labels:
        title += f" {atlas_labels[roi_id]}"

    print(f"Displaying region {roi_id}: {title}")
    return mask_img, plotting.view_img(
        mask_img, cmap="Reds", title=title, colorbar=False
    )


# %%
import atlasreader

# ch dir bc atlas reader creates files in pwd just for visu
pwd_scripts = os.path.join(PROJECT_DIR, "scripts")
os.chdir(os.path.join(pwd_scripts, "atlas_views"))
print("Current working directory:", os.getcwd())
from atlasreader import create_output

# %%
#
roi_idx = 76
masked_region, _plot = plot_single_roi(
    atlas, roi_idx, id_labels_dct, title=f"roi idx {roi_idx}"
)  # 30 = Default_Temp_3 LH
_plot
# %%
create_output(masked_region, voxel_thresh=0.99, cluster_extent=5)
atlas_df = pd.read_csv("atlasreader_peaks.csv")
atlas_df
# %%
plot_single_roi(atlas, 189, id_labels_dct)  # 30 = Default_Temp_3 LH
plot_filtered_atlas(*filter_atlas(atlas, id_labels_dct, []), cmap="tab20")
# ----------------------------------------

# %%
# IS-RSA SHSS~ISC
# ===============
# Load results and save tables

load_dir = os.path.join(
    PROJECT_DIR, "results/imaging/RSA/isc-RSA_ext_conds_sugg-pain_tian216_2tails"
)
RESULT_FILE = "rsa_SHSS-behav_isc-sugg10000perm.pkl"
rsa_isc_sugg = isc_utils.load_pickle(os.path.join(load_dir, RESULT_FILE))

# change keys in dict
rsa_isc_sugg["euclidean"] = rsa_isc_sugg.pop("euclidian")
    
# load_dir = os.path.join(
#     PROJECT_DIR, "results/imaging/RSA/isc-RSA_tian216_sugg-pain_combined_conds"
# )
# RESULT_FILE = "rsa_chge-pain-behav_isc-sugg10000perm.pkl"
# rsa_isc_sugg = isc_utils.load_pickle(os.path.join(load_dir, RESULT_FILE))

save_rsa = os.path.join(load_dir, "post-hoc_VISU-tables")
os.makedirs(save_rsa, exist_ok=True)

save_tables_rsa = os.path.join(save_rsa, "tables_rsa")
os.makedirs(save_tables_rsa, exist_ok=True)
save_plots_rsa = os.path.join(save_rsa, "plots_rsa")
os.makedirs(save_plots_rsa, exist_ok=True)

n_perm = 10000

rsa_tables_fdr = {}  # changed for `euclidean`` in original script
rsa_table_unc = {}
rsa_dfs = {}
models = ["annak", "euclidean"] #potentially euclidian !!
model = "annak"

rsa_fdr_imgs = {}
rsa_unc_stats_imgs = {}
rsa_liberal_imgs = {}
rsa_p_values = []

cond_p_thresh = {}
# Load rsa results and find FDR thresh
model_p_values = []
for cond in [
        "ana_run",
        "hyper_run",
    ]:  # , 'HYPER', 'ANA']: #, 'all_sugg', 'neutral']:
    for model in models:

        model_cond_key = f"{model}_{cond}"

        print("\ncond", model_cond_key,)
        rsa_df = rsa_isc_sugg[model][cond].sort_index(
            ascending=True
        )  # to match the atlas labels
        rsa_dfs[model_cond_key] = rsa_df

        # === Prepare variables for projection ===
        correlations = rsa_df["spearman_r"].values
        p_values = rsa_df["p_values"].values
        rsa_p_values.extend(p_values)

        model_p_values.extend(p_values)

        # !! Shift ROI id in df by 1, since starts at 0 and ends at 215 (atlas is 216)
        rsa_df["ROI"] = (
            rsa_df["ROI"] + 1
        )  # fixed in rsa-isc_ext_model.py, rm if re-run !!
        fdr_p = isc_utils.fdr(p_values, q=0.05)
        print(f"Within FDR threshold: {fdr_p:.4f}")
        print(f"Min p values : {p_values.min():.4f}")
        print(f"max spearman: {correlations.max():.4f}")

    cond_p_thresh[cond] = isc_utils.fdr(np.array(model_p_values), q=0.05)
    print(f"FDR threshold for condition {cond}: {cond_p_thresh[cond]:.4f}\n")

# compute across condition FDR thresh
fdr_all_rsa = isc_utils.fdr(np.array(rsa_p_values), q=0.05)
print(f"Across conditions FDR threshold: {fdr_all_rsa:.4f}")
#%%
# Project RSA values to brain and save tables
for correction in ["unc01", "FDR"]:
    for (
        rsa_cond
    ) in (
        rsa_dfs.keys()
    ):  # ['ana_run', 'hyper_run']: #, 'HYPER', 'ANA']: #, 'all_sugg', 'neutral']:

        rsa_df = rsa_dfs[rsa_cond]
        correlations = rsa_df["spearman_r"].values
        p_values = rsa_df["p_values"].values
        region_id = rsa_df["ROI"].values

        show = False
        if correction == "unc01":
            p_thresh = 0.01
            p_thresh_str = "0.01"
        if correction == "within_FDR":
            p_thresh = isc_utils.fdr(p_values, q=0.05)
            p_thresh_str = "within_FDR" + str(round(p_thresh, 4))
        if correction == "FDR":
            p_thresh = fdr_all_rsa
            p_thresh_str = "FDR" + str(round(p_thresh, 4))
            show = True
        if correction == "cond-wise_FDR":
            model = rsa_cond.split("_")[0]
            p_thresh = cond_p_thresh[cond]
            p_thresh_str = f"cond-wise_FDR_{cond}" + str(round(p_thresh, 4))
            show = True
        if correction == "bonf":
            p_thresh = isc_utils.bonferroni(p_values, alpha=0.05)
            p_thresh_str = "Bonf" + str(round(p_thresh, 4))

        rsa_img, rsa_thresh, sig_df, unc_img = visu_utils.project_isc_to_brain_perm(
            atlas_img=atlas,
            isc_median=correlations,
            atlas_labels=id_labels_dct,
            roi_coords=roi_coords,
            p_values=p_values,
            p_threshold=p_thresh,  #!!
            title=f"IS-RSA SHSS~{rsa_cond} ({p_thresh_str})",  # "RSA-ISC: Suggestion-Pain Similarity (FDR<.05)",
            save_path=None,
            show=show,
            display_mode="x",
            cut_coords_plot=(5),  # (-52, -40, 34),
            color="coolwarm",
        )

        rsa_unc_stats_imgs[rsa_cond] = (unc_img, rsa_thresh)

        if correction == "unc01":
            rsa_table_unc[rsa_cond] = sig_df
            rsa_liberal_imgs[rsa_cond] = (rsa_img, rsa_thresh)
        if correction == "FDR" and sig_df.shape[0] > 0:
            
            rsa_fdr_imgs[rsa_cond] = (rsa_img, rsa_thresh)
            rsa_tables_fdr[rsa_cond] = sig_df

            sig_df.columns = [
            "ROI",
            "Label",
            "Spearman r",
            "p-values",
            "Coordinates (X,Y,Z)",
            ]
            
            sig_df.to_csv(
                os.path.join(
                    save_tables_rsa, f"RSA_{rsa_cond}_table_{p_thresh_str}.csv"
                ),
                index=False,
            )
            # save uncorrected stats img
            rsa_img.to_filename(
                os.path.join(
                    save_tables_rsa, f"RSA_stats_img_{rsa_cond}_uncorrected.nii.gz"
                )
            )

            print("Sig ROIs : ", sig_df["Label"].to_list(), "\n")

#%%
# %%
# uncorrected maps with pruned contours
import warnings

for cond in rsa_fdr_imgs.keys():  # e.g., 'Analgesia', 'Hyperalgesia'

    liberal_img = rsa_liberal_imgs[cond][0]
    fdr_img = rsa_fdr_imgs[cond][0]
    unc_img = rsa_unc_stats_imgs[cond][0]
    display = plotting.plot_glass_brain(
        unc_img,
        display_mode="lyrz",
        colorbar=True,
        cmap="coolwarm",
        threshold=0,              # ensures transparent background
        plot_abs=False,
        vmax=np.max(all_iscs),
        vmin=-np.max(np.abs(unc_img.get_fdata())),
        title=f"ISC {cond}",
    )

    # Add liberal contours (gray, thin)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        display.add_contours(liberal_img, colors="gray", linewidths=0.7)

    # Add strict FDR contours (black, thick)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        display.add_contours(fdr_img, colors="black", linewidths=1.5)



# %%
# #%%
# reload(visu_utils)
# #rename conditions with full names
# conditions_full_names = {cond : full for cond, full in zip(conditions, ['Neutral (Analg.)', 'Analgesia', 'Neutral (Hyperalg.)', 'Hyperalgesia'])} #, 'Neutral (Hyper)', 'N_Ana', 'N_Hyper'])}
# # CAREFUL FOR NAME ORDER / CONDITIONS + full name
# visu_utils.plot_median_isc_dots(sig_isc_tables,conditions_full_names=conditions_full_names, title="Median ISC across significant ROIs")

# # Pie plot of sig regions in df
# #------------------------------
# # all_sugg pie-chart
# cond_pie = 'ANA'
# sig_rois_for_pie = list(sig_isc_tables[cond_pie]['Label'])
# sig_networks = [label.split('_')[2] if len(label.split('_')) > 2 else 'subcortical' for label in sig_rois_for_pie]

# visu_utils.plot_network_radar(sig_networks, title = f'Sig ISC during {cond_pie}')
# # # all_sugg pie-chart
# # cond_pie ='hyper_run'
# # sig_rois_for_pie = list(sig_df_conditions[cond_pie]['Label'])
# # sig_networks = [label.split('_')[2] if len(label.split('_')) > 2 else 'subcortical' for label in sig_rois_for_pie]
# visu_utils.plot_network_radar(sig_networks, title = f'Sig ISC during {cond_pie}')

reload(visu_utils)
renamed_networks = {
    "Cont": "Frontoparietal",
    "SomMot": "Somatomotor",
    "SalVentAttn": "SalVentAttn",
    "DorsAttn": "DorsAttn",
    "Default": "Default",
    "limbic": "Limbic",
    "Vis": "Visual",
    "subcortical": "Subcortical",
}

# Radar plot for IS-RSA annak / ANA-context condition
for radar_cond in["annak_ana_run", 'annak_hyper_run']:  # ['annak_ana_run', 'annak_hyper_run']:
    sig_rois_for_radar = list(rsa_tables_fdr[radar_cond]["Label"])
    correlation_scores = list(rsa_tables_fdr[radar_cond]["Spearman r"])
    sig_networks_orin = [
        label.split("_")[2] if len(label.split("_")) > 2 else "subcortical"
        for label in sig_rois_for_radar
    ]
    sig_networks = [renamed_networks[net] for net in sig_networks_orin]

    # Map network names to colors

    color_map = {
        "Frontoparietal": "#7f7f7f",
        "Somatomotor": "#4e81a6",  # orange
        "SalVentAttn": "#4e81a6",  # red
        "DorsAttn": "#7f7f7f",
        "Default": "#ff6e6e",  # yellow-green
        "Limbic": "#7f7f7f",  # green
        "Visual": "#7f7f7f",  # cyan      # pink
        "Subcortical": "#7f7f7f",  # brown      # gray
    }
    colors = [color_map.get(net, "lightgray") for net in sig_networks]
    all_networks = list(color_map.keys())

    visu_utils.plot_network_polarbar(
        sig_networks,
        all_networks=all_networks,
        title=None,
        max_radius=5,
        font_size=30,
        label_distance=1.10,
        color_map=color_map,
        tick_fontsize=30,
        show_only_max_tick=False,
        save_as=f"{radar_cond}_polar_chart.png",
    )


# visu_utils.plot_isc_colorbar(vmin=norm.vmin, vmax=norm.vmax, cmap="RdBu_r")
# %%
# SAVE ISC matrices for sig. regions in RSA
# -------------------------------------------
from src.visu_utils import schaeffer_region_mapping
reload(visu_utils)
# Fetch ISC mat from sgnificant IS-RSA regions across conditions
# Save and plot
save_sig_rsa_matrices = os.path.join(save_plots_rsa, "isc_mat_FDR_rsa")
os.makedirs(save_sig_rsa_matrices, exist_ok=True)

for rsa_cond in ['annak_ana_run', 'annak_hyper_run', 'euclidean_ana_run']:
    print(rsa_cond)
    sig_df = rsa_tables_fdr[rsa_cond]

    if "ana_run" in rsa_cond:  # to get ISC mat key
        sugg_cond = "ana_run"
    elif "hyper_run" in rsa_cond:
        sugg_cond = "hyper_run"

    for sig_region in sig_df["Label"]:

        if sig_region not in schaeffer_region_mapping:
            print(f"Region {sig_region} not in schaeffer mapping, skipping.")
            continue
        anatomical_name = schaeffer_region_mapping[sig_region]["anatomical_label"]
        region_id = schaeffer_region_mapping[sig_region]["region_index"]
        isc_mat = isc_matrices[sugg_cond][sig_region]  # get from 1 sample ISC dct
        ranked_indices = np.argsort(Y["SHSS_score"].values)
        isc_mat = isc_mat[np.ix_(ranked_indices, ranked_indices)]  # reorder

        vmax = np.max(np.abs(isc_mat))

        fig_path = os.path.join(
            save_sig_rsa_matrices, f"{rsa_cond}_{sig_region}_id{region_id}.png"
        )
        visu_utils.plot_simmat_isc(
            simmat=isc_mat,
            title=f"{anatomical_name}",
            x_label=None,  #' ranked subjects (SHSS)'
            colorbar_name=None,
            tick_fontsize=60,
            label_fontsize=70,
            vmin=-vmax,
            vmax=vmax,
            show_colorbar=False,
            save_path=fig_path,
            ticks= None,
            dpi=1000
        )

    # Save addition Glass brain of uncorected ISC maps
    unc_isc_img = rsa_unc_stats_imgs[rsa_cond][0]

#%%
# Save uncorrected RSA glass brain plots combined conditions
for rsa_cond in rsa_fdr_imgs.keys():
    unc_isc_img = rsa_unc_stats_imgs[rsa_cond][0]
    fig_path = os.path.join(
        save_sig_rsa_matrices, f"unc_ISC_glass-brain_{rsa_cond}.png"
    )
    fig_path_cb = os.path.join(
        save_sig_rsa_matrices, f"colorbar_unc_ISC_glass-brain_{rsa_cond}.png"
    )
    plotting.plot_glass_brain(
        unc_isc_img,
        threshold=None,
        colorbar=False,
        display_mode="z",
        plot_abs=False,
        cmap='RdBu_r',
        title=None,
        vmax=np.max(all_iscs),
        vmin=-np.max(all_iscs),
        output_file=fig_path
    )

    tick_fontsize=75
    tick_labels = [f"{np.max(all_iscs):.2f}", "", f"{-np.max(all_iscs):.2f}"]

    visu_utils.plot_isc_colorbar(   
        vmin=-np.max(all_iscs),
        vmax=np.max(all_iscs),
        cmap='RdBu_r',
        figsize=(8,3),
        label=None,
        tick_fontsize=tick_fontsize,
        tick_labels=tick_labels,
        horizontal=True,
        save_path=fig_path_cb,
        dpi=1000)
# %%
# View RSA stats images
views_rsa = []
for cond_rsa in rsa_fdr_imgs.keys():
    view = view_img(
        rsa_fdr_imgs[cond_rsa][0],
        threshold=rsa_fdr_imgs[cond_rsa][1],
        title=f"IS-RSA SHSS~{cond_rsa} (FDR<.05)",
    )
    views_rsa.append(view)

# %% Find which region to plot ISC matrix
plot_single_roi(atlas, 173)
keep_ids = list(rsa_tables_fdr["annak_ana_run"]["ROI"].astype(float))
filt_atlas_img, filt_labels = filter_atlas(atlas, id_labels_dct, keep_ids)
plot_filtered_atlas(filt_atlas_img, filt_labels)

# %%
# PUBLICATION BRAIN PLOTS for IS-RSA SHSS~ISC
# ========================
cmap = "RdBu_r"
# cmap = sns.color_palette("icefire", as_cmap=True)

# ANA-RUN
# --------
COORDS = {
    "x": [-54, 0, 8, 56, 64],
    "y": [-32, -26, -4, 6],
    "z": [42, 58],
}

SAVE_TO = os.path.join(save_plots_rsa, "annak_SHSS-ISC_ana_run")  #!!!
os.makedirs(SAVE_TO, exist_ok=True)
IMG = rsa_fdr_imgs["annak_ana_run"][0]
IMG_ID = "FDR_ana_run"

visu_utils.save_slices(IMG, COORDS, SAVE_TO, img_id=IMG_ID, cmap=cmap)
# save colorbar
fig_path = os.path.join(SAVE_TO, "colorbar.png")
cb_ab = visu_utils.save_colorbar(
    IMG,
    cmap=cmap,
    outpath=fig_path,
    symmetric_cbar=False,  # false because we dont keep negative coeff
    offset=0,
    n_ticks=3,
    transparent=True,
)
# %%
# HYPER-RUN
# ----------
views_rsa[1]
COORDS = {
    "x": [-54, -48, -44],
    "y": [-40],
    "z": [24],
}

SAVE_TO = os.path.join(save_plots_rsa, "annak_SHSS-ISC_hyper_run")  #!!!
os.makedirs(SAVE_TO, exist_ok=True)
IMG = rsa_fdr_imgs["annak_hyper_run"][0]  #!!
IMG_ID = "FDR_hyper_run"  #!!

visu_utils.save_slices(IMG, COORDS, SAVE_TO, img_id=IMG_ID, cmap=cmap)
# save colorbar
fig_path = os.path.join(SAVE_TO, "colorbar.png")
cb_ab = visu_utils.save_colorbar(
    IMG,
    cmap=cmap,
    outpath=fig_path,
    symmetric_cbar=False,  # false because we dont keep negative coeff
    offset=0,
    n_ticks=3,
    transparent=True,
)

# %%
# ANA-RUN EUCLIDEWAN MODEL
# ----------
views_rsa[2]
COORDS = {
    "x": [32, 36],
    "y": [36],
    "z": [50],
}

SAVE_TO = os.path.join(save_plots_rsa, "euclidean_SHSS-ISC_ana_run")  #!!!
os.makedirs(SAVE_TO, exist_ok=True)
IMG = rsa_fdr_imgs["euclidian_ana_run"][0]
IMG_ID = "FDR_ana_run"

visu_utils.save_slices(IMG, COORDS, SAVE_TO, img_id=IMG_ID, cmap=cmap)
# save colorbar
fig_path = os.path.join(SAVE_TO, "colorbar.png")
cb_ab = visu_utils.save_colorbar(
    IMG,
    cmap=cmap,
    outpath=fig_path,
    symmetric_cbar=False,  # false because we dont keep negative coeff
    offset=0,
    n_ticks=3,
    transparent=True,
)

# %%
# ====================================================================
# Big clean code: rm duplicates and previous round of visu

# ====================================================================
