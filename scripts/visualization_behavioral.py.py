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
import src.preproc_utils as preproc_utils
reload(visu_utils)
reload(isc_utils)


# %% Load the data
model_names = {
    "model2_sugg": "model2_sugg_23-sub_schafer_tian-200-2mm_mask-whole-brain_pairWise-True_preproc_reg-mvmnt-True-8",
    "model3_shock": "model3_shock_23-sub_schafer_tian-200-2mm_mask-whole-brain_pairWise-True_preproc_reg-mvmnt-True-8",
}

MODEL_IS = "model2_sugg"
project_dir = "/data/rainville/dSutterlin/projects/ISC_hypnotic_suggestions"
model_name = model_names[MODEL_IS]

results_dir = os.path.join(project_dir, f"results/imaging/ISC/{model_name}")
setup = isc_utils.load_json(os.path.join(results_dir, "setup_parameters.json"))

# create visualization dir.
save_visu = os.path.join(results_dir, "post-hoc_VISU-tables")
os.makedirs(save_visu, exist_ok=True)

save_behav = os.path.join(save_visu, "plots_from_visu_behav")
os.makedirs(save_behav, exist_ok=True)

parcel_name = setup["atlas_name"]
do_pairWise = setup["do_pairwise"]
n_boot = setup["n_boot"]

#%%
#==================================
# load Y data
behav_df = pd.read_csv(
    os.path.join(setup["project_dir"], f"results/behavioral_data_cleaned.csv"),
    index_col=0,
)
behav_df.index.name = "subjects"
behav_df = behav_df.sort_index()

xlsx_path = os.path.join(project_dir, 'masks/Hypnosis_variables_20190114_pr_jc.xlsx')
subjects = list(setup["subjects"])
apm_subjects = ["APM" + subj[4:] for subj in subjects]
Y = preproc_utils.load_process_y(xlsx_path, subjects)

corr_matrix = Y.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(
    corr_matrix,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    square=True,
    linewidths=0.5,
    cbar_kws={"label": "Pearson r"},
)
plt.title("Correlation Matrix of Behavioral Variables", fontsize=14)
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()

# %%
# PUBLICATION JOINTPLOTS
# regression SHSS ~ change in pain 
from src.visu_utils import jointplot
reload(visu_utils)
reload(isc_utils)

color_conds = {"Ana": "#004197", "Hyper": "#df265e", "Neutral": "#a1a0a0", "SHSS": "#79ab8d"}


red_blue = ["#df265e", "#004197"]  # red blue green blue
shss_color = "#98c3db"

x_shss = np.array(Y["SHSS_score"]).astype(float)


for y_name in ["raw_change_ANA", "raw_change_HYPER"]:

    fig_id = "jointplot_SHSS" + y_name + ".png"
    y_label = "Change in pain " if y_name == "raw_change_HYPER" else "Change in pain"
    y_color = red_blue[0] if y_name == "raw_change_HYPER" else red_blue[1]
    g = jointplot(
        x_shss,
        np.array(Y[y_name]).astype(float),
        x_label="Hypnotic susceptibility",
        y_label=y_label,
        title=None,
        density_color_x=shss_color,
        density_color_y=y_color,
        alpha=0.75,
    )
    g.savefig(os.path.join(save_behav, fig_id), dpi=300, bbox_inches="tight")

# %%
# REGRESSION STATS
# # regression SHSS ~ pain
import statsmodels.api as sm

x = np.array(Y["SHSS_score"]).astype(float)
X = sm.add_constant(x)  # add intercept
y = np.array(Y["raw_change_HYPER"]).astype(float)

model = sm.OLS(y, X).fit()
print(model.summary())

print("======NEXT=======")

y = np.array(Y["raw_change_ANA"]).astype(float)

model = sm.OLS(y, X).fit()
print(model.summary())

# %%
# PUBLICATION PAIN RATINGS PLOT
# T test + violin plot for pain ratings
from scipy.stats import ttest_rel
from itertools import combinations
reload(visu_utils)
# Perform paired t-test

variables = {
    "Analgesia": ["mean_VAS_ana_int", "mean_VAS_ana_UnP"],
    "Hyperalgesia": ["mean_VAS_hyper_int", "mean_VAS_hyper_UnP"],
    "Control (Ana.)": ["mean_VAS_Nana_int", "mean_VAS_Nana_UnP"],
    "Control (Hyper.)": ["mean_VAS_Nhyper_int", "mean_VAS_Nhyper_UnP"],
}


cond_means = {}
for label, cols in variables.items():
    print(cols)
    cond_means[label] = Y[cols].mean(axis=1)

# Convert to DataFrame
mean_df = pd.DataFrame(cond_means)
conditions = mean_df.columns.tolist()
# Perform paired t-tests between all condition pairs
print("Paired t-tests between conditions:")
ttest_results = []
for cond1, cond2 in combinations(mean_df.columns, 2):
    stat, p = ttest_rel(mean_df[cond1], mean_df[cond2])
    ttest_results.append(
        {"Comparison": f"{cond1} vs {cond2}", "t-stat": stat, "p-value": p}
    )
    print(f"{cond1} vs {cond2} : t = {stat:.3f}, p = {p:.4f}")

# Convert to DataFrame if needed
ttest_df = pd.DataFrame(ttest_results)
palette = ["#df265e", "#a1a0a0", "#004197", "#a1a0a0"]  # red blue green blue

reload(visu_utils)
fig = visu_utils.plot_four_condition_violin(
    data_dict={
        "Analgesia": mean_df["Analgesia"],
        "Control\n(Ana.)": mean_df["Control (Ana.)"],
        "Hyperalgesia": mean_df["Hyperalgesia"],
        "Control\n(Hyper.)": mean_df["Control (Hyper.)"],
    },
    palette=palette,
    title=None,
)

fig.savefig(
    os.path.join(save_behav, "pain_ratings_violin.png"), dpi=1000, bbox_inches="tight"
)


# %%
# Similarity matrices SHSS (ANNAK | Euclidean)
reload(visu_utils)
cmap = 'RdBu_r'
# UNIVARIATE pairwise similarities : NN & AnnaK
y = Y["SHSS_score"].values
y = (y - np.mean(y)) / np.std(y)
sim_mat_euclidean = isc_utils.compute_behav_similarity(
    y, metric="euclidean", vectorize=False
)
sim_mat_euclidean = (sim_mat_euclidean * 2) - 1
sim_mat_annak = isc_utils.compute_behav_similarity(
    y, metric="annak", vectorize=False
)
ranked_indices = np.argsort(Y["SHSS_score"].values)

sim_annak_ranked = sim_mat_annak[np.ix_(ranked_indices, ranked_indices)]
sim_euclidean_ranked = sim_mat_euclidean[np.ix_(ranked_indices, ranked_indices)]

visu_utils.plot_simmat_isc(
    simmat=sim_annak_ranked,
    x_label=None,  #' ranked subjects (SHSS)'
    colorbar_name=None,
    tick_fontsize=60,
    ticks=False,
    label_fontsize=70,
    title=f"",
    vmin=-1,
    vmax=1,
    show_colorbar=False,
    save_path=os.path.join(save_behav, "simmat_SHSS_AnnaK.png")
)

visu_utils.plot_simmat_isc(
    simmat=sim_euclidean_ranked,
    x_label=None,  #' ranked subjects (SHSS)'
    colorbar_name=None,
    tick_fontsize=60,
    ticks=False,
    label_fontsize=70,
    title=f"",
    vmin=-1,
    vmax=1,
    show_colorbar=False,
    save_path=os.path.join(save_behav, "simmat_SHSS_Euclidean.png")
)

#%%
reload(visu_utils)
tick_fontsize=75    
tick_labels = ['Low', "", 'High']
visu_utils.plot_isc_colorbar(
    vmin=-1,
    vmax=1,
    cmap=cmap,
    figsize=(2, 7),
    label=None,
    tick_fontsize=tick_fontsize,
    tick_labels=tick_labels,
    horizontal=False,)

visu_utils.plot_isc_colorbar(
    vmin=-1,
    vmax=1,
    cmap=cmap,
    figsize=(8,3),
    label=None,
    tick_fontsize=tick_fontsize,
    tick_labels=tick_labels,
    horizontal=True,)
#%%
visu_utils.plot_simmat_behav(
    simmat=sim_euclidean_ranked,
    title= " ", )


#%%
        isc_mat = isc_mat[np.ix_(ranked_indices, ranked_indices)]  # reorder

        vmax = np.max(np.abs(isc_mat))

        fig_path = os.path.join(
            save_sig_rsa_matrices, f"{rsa_cond}_{sig_region}_id{region_id}.png"
        )

