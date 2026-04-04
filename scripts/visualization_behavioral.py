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
Y, Yraw = preproc_utils.load_process_y(xlsx_path, subjects)

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
reload(visu_utils)
reload(isc_utils)
from src.visu_utils import jointplot

color_conds = {"Ana": "#004197", "Hyper": "#df265e", "Neutral_hypo": "#a1a0a0","Neutral_hyper" : "#a1a0a0", "SHSS": "#76afcf"}

red_blue = ["#df265e", "#004197"]  # red blue green blue
shss_color = "#98c3db"

x_shss = np.array(Y["SHSS_score"]).astype(float)


for y_name in ["raw_change_ANA", "raw_change_HYPER"]:

    fig_id = "jointplot_SHSS" + y_name + ".png"
    y_label = "Δ pain (Hyperalgesia) " if y_name == "raw_change_HYPER" else "Δ pain (Hypoalgesia)"
    y_color = color_conds['Hyper'] if y_name == "raw_change_HYPER" else color_conds['Ana']

    g = jointplot(
        x_shss,
        np.array(Y[y_name]).astype(float),
        x_label="Hypnotic suggestibility",
        y_label=y_label,
        title=None,
        density_color_x=color_conds['SHSS'],
        density_color_y=y_color,
        alpha=0.60,
    )
    g.savefig(os.path.join(save_behav, fig_id), dpi=1000, bbox_inches="tight")


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
#%%
# load preproc dataframe
# implementation ~10 oct. while testing filal IS-RSA pain models

from sklearn.preprocessing import StandardScaler
xlsx_preproc = os.path.join(project_dir, 'masks/Preprocessed_sugg_pain_behavioral.xlsx' )
behavioral_df = pd.read_excel(xlsx_preproc)  
scaler = StandardScaler(with_std=False) # keep variance, just mean center

behav_interest = {}
# behav_interest['pain_diff_Ana'] = scaler.fit_transform(np.array(behavioral_df['pain_diff_Ana'], dtype='float').reshape(-1,1))
# behav_interest['pain_diff_Hyper'] = scaler.fit_transform(np.array(behavioral_df['pain_diff_Hyper'], dtype='float').reshape(-1,1))
# behav_interest['total_change_pain'] = scaler.fit_transform(np.array(behavioral_df['total_chge_pain_hypAna'], dtype='float').reshape(-1,1))
# behav_interest['SHSS_score'] = scaler.fit_transform(np.array(behavioral_df['SHSS_score'], dtype='float').reshape(-1,1))
# behav_interest['resid_pain_diff_Ana'] = scaler.fit_transform(np.array(behavioral_df['resid_pain_diff_Ana'], dtype='float').reshape(-1,1))
# behav_interest['resid_pain_diff_Hyper'] = scaler.fit_transform(np.array(behavioral_df['resid_pain_diff_Hyper'], dtype='float').reshape(-1,1))

behav_interest['pain_diff_Ana'] = np.array(behavioral_df['pain_diff_Ana'], dtype='float')
behav_interest['pain_diff_Hyper'] = np.array(behavioral_df['pain_diff_Hyper'], dtype='float')
behav_interest['total_change_pain'] = np.array(behavioral_df['total_chge_pain_hypAna'], dtype='float')
behav_interest['SHSS_score'] = np.array(behavioral_df['SHSS_score'], dtype='float')
behav_interest['resid_pain_diff_Ana'] = np.array(behavioral_df['resid_pain_diff_Ana'], dtype='float')
behav_interest['resid_pain_diff_Hyper'] = np.array(behavioral_df['resid_pain_diff_Hyper'], dtype='float')

chge_ana = np.array(behavioral_df['pain_diff_Ana'], dtype='float')
chge_hyper = np.array(behavioral_df['pain_diff_Hyper'], dtype='float')
chge_ana[chge_ana > 0] = 0 #cap to 0 for no modulation
chge_hyper[chge_hyper < 0] = 0
absolute_pain_modulation = np.abs(chge_ana) + np.abs(chge_hyper)
sc_absolute_pain_modulation = scaler.fit_transform(absolute_pain_modulation.reshape(-1,1))

# secondary pain variables
ANNAK_VERSION = 'mean'
annak_sim_matrices = {}
annak_sim_matrices['SHSS_score'] = isc_utils.compute_behav_similarity(y, metric="annak", annak_version=ANNAK_VERSION, vectorize=False)
annak_sim_matrices['total_change_pain'] = isc_utils.compute_behav_similarity(behav_interest['total_change_pain'], metric="annak", annak_version=ANNAK_VERSION, vectorize=False)
# annak_sim_matrices['resid_pain_diff_Ana'] = isc_utils.compute_behav_similarity(behav_interest['resid_pain_diff_Ana'], metric="annak", vectorize=False)
# annak_sim_matrices['resid_pain_diff_Hyper'] = isc_utils.compute_behav_similarity(behav_interest['resid_pain_diff_Hyper'], metric="annak", vectorize=False)
annak_sim_matrices['pain_diff_Ana'] = isc_utils.compute_behav_similarity(behav_interest['pain_diff_Ana'], metric="annak", annak_version=ANNAK_VERSION, vectorize=False)
annak_sim_matrices['pain_diff_Hyper'] = isc_utils.compute_behav_similarity(behav_interest['pain_diff_Hyper'], metric="annak", annak_version=ANNAK_VERSION, vectorize=False)
# # annak_sim_matrices['Ana_diff_global_neutral'] = isc_utils.compute_behav_similarity(behav_interest['Ana_diff_global_neutral'], metric="annak", vectorize=False)

NN_sim_matrices = {}
NN_sim_matrices['SHSS_score'] = isc_utils.compute_behav_similarity(y, metric="euclidean", vectorize=False)
# NN_sim_matrices['total_change_pain'] = isc_utils.compute_behav_similarity(behav_interest['total_change_pain'], metric="euclidean", vectorize=False)
NN_sim_matrices['pain_diff_Ana'] = isc_utils.compute_behav_similarity(behav_interest['pain_diff_Ana'], metric="euclidean", vectorize=False)
NN_sim_matrices['pain_diff_Hyper'] = isc_utils.compute_behav_similarity(behav_interest['pain_diff_Hyper'], metric="euclidean", vectorize=False)


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
    "Neutral (Ana.)": ["mean_VAS_Nana_int", "mean_VAS_Nana_UnP"],
    "Neutral (Hyper.)": ["mean_VAS_Nhyper_int", "mean_VAS_Nhyper_UnP"],
}

cond_means = {}
for label, cols in variables.items():
    print(cols)
    cond_means[label] = Y[cols].mean(axis=1)

Y['int_diff_Ana'] = -(Y['mean_VAS_ana_int'] - Y['mean_VAS_Nana_int'])
Y['unp_diff_Ana'] = -(Y['mean_VAS_ana_UnP'] - Y['mean_VAS_Nana_UnP'])
Y['int_diff_Hyper'] = Y['mean_VAS_hyper_int'] - Y['mean_VAS_Nhyper_int']
Y['unp_diff_Hyper'] = Y['mean_VAS_hyper_UnP'] - Y['mean_VAS_Nhyper_UnP']

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
        "Hypoalgesia": mean_df["Analgesia"],
        "Neutral\n(Hypo)": mean_df["Neutral (Ana.)"],
        "Hyperalgesia": mean_df["Hyperalgesia"],
        "Neutral\n(Hyper)": mean_df["Neutral (Hyper.)"],
    },
    x_name = 'Verbal suggestion conditions',
    palette=[color_conds['Ana'], color_conds['Neutral_hypo'], color_conds['Hyper'], color_conds['Neutral_hyper']],
    title=None,
)

fig.savefig(
    os.path.join(save_behav, "pain_ratings_violin.png"), dpi=1000, bbox_inches="tight"
)

#%%
# REPRODUCING Desmarteaux et al., 2021:NEW VARIABLES IN DF : SAME as JENI 
Y['pain_diff_Ana'] =  -(cond_means["Neutral (Ana.)"] - cond_means["Analgesia"]) 
Y['pain_diff_Hyper'] = cond_means["Hyperalgesia"] - cond_means["Neutral (Hyper.)"] 

from sklearn.linear_model import LinearRegression

X = mean_df["Neutral (Ana.)"].values.reshape(-1, 1)
y = Y["pain_diff_Ana"].values
reg = LinearRegression().fit(X, y)
mod_resid = y - reg.predict(X)
Y['resid_pain_diff_Ana'] = mod_resid
X = mean_df["Neutral (Hyper.)"].values.reshape(-1, 1)
y = Y["pain_diff_Hyper"].values
reg = LinearRegression().fit(X, y)
mod_resid = y - reg.predict(X)
Y['resid_pain_diff_Hyper'] = mod_resid
Y['resid_total_chge_pain_hypAna'] = Y['resid_pain_diff_Hyper'] - Y['resid_pain_diff_Ana']

g = jointplot(
    np.array(Y['pain_diff_Ana']).astype(float),
    np.array(Y['raw_change_ANA']).astype(float),
    x_label="Hypnotic susceptibility",
    y_label=y_label,
    title=None,
    density_color_x=shss_color,
    density_color_y=y_color,
    alpha=0.75,
)

# SAVE MODIFIED Y
Y.to_excel(os.path.join(project_dir , 'masks/Preprocessed_sugg_pain_behavioral.xlsx' ))
Y.to_csv(os.path.join(project_dir , 'masks/Preprocessed_sugg_pain_behavioral.csv' ))

#%%
# REprocude DEsmaarteaux et al., 2021 (used pain INT only: r=0.56 with SHSS)
g = jointplot(
    np.array(Y['pain_diff_Ana']).astype(float),
    np.array(Y['SHSS_score']).astype(float),
    x_label="Hypnotic susceptibility",
    y_label=y_label,
    title=None,
    density_color_x=shss_color,
    density_color_y=y_color,
    alpha=0.75,
)

g = jointplot(
    np.array(behav_interest['pain_diff_Ana']).astype(float),
    np.array(Y['SHSS_score']).astype(float),
    x_label="Hypnotic susceptibility",
    y_label=y_label,
    title=None,
    density_color_x=shss_color,
    density_color_y=y_color,
    alpha=0.75,
)
#%% Correlation matrix for updated Y
corr_matrix = Y.corr()
plt.figure(figsize=(20, 15))
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

#%%
from sklearn.preprocessing import MinMaxScaler
from numpy.linalg import norm
reload(isc_utils)

minmax_scaler = MinMaxScaler(feature_range=(-1, 1))

var = pd.concat([Y['int_diff_Ana'], Y['unp_diff_Ana'] ], axis=1)
var = Y['pain_diff_Hyper']
# var = minmax_scaler.fit_transform(var.values.reshape(-1, 1)).ravel()
# min max scaler on each column

sim_mat = isc_utils.compute_behav_similarity(var, metric="annak", standardize = True,annak_version= 'mean', vectorize=False)
#rescale to [-1,1]
sim_mat = (sim_mat * 2) - 1

# ranked_indices = np.argsort(norm(np.asarray(var.values, dtype='float'), axis=1))

ranked_indices = np.argsort(var.values.ravel())
ord_mat = sim_mat[np.ix_(ranked_indices, ranked_indices)]
visu_utils.plot_simmat_isc(
    simmat=ord_mat,
    x_label=None,  #' ranked subjects (SHSS)'
    show_colorbar = True,
)

#%%


#%%
# checking correlation between int and unp
# Stack intensity and unpleasantness ratings across all conditions
intensity_cols = [v[0] for v in variables.values()]
unpleasantness_cols = [v[1] for v in variables.values()]

stacked_intensity = Y[intensity_cols].values.flatten()
stacked_unpleasantness = Y[unpleasantness_cols].values.flatten()

x = np.asarray(stacked_intensity, dtype=np.float64).ravel()
y = np.asarray(stacked_unpleasantness, dtype=np.float64).ravel()

# Compute correlation matrix (2x2)
corr_matrix = np.corrcoef(np.stack([x, y]))
r = corr_matrix[0, 1]
print(f"Global correlation between intensity and unpleasantness: r = {r:.3f}")

#%%
import matplotlib.pyplot as plt
import numpy as np

# Compute pain change scores
pain_change_ana = cond_means["Analgesia"].values - cond_means["Neutral (Ana.)"].values
pain_change_hyper = cond_means["Hyperalgesia"].values - cond_means["Neutral (Hyper.)"].values 

# Extract SHSS scores
shss_scores = Y["SHSS_score"].values

# Create scatterplot with color mapping
plt.figure(figsize=(8, 6))
scatter = plt.scatter(
    pain_change_ana,
    pain_change_hyper,
    c=shss_scores,
    cmap="viridis",  # or try "plasma", "coolwarm", etc.
    edgecolor="k",
    s=80
)

# Add colorbar
cbar = plt.colorbar(scatter)
cbar.set_label("SHSS Score (Suggestibility)")

# Add grid and axes
plt.axhline(0, color='grey', linestyle='--')
plt.axvline(0, color='grey', linestyle='--')
plt.xlabel("Analgesia effect (N_ANA - ANA)")
plt.ylabel("Hyperalgesia effect (HYPER - N_HYPER)")
plt.title("Pain Modulation Effects Colored by Suggestibility")
plt.grid(True)
plt.tight_layout()
plt.show()

#%%
# absolute change in pain ratings from Ana > Hyper
# !reproduced! the metric change in pain by Jeni !!!
total_change_pain = pain_change_hyper - pain_change_ana
plt.figure(figsize=(8, 6))
scatter = plt.scatter(
    total_change_pain,
    Y['total_chge_pain_hypAna'],
    c=shss_scores,
    cmap="viridis",  # or try "plasma", "coolwarm", etc.
    edgecolor="k",
    s=80
)
Y['total_chge_pain_hypAna_local'] = total_change_pain
# %%
# Similarity matrices SHSS (ANNAK | Euclidean)
reload(visu_utils)
all_sim_mats_unranked = {}
all_sim_vec = {}

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
all_sim_mats_unranked["SHSS_Euclidean"] = sim_mat_euclidean
all_sim_mats_unranked["SHSS_AnnaK"] = sim_mat_annak

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
# Pain variable
# compute pairwise beahv for pain modualtion
reload(isc_utils)
from sklearn.preprocessing import MinMaxScaler
def rescale_matrix(mat, feature_range=(-1, 1)):
    scaler = MinMaxScaler(feature_range=feature_range)
    flat = mat.flatten().reshape(-1, 1)
    scaled = scaler.fit_transform(flat).reshape(mat.shape)
    return scaled


clean_pain_var_names = {
    "pain_diff_Ana": "Hypo. pain modulation",
    "pain_diff_Hyper": "Hyper. pain modulation",
    "total_chge_pain_hypAna": "Total pain change",
    "raw_change_ANA": "Pain change Ana",
    "raw_change_HYPER": "Pain change Hyper",
}
for pain_var in ["pain_diff_Ana", "pain_diff_Hyper", "total_chge_pain_hypAna", "raw_change_ANA", "raw_change_HYPER"]:
    if pain_var == 'pain_diff_Ana':
        y_pain = behav_interest['pain_diff_Ana']
    elif pain_var == 'pain_diff_Hyper':
        y_pain = behav_interest['pain_diff_Hyper']
    else:
        y_pain = Y[pain_var].values
        y_pain = (y_pain - np.mean(y_pain)) / np.std(y_pain)

    sim_behav_pain = isc_utils.compute_behav_similarity(
        y_pain, metric="euclidean", vectorize=False
    )
    sim_behav_annak_pain = isc_utils.compute_behav_similarity(
        y_pain, metric="annak", vectorize=False
    )
    # sim_behav_pain = (sim_behav_pain * 2) - 1  # Rescale to [-1,1]
    sim_behav_pain = rescale_matrix(sim_behav_pain, feature_range=(-1, 1))
    # sim_behav_annak_pain = rescale_matrix(sim_behav_annak_pain, feature_range=(-1, 1))
    all_sim_mats_unranked[f"{pain_var}_Euclidean"] = sim_behav_pain
    all_sim_mats_unranked[f"{pain_var}_AnnaK"] = sim_behav_annak_pain

    ranked_indices = np.argsort(Y[pain_var].values)

    sim_annak_ranked = sim_behav_annak_pain[np.ix_(ranked_indices, ranked_indices)]
    sim_euclidean_ranked = sim_behav_pain[np.ix_(ranked_indices, ranked_indices)]

    visu_utils.plot_simmat_isc(
        simmat=sim_annak_ranked,
        x_label=None,  #' ranked subjects (SHSS)'
        colorbar_name=None,
        tick_fontsize=60,
        ticks=None,
        label_fontsize=80,
        title= None, # POSTER SPRclean_pain_var_names[pain_var],
        vmin=-1,
        vmax=1,
        show_colorbar=False,
    )

    visu_utils.plot_simmat_isc(
        simmat=sim_euclidean_ranked,
        x_label=None,  #' ranked subjects (SHSS)'
        colorbar_name=None,
        tick_fontsize=60,
        ticks=None,
        label_fontsize=80,
        title=f"NN {pain_var}",
        vmin=-1,
        vmax=1,
        show_colorbar=False,
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
# PUBLICATION
# the0retical similarity matrices for figures


scores = np.array([1, 3, 5.5, 8, 10])

euclidean_matrix = isc_utils.compute_behav_similarity(
    scores, metric="euclidean", vectorize=False
)
euclidean_matrix = (euclidean_matrix * 2) - 1  # Rescale to [-1,1]
# === AnnaK model matrix ===
annak_matrix = (scores[:, None] + scores[None, :]) / (2 * scores.max())  # Normalize to [0,1]
annak_matrix = annak_matrix * 2 - 1  # Rescale to [-1,1]

# Plot both
from src.visu_utils import plot_simmat_behav, plot_simmat_isc
reload(visu_utils)
cmap = 'RdBu_r'
plot_simmat_isc(euclidean_matrix, cmap=cmap)
plot_simmat_isc(annak_matrix, cmap=cmap)

#%%
random_scores= np.array([0.03142919, 0.63641041, 0.31435598, 0.50857069, 0.90756647,
       0.24929223]) #randomly geneated then copied
random_matrix = isc_utils.compute_behav_similarity(
    random_scores, metric="euclidean", vectorize=False
)
random_matrix = (random_matrix * 2) - 1  # Rescale to [-1,1]
plot_simmat_isc(random_matrix, cmap=cmap)

# %%
# BEHAVIORAL IS-RSA coparison


# Define which matrices belong to each model
euclidean_keys = [k for k in all_sim_mats_unranked if "Euclidean" in k]
annak_keys = [k for k in all_sim_mats_unranked if "AnnaK" in k]

# RSA parameters
n_perm_rsa = 5000
rsa_results = []

# Function to extract upper triangle as vector
def get_upper(mat):
    iu = np.triu_indices(mat.shape[0], k=1)
    return mat[iu]

# --- Compute RSA for each pair in the Euclidean group ---
print("RSA within Euclidean models:")
for i, key1 in enumerate(euclidean_keys):
    for key2 in euclidean_keys[i+1:]:
        vec1 = get_upper(all_sim_mats_unranked[key1])
        vec2 = get_upper(all_sim_mats_unranked[key2])
        
        r, p, _ = isc_utils.matrix_permutation(
            vec1, vec2,
            n_permute=n_perm_rsa,
            metric="spearman",
            how="upper",
            tail=2,
            return_perms=False,
        )
        print(f"RSA {key1} ~ {key2}: r = {r:.3f}, p = {p:.4f}")
        rsa_results.append((key1, key2, r, p))

# --- Compute RSA for each pair in the AnnaK group ---

print("RSA within AnnaK models:")
for i, key1 in enumerate(annak_keys):
    for key2 in annak_keys[i+1:]:
        vec1 = get_upper(all_sim_mats_unranked[key1])
        vec2 = get_upper(all_sim_mats_unranked[key2])
        
        r, p, _ = isc_utils.matrix_permutation(
            vec1, vec2,
            n_permute=n_perm_rsa,
            metric="spearman",
            how="upper",
            tail=2,
            return_perms=False,
        )
        print(f"RSA {key1} ~ {key2}: r = {r:.3f}, p = {p:.4f}")
        rsa_results.append((key1, key2, r, p))

pd.DataFrame(rsa_results, columns=["Model 1", "Model 2", "Spearman r", "p-value"])
