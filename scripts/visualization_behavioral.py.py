# behavioral Plots and analyses

save_path = "/data/rainville/dSutterlin/projects/ISC_hypnotic_suggestions/results/imaging/RSA/isc-RSA_ext_conds_sugg-pain_tian216_2tails"
save_figures = os.path.join(
    project_dir,
    "results/imaging/RSA/isc-RSA_ext_conds_sugg-pain_tian216_2tails/post-hoc_figures",
)
os.makedirs(save_path, exist_ok=True)


from src import preproc_utils

isc_results_roi = {}

behav_df = pd.read_csv(
    os.path.join(setup["project_dir"], f"results/behavioral_data_cleaned.csv"),
    index_col=0,
)
behav_df.index.name = "subjects"
behav_df = behav_df.sort_index()

xlsx_path = r"/data/rainville/dSutterlin/projects/ISC_hypnotic_suggestions/masks/Hypnosis_variables_20190114_pr_jc.xlsx"
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
import matplotlib.pyplot as plt
import seaborn as sns

red_blue = ["#df265e", "#004197"]  # red blue green blue
shss_color = "#a1a0a0"
# FINAL JOINTPLOTS
# plot jointplots
from src import isc_utils
import src.isc_utils as isc_utils
import src.visu_utils as visu_utils
from importlib import reload

reload(visu_utils)
reload(isc_utils)

from src.visu_utils import jointplot

x_shss = np.array(Y["SHSS_score"]).astype(float)


for y_name in ["raw_change_ANA", "raw_change_HYPER"]:

    fig_id = "jointplot_SHSS" + y_name + ".png"
    y_label = "Change in pain " if y_name == "raw_change_HYPER" else "Change in pain"
    y_color = red_blue[0] if y_name == "raw_change_HYPER" else red_blue[1]
    g = jointplot(
        x_shss,
        np.array(Y[y_name]).astype(float),
        x_label="Hypnotic Susceptibility",
        y_label=y_label,
        title=None,
        density_color_x=shss_color,
        density_color_y=y_color,
    )
    g.savefig(os.path.join(save_figures, fig_id), dpi=1000, bbox_inches="tight")

# %%
# REGRESSION
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
# FINAL PAIN RATINGS PLOT
# T test + violin plot for pain ratings
from scipy.stats import ttest_rel
from itertools import combinations

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
        "Neutral\n(Ana.)": mean_df["Neutral (Ana.)"],
        "Hyperalgesia": mean_df["Hyperalgesia"],
        "Neutral\n(Hyper.)": mean_df["Neutral (Hyper.)"],
    },
    palette=palette,
    title=None,
)

fig.savefig(
    os.path.join(save_figures, "pain_ratings_violin.png"), dpi=1000, bbox_inches="tight"
)


# %%
# cov matrix Y_sugg
corr_matrix_sugg = np.corrcoef(np.array(Y).astype(float), rowvar=False)
np.fill_diagonal(corr_matrix_sugg, 0)

plt.figure(figsize=(10, 8))
sns.heatmap(
    corr_matrix_sugg,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    vmin=-1,
    vmax=1,
    xticklabels=Y.columns,
    yticklabels=Y.columns,
    square=True,
    cbar_kws={"label": "Correlation", "shrink": 0.8},  # Adjust shrink to control size
)
cbar = plt.gca().collections[0].colorbar
cbar.ax.tick_params(labelsize=15)  # Increase the fontsize of the color bar
cbar.set_label("Correlation", fontsize=18)  # Increase the label size
plt.title("Correlation Matrix of hypnotic features", fontsize=22)
plt.xticks(rotation=45, ha="right", fontsize=15)
plt.yticks(rotation=0, fontsize=15)
