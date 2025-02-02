# %%
from nilearn.image import math_img
from nilearn.plotting import plot_stat_map
import os
import seaborn as sns
from nilearn import plotting
from nilearn.plotting import view_img 
import utils
import time
from importlib import reload
import nibabel as nib
import numpy as np
import pandas as pd
import json
from nilearn.image import concat_imgs
from brainiak.isc import isc, bootstrap_isc, permutation_isc, compute_summary_statistic, phaseshift_isc
from nilearn.maskers import MultiNiftiMapsMasker, MultiNiftiMasker
from nilearn.datasets import fetch_atlas_schaefer_2018
from sklearn.utils import Bunch
import visu_utils
from nilearn.plotting import view_img_on_surf
reload(visu_utils)
reload(utils)
# %% Load the data

project_dir = "/data/rainville/dSutterlin/projects/ISC_hypnotic_suggestions"
# base_path = "/data/rainville/dSutterlin/projects/ISC_hypnotic_suggestions/results/data/test_data_sugg_3sub"
preproc_model_data = '23subjects_zscore_sample_detrend_FWHM6_low-pass428_10-12-24/suggestion_blocks_concat_4D_23sub'
base_path = os.path.join(project_dir, 'results/imaging/preproc_data', preproc_model_data)

#jeni prepoc
base_path =  r'/data/rainville/Hypnosis_ISC/4D_data/segmented/concat_bks'
behav_path = os.path.join(project_dir, 'results/behavioral/behavioral_data_cleaned.csv')

model_name = f'model3_jeni_preproc-23sub'
model_name = f'model5_jeni_lvlpreproc-23sub_schafer100_2mm'
results_dir = os.path.join(project_dir, f'results/imaging/ISC/{model_name}')

setup = utils.load_json(os.path.join(results_dir, "setup_parameters.json"))

#%%all_results_paths = utils.load_json(os.path.join(results_dir, "result_paths.json"))
atlas_name = 'Difumo256' # change to setup['atlas_name']
n_sub = setup['n_sub']

atlas_data = fetch_atlas_schaefer_2018(n_rois = 100, resolution_mm=2)
atlas = nib.load(atlas_data['maps'])
atlas_path = atlas_data['maps'] #os.path.join(project_dir,os.path.join(project_dir, 'masks', 'k50_2mm', '*.nii*'))
labels = list(atlas_data['labels'])



# %%
# =====================================
# Bootstrap per condition visualization
reload(visu_utils)
reload(utils)
result_key = 'isc_results'
conditions = ['Hyper', 'Ana', 'NHyper', 'NAna']
    # cond = conditions[0]
views = {}
for cond in conditions:
    #masker =utils.load_pickle(os.path.join(results_dir, cond, f'maskers_{atlas_name}_{cond}_{n_sub}sub.pkl'))
    #isc_bootstrap = utils.load_pickle(all_results_paths[result_key][cond])
    masker = utils.load_pickle('/data/rainville/dSutterlin/projects/ISC_hypnotic_suggestions/results/imaging/ISC/model5_jeni_lvlpreproc-23sub_schafer100_2mm/Hyper/maskers_schafer100_2mm_Hyper_23sub.pkl')
    isc_bootstrap = utils.load_pickle(f'/data/rainville/dSutterlin/projects/ISC_hypnotic_suggestions/results/imaging/ISC/model5_jeni_lvlpreproc-23sub_schafer100_2mm/{cond}/isc_results_{cond}_5000boot_pairWiseTrue.pkl')
    isc_rois = pd.DataFrame(isc_bootstrap['isc'], columns=labels)
    isc_median = isc_bootstrap['observed']
    ci = isc_bootstrap['confidence_intervals']
    p_values = isc_bootstrap['p_values']
    dist = isc_bootstrap['distribution']
    n_boot = setup['n_boot']
    fdr_p = utils.fdr(p_values, q=0.05)
    bonf_p = utils.bonferroni(p_values, alpha=0.05)
   

    # to brain plot 
    reload(visu_utils)
    isc_img, isc_thresh = visu_utils.project_isc_to_brain(
        atlas_path=atlas_path,
        isc_median=isc_median   ,
        atlas_labels=labels,
        p_values=p_values,
        p_threshold=bonf_p,
        title = f"ISC Median per ROI for {cond}",
        save_path=None,
        show=True
    )
    views[cond] = view_img_on_surf(isc_img, threshold=isc_thresh, surf_mesh='fsaverage')


    # bar plots 
    # reload(visu_utils)
    # visu_utils.plot_isc_median_with_significance(
    #     isc_median=isc_median,
    #     p_values=p_values,
    #     atlas_labels=labels,
    #     p_threshold=bonf_p,
    #     save_path=None,
    #     show=True,
    #     fdr_correction=True
    # )
#%%


#%%
# ================
# Combinde conditions
reload(visu_utils)
reload(utils)
result_key = 'isc_results'
all_conditions = ['all_sugg', 'modulation', 'neutral']
n_scans = [438, 188, 250]
views = {}
for i, cond in enumerate(all_conditions):
    scans = n_scans[i]
    #masker =utils.load_pickle(os.path.join(results_dir, cond, f'maskers_{atlas_name}_{cond}_{n_sub}sub.pkl'))
    #isc_bootstrap = utils.load_pickle(all_results_paths[result_key][cond])
    masker = utils.load_pickle('/data/rainville/dSutterlin/projects/ISC_hypnotic_suggestions/results/imaging/ISC/model5_jeni_lvlpreproc-23sub_schafer100_2mm/Hyper/maskers_schafer100_2mm_Hyper_23sub.pkl')
    isc_bootstrap = utils.load_pickle(f'/data/rainville/dSutterlin/projects/ISC_hypnotic_suggestions/results/imaging/ISC/model5_jeni_lvlpreproc-23sub_schafer100_2mm/concat_suggs_1samp_boot/isc_results_{cond}_{scans}TRs_5000boot_pairWiseTrue.pkl')
    isc_rois = pd.DataFrame(isc_bootstrap['isc'], columns=labels)
    isc_median = isc_bootstrap['observed']
    ci = isc_bootstrap['confidence_intervals']
    p_values = isc_bootstrap['p_values']
    dist = isc_bootstrap['distribution']
    n_boot = setup['n_boot']
    fdr_p = utils.fdr(p_values, q=0.05)
    bonf_p = utils.bonferroni(p_values, alpha=0.05)


    # to brain plot 
    reload(visu_utils)
    isc_img, isc_thresh = visu_utils.project_isc_to_brain(
        atlas_path=atlas_path,
        isc_median=isc_median,
        atlas_labels=labels,
        p_values=p_values,
        p_threshold=bonf_p,
        title = f"ISC Median per ROI for {cond}",
        save_path=None,
        show=True
    )
    views[cond] = view_img_on_surf(isc_img, threshold=isc_thresh, surf_mesh='fsaverage')

#%%
import matplotlib.pyplot as plt
roi_index = 0
plt.hist(dist[:, roi_index], bins=50, alpha=0.7, label="Bootstrap Distribution")
plt.axvline(isc_median[roi_index], color='red', linestyle='--', label="Observed Median")
plt.xlabel("ISC Values")
plt.ylabel("Frequency")
plt.title(f"Bootstrap Distribution for ROI {roi_index}")
plt.legend()
plt.show()

shifted_dist = dist[:, roi_index] - np.median(dist[:, roi_index])
plt.hist(shifted_dist, bins=50, alpha=0.7, label="Shifted Bootstrap Distribution")
plt.axvline(isc_median[roi_index], color='red', linestyle='--', label="Observed Median")
plt.xlabel("Shifted ISC Values")
plt.ylabel("Frequency")
plt.title(f"Shifted Bootstrap Distribution for ROI {roi_index}")
plt.legend()
plt.show()


#%%
#=====================================
# Bootstrap per combined conditions
result_key = 'isc_combined_results'
folder = 'concat_suggs_1samp_boot'
#folder = 'concat_suggs_1samp_boot'
combined_conditions = ['all_sugg', 'modulation', 'neutral']
conditions = ['Hyper', 'Ana', 'NHyper', 'NAna']
cond = combined_conditions[1]
n_scans = 191

#for cond in conditions:
masker =utils.load_pickle(os.path.join(results_dir, conditions[0], f'maskers_{atlas_name}_{conditions[0]}_{n_sub}sub.pkl'))
file = f"isc_results_{cond}_{n_scans}TRs_{setup['n_perm']}boot_pairWiseTrue.pkl"
isc_bootstrap = utils.load_pickle(os.path.join(results_dir, folder, file))

isc_rois = isc_bootstrap['isc']
isc_median = isc_bootstrap['observed']
ci = isc_bootstrap['confidence_intervals']
p_values = isc_bootstrap['p_values']
dist = isc_bootstrap['distribution']
n_boot = setup['n_boot']

fdr_p = utils.fdr(p_values, q=0.05)

#%%

# %%
# ===========================
# Difference (Permutation) per condition visualization
result_key = 'cond_contrast_permutation'
conditions = ['Hyper', 'Ana', 'NHyper', 'NAna']
contrasts = ['Hyper-Ana', 'Ana-Hyper', 'NHyper-NAna']

reload(visu_utils)
n_scans = [94, 94, 125]
views = {}
sig_rois = {}
for i, cont in enumerate(contrasts):
    scans = n_scans[i]
    # masker =utils.load_pickle(os.path.join(results_dir, cond, f'maskers_{atlas_name}_{cond}_{n_sub}sub.pkl'))
    # file = f"isc_results_{contrast}_{n_scans}TRs_{setup['n_perm']}perm_pairWiseTrue.pkl"
    #isc_bootstrap = utils.load_pickle(all_results_paths[result_key][cond])
    file = f'/data/rainville/dSutterlin/projects/ISC_hypnotic_suggestions/results/imaging/ISC/model5_jeni_lvlpreproc-23sub_schafer100_2mm/cond_contrast_permutation/isc_results_{cont}_{scans}TRs_5000perm_pairWiseTrue.pkl'
    # isc_contrast = utils.load_pickle(os.path.join(results_dir, result_key, file))
    isc_contrast = utils.load_pickle(file)

    grouped_isc = isc_contrast['grouped_isc']        # Grouped ISC values
    observed_diff = isc_contrast['observed_diff']    # Observed differences in ISC
    p_values = isc_contrast['p_value']               # P-values for the ISC contrasts
    distribution = isc_contrast['distribution']      
    fdr_p = utils.fdr(p_values, q=0.05)
    unc_p = 0.05

    reload(visu_utils)
    diff_img, diff_thresh, sig_labels = visu_utils.project_isc_to_brain_perm(
        atlas_path=atlas_path,
        isc_median=observed_diff,
        atlas_labels=labels,
        p_values=p_values,
        p_threshold=unc_p,
        title = f"Difference in ISC between {cont}",
        save_path=None,
        show=True
    )
    views[cont] = view_img_on_surf(diff_img, threshold=diff_thresh, surf_mesh='fsaverage')
    sig_rois[cont] = sig_labels

#%%
#=======================
# Difference for high vs low SHSS
result_key = 'cond_contrast_permutation'
conditions = ['Hyper', 'Ana', 'NHyper', 'NAna']
contrasts = ['Hyper-Ana', 'Ana-Hyper', 'NHyper-NAna']

reload(visu_utils)
n_scans = [94, 94, 125]
views = {}
sig_rois = {}
interactive_views = []
for shss_grp, n_sub in zip(['high_shss', 'low_shss'], [11, 12]):
    print(f"Doing {shss_grp} with {n_sub} subjects")
    cont = 'Hyper-Ana'  
    # masker =utils.load_pickle(os.path.join(results_dir, cond, f'maskers_{atlas_name}_{cond}_{n_sub}sub.pkl'))
    # file = f"isc_results_{contrast}_{n_scans}TRs_{setup['n_perm']}perm_pairWiseTrue.pkl"
    #isc_bootstrap = utils.load_pickle(all_results_paths[result_key][cond])
    file = f'/data/rainville/dSutterlin/projects/ISC_hypnotic_suggestions/results/imaging/ISC/model5_jeni_lvlpreproc-23sub_schafer100_2mm/group_perm_{shss_grp}/isc_results_{n_sub}sub_{cont}_5000perm_pairWiseTrue.pkl'
    # isc_contrast = utils.load_pickle(os.path.join(results_dir, result_key, file))
    isc_contrast = utils.load_pickle(file)

    grouped_isc = isc_contrast['grouped_isc']        # Grouped ISC values
    observed_diff = isc_contrast['observed_diff']    # Observed differences in ISC
    p_values = isc_contrast['p_value']               # P-values for the ISC contrasts
    distribution = isc_contrast['distribution']      
    fdr_p = utils.fdr(p_values, q=0.05)
    unc_p = 0.05

    reload(visu_utils)
    diff_img, diff_thresh, sig_labels = visu_utils.project_isc_to_brain_perm(
        atlas_path=atlas_path,
        isc_median=observed_diff,
        atlas_labels=labels,
        p_values=p_values,
        p_threshold=unc_p,
        title = f"{shss_grp} ({n_sub} subj.) : Difference in ISC between {cont} (unc. p = {unc_p})",
        save_path=None,
        show=True
    )
    views[shss_grp] = view_img_on_surf(diff_img, threshold=diff_thresh, surf_mesh='fsaverage')
    sig_rois[shss_grp] = sig_labels

    # Plot Diff in high to display parietal
    coords = sig_rois['high_shss']['Coordinates'][0]
    diff_thresh = float(sig_rois['high_shss']['Difference'])
    plot_stat_map(
        diff_img,
        threshold=None,
        title="Diff SHSS High ",
        display_mode="z",
        cut_coords=coords,
        colorbar=True
        )
    interactive_view = view_img(
            diff_img,
            threshold=0,
            title='d'
        )
    interactive_views.append(interactive_view)

#%%
#=======================
# Group difference with behavioral


# %%
# ===========================
# grouped isc with behavioral
result_key = 'group_permutation_results'
conditions = ['Hyper', 'Ana', 'NHyper', 'NAna']
conditions = [ 'all_sugg', 'neutral', 'modulation', 'Hyper', 'Ana']

sig_rois_cond = {}
views = {}
for cond in conditions:
    #masker =utils.load_pickle(os.path.join(results_dir, conditions[0], f'maskers_{atlas_name}_{conditions[0]}_{n_sub}sub.pkl'))
    # isc_group = utils.load_pickle(all_results_paths[result_key][cond])
    isc_group = utils.load_pickle(f'/data/rainville/dSutterlin/projects/ISC_hypnotic_suggestions/results/imaging/ISC/model5_jeni_lvlpreproc-23sub_schafer100_2mm/behavioral_group_permutation/{cond}_group_permutation_results_5000perm.pkl')
    behav_ls = list(isc_group.keys())
    sig_rois_cond[cond] = {}
    views[cond] = {}
    for y in behav_ls:
    
        print(f"Condition: {cond}, Behavior: {y}")
        grouped_isc = isc_group[y]['grouped_isc']        # Grouped ISC values
        observed_diff = isc_group[y]['observed_diff']    # Observed differences in ISC
        p_values = isc_group[y]['p_value']               # P-values for the ISC contrasts
        distribution = isc_group[y]['distribution'] 
        fdr_p = utils.fdr(p_values, q=0.05)
        unc_p = 0.01 #0.05

        reload(visu_utils)
        diff_img, diff_thresh, sig_labels = visu_utils.project_isc_to_brain_perm(
            atlas_path=atlas_path,
            isc_median=observed_diff,
            atlas_labels=labels,
            p_values=p_values,
            p_threshold=unc_p,
            title = f"Difference in ISC for {cond} based on median split {y} (unc. p = {unc_p})",
            save_path=None,
            show=True
        )
        sig_rois_cond[cond][y] = sig_labels
        print(sig_labels)
        #views[cond][y] = view_img_on_surf(diff_img, threshold=diff_thresh, surf_mesh='fsaverage')
# %%

# %%

diff_img = masker.inverse_transform(observed_diff)
#p_img = masker.inverse_transform(p_values)
#%%
view = plotting.view_img_on_surf(diff_img, threshold='95%', surf_mesh='fsaverage')

# %%
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from scipy.stats import ttest_1samp
from scipy.stats import ttest_rel

# ===========================
# ISC-RSA
behav_df = pd.read_csv(f'/data/rainville/dSutterlin/projects/ISC_hypnotic_suggestions/results/behavioral_data_cleaned.csv')
X_pheno = behav_df
result_key = 'rsa_isc_results'
conditions = ['Hyper', 'Ana', 'NHyper', 'NAna']
#conditions = ['Hyper', 'Ana', 'NHyper', 'NAna', 'all_sugg', 'modulation', 'neutral']
behav_ls = ['SHSS_score', 'total_chge_pain_hypAna', 'Abs_diff_automaticity']
models =['euclidean', 'annak']

rsa_dict_2ttest = {}

# compare NN and AnnaK models using 2sample t tests
# tests differences ISC-RSA correlation between two models for each var.
n_test_p = []
for y_name in behav_ls:
    print(f'====Processing {y_name} across conditions======')
    rsa_dict_2ttest[y_name] = {}

    for cond in conditions:
        isc_bootstrap = utils.load_pickle(
            f'/data/rainville/dSutterlin/projects/ISC_hypnotic_suggestions/results/imaging/ISC/model5_jeni_lvlpreproc-23sub_schafer100_2mm/{cond}/isc_results_{cond}_5000boot_pairWiseTrue.pkl'
        )
        isc_rois = pd.DataFrame(isc_bootstrap['isc'], columns=labels)
        rsa_dict_2ttest[y_name][cond] = {}

        for simil_model in models:
            # Load RSA data
            rsa_df = pd.read_csv(
                f'/data/rainville/dSutterlin/projects/ISC_hypnotic_suggestions/results/imaging/ISC/model5_jeni_lvlpreproc-23sub_schafer100_2mm/rsa_isc_results_{simil_model}/rsa-isc_{cond}/{y_name}_rsa_isc_{simil_model}simil_10000perm_pvalues.csv'
            )
            p_values = np.array(rsa_df['p_value'])
            fdr_p = utils.fdr(p_values, q=0.05)

            rsa_dict_2ttest[y_name][cond][simil_model] = {
                'correlation': rsa_df['correlation'],
                'p_values': p_values,
                'fdr_p': fdr_p
            }

        # Perform paired t-test between the two models
        correl1 = rsa_dict_2ttest[y_name][cond][models[0]]['correlation']
        correl2 = rsa_dict_2ttest[y_name][cond][models[1]]['correlation']

        t_stat, p_value = ttest_rel(correl1, correl2)
        n_test_p.append(p_value) # for fdr correction across Ttest
        # Store results
        t_dict = {'t': t_stat, 'p': p_value, 'df': len(correl1) - 1}
        rsa_dict_2ttest[y_name][cond]['t_test'] = t_dict

        labels_map, yeo7_net = visu_utils.yeo_networks_from_schaeffer(labels)
        visu_utils.plot_scatter_legend(correl1, correl2, grp_id=yeo7_net,var_name = models, title = f'{y_name} {cond} RSA-ISC per ROI', save_path=None)
        
        # Print results
        if t_stat > 0 and p_value <= fdr_test_p:
            print(f'{models[0]} is better than {models[1]} in {cond}')
            print(f"t({t_dict['df']}) = {t_dict['t']:.2f}, p = {t_dict['p']:.4f}")
        elif t_stat < 0 and p_value < 0.05/12:
            print(f'{models[1]} is better than {models[0]} in {cond}')
            print(f"t({t_dict['df']}) = {t_dict['t']:.2f}, p = {t_dict['p']:.4f}")
        else:
            print(f'No significant difference between {models[0]} and {models[1]} in {cond}')
            print(f"t({t_dict['df']}) = {t_dict['t']:.2f}, p = {t_dict['p']:.4f}")

# Take home is that annak is better in Hyper related conditions while NN in Ana

#%%
reload(visu_utils)
fdr_test_p = utils.fdr(np.array(n_test_p), q=0.05)

#%%
heatmaps = {}
for i, simil_model in enumerate(models):
    # Load ISC bootstrap data
    isc_bootstrap = utils.load_pickle(
        f'/data/rainville/dSutterlin/projects/ISC_hypnotic_suggestions/results/imaging/ISC/model5_jeni_lvlpreproc-23sub_schafer100_2mm/{cond}/isc_results_{cond}_5000boot_pairWiseTrue.pkl'
    )
    isc_rois = pd.DataFrame(isc_bootstrap['isc'], columns=labels)

    # Compute similarity matrix
    y = np.array(behav_df[y_name])
    sim_behav = utils.compute_behav_similarity(y, metric=simil_model)

    # Normalize similarity matrix for better visualization
    sim_behav_norm = (sim_behav - sim_behav.min()) / (sim_behav.max() - sim_behav.min())
    heatmaps[simil_model] = sim_behav_norm

    if i == 1:  # Generate plots at the second iteration
        # Plot Heatmaps
        fig, axs = plt.subplots(1, 2, figsize=(12, 6))

        for ax, (model, matrix) in zip(axs, heatmaps.items()):
            sns.heatmap(
                matrix, ax=ax, cmap='viridis', square=True,
                cbar=True, xticklabels=False, yticklabels=False
            )
            ax.set_title(f"{model.capitalize()} Model Similarity")

        plt.tight_layout()
        plt.show()

        # Plot histograms of correlations
        plt.figure(figsize=(8, 6))
        for model, matrix in heatmaps.items():
            corr_values = matrix[np.triu_indices(matrix.shape[0], k=1)]
            plt.hist(corr_values, bins=20, alpha=0.6, label=f"{model.capitalize()} Model")

        plt.title("Correlation Distribution")
        plt.xlabel("Similarity Value")
        plt.ylabel("Frequency")
        plt.legend()
        plt.show()
#%%
reload(visu_utils)


# Plot the similarity matrix and RSA correlation histogram
visu_utils.plot_similarity_and_histogram(
    similarity_matrix=sim_behav,
    correlations=correl,
    p_values=p_values,
    atlas_labels=atlas_labels,
    behav_name=y_name,
    save_path=None
)

# %%

# ===========================
# supplementary ISC-RSA with other behavioral
X_pheno = pd.read_csv(f'/data/rainville/dSutterlin/projects/ISC_hypnotic_suggestions/results/behavioral_data_cleaned.csv')
behav_ls = ['mean_VAS_Hyper', 'mean_VAS_Ana', 'mean_VAS_NHyper', 'mean_VAS_Nana']
conditions = ['Hyper', 'Ana', 'NHyper', 'NAna']
models =['euclidean', 'annak']
atlas_labels = labels
rsa_dict_2ttest = {}
n_perm_rsa = 10000
save_cond_rsa = '/data/rainville/dSutterlin/projects/ISC_hypnotic_suggestions/results/imaging/ISC/model5_jeni_lvlpreproc-23sub_schafer100_2mm/rsa_isc_results_euclidean/supp_analyses'

# compare NN and AnnaK models using 2sample t tests
# tests differences ISC-RSA correlation between two models for each var.
n_test_p = []
for y_name, cond in zip(behav_ls, conditions):
    print(f'====Processing {y_name} across conditions======')
    key_name = f'{y_name}-{cond}'
    rsa_dict_2ttest[key_name] = {}
    isc_bootstrap = utils.load_pickle(
        f'/data/rainville/dSutterlin/projects/ISC_hypnotic_suggestions/results/imaging/ISC/model5_jeni_lvlpreproc-23sub_schafer100_2mm/{cond}/isc_results_{cond}_5000boot_pairWiseTrue.pkl'
    )
    isc_pairwise = isc_bootstrap['isc']
    isc_rois = pd.DataFrame(isc_bootstrap['isc'], columns=labels)
    
    values_rsa_perm = {}
    # Load RSA data
    y = np.array(X_pheno[y_name])
    y = (y - np.mean(y)) / np.std(y)
    for simil_model in models:
        sim_behav = utils.compute_behav_similarity(y, metric = simil_model)
        df_subjectwise_rsa = pd.DataFrame(index=range(isc_pairwise.shape[0]), columns=atlas_labels)

        for col_j in range(isc_pairwise.shape[1]): # j ROIs
            if atlas_name == 'voxelWise':
                roi_name = f'voxel_{col_j}'
            else:
                roi_name = atlas_labels[col_j]

            isc_roi_vec = isc_pairwise[:, col_j]
            rsa_results = utils.matrix_permutation(sim_behav, isc_roi_vec, n_permute=n_perm_rsa, metric="spearman", how="upper", tail=1, return_perms = True)
            values_rsa_perm[roi_name] = {'correlation': rsa_results['correlation'], 'p_value': rsa_results['p']}
            #distribution_rsa_perm[roi_name] = rsa_results['perm_dist']

        rsa_df = pd.DataFrame.from_dict(values_rsa_perm, orient='index')
        csv_path = os.path.join(save_cond_rsa, f'isc_rsa_{n_perm_rsa}perm_{y_name}_{simil_model}simil_pvalues.csv')
        rsa_df.to_csv(csv_path)
        
        p_values = np.array(rsa_df['p_value'])
        fdr_p = utils.fdr(p_values, q=0.05)

        rsa_dict_2ttest[key_name][simil_model] = {
            'correlation': rsa_df['correlation'],
            'p_values': p_values,
            'fdr_p': fdr_p
        }

    # Perform paired t-test between the two models
    correl1 = rsa_dict_2ttest[key_name][models[0]]['correlation']
    correl2 = rsa_dict_2ttest[key_name][models[1]]['correlation']

    t_stat, p_value = ttest_rel(correl1, correl2)
    n_test_p.append(p_value) # for fdr correction across Ttest
    # Store results
    t_dict = {'t': t_stat, 'p': p_value, 'df': len(correl1) - 1}
    rsa_dict_2ttest[key_name]['t_test'] = t_dict

    labels_map, yeo7_net = visu_utils.yeo_networks_from_schaeffer(labels)
    visu_utils.plot_scatter_legend(correl1, correl2, grp_id=yeo7_net,var_name = models, title = f'{y_name} {cond} RSA-ISC per ROI', save_path=None)
    
    if t_stat > 0 and p_value <= 0.01:
        print(f'{models[0]} is better than {models[1]} in {cond}')
        print(f"t({t_dict['df']}) = {t_dict['t']:.2f}, p = {t_dict['p']:.4f}")
    elif t_stat < 0 and p_value < 0.01:
        print(f'{models[1]} is better than {models[0]} in {cond}')
        print(f"t({t_dict['df']}) = {t_dict['t']:.2f}, p = {t_dict['p']:.4f}")
    else:
        print(f'No significant difference between {models[0]} and {models[1]} in {cond}')
        print(f"t({t_dict['df']}) = {t_dict['t']:.2f}, p = {t_dict['p']:.4f}")

# %%
