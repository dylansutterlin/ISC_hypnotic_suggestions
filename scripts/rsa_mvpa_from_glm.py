# %%
import os
import numpy as np
import pandas as pd
import sys
import nibabel as nib
import matplotlib.pyplot as plt

from nilearn.glm.first_level import FirstLevelModel
from nilearn.glm.second_level import SecondLevelModel
from nilearn.glm.contrasts import compute_contrast
from nilearn.image import mean_img, concat_imgs
from nilearn.plotting import plot_design_matrix, plot_stat_map
from nilearn import plotting
from nilearn.glm.thresholding import threshold_stats_img




print('current working dir : ', os.getcwd())
# %%
from src import preproc_utils, visu_utils, qc_utils
import src.glm_utils as utils

from sklearn.utils import Bunch
from importlib import reload
from nilearn import datasets,image
from datetime import datetime

# %% [markdown]
## load data
preproc_model_name = r'model2_23subjects_zscore_sample_detrend_25-02-25' #r'model2_3subjects_zscore_sample_detrend_25-02-25'
model_dir = rf'/data/rainville/dSutterlin/projects/ISC_hypnotic_suggestions/results/imaging/preproc_data/{preproc_model_name}'
model_name = "model3_23subjects_nuis_nodrift_{}".format(datetime.today().strftime("%d-%m-%y"))

model_name = 'model3_final-isc_23subjects_nuis_nodrift_31-03-25' #final model, reproduced Desmarteaux et al., 2019 !! 31 mars

#load manually
# model_name = r'model3_23subjects_allnuis_nodrift_10-03-25'

#%%
setup_dct = preproc_utils.load_json(os.path.join(model_dir, 'setup_parameters.json'))
data_info_dct = preproc_utils.load_pickle(os.path.join(model_dir, 'data_info_regressors.pkl'))
masker_params = preproc_utils.load_json(os.path.join(model_dir, 'preproc_params.json'))

MAX_ITER = None

setup = Bunch(**setup_dct)
setup.run_id = ['ANA', 'HYPER']
data = Bunch(**data_info_dct)
masker_params = Bunch(**masker_params)
glm_info = Bunch()
subjects = setup.subjects
subjects.sort()

if MAX_ITER == None :
    MAX_ITER= len(subjects) 
else: 
    subjects = subjects[:MAX_ITER]
    setup.subjects = subjects   

regressors_dct = data.regressors_per_conds
condition_names = setup.run_names
tr = setup.tr

reload(utils)
ref_img = nib.load(setup.ana_run[0])
single_img = image.index_img(ref_img, 0)
mask = utils.load_data_mask(ref_img)

mni_temp = datasets.load_mni152_template(resolution=1)
mni_bg = qc_utils.resamp_to_img_mask(mni_temp, mask)

save_glm = os.path.join(setup.project_dir, 'results', 'imaging', 'GLM', model_name)
os.makedirs(save_glm, exist_ok=True)
setup.save_dir = save_glm

# %%
#========================================
# Load 1st level maps for further analyses
#==========================================
from glob import glob as glob
from nilearn.maskers import NiftiLabelsMasker
from nilearn.image import binarize_img
from nilearn.plotting import view_img
from nilearn.datasets import fetch_atlas_schaefer_2018
from src import qc_utils, isc_utils

model_res = r'/data/rainville/dSutterlin/projects/ISC_hypnotic_suggestions/results/imaging/GLM/model3_23subjects_nuis_nodrift_31-03-25'
project_dir = setup.project_dir
results_dir = setup['save_dir']

mvpa_save_to = os.path.join(results_dir, 'mvpa_similarity')
os.makedirs(mvpa_save_to, exist_ok=True)

# load maps 
all_shock_maps = glob(os.path.join(model_res, 'all_shock', 'firstlev_localizer_*.nii.gz'))
all_sugg = glob(os.path.join(model_res, 'all_sugg', 'firstlev_localizer_*.nii.gz'))

def build_subject_dict(file_list):
    """Build a dict: {subject_id: filepath} from a list of NIfTI file paths."""
    subject_dict = {}
    for path in file_list:
        fname = os.path.basename(path)
        subj_id = fname.split('_')[-1].replace('.nii.gz', '')  # expects '..._sub-01.nii.gz'
        subject_dict[subj_id] = path
    return subject_dict

sugg_dict = build_subject_dict(all_sugg) # not sorted!!
shock_dict = build_subject_dict(all_shock_maps)

# Intersect 
shared_subjects = sorted(set(sugg_dict) & set(shock_dict))

# load atlas for ROI

#load Tian subcortical + combine with schaeffer 
tian_sub_cortical = nib.load(os.path.join(project_dir, 'masks/Tian2020_schaeffer200_subcortical16/Schaefer2018_200Parcels_17Networks_order_Tian_Subcortex_S1.dlabel.nii.gz'))
tian_data = tian_sub_cortical.get_fdata()

# Load the labels from a text file
labels_tian16 = os.path.join(project_dir, 'masks/Tian2020_schaeffer200_subcortical16/Schaefer2018_200Parcels_17Networks_order_Tian_Subcortex_S1_label.txt')
with open(labels_tian16, 'r') as f:
    tian16_labels = [line.strip() for line in f][::2][0:16] #!! getting only labels, only 16 !!


atlas_data = fetch_atlas_schaefer_2018(n_rois = 200, resolution_mm=2)
atlas = nib.load(atlas_data['maps'])
atlas_path = atlas_data['maps'] #os.path.join(project_dir,os.path.join(project_dir, 'masks', 'k50_2mm', '*.nii*'))
# labels_bytes = list(atlas_data['labels'])
full_labels = [str(label, 'utf-8') if isinstance(label, bytes) else str(label) for label in atlas_data['labels']]
roi_index = [full_labels.index(lbl)+1 for lbl in full_labels]
id_labels_dct = dict(zip(roi_index, full_labels))

#combined Tian + shaeffer
combined_data = atlas.get_fdata().copy()
combined_data[tian_data > 0] = tian_data[tian_data > 0] + 200  # Avoid index collision
combined_img = nib.Nifti1Image(combined_data, affine=atlas.affine, header=atlas.header)
nib.save(combined_img, os.path.join(project_dir, 'masks/Tian2020_schaeffer200_subcortical16/', 'combined_schaefer200_tian16.nii.gz'))

image.plot_roi(combined_img,bg_img=single_img, colorbar=True, display_mode = 'x', cut_coords=(-60,-50,0,50,60 ))

#%%



