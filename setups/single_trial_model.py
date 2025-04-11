

from setups.setup_base import ExperimentSetup
from importlib import reload
import os
# reload(ExperimentSetup)

def init():

    setup = ExperimentSetup()

    setup.exclude_sub = []
    setup.keep_n_sub = None  # Set to None to include all subjects

    setup.atlas_name = "schafer-200-2mm" #voexelWise 
    setup.apply_mask = 'whole-brain' #'lanA800'
    setup.prob_threshold = 0.20

    setup.model_is = 'sugg'
    setup.model_id = 'model_single-trial-wb'
    setup._preproc_model_data = 'model3_single-blocks_extraction-vols_23subjects_zscore_sample_detrend-False_08-04-25/extracted_4D_per_cond_23sub/'
    setup.base_path = os.path.join(setup.project_dir, 'results/imaging/preproc_data', setup._preproc_model_data)
    
    setup.conditions = [
    "ANA1_instrbk_1",
    "ANA2_instrbk_1",
    "HYPER1_instrbk_1",
    "HYPER2_instrbk_1",
    "N_ANA1_instrbk_1",
    "N_ANA2_instrbk_1",
    "N_ANA3_instrbk_1",
    "N_HYPER1_instrbk_1",
    "N_HYPER2_instrbk_1",
    "N_HYPER3_instrbk_1"
    ]

    setup.transform_imgs = True #False
    setup.pre_computed = False #'model5_sugg_schafer-200-2mm_mask-whole-brain_preproc_reg-mvmnt-True-6'
    setup.do_isc_analyses = True 
    setup.do_rsa = False  

    setup.n_boot = 5000
    setup.do_pairwise = True # !!
    setup.n_perm = 5000
    setup.n_perm_rsa = 5000  # Default, but can be overridden
    setup.reg_conf = True  # Whether to regress out movement confounds

    return setup

