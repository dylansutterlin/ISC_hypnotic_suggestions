


from setups.setup_base import ExperimentSetup

def init():
    setup = ExperimentSetup()

    setup.atlas_name = "schafer_tian-200-2mm"
    setup.model_id = 'model1-runs'
    setup.model_is = 'sugg' #!!
    setup.apply_mask = 'whole-brain'
    
    setup.transform_imgs = True
    setup.pre_computed = 'model1_sugg_23-sub_schafer-200-2mm_mask-whole-brain_pairWise-True_preproc_reg-mvmnt-True-8' 
    setup.do_isc_analyses = True
    setup.do_pairwise = True
    setup.do_isfc = False
    setup.do_contrast_permutation = True
    setup.do_group_permutation_from_median = True
    setup.do_low_high_shss_analyses = False # median split and perform 1 sample test + contrast ISC
    setup.do_rsa = False


    setup.n_boot = 10000
    setup.do_pairwise = True
    setup.n_perm = 10000
    setup.n_perm_rsa = 5000  
    setup.reg_conf = True  # Whether to regress out movement confounds
    setup.keep_n_conf = 8
    
    return setup


