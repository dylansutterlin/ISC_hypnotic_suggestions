from setups.setup_base import ExperimentSetup

def init():
    setup = ExperimentSetup()

    setup.atlas_name = "schafer-200-2mm"
    setup.model_id = 'model2'
    setup.model_is = 'sugg'

    setup.transform_imgs = False
    # 8 or 6 depending!!
    setup.pre_computed = 'model1_sugg_23-sub_schafer-200-2mm_mask-whole-brain_pairWise-True_preproc_reg-mvmnt-True-8'
   
    setup.do_isc_analyses = True
    setup.do_rsa = True

    setup.n_boot = 5000
    setup.do_pairwise = False # LOO
    setup.n_perm = 5000
    setup.n_perm_rsa = 10000  # Default, but can be overridden
    setup.reg_conf = True  # Whether to regress out movement confounds
    setup.keep_n_conf = 8

    
    return setup


