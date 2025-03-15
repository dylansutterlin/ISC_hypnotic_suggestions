


from setups.setup_base import ExperimentSetup

def init():
    setup = ExperimentSetup()

    setup.atlas_name = "schafer-200-2mm"
    setup.model_id = 'model1'
    setup.model_is = 'sugg'

    setup.transform_imgs = True
    setup.pre_computed = False 
    setup.do_isc_analyses = True
    setup.do_rsa = True

    setup.n_boot = 5000
    setup.do_pairwise = True
    setup.n_perm = 5000
    setup.n_perm_rsa = 5000  # Default, but can be overridden
    setup.reg_conf = True  # Whether to regress out movement confounds
    setup.keep_n_conf = 8
    
    return setup


