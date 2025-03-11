

from setups.setup_base import ExperimentSetup
from importlib import reload
# reload(ExperimentSetup)

def init():

    setup = ExperimentSetup()

    setup.exclude_sub = []
    setup.keep_n_sub = 3  # Set to None to include all subjects

    setup.atlas_name = "schafer100_2mm"
    setup.n_rois = 100

    setup.model_is = 'sugg'
    setup.conditions = ['HYPER', 'ANA', 'NHYPER', 'NANA']

    setup.transform_imgs = True #False
    setup.do_isc_analyses = True 
    setup.do_rsa = True   

    setup.n_boot = 100
    setup.do_pairwise = True
    setup.n_perm = 100
    setup.n_perm_rsa = 5000  # Default, but can be overridden
    setup.reg_conf = True  # Whether to regress out movement confounds

    # Save setup
    setup.save_to_json()

    return setup

