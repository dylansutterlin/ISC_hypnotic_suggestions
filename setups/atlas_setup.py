

from setups.setup_base import ExperimentSetup

setup = ExperimentSetup()
setup.atlas_name = "schafer100_2mm"
setup.n_rois = 200

setup.model_is = 'sugg'
setup.conditions = ['HYPER', 'ANA', 'NHYPER', 'NANA']
setup.n_boot = 5000
setup.do_pairwise = True
setup.n_perm = 5000
setup.n_perm_rsa = 5000  # Default, but can be overridden
setup.reg_conf = True  # Whether to regress out movement confounds


# Save setup
setup.save_to_json()


