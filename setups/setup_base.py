import os
import json
from sklearn.utils import Bunch 

class ExperimentSetup():

    def __init__(self):

        self.project_dir = "/data/rainville/dSutterlin/projects/ISC_hypnotic_suggestions"
        self._preproc_model_data = 'model2_23subjects_zscore_sample_detrend_25-02-25/extracted_4D_per_cond_23sub'
        self.base_path = os.path.join(self.project_dir, 'results/imaging/preproc_data', self._preproc_model_data)
        self.behav_path = os.path.join(self.project_dir, 'results/behavioral/behavioral_data_cleaned.csv')
        self.results_branch = f'results/imaging/ISC/'

        
        # Subject selection
        self.exclude_sub = []
        self.keep_n_sub = None  # Set to None to include all subjects
        self._NSUB = 23
        self.n_sub =23

        # Model parameters
        self.model_is = 'sugg' # or shock
        self.conditions = ['HYPER', 'ANA', 'NHYPER', 'NANA']
        self.combined_conditions = ['all_sugg', 'modulation', 'neutral']
        self.combined_task_to_test = [self.conditions, self.conditions[0:2], self.conditions[2:4]]
        self.all_conditions = ['HYPER', 'ANA', 'NHYPER', 'NANA', 'all_sugg', 'modulation', 'neutral']

        self.transform_imgs = True #False
        self.pre_computed = False # used if transform_imgs is False, can specify diff model to load
        self.do_isc_analyses = True
        self.do_pairwise = True 
        self.do_rsa = True
    
        self.n_boot = 5000
        self.n_perm = 5000
        self.n_perm_rsa = 5000  # Default, but can be overridden
        self.reg_conf = True  # Whether to regress out movement confounds
        self.keep_n_conf = 8 # only movement params *

        # Atlas selection
        # self.n_rois = 200
        self.atlas_name = f'schafer-200-2mm' 
        self.apply_mask = 'whole-brain'
        self.prob_threshold = 0.30 # only used in 'voxelWise_lanA800'

        # Model naming
        self.model_id = 'model5'

    # @property
    # def n_sub(self):
    #     """Calculate number of subjects dynamically"""
    #     real_n_sub = self._NSUB 

    #     if len(self.exclude_sub) > 0: # not exclusively correct bc keep_n_sub is ordered
    #         real_n_sub = self._NSUB - len(self.exclude_sub)

    #     elif self.keep_n_sub is not None:
    #         real_n_sub = self.keep_n_sub

    #     return real_n_sub


    @property
    def model_name(self):
        """Dynamically updates the model name whenever attributes change."""
        return f'{self.model_id}_{self.model_is}_{self.n_sub}-sub_{self.atlas_name}_mask-{self.apply_mask}_pairWise-{self.do_pairwise}_preproc_reg-mvmnt-{self.reg_conf}-{self.keep_n_conf}'

    @property
    def results_dir(self):
        """Dynamically updates the results directory whenever model parameters change."""
        return os.path.join(self.project_dir, f'results/imaging/ISC/{self.model_name}')

        # self.model_name = f'{self.model_id}_{self.model_is}_{self.n_sub}-sub_{self.atlas_name}_mask-{self.apply_mask}_pairWise-{self.do_pairwise}_preproc_reg-mvmnt-{self.reg_conf}-{self.keep_n_conf}'
        # self.results_branch = f'results/imaging/ISC/'
        # self.results_dir = os.path.join(self.project_dir, f'results/imaging/ISC/{self.model_name}')

    def to_dict(self):
        """Convert the setup object to a dictionary"""
        return vars(self)  # or self.__dict__

    def keys(self):
        """Return the keys of the setup dictionary"""
        return self.to_dict().keys()
    
    def values(self):
        """Return the values of the setup dictionary"""
        return self.to_dict().values()
    
    def items(self):
        """Return the items of the setup dictionary"""
        return self.to_dict().items()
    


    def check_and_create_results_dir(self):
        """Check if the results directory exists and handle overwriting logic."""
        if not os.path.exists(self.results_dir):
            os.makedirs(self.results_dir)
            print(f"Created results directory: {self.results_dir}")
        elif not self.transform_imgs:
            print(f"Results directory {self.results_dir} already exists and will be used for saving new ISC results.")
        else:
            print(f"Results directory {self.results_dir} already exists and will be overwritten!!")
            print("Press 'y' to overwrite or 'n' to exit")
            while True:
                user_input = input("Enter your choice: ").lower()
                if user_input == 'y':
                    break
                elif user_input == 'n':
                    print("Exiting...")
                    exit()
                else:
                    print("Invalid input. Please enter 'y' or 'n'.")

    def save_to_json(self):
        """Save setup parameters to a JSON file for reproducibility."""
        save_path = os.path.join(self.results_dir, "setup_parameters.json")
        
        import numpy as np
        # Convert NumPy arrays before saving
        setup_dict = {key: (value.tolist() if isinstance(value, np.ndarray) else value)
                    for key, value in self.__dict__.items()}
        
        with open(save_path, 'w') as fp:
            json.dump(setup_dict, fp, indent=4)

        print(f"Setup parameters saved to {save_path}")












