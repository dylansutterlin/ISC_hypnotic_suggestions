import sys
import nibabel as nib
import pandas as pd
from nilearn import datasets, maskers, plotting

sys.path.append("../masks")

atlas_options = [
    "yeo2011",
    "schaefer2018",
    "difumo2020",
    "ajd2021",
    "harvox2006",
    "sensaas",
    "voxelWise_lanA800"
]

class Atlas:
    def __init__(self, atlas, masker_params=None, mask_img=None):
        """
        :param atlas: str, name of the atlas (e.g., "sensaas", "voxelWise_lanA800", etc.)
        :param masker_params: dict, overrides for NiftiMapsMasker/NiftiLabelsMasker parameters
        :param mask_img: nibabel.Nifti1Image or file path, optional mask to restrict extraction
        """
        self.title = atlas
        self.mask_img = mask_img
        self.masker_params = masker_params or {}
        self.maps, self.df, self.probabilistic = self.get_data()
        self.df[["x", "y", "z"]] = self.get_coords()
        self.fig = self.get_fig()

    def get_data(self, n_rois=None):
        """Load atlas maps and corresponding labels/DF based on self.title."""
        if self.title == "yeo2011":
            fetcher = datasets.fetch_atlas_yeo_2011()
            maps = fetcher.thick_7
            df = pd.read_csv("atlases/atlas-yeo2011.csv")
            probabilistic = False

        elif self.title == "schaefer2018":
            if n_rois is None: n_rois = 200
            fetcher = datasets.fetch_atlas_schaefer_2018(
                n_rois=n_rois, yeo_networks=7, resolution_mm=2
            )
            maps = fetcher.maps
            df = [label.decode() for label in fetcher.labels]
            df = pd.DataFrame(df, columns=["labels"])
            probabilistic = False

        elif self.title == "difumo2020":
            if n_rois is None: n_rois = 64
            fetcher = datasets.fetch_atlas_difumo(dimension=n_rois)
            maps = fetcher.maps
            df = pd.DataFrame([lbl[1] for lbl in fetcher.labels], columns=["labels"])
            probabilistic = True

        elif self.title == "ajd2021":
            maps = "atlases/atlas-ajd2021.nii.gz"
            df = pd.read_csv("atlases/atlas-ajd2021.csv")
            probabilistic = False

        elif self.title == "harvox2006":
            fetcher = datasets.fetch_atlas_harvard_oxford(
                "cort-maxprob-thr25-2mm", symmetric_split=True
            )
            maps = fetcher.maps
            # The first label is "Background," so we skip it
            df = pd.DataFrame(fetcher.labels[1:], columns=["labels"])
            probabilistic = False

        elif self.title == "sensaas":
            folder = "sensaas"

            df = pd.read_csv(f"{folder}/SENSAAS_description.csv")
            maps = nib.load(f"{folder}/SENSAAS_MNI_ICBM_152_2mm.nii")
            probabilistic = False

        # elif self.title == "voxelWise_lanA800":
        #     # Lipkin et al. 2022
        #     maps = nib.load('lipkin2022_lanA800', 'LanA_n806.nii')
        #     df = pd.DataFrame(
        #         {"labels": [f"voxel_{i}" for i in range(1, maps.shape[-1] + 1)]}
        #     )
        #     probabilistic = False

        else:
            raise ValueError(f"Atlas '{self.title}' not recognized or not yet supported.")

        return maps, df, probabilistic

    def get_coords(self):

        if self.probabilistic:
            return plotting.find_probabilistic_atlas_cut_coords(maps_img=self.maps)
        else:
            return plotting.find_parcellation_cut_coords(labels_img=self.maps)

    def get_fig(self):

        kwargs = {"display_mode": "z", "annotate": False, "draw_cross": False}

        if self.probabilistic:
            return plotting.plot_prob_atlas(self.maps, **kwargs)
        else:
            return plotting.plot_roi(self.maps, **kwargs)

    def get_masker(self, smooth_fwhm=None, memory_path="wb-ppi-cache"):

        default_kwargs = {
            "smoothing_fwhm": smooth_fwhm,
            "standardize": True,
            "standardize_confounds": True,
            "memory": str(memory_path),
            "memory_level": 3,
        }
        # Merge user-supplied parameters:
        default_kwargs.update(self.masker_params)
        
        # If user provided a separate mask to restrict analysis, include it:
        if self.mask_img is not None:
            default_kwargs["mask_img"] = self.mask_img

        if self.probabilistic:
            return maskers.NiftiMapsMasker(maps_img=self.maps, **default_kwargs)
        else:
            return maskers.NiftiLabelsMasker(labels_img=self.maps, **default_kwargs)
