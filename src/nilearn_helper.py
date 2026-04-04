import pandas as pd
from nilearn import datasets, input_data, plotting

atlas_options = ["yeo2011", "schaefer2018", "difumo2020", "ajd2021", "harvox2006"]

import nibabel as nib
import pandas as pd
import json
from nilearn import datasets, plotting, input_data
from nilearn.plotting import find_parcellation_cut_coords
from nilearn.input_data import NiftiLabelsMasker


class Atlas:
    def __init__(self, atlas):

        self.title = atlas
        self.maps, self.df, self.probabilistic = self.get_data()
        self.df[["x", "y", "z"]] = self.get_coords()
        self.fig = self.get_fig()

        # --- labels ---
        self.labels = self.df["labels"].tolist()
        self.roi_index = list(range(1, len(self.labels) + 1))
        self.id_labels_dct = dict(zip(self.roi_index, self.labels))

    # ------------------------------------------------------------------ #
    # --------------------------- DATA LOAD ---------------------------- #
    # ------------------------------------------------------------------ #
    def get_data(self):

        if self.title == "yeo2011":
            fetcher = datasets.fetch_atlas_yeo_2011()
            maps = nib.load(fetcher.thick_7)
            df = pd.read_csv("atlases/atlas-yeo2011.csv")
            probabilistic = False

        elif self.title == "schaefer2018":
            fetcher = datasets.fetch_atlas_schaefer_2018(
                n_rois=100, yeo_networks=7, resolution_mm=2
            )
            maps = nib.load(fetcher.maps)
            df = [label.decode() for label in fetcher.labels]
            df = pd.DataFrame(df, columns=["labels"])
            probabilistic = False

        elif self.title == "difumo2020":
            fetcher = datasets.fetch_atlas_difumo(dimension=64)
            maps = nib.load(fetcher.maps)
            df = pd.DataFrame(
                [label[1] for label in fetcher.labels], columns=["labels"]
            )
            probabilistic = True

        elif self.title == "harvox2006":
            fetcher = datasets.fetch_atlas_harvard_oxford(
                "cort-maxprob-thr25-2mm", symmetric_split=True
            )
            maps = fetcher.maps
            df = pd.DataFrame(fetcher.labels[1:], columns=["labels"])
            probabilistic = False

        elif self.title == "schaefer_tian216":
            atlas_path = (
                "/data/rainville/dSutterlin/projects/ISC_hypnotic_suggestions/"
                "masks/Tian2020_schaeffer200_subcortical16/"
                "combined_schaefer200_tian16_DSG.nii.gz"
            )
            labels_json = (
                "/data/rainville/dSutterlin/projects/ISC_hypnotic_suggestions/"
                "masks/Tian2020_schaeffer200_subcortical16/"
                "roi_labels_dict_DSG.json"
            )
            maps = nib.load(atlas_path)

            with open(labels_json, "r") as f:
                id_labels_dct = json.load(f)

            # remove background key (id "0")
            id_labels_dct.pop("0", None)

            # --- ensure proper numeric sorting by ROI id ---
            id_labels_dct = {
                int(k): v for k, v in sorted(id_labels_dct.items(), key=lambda x: int(x[0]))
            }

            # --- extract ordered ROI indices and labels ---
            roi_index = list(id_labels_dct.keys())
            labels = list(id_labels_dct.values())

            # --- build dataframe with guaranteed correct order ---
            df = pd.DataFrame({
                "roi": roi_index,
                "labels": labels
            })

            probabilistic = False



        else:
            raise ValueError(f"Unknown atlas: {self.title}")

        return maps, df, probabilistic

    # ------------------------------------------------------------------ #
    # -------------------------- COORDINATES --------------------------- #
    # ------------------------------------------------------------------ #
    def get_coords(self):
        if self.probabilistic:
            return plotting.find_probabilistic_atlas_cut_coords(maps_img=self.maps)
        else:
            return find_parcellation_cut_coords(labels_img=self.maps)

    # ------------------------------------------------------------------ #
    # ------------------------- VISUALIZATION -------------------------- #
    # ------------------------------------------------------------------ #
    def get_fig(self):

        kwargs = {"display_mode": "z", "annotate": False, "draw_cross": False, "colorbar" : True}

        if self.probabilistic:
            return plotting.plot_prob_atlas(self.maps, **kwargs)
        else:
            return plotting.plot_roi(self.maps, **kwargs)

    # ------------------------------------------------------------------ #
    # --------------------------- MASKER ------------------------------- #
    # ------------------------------------------------------------------ #
    def get_masker(self, maps, probabilistic, kwargs=None):

        if kwargs is None:
            kwargs = {
                "smoothing_fwhm": None,
                "standardize": True,
                "standardize_confounds": True,
                "memory_level": 3,
            }
        

        if probabilistic:
            return input_data.NiftiMapsMasker(maps_img=maps, **kwargs)
        else:
            return input_data.NiftiLabelsMasker(
                labels_img=maps, strategy="mean", **kwargs
            )
