import os
import nibabel as nib
from typing import *
import numpy as np
import torchio as tio

from .base import OurDataset, read_config


class CTCSF(OurDataset):
    def __init__(self, resolution: str = "5.0mm") -> None:
        config = read_config("CTCSF")
        database = config["database"]
        database = os.path.expanduser(database)
        self.resolution = resolution
        super().__init__(
            database, "CTCSF", has_seg=True, has_info=False, has_label=False
        )

    def get_samples(self, database):
        path = os.path.join(self.database, self.resolution)
        segs = os.path.join(path, "seg")
        sids = os.listdir(segs)
        sids = list(map(lambda x: x[:-7], sids))
        return sids

    def img_file(self, sid):
        return os.path.join(self.database, self.resolution, "img", f"{sid}.nii.gz")

    def seg_file(self, sid):
        return os.path.join(self.database, self.resolution, "seg", f"{sid}.nii.gz")
