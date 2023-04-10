import os
from typing import *
import json

import numpy.random as nprand

from .base import OurDataset,read_config

class Kits21(OurDataset):
    def __init__(self,seg_mode:str="AND") -> None:
        assert seg_mode in ["AND","MAJ","OR","RAND"]
        self.SMOD = ["AND","MAJ","OR"]
        self.seg_mode = seg_mode
        config = read_config("Kits21")
        database = config["database"]
        info = config["metainfo"]
        super().__init__(
            database,
            "Kits21",
            has_seg=True,
            has_info=True,
            has_label=False
        )
        with open(os.path.join(self.database,info)) as info:
            self.info = json.load(info)

    def get_samples(self, database):
        return list(str(i).zfill(5) for i in range(300))
    
    def img_file(self, sid):
        file_name = "imaging.nii.gz"
        return os.path.join(self.database,f"case_{sid}",file_name)

    def seg_file(self, sid):
        if self.seg_mode == "RAND":
            seed = nprand.randint(3)
            seg_mode = self.SMOD[seed]
            file_name = f"aggregated_{seg_mode}_seg.nii.gz"
        else:
            file_name = f"aggregated_{self.seg_mode}_seg.nii.gz"
        return os.path.join(self.database,f"case_{sid}",file_name)
    


    