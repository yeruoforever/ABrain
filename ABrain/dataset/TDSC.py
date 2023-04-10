import os
from typing import *

import pandas as pds
from .base import OurDataset,read_config


class ABUS23(OurDataset):
    def __init__(self) -> None:
        config = read_config()
        database = config["TDSC-ABUS"]["database"]
        super().__init__(database, has_seg=True,has_label=True)
        label_file = config["TDSC-ABUS"]["labels"]
        label_file = os.path.join(database,label_file)
        df = pds.read_csv(label_file)
        self.labels = df["label"].tolist()

    def get_samples(self, database):
        return list(str(i).zfill(3) for i in range(100))
    
    def img_file(self, sid):
        filename = f"DATA_{sid}.nrrd"
        return os.path.join(self.database,"DATA",filename)

    def seg_file(self, sid):
        filename = f"MASK_{sid}.nrrd"
        return os.path.join(self.database,"MASK",filename)
    
