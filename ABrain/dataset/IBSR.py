import os
from typing import Optional
import torchio as tio
from torch.utils.data import Dataset

from .base import OurDataset


class IBSR18(OurDataset):
    def __init__(self, database, transforms: Optional[tio.Transform] = None, target: int = 2) -> None:
        super().__init__(
            database,
            "IBSR18",
            has_info=False,
            has_seg=True,
            has_label=False
        )
        self.ts = transforms
        seg_mode = ["TRI_fill", "TRI", ""]
        self.mode = seg_mode[target]

    def get_samples(self, database):
        sids = os.listdir(database)
        sids = list(filter(lambda x: x.startswith("IBSR"), sids))
        sids.sort()
        return sids

    def img_file(self, sid):
        file = "%s_ana.nii.gz" % sid
        return os.path.join(self.database, sid, file)

    def seg_file(self, sid):
        file = "%s_seg%s_ana.nii.gz" % (sid, self.mode)
        return os.path.join(self.database, sid, file)
