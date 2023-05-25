import os

from .base import OurDataset


class Kits19(OurDataset):
    def __init__(self, database, has_seg: bool = True) -> None:
        super().__init__(database, has_seg)

    def get_samples(self, database):
        sids = os.listdir(database)
        sids = list(filter(lambda x: x.startswith("case"), sids))
        return sids

    def img_file(self, sid):
        return os.path.join(self.database, sid, "imaging.nii.gz")

    def seg_file(self, sid):
        return os.path.join(self.database, sid, "segmentation.nii.gz")
