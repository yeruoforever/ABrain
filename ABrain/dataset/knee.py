import os

from .base import OurDataset


class KneeHighResTrain(OurDataset):
    def __init__(self, database) -> None:
        super().__init__(database, has_seg=True)

    def get_samples(self, database):
        dir_img = os.path.join(database, "image")
        sids = os.listdir(dir_img)
        sids = list(map(lambda x: x.split('.')[0], sids))
        return sids

    def img_file(self, sid):
        return os.path.join(self.database, 'image', sid+'.nii.gz')

    def seg_file(self, sid):
        return os.path.join(self.database, 'mask', sid+'mask.nii.gz')


class KneeHighResVal(OurDataset):
    def __init__(self, database) -> None:
        super().__init__(database, has_seg=True)

    def get_samples(self, database):
        dir_img = os.path.join(database, "image_02")
        sids = os.listdir(dir_img)
        sids = list(map(lambda x: x.split('.')[0], sids))
        return sids

    def img_file(self, sid):
        return os.path.join(self.database, 'image_02', sid+'.nii.gz')

    def seg_file(self, sid):
        return os.path.join(self.database, 'mask_02', sid+'mask.nii.gz')