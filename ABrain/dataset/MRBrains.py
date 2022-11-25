import os
from typing import Optional
import torchio as tio
from torchio.transforms import Transform, Resample, ToCanonical
from torch.utils.data import Dataset


class MRBrains18(Dataset):
    def __init__(self, database: str, transforms: Optional[Transform] = None, preprocess: Optional[Transform] = Resample("img")) -> None:
        super().__init__()
        self.database = database
        self.preprocess = preprocess
        self.transforms = transforms
        self.to_canonical = ToCanonical()
        self.sids = os.listdir(database)

    def __getitem__(self, index):
        sid = self.sids[index]
        img = os.path.join(self.database, sid, "pre", "T1.nii.gz")
        seg = os.path.join(self.database, sid, "segm.nii.gz")
        subject = tio.Subject(
            name=sid,
            img=tio.ScalarImage(img),
            seg=tio.LabelMap(seg)
        )

        if self.preprocess is not None:
            subject = self.preprocess(subject)
        if self.transforms is not None:
            subject = self.transforms(subject)

        return subject

    def __len__(self):
        return len(self.sids)
