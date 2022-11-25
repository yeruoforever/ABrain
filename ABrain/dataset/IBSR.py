import os
from typing import Optional
import torchio as tio
from torch.utils.data import Dataset


class IBSR18(Dataset):
    def __init__(self, database, target: int = 2, transforms: Optional[tio.Transform] = None) -> None:
        super().__init__()
        seg_mode = ["TRI_fill", "TRI", ""]
        self.sids = os.listdir(database)
        self.database = database
        self.mode = seg_mode[target]
        self.preprocess = tio.transforms.ToCanonical()
        self.transforms = transforms

    def __getitem__(self, index):

        sid = self.sids[index]
        img_name = "%s_ana.nii.gz" % sid
        seg_name = "%s_seg%s_ana.nii.gz" % (sid, self.mode)
        img = os.path.join(self.database, sid, img_name)
        seg = os.path.join(self.database, sid, seg_name)

        subject = tio.Subject(
            name=sid,
            img=self.preprocess(tio.ScalarImage(img)),
            seg=self.preprocess(tio.LabelMap(seg))
        )

        if self.transforms is not None:
            subject = self.transforms(subject)

        return subject

    def __len__(self):
        return len(self.sids)
