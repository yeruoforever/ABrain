import torch
import nibabel as nib
import numpy as np
from typing import Optional, Any
from numpy import ndarray
from torch.multiprocessing import Event, Queue, Value
import monai.networks.nets as nets
import os

import tempfile


__all__ = ["get_img", "get_seg", "start_segment"]


def normalize_min_max(img: ndarray, mi: int, mx: int):
    return (img - mi) / (mx - mi)


class Preprocess(object):
    def __call__(self, img: ndarray, seg: Optional[ndarray] = None, **kwds: Any) -> Any:
        img_csf = img.clip(0, 80)
        img_csf = normalize_min_max(img_csf, 0, 80)
        img_soft = img.clip(-20, 150)
        img_soft = normalize_min_max(img_soft, -20, 150)
        # img_brain = img.clip(-80, 200)
        # img_brain = normalize_min_max(img_brain, -80, 200)
        img_bone = img.clip(80, 300)
        img_bone = normalize_min_max(img_bone, 80, 300)
        # img = np.stack((img_csf, img_soft, img_brain, img_bone), axis=0)
        img = np.stack((img_csf, img_soft, img_bone), axis=0)
        img = torch.tensor(img).float()
        if seg:
            seg = torch.tensor(seg).long().unsqueeze(dim=0)
        return img, seg


def get_img(img: str):
    nii = nib.load(img)
    spacing = nii.header.get_zooms()
    affine = nii._affine
    directions = nib.aff2axcodes(affine)
    data = np.array(nii.dataobj, dtype=np.float32)
    return data, spacing, directions


def start_segment(img: str, prog: Value, finished: Event, queue: Queue):
    prog.value = 0.0
    nii = nib.load(img)
    data = np.array(nii.dataobj, dtype=np.float32)
    n = data.shape[2]
    model = nets.BasicUNet(spatial_dims=2, in_channels=3, out_channels=2)
    state = torch.load(
        os.path.join(os.path.dirname(__file__), "best.model"),
        map_location=torch.device("cpu"),
    )
    model.load_state_dict(state)
    pred = []
    processing = Preprocess()
    model.eval()
    with torch.no_grad():
        for i in range(n):
            img = data[:, :, i]
            img, _ = processing(img, None)
            img = img.unsqueeze(dim=0)
            out = model(img)
            pred.append(out)
            prog.value += 1 / n
        output = torch.cat(pred, dim=0)  # B,C,W,H
        seg: torch.Tensor = output.argmax(dim=1)
    seg = seg.permute(1, 2, 0)
    seg = nib.nifti1.Nifti1Image(seg.numpy().astype(np.uint8), nii._affine)
    tmp_nii = os.path.join(tempfile.gettempdir(), "tmp.nii.gz")
    nib.save(seg, tmp_nii)
    queue.put(tmp_nii)
    prog.value = 1.0
    finished.set()


def get_seg(img: str):
    path = img.replace("img", "seg")
    nii = nib.load(path)
    return nii.get_fdata().astype(np.float32)
