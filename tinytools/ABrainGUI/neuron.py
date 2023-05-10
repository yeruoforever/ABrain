import torch
import nibabel as nib
import numpy as np

__all__ = [
    "get_img",
    "get_seg",
]


def get_img(img: str):
    nii = nib.load(img)
    spacing = nii.header.get_zooms()
    data = nii.get_fdata().astype(np.float)
    return data, spacing


def get_seg(img: str):
    path = img.replace("img", "seg")
    nii = nib.load(path)
    return nii.get_fdata().astype(np.float32)
