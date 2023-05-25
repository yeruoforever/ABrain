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
    affine = nii._affine
    directions = nib.aff2axcodes(affine)
    data = nii.get_fdata().astype(np.float32)
    print(directions)
    return data, spacing, directions


def get_seg(img: str):
    path = img.replace("img", "seg")
    nii = nib.load(path)
    return nii.get_fdata().astype(np.float32)
