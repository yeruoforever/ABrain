import nibabel as nib
import numpy as np

__all__ = ["load_nii"]


def load_nii(path: str):
    nii = nib.load(path)
    img = nii.get_fdata()
    spacing = nii.header.get_zooms()
    return img.astype(np.float32), spacing
