import numpy as np
import nibabel as nib
from OpenGL.GL import *


def load_nii(path: str):
    nii_img = nib.load(path)
    img = nii_img.get_fdata()
    spacing = nii_img.header.get_zooms()
    img = img.astype(np.float32)
    return img, spacing


def volume_texture(img: np.ndarray) -> GLuint:
    texture = glGenTextures(1)
    glBindTexture(GL_TEXTURE_3D, texture)
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR)
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    bg_img = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_BORDER)
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER)
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER)
    glTexParameterfv(GL_TEXTURE_3D, GL_TEXTURE_BORDER_COLOR, bg_img)
    glTexImage3D(
        GL_TEXTURE_3D,
        0,
        GL_R32F,
        *img.shape,
        0,
        GL_RED,
        GL_FLOAT,
        img.ctypes.data_as(GLvoidp)
    )
    glGenerateMipmap(GL_TEXTURE_3D)
    glBindTexture(GL_TEXTURE_3D, 0)

    return texture


__all__ = [
    "load_nii",
    "volume_texture",
]
