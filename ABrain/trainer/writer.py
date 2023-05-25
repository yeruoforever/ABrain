import os
from torch import Tensor
from typing import Union, List
import nibabel as nib
import numpy as np
from collections import namedtuple


Meta = namedtuple("SegmentationMeta", ["name", "affine"])


class InferenceWriter(object):
    def __init__(
        self,
        dir_out: str,
    ) -> None:
        if not os.path.exists(dir_out):
            print("%s not exists, creating..." % dir_out)
            os.makedirs(dir_out)
        self.dir_out = dir_out

    def save_as(self, path, data: Tensor, affine: Tensor):
        raise NotImplementedError()

    def parse_path(self, name: str):
        raise NotImplementedError()

    def __save_one(self, name: str, pred: Tensor, affine: Tensor):
        path = self.parse_path(name)
        self.save_as(path, pred, affine)

    def save(self, meta: Meta, pred: Tensor):
        data = pred.cpu()
        names, affine = meta
        if isinstance(names, str):
            self.__save_one(names, data, affine)
        else:
            for i in range(len(names)):
                self.__save_one(names[i], data[i], affine[i])


class NIfTIWriter(InferenceWriter):
    def __init__(self, dir_out: str) -> None:
        super().__init__(dir_out)

    def parse_path(self, name: str):
        file = os.path.join(self.dir_out, name + ".pred.nii.gz")
        parent = os.sep.join(file.split(os.sep)[:-1])
        if not os.path.exists(parent):
            os.makedirs(parent)
        return file

    def save_as(self, path, data: Tensor, affine: Tensor):
        data = data.argmax(dim=0)
        if isinstance(data, Tensor):
            data = data.numpy()
        if isinstance(affine, Tensor):
            affine = affine.numpy()
        out = nib.Nifti1Image(data.astype(np.int8), affine)
        nib.save(out, path)


class CAMWriter(InferenceWriter):
    def __init__(self, dir_out: str) -> None:
        super().__init__(dir_out)

    def parse_path(self, name: str):
        file = os.path.join(self.dir_out, name + ".cam.npz")
        parent = os.sep.join(file.split(os.sep)[:-1])
        if not os.path.exists(parent):
            os.makedirs(parent)
        return file

    def save_as(self, path, data: Tensor, affine: Tensor):
        out = data.softmax(dim=0).numpy()
        np.savez_compressed(path, out)


class MultiWriter(InferenceWriter):
    def __init__(self, writers: List[InferenceWriter]) -> None:
        self.writers = writers

    def save(self, meta: Meta, data: Tensor):
        for writer in self.writers:
            writer.save(meta, data)
