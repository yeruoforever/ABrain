from typing import Union, Tuple

import torch
from nibabel.affines import apply_affine
from torchio.data import Subject
from torchio.transforms.augmentation.random_transform import RandomTransform
from torchio.transforms.spatial_transform import SpatialTransform


class RandomCrop(RandomTransform, SpatialTransform):
    def __init__(
            self,
            shape: Union[int, Tuple[int, ...]],
            only_target: bool = False,
            **kwargs):
        super().__init__(**kwargs)
        self.shape = shape
        self.only_target = only_target

    def apply_transform(self, subject: Subject) -> Subject:
        if self.only_target:
            (i0, j0, k0), (i1, j1, k1) = self._try_a_patch(subject)
        else:
            (i0, j0, k0), (i1, j1, k1) = self._random_patch(subject)
        w, h, d = self.shape
        for image in self.get_images(subject):
            new_origin = apply_affine(image.affine, (i0, j0, k0))
            new_affine = image.affine.copy()
            new_affine[:3, 3] = new_origin
            c = image.shape[0]
            new_data = torch.zeros(c, w, h, d)
            new_data[:, :i1, :j1, :k1] = image.data[:, i0:i1, j0:j1, k0:k1]
            image.set_data(new_data)
            image.affine = new_affine
        return subject

    def _random_patch(self, subject: Subject):
        _, W, H, D = subject.shape
        shape = torch.tensor([x for x in self.shape], dtype=int)
        bounds = torch.tensor([W, H, D], dtype=int)
        st = torch.rand(3)*bounds
        st = st.floor_().to(int)
        st = torch.where(st > bounds-shape, bounds-shape, st)
        st = torch.where(st < 0, 0, st)
        ed = st+shape
        return st.tolist(), ed.tolist()

    def _try_a_patch(self, subject: Subject):
        ids = torch.argwhere(subject['seg'].data > 0)[:, 1:]
        shape = torch.tensor([x for x in self.shape], dtype=int)
        _, W, H, D = subject.shape
        bounds = torch.tensor([W, H, D], dtype=int)
        n = torch.randint(ids.shape[0], (1,)).item()
        center = ids[n, :]
        st = center-torch.div(shape, 2, rounding_mode='floor')
        st = torch.where(st > bounds-shape, bounds-shape, st)
        st = torch.where(st < 0, 0, st)
        ed = st+shape
        return st.tolist(), ed.tolist()

    @staticmethod
    def is_invertible():
        return False
