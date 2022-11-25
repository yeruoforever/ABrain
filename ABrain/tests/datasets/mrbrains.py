import unittest

import os
import torchio as tio
from torch.utils.data import DataLoader


from ...dataset.MRBrains import MRBrains18


class TestMRBrains(unittest.TestCase):

    def setUp(self) -> None:
        pass

    def test_MRBrains18(self):
        B = 2
        ds = MRBrains18("/dev/shm/tinyset/MRBrains18")
        for data in DataLoader(ds, batch_size=B, shuffle=True):
            names = data['name']
            imgs = data['img'][tio.DATA]
            segs = data['seg'][tio.DATA]
            self.assertEqual(imgs.shape[-3:], (256, 256, 192))
            self.assertEqual(segs.shape[-3:], (256, 256, 192))
