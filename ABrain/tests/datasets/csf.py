import unittest
import torchio as tio

from ...dataset.csf import CTCSF
from ...dataset.base import DatasetWapper


class TestCTCSF(unittest.TestCase):
    def setUp(self) -> None:
        return super().setUp()

    def test_ct_csf(self):
        ds = CTCSF()
        ds = DatasetWapper(ds, lambda x: x)
        for each in ds:
            img = each["img"][tio.DATA]
            seg = each["seg"][tio.DATA]
            print(img.shape, img.dtype, seg.shape, seg.dtype)
