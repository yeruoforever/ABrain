import unittest
import os

from ...dataset.base import DatasetWapper
from ...dataset.TDSC import ABUS23

class TestTDSC(unittest.TestCase):
    def setUp(self) -> None:
        pass
    def test_1(self):
        ds = ABUS23()
        d = {
            "M":0,
            "B":0
        }
        for each in ds.labels:
            d[each]+=1
        print(d) # {'M': 58, 'B': 42}

        ds = DatasetWapper(ds)
        for each in ds:
            print(each.spacing,each.shape)
