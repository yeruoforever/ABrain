import unittest
import os

import torch.utils.data as data
import torchio as tio

from ...dataset.IBSR import IBSR18

class TestIBSR(unittest.TestCase):

    def setUp(self) -> None:
        self.database="/dev/shm/tinyset"

    def test_ibsr18(self):
        B = 2
        for mode_id in range(3):
            dataset= IBSR18(os.path.join(self.database,"IBSR18"),mode_id)
            for x in data.DataLoader(dataset,batch_size=B,shuffle=True):
                names = x["name"]
                imgs = x["img"][tio.DATA]
                segs = x["seg"][tio.DATA]
                self.assertEqual(len(names),B)
                self.assertEqual(imgs.shape,(B,1,256,128,256))
                self.assertEqual(segs.shape,(B,1,256,128,256))