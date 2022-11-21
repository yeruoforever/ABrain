import unittest
import torch

from ...transforms.RGB_PCA import RGBPCA


class TestTransform(unittest.TestCase):
    
    def setUp(self) -> None:
        self.shape=3,512,512
        C,W,H = self.shape
        self.img=torch.rand(C,W,H)
    
    def test_rgb_pca(self):
        trans=RGBPCA()
        transformed=trans(self.img)
        self.assertEqual(self.shape,transformed.shape)