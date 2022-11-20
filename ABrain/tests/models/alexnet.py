import unittest
import torch

from ...modelzoo.AlexNet import AlexNet


class TestAlexNet(unittest.TestCase):

    def setUp(self) -> None:
        B, C, W, H = 2, 3, 224, 224
        self.in_shape = (B, C, W, H)
        self.n_class = 6
        self.data = torch.rand(B, C, W, H)
        self.labels = torch.rand(B, self.n_class)
        self.lossfunc = torch.nn.CrossEntropyLoss()

    def test_alexnet_cpu(self):
        B, C, W, H = self.in_shape
        model = AlexNet(self.n_class, C)
        y_hat = model(self.data)
        y = self.labels
        # print(y_hat.shape)
        loss: torch.Tensor = self.lossfunc(y, y_hat)
        self.assertEqual(y_hat.shape, (B, self.n_class))
        loss.backward()
        self.assertTrue(True)

    def test_alexnet_gpu(self):
        B, C, W, H = self.in_shape
        gpu = torch.device("cuda:0")
        model = AlexNet(self.n_class, in_channels=C).to(gpu)
        y_hat = model(self.data.to(gpu))
        y = self.labels.to(gpu)
        loss: torch.Tensor = self.lossfunc(y, y_hat)
        loss.backward()
        self.assertTrue(True)

    def tearDown(self) -> None:
        pass
