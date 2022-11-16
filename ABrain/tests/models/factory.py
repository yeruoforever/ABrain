import unittest

from ...model import ModelFactory


class TestFactory(unittest.TestCase):
    def test_create(self):
        d = {'n_class': 10, 'in_channels': 1, 'hidden': 5, "other": 999}
        ModelFactory.create("LeNet5", **d)

    def test_list(self):
        ModelFactory.list()

    def test_LeNet5(self):
        import torch
        from ...modelzoo.base import ClassifyModel
        x = torch.rand(32, 1, 32, 32)
        model = ModelFactory.create("LeNet5", 10, 1)
        self.assertTrue(isinstance(model,ClassifyModel))
        y = model(x)
        print(y.shape)
        self.assertEqual(y.shape,(32,10))
        self.assertTrue(torch.all(torch.sum(y,dim=1)-1<=1e-4))
