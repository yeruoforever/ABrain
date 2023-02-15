import unittest

from ...factory import *
from ...pipeline.base import optional



class TestFactoryModel(unittest.TestCase):
    def test_create(self):
        d = {'n_class': 10, 'in_channels': 1, 'hidden': 5, "other": 999}
        ModelFactory.get("LeNet5")(**d)

    def test_list(self):
        ModelFactory.list()

    def test_LeNet5(self):
        import torch
        from ...modelzoo.base import ClassifyModel
        x = torch.rand(32, 1, 32, 32)
        model = optional(ModelFactory.get("LeNet5"), 10, 1)

        self.assertTrue(isinstance(model, ClassifyModel))
        y = model(x)
        self.assertEqual(y.shape, (32, 10))
        self.assertTrue(torch.all(torch.sum(y, dim=1)-1 <= 1e-4))


class TestFactoryDataset(unittest.TestCase):
    def test_get(self):
        ds = DataFactory.get("IBSR18")
        print(ds)
    def test_list(self):
        DataFactory.list()

class TestFactoryTransform(unittest.TestCase):
    def test_get(self):
        ts = TransfromFactory.get('RGBPCA')
        print(ts)

    def test_list(self):
        TransfromFactory.list()

class TestFactoryOptim(unittest.TestCase):
    def test_get(self):
        opt =OptimFactory.get('SGD')
        print(opt)
    def test_list(self):
        OptimFactory.list()

class TestFactoryScheduler(unittest.TestCase):
    def test_get(self):
        sch=SchedulerFactory.get('CosineAnnealingWarmRestarts')
        print(sch)

    def test_list(self):
        SchedulerFactory.list()