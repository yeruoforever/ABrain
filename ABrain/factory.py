import os
import json

import torch.optim as opt
from torch.optim import lr_scheduler as lrs

from . import modelzoo
from .modelzoo import AbstractModel
from . import dataset
from .dataset import Dataset
from . import transforms
from .transforms import Transform


class Factory(object):
    products = {}

    @classmethod
    def get(cls, name) -> AbstractModel:
        assert name in cls.products, KeyError(
            f"`{name}` not find in Factory.")
        return cls.products[name]

    @classmethod
    def list(cls):
        print("Currently supported products are：")
        for name in cls.products.keys():
            print('\t', name)


class ModelFactory(Factory):
    '''模型工厂'''
    products = {name: getattr(modelzoo, name) for name in modelzoo.__all__}
    del products['AbstractModel']
    del products['ClassifyModel']
    del products['SegmentationModel']
    del products['DeepSupervisedClassify']
    del products['DeepSupervisedSegmentation']


class DataFactory(Factory):
    products = {name: getattr(dataset, name) for name in dataset.__all__}
    del products['Dataset']


class TransfromFactory(Factory):
    products = {
        name: getattr(transforms, name)
        for name in transforms.__all__
    }
    del products['Transform']


class OptimFactory(Factory):
    products = {
        name: getattr(opt, name) for name in opt.__dict__ if name[0].isupper()
    }
    del products['Optimizer']


class SchedulerFactory(Factory):
    products = {
        'LambdaLR': lrs.LambdaLR,
        # 'MultiplicativeLR': lrs.MultiplicativeLR,
        'StepLR': lrs.StepLR,
        'MultiStepLR': lrs.MultiStepLR,
        'ConstantLR': lrs.ConstantLR,
        'LinearLR': lrs.LinearLR,
        'ExponentialLR': lrs.ExponentialLR,
        'SequentialLR': lrs.SequentialLR,
        'CosineAnnealingLR': lrs.CosineAnnealingLR,
        'ChainedScheduler': lrs.ChainedScheduler,
        'ReduceLROnPlateau': lrs.ReduceLROnPlateau,
        'CyclicLR': lrs.CyclicLR,
        'CosineAnnealingWarmRestarts': lrs.CosineAnnealingWarmRestarts,
        # 'OneCycleLR': lrs.OneCycleLR
    }


__all__ = [
    'ModelFactory',
    'DataFactory',
    'TransfromFactory',
    'OptimFactory',
    'SchedulerFactory'
]
