import os
import json

from . import modelzoo
from .modelzoo import AbstractModel
from . import dataset
from .dataset import Dataset
from . import transforms
from .transforms import Transform


class ModelFactory(object):
    '''模型工厂'''
    model_dict = {name: getattr(modelzoo, name) for name in modelzoo.__all__}
    del model_dict['AbstractModel']
    del model_dict['ClassifyModel']
    del model_dict['SegmentationModel']
    del model_dict['DeepSupervisedClassify']
    del model_dict['DeepSupervisedSegmentation']

    @staticmethod
    def create(model_name, *args, **kwargs) -> AbstractModel:
        '''创建模型'''
        assert model_name in ModelFactory.model_dict, \
            (f"`{model_name}` not find in Model Zoo.")
        initialler = ModelFactory.model_dict[model_name]
        return initialler(*args, **kwargs)

    @staticmethod
    def create_from_config(model_name: str, file_name: str) -> AbstractModel:
        assert os.path.exists(file_name), "`{file_name}` not in file system."
        config = json.load(file_name)
        return ModelFactory.create(model_name, **config)

    @staticmethod
    def list() -> None:
        print("Currently supported models are：")
        for name in ModelFactory.model_dict.keys():
            print('\t', name)


class DataFactory(object):
    dataset_dict = {name: getattr(dataset, name) for name in dataset.__all__}
    del dataset_dict['Dataset']

    @staticmethod
    def get(ds: str, *args, **kwargs) -> Dataset:
        assert ds in DataFactory.dataset_dict, \
            (f"`{ds}` don't support.")
        return DataFactory.dataset_dict[ds](*args, **kwargs)

    @staticmethod
    def list():
        print("Currently supported datasets are：")
        for name in DataFactory.dataset_dict.keys():
            print('\t', name)


class TransfromFactory(object):
    transforms_dict = {
        name: getattr(transforms, name)
        for name in transforms.__all__
    }
    del transforms_dict['Transform']
    @staticmethod
    def get(ts: str, *args, **kwargs):
        return TransfromFactory.transforms_dict[ts](*args, *kwargs)

    @staticmethod
    def list():
        print("Currently supported transforms are：")
        for name in TransfromFactory.transforms_dict.keys():
            print('\t', name)
