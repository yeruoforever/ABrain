import os
import json

from ABrain.modelzoo.base import AbstractModel
from ABrain.modelzoo.LeNet5 import LeNet5


class ModelFactory(object):
    '''模型工厂'''
    model_dict = {
        "LeNet5": LeNet5
    }

    @staticmethod
    def create(model_name, *args, **kwargs) -> AbstractModel:
        '''创建模型'''
        assert model_name in ModelFactory.model_dict, \
            ("`{model_name}` not find in Model Zoo.")
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
