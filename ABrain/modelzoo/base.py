from torch.nn import Module


class AbstractModel(Module):
    '''模型抽象类'''
    pass


class ClassifyModel(AbstractModel):
    '''分类模型'''

    def __init__(self, n_class: int) -> None:
        super().__init__()
        self.n_class = n_class
    pass


class SegmentationModel(AbstractModel):
    '''分割模型'''

    def __init__(self, n_class: int) -> None:
        super().__init__()
        self.n_class = n_class
    pass


class DeepSupervisedClassify(ClassifyModel):
    '''深度监督分类模型'''

    def __init__(self, n_class: int, depth: int) -> None:
        super().__init__(n_class)
        self.depth = depth


class DeepSupervisedSegmentation(SegmentationModel):
    '''深度监督分割模型'''

    def __init__(self, n_class: int, depth: int) -> None:
        super().__init__(n_class)
        self.depth = depth
