from torchio import Subject
from typing import Tuple, Dict, Iterable

from .base import Analyzer, tio
from .functional import *


class CompareAnalyzer(Analyzer):
    def __init__(self, name: str) -> None:
        super().__init__(name)

    def parse_data(self, subject: Subject) -> Tuple[Tensor]:
        if not isinstance(subject['pred'], torch.Tensor):    # Monai API
            pred: Tensor = subject['pred'][tio.DATA]
        else:
            pred: Tensor = subject['pred']
        if not isinstance(subject['true'], torch.Tensor):
            true: Tensor = subject['true'][tio.DATA]
        else:
            true: Tensor = subject['true']

        pred = pred.long()
        true = true.long()
        return pred, true


class ASSD(CompareAnalyzer):
    '''Average Symmetric Surface Distance'''

    def __init__(self,
                 n_labels: int,
                 bg_ignore: bool = False,
                 name: str = "ASSD",
                 ) -> None:
        super().__init__(name)
        self.n_labels = n_labels
        self.bg_ignore = bg_ignore

    def analyze(self, subject: Subject):
        pred, true = self.parse_data(subject)
        # out is a list
        out = assd(pred, true, self.n_labels, self.bg_ignore)[0]
        return out


class HausdorffDistance95(CompareAnalyzer):
    '''hausdorff distance'''

    def __init__(self,
                 n_labels: int,
                 bg_ignore: bool = False,
                 name: str = "hausdorff_distance 95",
                 ) -> None:
        super().__init__(name)
        self.n_labels = n_labels
        self.bg_ignore = bg_ignore

    def analyze(self, subject: Subject):
        pred, true = self.parse_data(subject)
        out = hausdorff_distance(pred, true, self.n_labels, self.bg_ignore)[0]
        return out


class Dice(CompareAnalyzer):
    '''Dice'''

    def __init__(self,
                 n_labels: int,
                 bg_ignore: bool = False,
                 name: str = "Dice",
                 ) -> None:
        super().__init__(name)
        self.n_labels = n_labels
        self.bg_ignore = bg_ignore

    def analyze(self, subject: Subject):
        pred, true = self.parse_data(subject)
        out = dice(pred, true, self.n_labels, self.bg_ignore)
        return out


class IoU(CompareAnalyzer):
    '''IoU'''

    def __init__(self,
                 n_labels: int,
                 bg_ignore: bool = False,
                 name: str = "IoU",
                 ) -> None:
        super().__init__(name)
        self.n_labels = n_labels
        self.bg_ignore = bg_ignore

    def analyze(self, subject: Subject):
        pred, true = self.parse_data(subject)
        out = iou(pred, true, self.n_labels, self.bg_ignore)
        return out


class F1Score(CompareAnalyzer):
    def __init__(self,
                 n_labels: int,
                 bg_ignore: bool = False,
                 name: str = "F1 Score",
                 ) -> None:
        super().__init__(name)
        self.n_labels = n_labels
        self.bg_ignore = bg_ignore

    def analyze(self, subject: Subject):
        pred, true = self.parse_data(subject)
        out = f1score(pred, true, self.n_labels, self.bg_ignore)
        return out


class VD(CompareAnalyzer):
    '''Volume Difference'''

    def __init__(self,
                 n_labels: int,
                 bg_ignore: bool = False,
                 name: str = "VD",
                 ) -> None:
        super().__init__(name)
        self.n_labels = n_labels
        self.bg_ignore = bg_ignore

    def analyze(self, subject: Subject):
        pred, true = self.parse_data(subject)
        out = vd(pred, true, self.n_labels, self.bg_ignore)
        return out


class RVD(CompareAnalyzer):
    '''Relative Volume Difference'''

    def __init__(self,
                 n_labels: int,
                 bg_ignore: bool = False,
                 name: str = "RVD",
                 ) -> None:
        super().__init__(name)
        self.n_labels = n_labels
        self.bg_ignore = bg_ignore

    def analyze(self, subject: Subject):
        pred, true = self.parse_data(subject)
        out = rvd(pred, true, self.n_labels, self.bg_ignore)
        return out


class VOE(CompareAnalyzer):
    '''volumetric overlap error'''

    def __init__(self,
                 n_labels: int,
                 bg_ignore: bool = False,
                 name: str = "VOE",
                 ) -> None:
        super().__init__(name)
        self.n_labels = n_labels
        self.bg_ignore = bg_ignore

    def analyze(self, subject: Subject):
        pred, true = self.parse_data(subject)
        out = voe(pred, true, self.n_labels, self.bg_ignore)
        return out


VoxelMethods = {
    "IoU": iou_on_confusion_matrix,
    "Dice": dice_on_confusion_matrix,
    "F1-Score": f1_on_confusion_matrix,
    "VD": vd_on_confusion_matrix,
    "RVD": rvd_on_confusion_matrix,
    "VOE": voe_on_confusion_matrix,
}


def compute_confusion_matrix(pred: Tensor, target: Tensor, n_labels: int):
    ids = target*n_labels+pred
    cnt = ids.bincount(minlength=n_labels*n_labels)
    mat = cnt.reshape(n_labels, n_labels)
    return mat


class VoxelMetrics(Analyzer):
    def __init__(self, metrics:Iterable[str], n_labels: int, bg_ignore: bool = False,smooth:bool=1.) -> None:
        super().__init__()
        for each in metrics:
            assert each in VoxelMethods.keys(
            ), f"`{each}` not in `{VoxelMethods.keys()}`"
        self.metrics = metrics
        self.n_labels= n_labels
        self.bg_ignore = bg_ignore
        self.smooth=smooth

    def parse_data(self, subject: Subject) -> Tuple[Tensor]:
        if not isinstance(subject['pred'], torch.Tensor):    # Monai API
            pred: Tensor = subject['pred'][tio.DATA]
        else:
            pred: Tensor = subject['pred']
        if not isinstance(subject['true'], torch.Tensor):
            true: Tensor = subject['true'][tio.DATA]
        else:
            true: Tensor = subject['true']

        pred = pred.long()
        true = true.long()
        return pred, true
    
    def parse_result(name:str,out):
        """拼接分析结果。

        `{"Subject":sid, "A":a, "B":b, ...}`
        """
        if isinstance(out, Iterable) and not isinstance(out, str):
            return {name + ' ' + str(i): e for i, e in enumerate(out)}
        else:
            return {name: out}

    def __call__(self, subject: Subject) -> Dict:
        pred, true = self.parse_data(subject)
        confusion_matrix = compute_confusion_matrix(
            pred,
            true,
            self.n_labels
        )
        result = {'Subject':subject['name']}
        for name in self.metrics:
            func = VoxelMethods[name]
            out = func(
                confusion_matrix,
                self.bg_ignore,
                self.smooth
            )
            out = self.parse_result(name,out)
            result.update(out)
        return result

