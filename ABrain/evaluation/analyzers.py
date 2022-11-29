from torchio import Subject
from typing import Tuple

from .base import Analyzer,tio
from .functional import *


class CompareAnalyzer(Analyzer):
    def __init__(self, name: str) -> None:
        super().__init__(name)

    def parse_data(self, subject: Subject) -> Tuple[Tensor]:
        pred: Tensor = subject['pred'][tio.DATA]
        true: Tensor = subject['true'][tio.DATA]
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
