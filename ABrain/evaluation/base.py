import multiprocessing
import os
from typing import Callable, Dict, Iterable, Optional, Tuple

import pandas as pds
import torchio as tio
from torch import Tensor
from torchio import LabelMap
from torchio.data import Subject
from tqdm import tqdm


class Analyzer(object):
    def __init__(self, name: str) -> None:
        self.name = name

    def analyze(self, subject: Subject):
        """用于实现对`subject`的分析"""
        class_name = self.__class__.__name__
        why = "%s must implement `self.analyze(subject)`!" % (class_name)
        raise NotImplementedError(why)

    def parse_result(self, out):
        """拼接分析结果。

        `{"Subject":sid, "A":a, "B":b, ...}`
        """
        if isinstance(out, Iterable):
            return {self.name + ' ' + str(i): e for i, e in enumerate(out)}
        else:
            return {self.name: out}

    def __call__(self, subject: Subject) -> Dict:
        '''执行分析并返回结果
        
        subject --> analyze --> parse_result --> result
        '''
        out = self.analyze(subject)
        out = self.parse_result(out)
        result = {'Subject': subject['name']}
        result.update(out)
        return result


class ComposeAnalyzer(Analyzer):
    def __init__(self,
                 *analyzers: Iterable[Analyzer]
    ) -> None:
        super().__init__(name="ComposeAnalyzer")
        self.analyzers = analyzers

    def __call__(self, subject) -> Dict:
        ans = {}
        for analyzer in self.analyzers:
            ans.update(analyzer(subject))
        return ans


class Evaluator(object):
    def __init__(
            self,
            dir_pred: str,
            dir_true: str,
            analyzes: Iterable[Analyzer]
    ) -> None:
        self.df = pds.DataFrame()
        self.dir_pred = dir_pred
        self.dir_true = dir_true
        self.analyzes = analyzes

    def get_sids(self):
        sids = os.listdir(self.dir_true)
        sids = list(map(lambda x: x.split('.nii.gz')[0], sids))
        return sids

    def get_file_name(self, sid: str):
        return sid+'.nii.gz'

    def get_pred(self, sid: str):
        '''只返回文件名'''
        return self.get_file_name(sid)

    def get_true(self, sid: str):
        '''只返回文件名'''
        return self.get_file_name(sid)

    def show(self):
        print(self.df)

    def save(self, path: str):
        self.df.to_csv(path)

    def analyze_one(self, sid, pred, true):
        subject = Subject(
            name=sid,
            pred=LabelMap(os.path.join(self.dir_pred, pred)),
            true=LabelMap(os.path.join(self.dir_true, true))
        )
        if isinstance(self.analyzes, Callable):
            msgs = self.analyzes(subject)
        else:
            msgs = {}
            for analyze in self.analyzes:
                msgs.update(analyze(subject))
        return msgs

    def run(self, num_works: Optional[int] = None):
        sids = self.get_sids()
        preds = list(map(self.get_pred, sids))
        trues = list(map(self.get_true, sids))
        results = []
        for sid, pred, true in tqdm(zip(sids, preds, trues)):
            msgs = self.analyze_one(sid, pred, true)
            results.append(msgs)
        # TODO 多进程处理，可能存在死锁
        # pool = multiprocessing.Pool(num_works)
        # results = pool.starmap(self.analyze_one, zip(sids, preds, trues))
        # pool.close()
        self.df = pds.DataFrame(results)
        self.show()


class CompareAnalyzer(Analyzer):
    def __init__(self, name: str) -> None:
        super().__init__(name)

    def parse_data(self, subject: Subject) -> Tuple[Tensor]:
        pred: Tensor = subject['pred'][tio.DATA]
        true: Tensor = subject['true'][tio.DATA]
        pred = pred.long()
        true = true.long()
        return pred, true
