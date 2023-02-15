import multiprocessing
import os
from typing import Callable, Dict, Iterable, Optional, Tuple

import pandas as pds
import torchio as tio
from torch import Tensor
from torchio import LabelMap, ScalarImage
from torchio.data import Subject
from tqdm import tqdm

from ..dataset import OurDataset
from ..trainer.writer import InferenceWriter


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
        if isinstance(out, Iterable) and not isinstance(out, str):
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
            dataset: OurDataset,
            dir_pred: str,
            analyzes: Iterable[Analyzer],
            process: tio.Transform,
    ) -> None:
        self.df = pds.DataFrame()
        self.dir_pred = dir_pred
        self.analyzes = analyzes
        self.process = process
        self.dataset = dataset

    def get_file_name(self, sid: str):
        file =  sid+'.pred.nii.gz'
        return os.path.join(self.dir_pred, file)

    def get_pred(self, sid: str):
        '''返回预测文件完整路径'''
        return self.get_file_name(sid)

    def get_true(self, sid: str):
        '''返回文件完整路径'''
        return self.dataset.seg_file(sid)

    def show(self):
        print(self.df)
        print(self.df.mean())

    def save(self, path: str):
        self.df.to_csv(path)

    def analyze_one(self, sid, img, pred, true):
        subject = Subject(
            name=sid,
            img=ScalarImage(img),
            pred=LabelMap(pred).tensor,
            true=LabelMap(true).tensor
        )
        if self.process:
            subject = self.process(subject)
        if isinstance(self.analyzes, Callable):
            msgs = self.analyzes(subject)
        else:
            msgs = {}
            for analyze in self.analyzes:
                msgs.update(analyze(subject))
        return msgs

    def run(self, num_works: Optional[int] = None, ):
        sids = self.dataset.sids
        imgs = list(map(self.dataset.img_file, sids))
        preds = list(map(self.get_pred, sids))
        trues = list(map(self.get_true, sids))
        results = []
        for sid, img, pred, true in tqdm(zip(sids,imgs, preds, trues)):
            msgs = self.analyze_one(sid, img, pred, true)
            results.append(msgs)
        # TODO 多进程处理，可能存在死锁
        # pool = multiprocessing.Pool(num_works)
        # results = pool.starmap(self.analyze_one, zip(sids, preds, trues))
        # pool.close()
        self.df = pds.DataFrame(results)
        self.show()
