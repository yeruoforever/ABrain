import os

import torch
import numpy as np


def collect(target: dict, data: dict):
    for k, v in data.items():
        if k not in target:
            target[k] = [v]
        else:
            target[k].append(v)


class Sniffer(object):
    def __init__(self) -> None:
        pass

    def is_better(self, current, history) -> bool:
        return False

    def sniffing(self, pred, target) -> dict:
        return {}


class WatchDog(object):
    '''在训练及验证阶段，用于收集和管理相关数据和指标'''

    def __init__(self, sniffer: Sniffer) -> None:
        self.nose = sniffer
        self._kennel = {
            "train": {},
            "validate": {},
            "epochs": {
                "train": {},
                "validate": {}
            },
            "current": {}
        }
        self._happy = False
        self._st_t = 0
        self._st_v = 0
        self._cur_t = 0
        self._cur_v = 0

    def catch(self, loss, pred, target, mode: str = 'train'):
        smell = self.nose.sniffing(pred, target)
        smell['loss'] = loss
        collect(self._kennel[mode], smell)
        if mode == 'train':
            self._cur_t += 1
        else:
            self._cur_v += 1

    def step(self):
        current = {}
        for k, v in self._kennel["train"].items():
            current[k] = np.array(v[self._st_v:]).mean()
        collect(self._kennel['epochs']['train'], current)
        current = {}
        for k, v in self._kennel["validate"].items():
            current[k] = np.array(v[self._st_v:]).mean()
        collect(self._kennel['epochs']['validate'], current)
        if len(self._kennel["current"]) == 0:
            self._kennel["current"] = current
        else:
            self._happy = self.nose.is_better(current, self._kennel["current"])
            if self._happy:
                self._kennel["current"] = current
        self._st_v = self._cur_v
        self._st_t = self._cur_t
        return current

    def happy(self) -> bool:
        flag = self._happy
        self._happy = False
        return flag

    def current(self):
        return self._kennel["current"]

    def state_dict(self):
        return {
            "kennel": self._kennel,
            "st_t": self._st_t,
            "st_v": self._st_v,
            "cur_t": self._cur_t,
            "cur_v": self._cur_v,
        }

    def load_state(self, state: dict):
        self._kennel = state['kennel']
        self._st_t = state['st_t']
        self._st_v = state['st_v']
        self._cur_t = state['cur_t']
        self._cur_v = state['cur_v']
