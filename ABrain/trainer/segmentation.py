from collections import namedtuple
from torch.nn import Module
from .base import Trainer
from .watchdog import WatchDog
import torch
from torch import Tensor
import torchio as tio


Meta = namedtuple("SegmentationMeta", ["name", "affine"])


def _check_bounds(b):
    assert isinstance(b, (tuple, int)), "Bounds must be `int` or `tuple[int]`"
    if isinstance(b, int):
        return tuple(b for b in range(3))
    return b


def _patch_range(start, end, step, width):
    i = start
    while i < end:
        if i+width < end:
            yield i
        else:
            yield end-width
        i = i+step


class SegmentationTrainer(Trainer):
    '''通用基础训练器'''

    def __init__(self,
                 model: Module,
                 n_label: int,
                 watcher: WatchDog,
                 gpu: torch.device,
                 deep_supervise: bool = False,
                 ) -> None:
        super().__init__(model, watcher, gpu)
        self.deep_supervise = deep_supervise
        self.n_label = n_label

    def parse_data(self, subject):
        data:Tensor = subject["img"][tio.DATA].float()
        target:Tensor = subject["seg"][tio.DATA].long().squeeze(1)
        data = data.to(self.device)
        target = target.to(self.device)
        return data, target

    def parse_data_train(self, subject):
        return self.parse_data(subject)

    def parse_data_validate(self, subject):
        return self.parse_data(subject)

    def parse_data_inference(self, subject):
        name = subject['name']
        data:Tensor = subject["img"][tio.DATA]
        data:Tensor = data.to(self.device)
        affine = subject['img'][tio.AFFINE]
        return data, Meta(name=name, affine=affine)

    def forward_iterate(self, data, target, loss_fun):
        state, out = self.model(data)
        if self.deep_supervise:
            # [0.5, 0.07, 0.14, 0,29]
            loss_out = loss_fun(out, target)*0.5
            loss_ds1 = loss_fun(state[0], target)*0.07
            loss_ds2 = loss_fun(state[1], target)*0.14
            loss_ds3 = loss_fun(state[2], target)*0.29
            loss = loss_out+loss_ds1+loss_ds2+loss_ds3
            pred = out
        # elif fine_coarse:
        #     loss_fine = loss_fun(ou,t[0], target)
        #     loss_coarse = loss_fun(out[1], target)
        #     loss = loss_fine+loss_coarse
        #     pred = out[0]
        else:
            loss = loss_fun(out, target)
            pred = out
        return loss, pred

    def make_patch(self, bounds, patch, stride):
        W, H, D = bounds
        for i in _patch_range(0, W, stride[0], patch[0]):
            for j in _patch_range(0, H, stride[1], patch[1]):
                for k in _patch_range(0, D, stride[2], patch[2]):
                    yield (i, j, k)

    def inference_iterate(self, data):
        state, out = self.model(data)
        if self.deep_supervise:
            return out
        # elif fine_coarse:
        #     loss_fine = loss_fun(out[0], target)
        #     loss_coarse = loss_fun(out[1], target)
        #     loss = loss_fine+loss_coarse
        #     pred = out[0]
        else:
            return out

    def inference_batch(self, data, patch_size, stride):
        patch = _check_bounds(patch_size)
        stride = _check_bounds(stride)
        C = self.n_label
        B, _, W, H, D = data.shape
        w, h, d = patch
        pred = torch.zeros((B, C, W, H, D), dtype=float, device=self.device)
        cnt = torch.zeros((B, C, W, H, D), dtype=float, device=self.device)
        patchs = self.make_patch((W, H, D), patch, stride)
        for i, j, k in patchs:
            out = self.inference_iterate(data[:, :, i:i+w, j:j+h, k:k+d])
            pred[:, :, i:i+w, j:j+h, k:k+d].add_(out)
            cnt[:, :, i:i+w, j:j+h, k:k+d].add_(1.)
        return pred.div_(cnt)

