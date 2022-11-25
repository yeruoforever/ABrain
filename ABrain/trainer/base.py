from typing import Callable, Optional, Union

import torch
import torch.amp as amp
import torchio as tio
from torch.optim import Optimizer
from torch.nn import Module
from torch.utils.data import DataLoader, IterDataPipe
from tqdm import tqdm

from .watchdog import WatchDog


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


class Trainer(object):
    '''定义训练器的基础属性和行为'''

    def __init__(self,
                 model: Module,
                 watcher: WatchDog,
                 gpu: torch.device,
                 ) -> None:
        self.model = model.to(gpu)
        self.watcher = watcher
        self.device = gpu

    def reload_model(self, model: Module):
        self.model = model.to(self.device)


    def train(self, dataloader: DataLoader, loss_fun: Callable, optimer: Optimizer, info: Optional[str] = None):
        scaler = amp.GradScaler()
        self.model.train()
        progress = tqdm(dataloader, desc=info)
        for subject in progress:
            data, target = self.parse_data_train(subject)
            with amp.autocast():
                loss, pred = self.forward_iterate(data, target, loss_fun)
            scaler.scale(loss).backward()
            scaler.step(optimer)
            scaler.update()
            optimer.zero_grad()  # TODO
            progress.set_description("train loss: %.6f" % loss.item())
            # TODO
            # self.train_metrics(pred,target)

    def validate(self, dataloader: DataLoader, loss_fun: Callable, info: Optional[str] = None):
        self.model.eval()
        progress = tqdm(dataloader, desc=info)
        for subject in progress:
            data, target = self.parse_data_validate(subject)
            with torch.no_grad():
                loss, pred = self.forward_iterate(data, target, loss_fun)
            progress.set_description("val loss: %.6f" % loss.item())
            # TODO
            # self.validate_metrics(pred,target)

    def inference(self, dataloader: Union[DataLoader, IterDataPipe], patch_size, stride, writer):
        self.model.eval()
        for subject in tqdm(dataloader):
            name, (data, affine) = self.parse_data_inference(subject)
            with torch.no_grad():
                with amp.autocast():
                    pred = self.inference_by_patch(data, patch_size, stride)
            if writer:
                writer.save(name, pred, affine)

    def epoch_step(self):
        self.watcher.epoch_end()


class SegmentationTrainer(Trainer):
    '''通用基础训练器'''
    def __init__(self,
                 model: Module,
                 n_label: int,
                 watcher: WatchDog,
                 gpu: torch.device,
                 deep_supervise: bool = False,
                 ) -> None:
        self.model = model.to(gpu)
        self.watcher = watcher
        self.device = gpu
        self.deep_supervise = deep_supervise
        self.n_label = n_label


    def parse_data(self, subject):
        data = subject["img"][tio.DATA]
        target = subject["seg"][tio.DATA].long().squeeze(1)
        data = data.to(self.device)
        target = target.to(self.device)
        return data, target

    def parse_data_train(self, subject):
        return self.parse_data(subject)

    def parse_data_validate(self, subject):
        return self.parse_data(subject)

    def parse_data_inference(self, subject):
        name = subject['name']
        data = subject["img"][tio.DATA]
        data = data.to(self.device)
        affine = subject['img'][tio.AFFINE]
        return name, (data, affine)

    def forward_iterate(self, data, target, loss_fun):
        out = self.model(data)
        if self.deep_supervise:
            loss_out = loss_fun(out[0], target)*0.5
            loss_ds1 = loss_fun(out[1], target)*0.07
            loss_ds2 = loss_fun(out[2], target)*0.14
            loss_ds3 = loss_fun(out[3], target)*0.29
            loss = loss_out+loss_ds1+loss_ds2+loss_ds3
            pred = out[0]
        # elif fine_coarse:
        #     loss_fine = loss_fun(out[0], target)
        #     loss_coarse = loss_fun(out[1], target)
        #     loss = loss_fine+loss_coarse
        #     pred = out[0]
        else:
            loss = loss_fun(out[0], target)
            pred = out[0]
        return loss, pred

    def make_patch(self, bounds, patch, stride):
        W, H, D = bounds
        for i in _patch_range(0, W, stride[0], patch[0]):
            for j in _patch_range(0, H, stride[1], patch[1]):
                for k in _patch_range(0, D, stride[2], patch[2]):
                    yield (i, j, k)

    def inference_iterate(self, data):
        # data = data.to(self.device)
        if self.deep_supervise:
            out = self.model(data)[0]
        # elif fine_coarse:
        #     loss_fine = loss_fun(out[0], target)
        #     loss_coarse = loss_fun(out[1], target)
        #     loss = loss_fine+loss_coarse
        #     pred = out[0]
        else:
            out = self.model(data)
        return out

    def inference_by_patch(self, data, patch_size, stride):
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
        return pred.div_(cnt).argmax(dim=1)

    def epoch_step(self):
        self.watcher.epoch_end()


class ClassifyTrainer(Trainer):
    pass 

class DetectionTrainer(Trainer):
    pass

class CascadedTrainer(Trainer):
    '''通用级联网络训练器'''
    pass
