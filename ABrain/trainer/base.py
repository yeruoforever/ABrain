from typing import Callable, Optional, Union, Tuple

import torch
import torch.cuda.amp as amp
from torch import Tensor
from torch.nn import Module
from torch.optim import Optimizer
from torch.utils.data import DataLoader, IterDataPipe
from tqdm import tqdm

from .watchdog import WatchDog


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

    def reload_model(self, model: Module) -> None:
        self.model = model.to(self.device)

    def parse_data_train(package: dict) -> Tuple:
        """
        ### Returns:
            - `data`
            - `target`
        """
        raise NotImplementedError()

    def parse_data_validate(package: dict) -> Tuple:
        """
        ### Returns:
            - `data`
            - `target`
        """
        raise NotImplementedError()

    def parse_data_inference(package: dict) -> Tuple:
        """
        ### Returns:
            - `data`
            - `meta`
        """
        raise NotImplementedError()

    def forward_iterate(data: Tensor, target: Tensor, loss_fun: Callable) -> Tuple:
        '''
        ### Returns:
            - `loss`
            - `predict`
        '''
        raise NotImplementedError()

    def inference_batch(data, **kwargs)->Tensor:
        raise NotImplementedError()

    def train(self, dataloader: DataLoader, loss_fun: Callable, optimer: Optimizer, info: Optional[str] = None):
        scaler = amp.GradScaler()
        self.model.train()
        progress = tqdm(dataloader, desc=info)
        for subject in progress:
            data, target = self.parse_data_train(subject)
            with amp.autocast():
                loss, pred = self.forward_iterate(data, target, loss_fun)
            optimer.zero_grad()  
            scaler.scale(loss).backward()
            scaler.step(optimer)
            scaler.update()
            progress.set_description("train loss: %.6f" % loss.item())
            # TODO record states for each iters.
            # self.train_callback(pred,target)

    def validate(self, dataloader: DataLoader, loss_fun: Callable, info: Optional[str] = None):
        self.model.eval()
        progress = tqdm(dataloader, desc=info)
        for subject in progress:
            data, target = self.parse_data_validate(subject)
            with torch.no_grad():
                loss, pred = self.forward_iterate(data, target, loss_fun)
            progress.set_description("val loss: %.6f" % loss.item())
        # TODO
        # self.validate_callback(pred,target)

    def inference(self, dataloader: Union[DataLoader, IterDataPipe], writer, **kwargs):
        self.model.eval()
        for subject in tqdm(dataloader):
            data, meta = self.parse_data_inference(subject)
            with torch.no_grad():
                with amp.autocast():
                    pred = self.inference_batch(data, **kwargs)
            if writer:
                writer.save(meta, pred)

    def epoch_step(self):
        self.watcher.epoch_end()


class CascadedTrainer(Trainer):
    '''通用级联网络训练器'''
    pass
