from typing import Callable, Optional, Union, Tuple

import torch
import torch.cuda.amp as amp
from torch import Tensor
from torch.nn import Module
from torch.optim import Optimizer
from torch.utils.data import DataLoader, IterDataPipe
from tqdm import tqdm
import time

from .watchdog import WatchDog, Sniffer
from .writer import InferenceWriter


class Trainer(object):
    '''定义训练器的基础属性和行为'''

    def __init__(self,
                 model: Module,
                 sniffer: Sniffer,
                 gpu: torch.device,
                 ) -> None:
        self.model = model.to(gpu)
        self.watcher = WatchDog(sniffer)
        self.device = gpu

    def reload_model(self, state_dict: dict) -> None:
        self.model.load_state_dict(state_dict)

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

    def inference_batch(data, **kwargs) -> Tensor:
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
            self.watcher.catch(loss.item(), pred, target, mode="train")

    def validate(self, dataloader: DataLoader, loss_fun: Callable, info: Optional[str] = None):
        self.model.eval()
        progress = tqdm(dataloader, desc=info)
        for subject in progress:
            data, target = self.parse_data_validate(subject)
            with torch.no_grad():
                loss, pred = self.forward_iterate(data, target, loss_fun)
            progress.set_description("val loss: %.6f" % loss.item())
            self.watcher.catch(loss.item(), pred, target, mode="validate")

    def inference(self, dataloader: Union[DataLoader, IterDataPipe], writer:InferenceWriter, **kwargs):
        self.model.eval()
        for subject in tqdm(dataloader):
            data, meta = self.parse_data_inference(subject)
            with torch.no_grad():
                with amp.autocast():
                    pred = self.inference_batch(data, **kwargs)
            if writer:
                writer.save(meta, pred)

    def epoch_step(self):
        return self.watcher.step()


    def save(self,location:str):
        pass

    def state_dict(self):
        state={
            'watcher':self.watcher.state_dict()
        }
        return state

    def load_state(self,state):
        self.watcher.load_state(state['watcher'])




class CascadedTrainer(Trainer):
    '''通用级联网络训练器'''
    pass
